"""Dolma context collection for shortlisted transcript features."""

from __future__ import annotations

import heapq
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from thesis_neuro.config import AppConfig
from thesis_neuro.datasets import DolmaStreamingAdapter, RawDocument
from thesis_neuro.models import GemmaModelAdapter, WindowBatch
from thesis_neuro.pipelines.extraction import DolmaContextStats, ExtractionPipeline
from thesis_neuro.sae import GemmaScopeAdapter
from thesis_neuro.storage import JsonlArtifactStore


class DolmaContextCollectionPipeline(ExtractionPipeline):
    def __init__(self, config: AppConfig, shortlist_path: str | None = None) -> None:
        self.config = config
        self.dataset = DolmaStreamingAdapter(config)
        self.model = GemmaModelAdapter(config)
        self.sae = GemmaScopeAdapter(config, device=self.model.device, dtype=self.model.dtype)
        self.store = JsonlArtifactStore(config.output_dir)
        self.shortlist_path = (
            Path(shortlist_path)
            if shortlist_path is not None
            else self.store.transcript_feature_shortlist_path
        )

    def run(self) -> dict[str, Any]:
        if not self.config.dolma_query.enabled:
            raise ValueError("dolma_query.enabled is false; enable it before collecting Dolma contexts.")
        if not self.shortlist_path.exists():
            raise FileNotFoundError(
                f"Transcript shortlist not found: {self.shortlist_path}. "
                "Run discover-transcript-features first or pass --shortlist."
            )

        stats = DolmaContextStats()
        print(f"[dolma] loading model {self.config.model.base_model_id}")
        model_info = self.model.describe_model()
        print(f"[dolma] resolving SAE layers for {self.config.model.scope_release}")
        available_layers = set(self.sae.available_layers(model_info.get("num_hidden_layers")))
        selected_by_layer = self._load_shortlist()
        selected_by_layer = {
            layer: feature_ids
            for layer, feature_ids in selected_by_layer.items()
            if layer in available_layers and feature_ids
        }
        self.store.reset_dolma_contexts_file()

        heaps: dict[tuple[int, int], list[tuple[float, str, dict[str, Any]]]] = {}
        entry_counter = 0

        for raw_document in self.dataset.stream_documents():
            if stats.windows_seen >= self.config.dolma_query.max_windows:
                break
            stats.documents_seen += 1
            token_ids = self.model.tokenize_document(raw_document.text)
            windows = self.model.make_windows(token_ids)
            for window_idx, window in enumerate(windows):
                if stats.windows_seen >= self.config.dolma_query.max_windows:
                    break
                stats.windows_seen += 1
                sample_id = f"{raw_document.doc_id}:{window.window_start}:{window.window_end}:{window_idx}"
                outputs, _ = self.model.forward_outputs(window.input_ids, require_grad=False)
                hidden_states = outputs.hidden_states
                for layer_idx, feature_ids in selected_by_layer.items():
                    residual = hidden_states[layer_idx + 1]
                    latents = self.sae.encode_layer(layer_idx, residual).squeeze(0).detach().to("cpu").clamp_min(0)
                    for feature_id in feature_ids:
                        if feature_id < 0 or feature_id >= latents.shape[-1]:
                            continue
                        token_values = latents[:, feature_id]
                        total_activation = float(token_values.sum().item())
                        peak_activation = float(token_values.max().item())
                        if max(total_activation, peak_activation) <= self.config.dolma_query.min_activation_threshold:
                            continue
                        active_fraction = float((token_values > 0).float().mean().item())
                        context_row = self._build_dolma_context_row(
                            raw_document=raw_document,
                            sample_id=sample_id,
                            window=window,
                            layer_idx=layer_idx,
                            feature_id=feature_id,
                            token_values=token_values,
                            total_activation=total_activation,
                            peak_activation=peak_activation,
                            active_fraction=active_fraction,
                        )
                        key = (layer_idx, feature_id)
                        heap = heaps.setdefault(key, [])
                        heapq.heappush(heap, (total_activation, f"{sample_id}:{entry_counter}", context_row))
                        if len(heap) > self.config.dolma_query.top_contexts_per_feature:
                            heapq.heappop(heap)
                        entry_counter += 1

        context_rows: list[dict[str, Any]] = []
        for key in sorted(heaps):
            contexts = sorted(heaps[key], key=lambda item: item[0], reverse=True)
            for _, _, row in contexts:
                context_rows.append(row)
                self.store.append_dolma_context_row(row)
                stats.contexts_written += 1

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "collect_dolma_contexts",
            "config": self.config.to_dict(),
            "model": model_info,
            "documents_seen": stats.documents_seen,
            "windows_seen": stats.windows_seen,
            "contexts_written": stats.contexts_written,
            "artifacts": {
                "transcript_feature_shortlist": str(self.shortlist_path),
                "dolma_feature_contexts": str(self.store.dolma_contexts_path),
                "manifest": str(self.store.manifest_path),
            },
        }
        if self.config.output.write_manifest:
            self.store.write_manifest(manifest)
        return manifest

    def _load_shortlist(self) -> dict[int, list[int]]:
        selected_by_layer: dict[int, list[int]] = defaultdict(list)
        for row in self.store.iter_records(self.shortlist_path):
            selected_by_layer[int(row["layer"])].append(int(row["feature_id"]))
        return selected_by_layer

    def _build_dolma_context_row(
        self,
        raw_document: RawDocument,
        sample_id: str,
        window: WindowBatch,
        layer_idx: int,
        feature_id: int,
        token_values: torch.Tensor,
        total_activation: float,
        peak_activation: float,
        active_fraction: float,
    ) -> dict[str, Any]:
        top_token_positions = self._top_token_positions(token_values)
        top_token_snippets = self._top_token_snippets(window, token_values, top_token_positions)
        top_span_snippets = self._top_span_snippets(window, token_values)
        top_sentence_snippets = self._top_sentence_snippets(window, token_values)
        scale_scores = {
            "token": max(
                [float(item.get("activation", 0.0)) for item in top_token_snippets],
                default=0.0,
            ),
            "span": max(
                [float(item.get("activation_total", 0.0)) for item in top_span_snippets],
                default=0.0,
            ),
            "sentence": max(
                [float(item.get("activation_total", 0.0)) for item in top_sentence_snippets],
                default=0.0,
            ),
            "pooled_window": float(total_activation),
        }
        return {
            "sample_id": sample_id,
            "model_id": self.config.model.base_model_id,
            "scope_release": self.config.model.scope_release,
            "layer": layer_idx,
            "feature_id": feature_id,
            "feature_activation_total": total_activation,
            "feature_activation_peak": peak_activation,
            "active_token_fraction": active_fraction,
            "window_text": window.text,
            "window_tokens": window.tokens,
            "top_token_positions": top_token_positions,
            "top_token_snippets": top_token_snippets,
            "top_span_snippets": top_span_snippets,
            "top_sentence_snippets": top_sentence_snippets,
            "dominant_scale": self._dominant_dolma_scale(scale_scores),
            "scale_scores": scale_scores,
            "document_snippet": window.text,
            "provenance": raw_document.provenance,
        }

    @staticmethod
    def _dominant_dolma_scale(scale_scores: dict[str, float]) -> str:
        return max(
            scale_scores.items(),
            key=lambda item: (float(item[1]), item[0] == "pooled_window"),
        )[0]

    def _top_token_positions(self, token_values: torch.Tensor) -> list[int]:
        k = min(self.config.dolma_query.top_tokens_per_context, token_values.shape[0])
        values, indices = torch.topk(token_values, k=k)
        positions: list[int] = []
        for activation, index in zip(values.tolist(), indices.tolist()):
            if activation <= 0:
                continue
            positions.append(int(index))
        return positions

    def _top_token_snippets(
        self,
        window: WindowBatch,
        token_values: torch.Tensor,
        positions: list[int],
    ) -> list[dict[str, Any]]:
        snippets: list[dict[str, Any]] = []
        radius = self.config.dolma_query.context_window_tokens
        for position in positions:
            snippets.append(
                {
                    "token_position": position,
                    "token": window.tokens[position],
                    "activation": float(token_values[position].item()),
                    "snippet_tokens": self._token_snippet(window.tokens, position, radius),
                }
            )
        return snippets

    def _top_span_snippets(
        self,
        window: WindowBatch,
        token_values: torch.Tensor,
    ) -> list[dict[str, Any]]:
        active_positions = [idx for idx, value in enumerate(token_values.tolist()) if value > 0]
        if not active_positions:
            return []

        spans: list[dict[str, Any]] = []
        current = [active_positions[0]]
        for position in active_positions[1:]:
            if position == current[-1] + 1:
                current.append(position)
                continue
            spans.append(self._span_snippet(window, token_values, current))
            current = [position]
        spans.append(self._span_snippet(window, token_values, current))
        ranked = sorted(spans, key=lambda item: item["activation_total"], reverse=True)
        return ranked[: self.config.dolma_query.top_spans_per_context]

    def _span_snippet(
        self,
        window: WindowBatch,
        token_values: torch.Tensor,
        positions: list[int],
    ) -> dict[str, Any]:
        start = positions[0]
        end = positions[-1]
        return {
            "start_token_position": start,
            "end_token_position": end,
            "tokens": window.tokens[start : end + 1],
            "activation_total": float(token_values[start : end + 1].sum().item()),
            "activation_peak": float(token_values[start : end + 1].max().item()),
        }

    def _top_sentence_snippets(
        self,
        window: WindowBatch,
        token_values: torch.Tensor,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for sentence in window.sentence_spans:
            start = int(sentence["start_token_index"])
            end = int(sentence["end_token_index"]) + 1
            sentence_values = token_values[start:end]
            total = float(sentence_values.sum().item())
            if total <= 0:
                continue
            rows.append(
                {
                    "sentence_id": int(sentence["sentence_id"]),
                    "start_token_position": start,
                    "end_token_position": end - 1,
                    "tokens": window.tokens[start:end],
                    "text": self.model.tokenizer.decode(window.input_ids[start:end], skip_special_tokens=False),
                    "activation_total": total,
                    "activation_peak": float(sentence_values.max().item()),
                }
            )
        ranked = sorted(rows, key=lambda item: item["activation_total"], reverse=True)
        return ranked[: self.config.dolma_query.top_sentences_per_context]

