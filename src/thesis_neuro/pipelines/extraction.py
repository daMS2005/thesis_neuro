from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import torch

from thesis_neuro.config import AppConfig
from thesis_neuro.datasets import DolmaStreamingAdapter, RawDocument
from thesis_neuro.models import GemmaModelAdapter, WindowBatch
from thesis_neuro.pipelines.summary import summarize_feature_records
from thesis_neuro.sae import GemmaScopeAdapter
from thesis_neuro.storage import JsonlArtifactStore


@dataclass(slots=True)
class ExtractionStats:
    documents_seen: int = 0
    windows_seen: int = 0
    records_written: int = 0


@dataclass(slots=True)
class AlignmentStats:
    window_layers_seen: int = 0
    features_written: int = 0


@dataclass(slots=True)
class TranscriptDiscoveryStats:
    documents_seen: int = 0
    windows_seen: int = 0
    records_written: int = 0
    features_written: int = 0
    shortlist_written: int = 0


@dataclass(slots=True)
class DolmaContextStats:
    documents_seen: int = 0
    windows_seen: int = 0
    contexts_written: int = 0


class ExtractionPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.dataset = DolmaStreamingAdapter(config)
        self.model = GemmaModelAdapter(config)
        self.sae = GemmaScopeAdapter(config, device=self.model.device, dtype=self.model.dtype)
        self.store = JsonlArtifactStore(config.output_dir)

    def run(self) -> dict[str, Any]:
        stats = ExtractionStats()
        print(f"[extract] loading model {self.config.model.base_model_id}")
        model_info = self.model.describe_model()
        print(f"[extract] resolving SAE layers for {self.config.model.scope_release}")
        available_layers = self.sae.available_layers(model_info.get("num_hidden_layers"))
        self.store.reset_run_files(include_summary=self.config.output.write_summary)

        for raw_document in self.dataset.stream_documents():
            if stats.windows_seen >= self.config.dataset.max_windows:
                break
            stats.documents_seen += 1
            print(f"[extract] tokenizing document {stats.documents_seen}: {raw_document.doc_id}")

            token_ids = self.model.tokenize_document(raw_document.text)
            windows = self.model.make_windows(token_ids)
            for window_idx, window in enumerate(windows):
                if stats.windows_seen >= self.config.dataset.max_windows:
                    break
                print(
                    f"[extract] processing window {stats.windows_seen + 1}/"
                    f"{self.config.dataset.max_windows} for {raw_document.doc_id}"
                )
                token_records = self._process_window(raw_document, window_idx, window, model_info)
                for record in token_records:
                    self.store.append_record(record)
                    stats.records_written += 1
                stats.windows_seen += 1

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config.to_dict(),
            "model": model_info,
            "available_layers": available_layers,
            "documents_seen": stats.documents_seen,
            "windows_seen": stats.windows_seen,
            "records_written": stats.records_written,
            "artifacts": {
                "paired_records": str(self.store.paired_path),
                "manifest": str(self.store.manifest_path),
            },
        }
        if self.config.output.write_manifest:
            self.store.write_manifest(manifest)
        if self.config.output.write_summary:
            summary_rows = summarize_feature_records(self.store.iter_records())
            self.store.write_summary(summary_rows)
            manifest["artifacts"]["feature_summary"] = str(self.store.summary_path)
            if self.config.output.write_manifest:
                self.store.write_manifest(manifest)
        return manifest

    def _process_window(
        self,
        raw_document: RawDocument,
        window_idx: int,
        window: WindowBatch,
        model_info: dict[str, Any],
    ) -> list[dict[str, Any]]:
        outputs, _ = self.model.forward_outputs(window.input_ids, require_grad=False)
        hidden_states = outputs.hidden_states
        logits = outputs.logits
        sample_id = f"{raw_document.doc_id}:{window.window_start}:{window.window_end}:{window_idx}"
        token_records: list[dict[str, Any]] = []

        for layer_idx in self.sae.available_layers(model_info.get("num_hidden_layers")):
            residual = hidden_states[layer_idx + 1]
            latents = self.sae.encode_layer(layer_idx, residual).squeeze(0).detach().to("cpu")
            sentence_feature_summaries = self._sentence_feature_summaries(
                latents=latents,
                sentence_spans=window.sentence_spans,
            )
            window_feature_summaries = self._pooled_feature_summaries(latents)
            for token_position, token_id in enumerate(window.input_ids):
                positive_token_id = (
                    window.input_ids[token_position + 1]
                    if token_position + 1 < len(window.input_ids)
                    else None
                )
                top_logits = self._top_logits(logits[0, token_position], self.config.latents.top_n_logits)
                negative_token_id = self._choose_negative_token_id(top_logits, positive_token_id)
                latent_records = self.sae.select_top_latents(
                    latents[token_position],
                    self.config.latents.token_top_k,
                )
                sentence_id = int(window.token_sentence_ids[token_position])
                sentence_span = self._sentence_span_lookup(window.sentence_spans, sentence_id)

                token_records.append(
                    {
                        "sample_id": sample_id,
                        "model_id": self.config.model.base_model_id,
                        "scope_release": self.config.model.scope_release,
                        "layer": layer_idx,
                        "token_position": token_position,
                        "token_id": token_id,
                        "token": window.tokens[token_position],
                        "window_token_ids": window.input_ids,
                        "window_tokens": window.tokens,
                        "text": window.text,
                        "window_start": window.window_start,
                        "window_end": window.window_end,
                        "sentence_id": sentence_id,
                        "sentence_start_token_index": sentence_span["start_token_index"],
                        "sentence_end_token_index": sentence_span["end_token_index"],
                        "token_sentence_ids": window.token_sentence_ids,
                        "window_sentences": window.sentence_spans,
                        "window_clauses": window.clause_spans,
                        "sentence_feature_summaries": sentence_feature_summaries.get(sentence_id, []),
                        "window_feature_summaries": window_feature_summaries,
                        "latent_activations": latent_records,
                        "top_logits": top_logits,
                        "chosen_positive_token_id": positive_token_id,
                        "chosen_negative_token_id": negative_token_id,
                        "provenance": raw_document.provenance,
                    }
                )
        return token_records

    def _pooled_feature_summaries(self, latents: torch.Tensor) -> list[dict[str, float | int]]:
        return self._feature_summaries_for_slice(latents)

    def _sentence_feature_summaries(
        self,
        latents: torch.Tensor,
        sentence_spans: list[dict[str, Any]],
    ) -> dict[int, list[dict[str, float | int]]]:
        summaries: dict[int, list[dict[str, float | int]]] = {}
        for sentence in sentence_spans:
            start = int(sentence["start_token_index"])
            end = int(sentence["end_token_index"]) + 1
            sentence_latents = latents[start:end]
            summaries[int(sentence["sentence_id"])] = self._feature_summaries_for_slice(sentence_latents)
        return summaries

    def _feature_summaries_for_slice(self, latents: torch.Tensor) -> list[dict[str, float | int]]:
        if latents.numel() == 0:
            return []
        positive = latents.clamp_min(0)
        total_activation = positive.sum(dim=0)
        max_activation = positive.max(dim=0).values
        mean_activation = positive.mean(dim=0)
        active_fraction = (positive > 0).float().mean(dim=0)

        top_n = min(self.config.latents.pooled_top_k, positive.shape[-1])
        by_total = torch.topk(total_activation, k=top_n).indices.tolist()
        by_max = torch.topk(max_activation, k=top_n).indices.tolist()
        ordered_feature_ids: list[int] = []
        seen: set[int] = set()
        for feature_id in by_total + by_max:
            feature_id = int(feature_id)
            if feature_id in seen:
                continue
            seen.add(feature_id)
            if total_activation[feature_id].item() <= 0 and max_activation[feature_id].item() <= 0:
                continue
            ordered_feature_ids.append(feature_id)
            if len(ordered_feature_ids) >= top_n:
                break

        return [
            {
                "feature_id": feature_id,
                "total_activation": float(total_activation[feature_id].item()),
                "max_activation": float(max_activation[feature_id].item()),
                "mean_activation": float(mean_activation[feature_id].item()),
                "active_token_fraction": float(active_fraction[feature_id].item()),
            }
            for feature_id in ordered_feature_ids
        ]

    @staticmethod
    def _sentence_span_lookup(
        sentence_spans: list[dict[str, Any]],
        sentence_id: int,
    ) -> dict[str, Any]:
        for span in sentence_spans:
            if int(span["sentence_id"]) == sentence_id:
                return span
        return sentence_spans[0]

    def _top_logits(self, token_logits: torch.Tensor, top_n: int) -> list[dict[str, float | int | str]]:
        k = min(top_n, token_logits.shape[-1])
        values, indices = torch.topk(token_logits.detach().to("cpu"), k=k)
        return [
            {
                "token_id": int(token_id),
                "token": self.model.tokenizer.convert_ids_to_tokens([int(token_id)])[0],
                "logit": float(logit),
            }
            for logit, token_id in zip(values.tolist(), indices.tolist())
        ]

    @staticmethod
    def _choose_negative_token_id(
        top_logits: list[dict[str, float | int | str]],
        positive_token_id: int | None,
    ) -> int | None:
        if positive_token_id is None:
            return None
        for candidate in top_logits:
            if int(candidate["token_id"]) != positive_token_id:
                return int(candidate["token_id"])
        return None

    @staticmethod
    def _token_snippet(tokens: list[str], token_position: int, radius: int) -> list[str]:
        start = max(0, token_position - radius)
        end = min(len(tokens), token_position + radius + 1)
        return tokens[start:end]


class PrefetchPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model = GemmaModelAdapter(config)
        self.sae = GemmaScopeAdapter(config, device=self.model.device, dtype=self.model.dtype)

    def run(self) -> dict[str, Any]:
        print(f"[prefetch] caching model {self.config.model.base_model_id}")
        model_info = self.model.prefetch()
        print(f"[prefetch] caching SAE weights from {self.config.model.scope_release}")
        cached_sae_ids = self.sae.prefetch_layers(model_info.get("num_hidden_layers"))
        return {
            "model": model_info,
            "scope_release": self.config.model.scope_release,
            "cached_sae_ids": cached_sae_ids,
            "local_files_only_ready": True,
        }

