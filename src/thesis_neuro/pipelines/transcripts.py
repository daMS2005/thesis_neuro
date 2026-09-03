"""Transcript-first feature discovery, aggregation, and shortlist construction."""

from __future__ import annotations

import heapq
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import torch

from thesis_neuro.config import AppConfig
from thesis_neuro.datasets import RawDocument, TranscriptDirectoryAdapter
from thesis_neuro.models import GemmaModelAdapter, WindowBatch
from thesis_neuro.pipelines.extraction import ExtractionPipeline, TranscriptDiscoveryStats
from thesis_neuro.sae import GemmaScopeAdapter
from thesis_neuro.storage import JsonlArtifactStore


class TranscriptFeatureDiscoveryPipeline(ExtractionPipeline):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.dataset = TranscriptDirectoryAdapter(config)
        self.model = GemmaModelAdapter(config)
        self.sae = GemmaScopeAdapter(config, device=self.model.device, dtype=self.model.dtype)
        self.store = JsonlArtifactStore(config.output_dir)

    def run(self) -> dict[str, Any]:
        stats = TranscriptDiscoveryStats()
        print(f"[transcripts] loading model {self.config.model.base_model_id}")
        model_info = self.model.describe_model()
        print(f"[transcripts] resolving SAE layers for {self.config.model.scope_release}")
        available_layers = self.sae.available_layers(model_info.get("num_hidden_layers"))
        self.store.reset_transcript_files()

        total_tokens_by_layer: dict[int, int] = defaultdict(int)
        feature_stats: dict[tuple[int, int], dict[str, Any]] = {}
        example_heaps: dict[tuple[int, int], list[tuple[float, str, dict[str, Any]]]] = {}
        entry_counter = 0

        for raw_document in self.dataset.stream_documents():
            stats.documents_seen += 1
            print(f"[transcripts] tokenizing transcript {stats.documents_seen}: {raw_document.doc_id}")
            token_ids = self.model.tokenize_document(raw_document.text)
            transcript_window_len = min(
                self.model.max_context_window_tokens(),
                self.config.tokenization.seq_len,
            )
            if len(token_ids) > transcript_window_len:
                print(
                    f"[transcripts] transcript {raw_document.doc_id} exceeds model context "
                    f"({len(token_ids)} > {transcript_window_len}); chunking is still required"
                )
            windows = self.model.make_windows(
                token_ids,
                metadata_mode="spacy",
                window_len=transcript_window_len,
            )
            for window_idx, window in enumerate(windows):
                stats.windows_seen += 1
                transcript_rows, window_updates, total_tokens = self._process_transcript_window(
                    raw_document=raw_document,
                    window_idx=window_idx,
                    window=window,
                    model_info=model_info,
                    entry_counter_start=entry_counter,
                )
                entry_counter += sum(len(update["examples"]) for update in window_updates.values())
                for row in transcript_rows:
                    self.store.append_transcript_record(row)
                    stats.records_written += 1
                for layer_idx, token_count in total_tokens.items():
                    total_tokens_by_layer[layer_idx] += token_count
                self._merge_feature_updates(
                    feature_stats=feature_stats,
                    example_heaps=example_heaps,
                    window_updates=window_updates,
                )

        stats_rows = self._build_transcript_feature_stats(
            feature_stats=feature_stats,
            total_tokens_by_layer=total_tokens_by_layer,
            example_heaps=example_heaps,
        )
        shortlist_rows = self._build_transcript_feature_shortlist(stats_rows)
        self.store.write_jsonl(self.store.transcript_feature_stats_path, stats_rows)
        self.store.write_jsonl(self.store.transcript_feature_shortlist_path, shortlist_rows)
        stats.features_written = len(stats_rows)
        stats.shortlist_written = len(shortlist_rows)

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "discover_transcript_features",
            "config": self.config.to_dict(),
            "model": model_info,
            "available_layers": available_layers,
            "documents_seen": stats.documents_seen,
            "windows_seen": stats.windows_seen,
            "records_written": stats.records_written,
            "features_written": stats.features_written,
            "shortlist_written": stats.shortlist_written,
            "artifacts": {
                "transcript_paired_records": str(self.store.transcript_paired_path),
                "transcript_feature_stats": str(self.store.transcript_feature_stats_path),
                "transcript_feature_shortlist": str(self.store.transcript_feature_shortlist_path),
                "manifest": str(self.store.manifest_path),
            },
        }
        if self.config.output.write_manifest:
            self.store.write_manifest(manifest)
        return manifest

    def _process_transcript_window(
        self,
        raw_document: RawDocument,
        window_idx: int,
        window: WindowBatch,
        model_info: dict[str, Any],
        entry_counter_start: int,
    ) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]], dict[int, int]]:
        outputs, _ = self.model.forward_outputs(window.input_ids, require_grad=False)
        hidden_states = outputs.hidden_states
        logits = outputs.logits
        sample_id = f"{raw_document.doc_id}:{window.window_start}:{window.window_end}:{window_idx}"
        token_records: list[dict[str, Any]] = []
        feature_updates: dict[tuple[int, int], dict[str, Any]] = {}
        total_tokens: dict[int, int] = {}
        entry_counter = entry_counter_start

        for layer_idx in self.sae.available_layers(model_info.get("num_hidden_layers")):
            residual = hidden_states[layer_idx + 1]
            latents = self.sae.encode_layer(layer_idx, residual).squeeze(0).detach().to("cpu")
            positive = latents.clamp_min(0)
            total_tokens[layer_idx] = int(positive.shape[0])
            sentence_feature_summaries = self._sentence_feature_summaries(
                latents=latents,
                sentence_spans=window.sentence_spans,
            )
            window_feature_summaries = self._pooled_feature_summaries(latents)

            self._update_transcript_feature_stats(
                feature_updates=feature_updates,
                layer_idx=layer_idx,
                positive=positive,
                window=window,
                sample_id=sample_id,
                raw_document=raw_document,
                example_heaps_entry_counter=entry_counter,
            )
            entry_counter += positive.shape[0] * max(
                1,
                min(self.config.latents.token_top_k, positive.shape[-1]),
            )

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

        return token_records, feature_updates, total_tokens

    def _update_transcript_feature_stats(
        self,
        feature_updates: dict[tuple[int, int], dict[str, Any]],
        layer_idx: int,
        positive: torch.Tensor,
        window: WindowBatch,
        sample_id: str,
        raw_document: RawDocument,
        example_heaps_entry_counter: int,
    ) -> None:
        token_totals = positive.sum(dim=0)
        token_max = positive.max(dim=0).values
        active_token_counts = (positive > 0).sum(dim=0)
        active_ids = torch.nonzero(token_totals > 0, as_tuple=False).flatten().tolist()

        for feature_id in active_ids:
            key = (layer_idx, int(feature_id))
            update = feature_updates.setdefault(
                key,
                {
                    "model_id": self.config.model.base_model_id,
                    "scope_release": self.config.model.scope_release,
                    "layer": layer_idx,
                    "feature_id": int(feature_id),
                    "max_token_activation": 0.0,
                    "total_activation": 0.0,
                    "active_token_count": 0,
                    "active_window_count": 0,
                    "active_sentence_count": 0,
                    "sentence_pooled_total_activation": 0.0,
                    "examples": [],
                },
            )
            update["max_token_activation"] = max(
                float(update["max_token_activation"]),
                float(token_max[feature_id].item()),
            )
            update["total_activation"] += float(token_totals[feature_id].item())
            update["active_token_count"] += int(active_token_counts[feature_id].item())
            update["active_window_count"] += 1

        for sentence in window.sentence_spans:
            start = int(sentence["start_token_index"])
            end = int(sentence["end_token_index"]) + 1
            sentence_positive = positive[start:end]
            sentence_totals = sentence_positive.sum(dim=0)
            sentence_means = sentence_positive.mean(dim=0)
            sentence_active_ids = torch.nonzero(sentence_totals > 0, as_tuple=False).flatten().tolist()
            for feature_id in sentence_active_ids:
                key = (layer_idx, int(feature_id))
                update = feature_updates.setdefault(
                    key,
                    {
                        "model_id": self.config.model.base_model_id,
                        "scope_release": self.config.model.scope_release,
                        "layer": layer_idx,
                        "feature_id": int(feature_id),
                        "max_token_activation": 0.0,
                        "total_activation": 0.0,
                        "active_token_count": 0,
                        "active_window_count": 0,
                        "active_sentence_count": 0,
                        "sentence_pooled_total_activation": 0.0,
                        "examples": [],
                    },
                )
                update["active_sentence_count"] += 1
                update["sentence_pooled_total_activation"] += float(sentence_means[feature_id].item())

        per_token_top_k = min(self.config.latents.token_top_k, positive.shape[-1])
        for token_position in range(positive.shape[0]):
            token_values = positive[token_position]
            values, indices = torch.topk(token_values, k=per_token_top_k)
            for offset, (activation, feature_id) in enumerate(zip(values.tolist(), indices.tolist())):
                if activation <= 0:
                    continue
                key = (layer_idx, int(feature_id))
                update = feature_updates.setdefault(
                    key,
                    {
                        "model_id": self.config.model.base_model_id,
                        "scope_release": self.config.model.scope_release,
                        "layer": layer_idx,
                        "feature_id": int(feature_id),
                        "max_token_activation": 0.0,
                        "total_activation": 0.0,
                        "active_token_count": 0,
                        "active_window_count": 0,
                        "active_sentence_count": 0,
                        "sentence_pooled_total_activation": 0.0,
                        "examples": [],
                    },
                )
                snippet = self._token_snippet(window.tokens, token_position, self.config.dolma_query.context_window_tokens)
                update["examples"].append(
                    (
                        float(activation),
                        f"{sample_id}:{token_position}:{offset}:{example_heaps_entry_counter}",
                        {
                            "sample_id": sample_id,
                            "stimulus_id": raw_document.provenance.get("stimulus_id"),
                            "token_position": token_position,
                            "token": window.tokens[token_position],
                            "activation": float(activation),
                            "snippet_tokens": snippet,
                            "text": window.text,
                        },
                    )
                )
                example_heaps_entry_counter += 1

    def _merge_feature_updates(
        self,
        feature_stats: dict[tuple[int, int], dict[str, Any]],
        example_heaps: dict[tuple[int, int], list[tuple[float, str, dict[str, Any]]]],
        window_updates: dict[tuple[int, int], dict[str, Any]],
    ) -> None:
        for key, update in window_updates.items():
            merged = feature_stats.setdefault(
                key,
                {
                    "model_id": update["model_id"],
                    "scope_release": update["scope_release"],
                    "layer": update["layer"],
                    "feature_id": update["feature_id"],
                    "max_token_activation": 0.0,
                    "total_activation": 0.0,
                    "active_token_count": 0,
                    "active_window_count": 0,
                    "active_sentence_count": 0,
                    "sentence_pooled_total_activation": 0.0,
                },
            )
            merged["max_token_activation"] = max(
                float(merged["max_token_activation"]),
                float(update["max_token_activation"]),
            )
            merged["total_activation"] += float(update["total_activation"])
            merged["active_token_count"] += int(update["active_token_count"])
            merged["active_window_count"] += int(update["active_window_count"])
            merged["active_sentence_count"] += int(update["active_sentence_count"])
            merged["sentence_pooled_total_activation"] += float(update["sentence_pooled_total_activation"])

            heap = example_heaps.setdefault(key, [])
            for entry in update["examples"]:
                heapq.heappush(heap, entry)
                if len(heap) > self.config.feature_selection.top_examples_per_feature:
                    heapq.heappop(heap)

    def _build_transcript_feature_stats(
        self,
        feature_stats: dict[tuple[int, int], dict[str, Any]],
        total_tokens_by_layer: dict[int, int],
        example_heaps: dict[tuple[int, int], list[tuple[float, str, dict[str, Any]]]],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in sorted(feature_stats):
            row = dict(feature_stats[key])
            layer_idx = int(row["layer"])
            total_tokens = max(1, int(total_tokens_by_layer.get(layer_idx, 0)))
            row["active_token_fraction"] = float(row["active_token_count"]) / total_tokens
            contexts = [item[2] for item in sorted(example_heaps.get(key, []), key=lambda item: item[0], reverse=True)]
            row["top_transcript_examples"] = contexts
            rows.append(row)
        return rows

    def _build_transcript_feature_shortlist(
        self,
        stats_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in stats_rows:
            rows_by_layer[int(row["layer"])].append(row)

        shortlist_rows: list[dict[str, Any]] = []
        for layer_idx, layer_rows in sorted(rows_by_layer.items()):
            ranking_specs = {
                "peak_rank": (
                    sorted(layer_rows, key=lambda item: item["max_token_activation"], reverse=True),
                    self.config.feature_selection.top_by_peak,
                ),
                "total_rank": (
                    sorted(layer_rows, key=lambda item: item["total_activation"], reverse=True),
                    self.config.feature_selection.top_by_total,
                ),
                "persistence_rank": (
                    sorted(layer_rows, key=lambda item: item["active_token_fraction"], reverse=True),
                    self.config.feature_selection.top_by_persistence,
                ),
                "sentence_pool_rank": (
                    sorted(layer_rows, key=lambda item: item["sentence_pooled_total_activation"], reverse=True),
                    self.config.feature_selection.top_by_sentence_pool,
                ),
            }

            selected_ids: set[int] = set()
            rank_lookup: dict[str, dict[int, int]] = {}
            for rank_name, (ranking_rows, limit) in ranking_specs.items():
                lookup: dict[int, int] = {}
                for rank_position, row in enumerate(ranking_rows, start=1):
                    feature_id = int(row["feature_id"])
                    lookup[feature_id] = rank_position
                    if rank_position <= limit:
                        selected_ids.add(feature_id)
                rank_lookup[rank_name] = lookup

            candidates = [row for row in layer_rows if int(row["feature_id"]) in selected_ids]
            for row in candidates:
                feature_id = int(row["feature_id"])
                default_rank = len(layer_rows) + 1
                row["combined_rank_score"] = int(
                    rank_lookup["peak_rank"].get(feature_id, default_rank)
                    + rank_lookup["total_rank"].get(feature_id, default_rank)
                    + rank_lookup["persistence_rank"].get(feature_id, default_rank)
                    + rank_lookup["sentence_pool_rank"].get(feature_id, default_rank)
                )
                row["rank_positions"] = {
                    "peak": rank_lookup["peak_rank"].get(feature_id),
                    "total": rank_lookup["total_rank"].get(feature_id),
                    "persistence": rank_lookup["persistence_rank"].get(feature_id),
                    "sentence_pool": rank_lookup["sentence_pool_rank"].get(feature_id),
                }

            ranked_candidates = sorted(
                candidates,
                key=lambda item: (
                    int(item["combined_rank_score"]),
                    -float(item["total_activation"]),
                    -float(item["max_token_activation"]),
                ),
            )
            for shortlist_rank, row in enumerate(
                ranked_candidates[: self.config.feature_selection.final_top_per_layer],
                start=1,
            ):
                shortlist_rows.append(
                    {
                        "model_id": row["model_id"],
                        "scope_release": row["scope_release"],
                        "layer": layer_idx,
                        "feature_id": int(row["feature_id"]),
                        "shortlist_rank": shortlist_rank,
                        "combined_rank_score": int(row["combined_rank_score"]),
                        "selection_metrics": {
                            "max_token_activation": float(row["max_token_activation"]),
                            "total_activation": float(row["total_activation"]),
                            "active_token_fraction": float(row["active_token_fraction"]),
                            "active_window_count": int(row["active_window_count"]),
                            "active_sentence_count": int(row["active_sentence_count"]),
                            "sentence_pooled_total_activation": float(row["sentence_pooled_total_activation"]),
                        },
                        "rank_positions": row["rank_positions"],
                        "top_transcript_examples": row["top_transcript_examples"],
                    }
                )
        return shortlist_rows
