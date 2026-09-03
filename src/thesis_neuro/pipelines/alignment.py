"""Counterfactual token and span alignment of selected features within transcript windows."""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from thesis_neuro.config import AppConfig
from thesis_neuro.models import GemmaModelAdapter
from thesis_neuro.pipelines.extraction import AlignmentStats
from thesis_neuro.sae import GemmaScopeAdapter
from thesis_neuro.storage import JsonlArtifactStore


class FeatureAlignmentPipeline:
    def __init__(
        self,
        config: AppConfig,
        input_path: str | None = None,
        wait_for_input_seconds: float = 30.0,
        focus_features_path: str | None = None,
    ) -> None:
        self.config = config
        self.store = JsonlArtifactStore(config.output_dir)
        self.input_path = input_path
        self.wait_for_input_seconds = max(0.0, float(wait_for_input_seconds))
        self.focus_features_path = Path(focus_features_path) if focus_features_path is not None else None
        self.focus_features: dict[int, set[int]] | None = None
        self.model: GemmaModelAdapter | None = None
        self.sae: GemmaScopeAdapter | None = None

    def run(self) -> dict[str, Any]:
        stats = AlignmentStats()
        source_path = self._wait_for_input_file()
        self.focus_features = self._load_focus_features()

        print(f"[align] loading model {self.config.model.base_model_id}")
        self.model = GemmaModelAdapter(self.config)
        self.sae = GemmaScopeAdapter(self.config, device=self.model.device, dtype=self.model.dtype)
        model_info = self.model.describe_model()
        print(f"[align] resolving SAE layers for {self.config.model.scope_release}")
        available_layers = set(self.sae.available_layers(model_info.get("num_hidden_layers")))
        self.store.reset_alignment_file()

        grouped_records = self._group_records(self.store.iter_records(source_path))
        if self.focus_features is not None:
            feature_windows = self._plan_feature_windows(grouped_records, available_layers)
            for feature_window in feature_windows:
                stats.window_layers_seen += 1
                row = self._build_feature_centric_alignment_row(feature_window)
                if row is None:
                    continue
                self.store.append_alignment_row(row)
                stats.features_written += 1
        else:
            for group in grouped_records.values():
                if group["layer"] not in available_layers:
                    continue
                stats.window_layers_seen += 1
                alignment_rows = self._build_alignment_rows(group)
                for row in alignment_rows:
                    self.store.append_alignment_row(row)
                    stats.features_written += 1

        result = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config.to_dict(),
            "model": model_info,
            "window_layers_seen": stats.window_layers_seen,
            "features_written": stats.features_written,
            "artifacts": {
                "paired_records": str(source_path),
                "feature_alignment": str(self.store.alignment_path),
            },
        }
        return result

    def _wait_for_input_file(self) -> Path:
        source_path = Path(self.input_path) if self.input_path is not None else self.store.paired_path
        deadline = time.monotonic() + self.wait_for_input_seconds
        announced_wait = False

        while True:
            if source_path.exists():
                if source_path.stat().st_size > 0:
                    return source_path
                raise ValueError(
                    f"Alignment input exists but is empty: {source_path}. "
                    "Run extract first, or point --input to a non-empty paired JSONL artifact."
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if not announced_wait:
                print(
                    f"[align] waiting up to {self.wait_for_input_seconds:.0f}s for input file "
                    f"{source_path}"
                )
                announced_wait = True
            time.sleep(min(1.0, remaining))

        raise FileNotFoundError(
            f"Alignment input not found: {source_path}. "
            "Run extract first, pass --input to an existing paired JSONL file, "
            "or increase --wait-for-input-seconds."
        )

    def _load_focus_features(self) -> dict[int, set[int]] | None:
        if self.focus_features_path is None:
            return None
        if not self.focus_features_path.exists():
            raise FileNotFoundError(f"Focus feature file not found: {self.focus_features_path}")
        mapping: dict[int, set[int]] = defaultdict(set)
        for row in self.store.iter_records(self.focus_features_path):
            mapping[int(row["layer"])].add(int(row["feature_id"]))
        return mapping

    def _group_records(self, records: Any) -> dict[tuple[str, int], dict[str, Any]]:
        grouped: dict[tuple[str, int], dict[str, Any]] = {}
        for record in records:
            key = (record["sample_id"], int(record["layer"]))
            group = grouped.setdefault(
                key,
                {
                    "sample_id": record["sample_id"],
                    "model_id": record["model_id"],
                    "scope_release": record["scope_release"],
                    "layer": int(record["layer"]),
                    "window_token_ids": list(record["window_token_ids"]),
                    "window_tokens": list(record["window_tokens"]),
                    "text": record["text"],
                    "window_start": int(record["window_start"]),
                    "window_end": int(record["window_end"]),
                    "token_sentence_ids": list(record.get("token_sentence_ids", [])),
                    "window_sentences": list(record.get("window_sentences", [])),
                    "window_clauses": list(record.get("window_clauses", [])),
                    "provenance": record["provenance"],
                    "rows": [],
                },
            )
            group["rows"].append(record)
        for group in grouped.values():
            group["rows"].sort(key=lambda item: int(item["token_position"]))
            group["positive_token_ids"] = [row.get("chosen_positive_token_id") for row in group["rows"]]
            group["negative_token_ids"] = [row.get("chosen_negative_token_id") for row in group["rows"]]
        return grouped

    def _plan_feature_windows(
        self,
        grouped_records: dict[tuple[str, int], dict[str, Any]],
        available_layers: set[int],
    ) -> list[dict[str, Any]]:
        windows_by_feature: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for group in grouped_records.values():
            layer_idx = int(group["layer"])
            if layer_idx not in available_layers:
                continue
            allowed_features = self.focus_features.get(layer_idx, set()) if self.focus_features is not None else set()
            if not allowed_features:
                continue
            for feature_id in allowed_features:
                metrics = self._group_feature_metrics(group, feature_id)
                if max(
                    metrics["window_total_activation"],
                    metrics["sentence_total_activation"],
                    metrics["max_token_activation"],
                ) <= 0:
                    continue
                windows_by_feature[(layer_idx, feature_id)].append(
                    {
                        "group": group,
                        "feature_id": feature_id,
                        "window_metrics": metrics,
                    }
                )

        planned: list[dict[str, Any]] = []
        for (_, feature_id), entries in sorted(windows_by_feature.items()):
            ranked_entries = sorted(
                entries,
                key=lambda item: (
                    float(item["window_metrics"]["window_total_activation"]),
                    float(item["window_metrics"]["sentence_total_activation"]),
                    float(item["window_metrics"]["max_token_activation"]),
                ),
                reverse=True,
            )
            for window_rank, entry in enumerate(
                ranked_entries[: self.config.alignment.top_windows_per_feature],
                start=1,
            ):
                planned.append(
                    {
                        **entry,
                        "window_rank_for_feature": window_rank,
                    }
                )

        return sorted(
            planned,
            key=lambda item: (
                int(item["group"]["layer"]),
                int(item["feature_id"]),
                int(item["window_rank_for_feature"]),
            ),
        )

    def _group_feature_metrics(self, group: dict[str, Any], feature_id: int) -> dict[str, float]:
        feature_metrics = self._feature_selection_metrics(group, feature_id)
        window_total = feature_metrics["window_total_activation"] or 0.0
        return {
            "max_token_activation": float(feature_metrics["max_token_activation"]),
            "window_total_activation": float(window_total),
            "sentence_total_activation": float(feature_metrics["sentence_total_activation"]),
        }

    def _build_feature_centric_alignment_row(self, feature_window: dict[str, Any]) -> dict[str, Any] | None:
        group = feature_window["group"]
        feature_id = int(feature_window["feature_id"])
        layer_idx = int(group["layer"])
        original_ids = list(group["window_token_ids"])
        original_outputs = self._run_single_feature_window(original_ids, layer_idx, feature_id)
        original_total = float(original_outputs["feature_total"])
        if original_total <= 0:
            return None

        token_values = original_outputs["token_values"]
        token_positions = self._top_feature_token_positions(token_values)
        if not token_positions and not any(method in self.config.alignment.methods for method in self._span_methods()):
            return None

        token_scores: list[dict[str, Any]] = []
        for token_method in self._token_methods():
            for token_position in token_positions:
                ablated_ids = self._ablate_range_ids(
                    original_ids=original_ids,
                    start_index=token_position,
                    end_index=token_position,
                    method=token_method,
                )
                ablated_outputs = self._run_single_feature_window(ablated_ids, layer_idx, feature_id)
                score = float(original_total - ablated_outputs["feature_total"])
                relative_drop = score / original_total if original_total > 0 else None
                logit_changes = self._logit_changes(
                    group=group,
                    original_logits=original_outputs["logits"],
                    ablated_logits=ablated_outputs["logits"],
                    anchor_position=token_position,
                )
                token_scores.append(
                    {
                        "token_position": token_position,
                        "token_id": int(group["window_token_ids"][token_position]),
                        "token": group["window_tokens"][token_position],
                        "method": token_method,
                        "score": score,
                        "relative_drop": relative_drop,
                        "positive_logit_change": logit_changes["positive_logit_change"],
                        "negative_logit_change": logit_changes["negative_logit_change"],
                    }
                )

        direct_span_scores: list[dict[str, Any]] = []
        for intervention in self._feature_span_interventions(group, token_values):
            ablated_ids = self._ablate_range_ids(
                original_ids=original_ids,
                start_index=int(intervention["start_token_position"]),
                end_index=int(intervention["end_token_position"]),
                method=str(intervention["method"]),
            )
            ablated_outputs = self._run_single_feature_window(ablated_ids, layer_idx, feature_id)
            score = float(original_total - ablated_outputs["feature_total"])
            if score <= 0:
                continue
            relative_drop = score / original_total if original_total > 0 else None
            logit_changes = self._logit_changes(
                group=group,
                original_logits=original_outputs["logits"],
                ablated_logits=ablated_outputs["logits"],
                anchor_position=int(intervention["start_token_position"]),
            )
            direct_span_scores.append(
                {
                    **intervention,
                    "score": score,
                    "relative_drop": relative_drop,
                    "positive_logit_change": logit_changes["positive_logit_change"],
                    "negative_logit_change": logit_changes["negative_logit_change"],
                }
            )

        return {
            "sample_id": group["sample_id"],
            "model_id": group["model_id"],
            "scope_release": group["scope_release"],
            "layer": layer_idx,
            "feature_id": feature_id,
            "window_token_ids": group["window_token_ids"],
            "window_tokens": group["window_tokens"],
            "text": group["text"],
            "feature_total_activation": original_total,
            "feature_selection_metrics": self._feature_selection_metrics(group, feature_id),
            "window_rank_for_feature": int(feature_window["window_rank_for_feature"]),
            "window_feature_metrics": feature_window["window_metrics"],
            "ablation_methods": list(self.config.alignment.methods),
            "top_token_alignments": self._top_token_alignments(token_scores),
            "top_span_alignments": self._top_span_alignments(
                token_scores=token_scores,
                direct_span_scores=direct_span_scores,
                original_total=original_total,
                token_ids=group["window_token_ids"],
                tokens=group["window_tokens"],
            ),
            "provenance": group["provenance"],
        }

    def _build_alignment_rows(self, group: dict[str, Any]) -> list[dict[str, Any]]:
        feature_candidates = self._candidate_feature_ids(group)
        if not feature_candidates:
            return []

        original_ids = list(group["window_token_ids"])
        layer_idx = int(group["layer"])
        original_outputs = self._run_window(original_ids, layer_idx, feature_candidates)
        original_totals = original_outputs["feature_totals"]
        selected_features = [
            feature_id
            for feature_id, total in sorted(original_totals.items(), key=lambda item: item[1], reverse=True)
            if total > 0
        ][: self.config.alignment.top_features_per_window]
        if self.focus_features is not None:
            allowed = self.focus_features.get(layer_idx, set())
            selected_features = [feature_id for feature_id in selected_features if feature_id in allowed]
        if not selected_features:
            return []

        token_scores_by_feature: dict[int, list[dict[str, Any]]] = defaultdict(list)
        direct_span_scores_by_feature: dict[int, list[dict[str, Any]]] = defaultdict(list)

        for token_method in self._token_methods():
            for token_position, token_id in enumerate(original_ids):
                ablated_ids = self._ablate_range_ids(
                    original_ids=original_ids,
                    start_index=token_position,
                    end_index=token_position,
                    method=token_method,
                )
                ablated_outputs = self._run_window(ablated_ids, layer_idx, selected_features)
                logit_changes = self._logit_changes(
                    group=group,
                    original_logits=original_outputs["logits"],
                    ablated_logits=ablated_outputs["logits"],
                    anchor_position=token_position,
                )
                for feature_id in selected_features:
                    original_total = original_totals[feature_id]
                    ablated_total = ablated_outputs["feature_totals"].get(feature_id, 0.0)
                    score = float(original_total - ablated_total)
                    relative_drop = score / original_total if original_total > 0 else None
                    token_scores_by_feature[feature_id].append(
                        {
                            "token_position": token_position,
                            "token_id": int(token_id),
                            "token": group["window_tokens"][token_position],
                            "method": token_method,
                            "score": score,
                            "relative_drop": relative_drop,
                            "positive_logit_change": logit_changes["positive_logit_change"],
                            "negative_logit_change": logit_changes["negative_logit_change"],
                        }
                    )

        span_interventions = self._span_interventions(group)
        for intervention in span_interventions:
            ablated_ids = self._ablate_range_ids(
                original_ids=original_ids,
                start_index=int(intervention["start_token_position"]),
                end_index=int(intervention["end_token_position"]),
                method=str(intervention["method"]),
            )
            ablated_outputs = self._run_window(ablated_ids, layer_idx, selected_features)
            logit_changes = self._logit_changes(
                group=group,
                original_logits=original_outputs["logits"],
                ablated_logits=ablated_outputs["logits"],
                anchor_position=int(intervention["start_token_position"]),
            )
            for feature_id in selected_features:
                original_total = original_totals[feature_id]
                ablated_total = ablated_outputs["feature_totals"].get(feature_id, 0.0)
                score = float(original_total - ablated_total)
                relative_drop = score / original_total if original_total > 0 else None
                direct_span_scores_by_feature[feature_id].append(
                    {
                        **intervention,
                        "score": score,
                        "relative_drop": relative_drop,
                        "positive_logit_change": logit_changes["positive_logit_change"],
                        "negative_logit_change": logit_changes["negative_logit_change"],
                    }
                )

        alignment_rows: list[dict[str, Any]] = []
        for feature_id in selected_features:
            token_scores = token_scores_by_feature[feature_id]
            top_token_alignments = self._top_token_alignments(token_scores)
            top_span_alignments = self._top_span_alignments(
                token_scores=token_scores,
                direct_span_scores=direct_span_scores_by_feature[feature_id],
                original_total=original_totals[feature_id],
                token_ids=group["window_token_ids"],
                tokens=group["window_tokens"],
            )
            alignment_rows.append(
                {
                    "sample_id": group["sample_id"],
                    "model_id": group["model_id"],
                    "scope_release": group["scope_release"],
                    "layer": layer_idx,
                    "feature_id": feature_id,
                    "window_token_ids": group["window_token_ids"],
                    "window_tokens": group["window_tokens"],
                    "text": group["text"],
                    "feature_total_activation": original_totals[feature_id],
                    "feature_selection_metrics": self._feature_selection_metrics(group, feature_id),
                    "ablation_methods": list(self.config.alignment.methods),
                    "top_token_alignments": top_token_alignments,
                    "top_span_alignments": top_span_alignments,
                    "provenance": group["provenance"],
                }
            )
        return alignment_rows

    def _candidate_feature_ids(self, group: dict[str, Any]) -> list[int]:
        rows = group["rows"]
        max_token_scores: dict[int, float] = defaultdict(float)
        for row in rows:
            for latent in row["latent_activations"]:
                feature_id = int(latent["latent_id"])
                activation = float(latent["activation"])
                if activation > max_token_scores[feature_id]:
                    max_token_scores[feature_id] = activation

        pooled_window_scores: dict[int, float] = {}
        if rows:
            for feature in rows[0].get("window_feature_summaries", []):
                pooled_window_scores[int(feature["feature_id"])] = float(feature["total_activation"])

        pooled_sentence_scores: dict[int, float] = defaultdict(float)
        seen_sentences: set[int] = set()
        for row in rows:
            sentence_id = int(row.get("sentence_id", -1))
            if sentence_id in seen_sentences:
                continue
            seen_sentences.add(sentence_id)
            for feature in row.get("sentence_feature_summaries", []):
                feature_id = int(feature["feature_id"])
                pooled_sentence_scores[feature_id] += float(feature["total_activation"])

        candidate_ids: list[int] = []
        seen_features: set[int] = set()
        ranking_sources = (
            sorted(max_token_scores.items(), key=lambda item: item[1], reverse=True),
            sorted(pooled_window_scores.items(), key=lambda item: item[1], reverse=True),
            sorted(pooled_sentence_scores.items(), key=lambda item: item[1], reverse=True),
        )
        for ranking in ranking_sources:
            for feature_id, _ in ranking[: self.config.alignment.top_features_per_window]:
                if feature_id in seen_features:
                    continue
                seen_features.add(feature_id)
                candidate_ids.append(feature_id)

        return candidate_ids

    def _top_feature_token_positions(self, token_values: torch.Tensor) -> list[int]:
        if token_values.numel() == 0:
            return []
        k = min(
            self.config.alignment.top_token_positions_per_feature_window,
            int(token_values.shape[0]),
        )
        values, indices = torch.topk(token_values, k=k)
        positions: list[int] = []
        for activation, index in zip(values.tolist(), indices.tolist()):
            if float(activation) <= 0:
                continue
            positions.append(int(index))
        return sorted(set(positions))

    def _feature_span_interventions(
        self,
        group: dict[str, Any],
        token_values: torch.Tensor,
    ) -> list[dict[str, Any]]:
        interventions: list[dict[str, Any]] = []
        scored: list[tuple[float, dict[str, Any]]] = []
        for span in group.get("window_sentences", []):
            start = int(span["start_token_index"])
            end = int(span["end_token_index"])
            total = float(token_values[start : end + 1].clamp_min(0).sum().item())
            if total <= 0:
                continue
            for method in self.config.alignment.methods:
                if method not in {"delete_sentence_retokenize", "pad_sentence_mask"}:
                    continue
                scored.append(
                    (
                        total,
                        {
                            "method": method,
                            "span_type": "sentence",
                            "start_token_position": start,
                            "end_token_position": end,
                            "token_ids": group["window_token_ids"][start : end + 1],
                            "tokens": group["window_tokens"][start : end + 1],
                        },
                    )
                )
        for span in group.get("window_clauses", []):
            start = int(span["start_token_index"])
            end = int(span["end_token_index"])
            total = float(token_values[start : end + 1].clamp_min(0).sum().item())
            if total <= 0:
                continue
            for method in self.config.alignment.methods:
                if method not in {"delete_clause_retokenize", "pad_clause_mask"}:
                    continue
                scored.append(
                    (
                        total,
                        {
                            "method": method,
                            "span_type": "clause",
                            "start_token_position": start,
                            "end_token_position": end,
                            "token_ids": group["window_token_ids"][start : end + 1],
                            "tokens": group["window_tokens"][start : end + 1],
                        },
                    )
                )

        seen: set[tuple[str, int, int]] = set()
        for _, row in sorted(scored, key=lambda item: item[0], reverse=True):
            key = (str(row["method"]), int(row["start_token_position"]), int(row["end_token_position"]))
            if key in seen:
                continue
            seen.add(key)
            interventions.append(row)
            if len(interventions) >= self.config.alignment.top_span_alignments:
                break
        return interventions

    @staticmethod
    def _span_methods() -> set[str]:
        return {
            "delete_sentence_retokenize",
            "pad_sentence_mask",
            "delete_clause_retokenize",
            "pad_clause_mask",
        }

    def _run_window(self, input_ids: list[int], layer_idx: int, feature_ids: list[int]) -> dict[str, Any]:
        if not input_ids:
            return {
                "feature_totals": {feature_id: 0.0 for feature_id in feature_ids},
                "logits": None,
            }
        if self.model is None or self.sae is None:
            raise RuntimeError("Alignment runtime is not initialized. Call run() first.")
        outputs, _ = self.model.forward_outputs(input_ids, require_grad=False)
        residual = outputs.hidden_states[layer_idx + 1]
        latents = self.sae.encode_layer(layer_idx, residual).squeeze(0).detach().to("cpu")
        totals: dict[int, float] = {}
        for feature_id in feature_ids:
            if feature_id >= latents.shape[-1]:
                totals[feature_id] = 0.0
                continue
            totals[feature_id] = float(latents[:, feature_id].clamp_min(0).sum().item())
        return {
            "feature_totals": totals,
            "logits": outputs.logits[0].detach().to("cpu"),
        }

    def _run_single_feature_window(
        self,
        input_ids: list[int],
        layer_idx: int,
        feature_id: int,
    ) -> dict[str, Any]:
        if not input_ids:
            return {
                "feature_total": 0.0,
                "token_values": torch.zeros(0),
                "logits": None,
            }
        if self.model is None or self.sae is None:
            raise RuntimeError("Alignment runtime is not initialized. Call run() first.")
        outputs, _ = self.model.forward_outputs(input_ids, require_grad=False)
        residual = outputs.hidden_states[layer_idx + 1]
        latents = self.sae.encode_layer(layer_idx, residual).squeeze(0).detach().to("cpu")
        if feature_id >= latents.shape[-1]:
            token_values = torch.zeros(latents.shape[0])
        else:
            token_values = latents[:, feature_id].clamp_min(0)
        return {
            "feature_total": float(token_values.sum().item()),
            "token_values": token_values,
            "logits": outputs.logits[0].detach().to("cpu"),
        }

    def _ablate_range_ids(
        self,
        original_ids: list[int],
        start_index: int,
        end_index: int,
        method: str,
    ) -> list[int]:
        if self.model is None:
            raise RuntimeError("Alignment runtime is not initialized. Call run() first.")
        if method in {"deletion_retokenize", "delete_sentence_retokenize", "delete_clause_retokenize"}:
            kept_ids = original_ids[:start_index] + original_ids[end_index + 1 :]
            if not kept_ids:
                return []
            ablated_text = self.model.tokenizer.decode(kept_ids, skip_special_tokens=False)
            return self.model.tokenize_document(ablated_text)

        if method in {"pad_eos_mask", "pad_sentence_mask", "pad_clause_mask"}:
            masked = list(original_ids)
            pad_token_id = self.model.tokenizer.pad_token_id
            if pad_token_id is None:
                raise ValueError("Tokenizer pad_token_id is required for masking alignment methods.")
            for index in range(start_index, end_index + 1):
                masked[index] = int(pad_token_id)
            return masked

        raise ValueError(f"Unsupported alignment method: {method}")

    def _top_token_alignments(self, token_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked = sorted(token_scores, key=lambda item: item["score"], reverse=True)
        positive = [item for item in ranked if item["score"] > 0]
        return positive[: self.config.alignment.top_token_alignments]

    def _top_span_alignments(
        self,
        token_scores: list[dict[str, Any]],
        direct_span_scores: list[dict[str, Any]],
        original_total: float,
        token_ids: list[int],
        tokens: list[str],
    ) -> list[dict[str, Any]]:
        spans: list[dict[str, Any]] = []
        for method in self.config.alignment.methods:
            method_scores = sorted(
                [item for item in token_scores if item["method"] == method and item["score"] > 0],
                key=lambda item: item["token_position"],
            )
            if not method_scores:
                continue

            current_positions = [method_scores[0]["token_position"]]
            current_score = float(method_scores[0]["score"])
            for item in method_scores[1:]:
                position = int(item["token_position"])
                if position == current_positions[-1] + 1:
                    current_positions.append(position)
                    current_score += float(item["score"])
                    continue
                spans.append(
                    self._make_span_record(
                        method=method,
                        positions=current_positions,
                        score=current_score,
                        original_total=original_total,
                        token_ids=token_ids,
                        tokens=tokens,
                    )
                )
                current_positions = [position]
                current_score = float(item["score"])
            spans.append(
                self._make_span_record(
                    method=method,
                    positions=current_positions,
                    score=current_score,
                    original_total=original_total,
                    token_ids=token_ids,
                    tokens=tokens,
                )
            )

        spans.extend(item for item in direct_span_scores if item["score"] > 0)
        ranked_spans = sorted(spans, key=lambda item: item["score"], reverse=True)
        return ranked_spans[: self.config.alignment.top_span_alignments]

    def _make_span_record(
        self,
        method: str,
        positions: list[int],
        score: float,
        original_total: float,
        token_ids: list[int],
        tokens: list[str],
    ) -> dict[str, Any]:
        start = positions[0]
        end = positions[-1]
        relative_drop = None
        if original_total > 0:
            relative_drop = score / original_total
        return {
            "start_token_position": start,
            "end_token_position": end,
            "token_ids": token_ids[start : end + 1],
            "tokens": tokens[start : end + 1],
            "method": method,
            "span_type": "token_merge",
            "score": score,
            "relative_drop": relative_drop,
            "positive_logit_change": None,
            "negative_logit_change": None,
        }

    def _feature_selection_metrics(self, group: dict[str, Any], feature_id: int) -> dict[str, float | int | None]:
        max_token_activation = 0.0
        for row in group["rows"]:
            for latent in row["latent_activations"]:
                if int(latent["latent_id"]) == feature_id:
                    max_token_activation = max(max_token_activation, float(latent["activation"]))

        window_total_activation = None
        if group["rows"]:
            for feature in group["rows"][0].get("window_feature_summaries", []):
                if int(feature["feature_id"]) == feature_id:
                    window_total_activation = float(feature["total_activation"])
                    break

        sentence_total_activation = 0.0
        seen_sentences: set[int] = set()
        for row in group["rows"]:
            sentence_id = int(row.get("sentence_id", -1))
            if sentence_id in seen_sentences:
                continue
            seen_sentences.add(sentence_id)
            for feature in row.get("sentence_feature_summaries", []):
                if int(feature["feature_id"]) == feature_id:
                    sentence_total_activation += float(feature["total_activation"])

        return {
            "feature_id": feature_id,
            "max_token_activation": max_token_activation,
            "window_total_activation": window_total_activation,
            "sentence_total_activation": sentence_total_activation,
        }

    def _token_methods(self) -> list[str]:
        return [
            method
            for method in self.config.alignment.methods
            if method in {"deletion_retokenize", "pad_eos_mask"}
        ]

    def _span_interventions(self, group: dict[str, Any]) -> list[dict[str, Any]]:
        interventions: list[dict[str, Any]] = []
        seen: set[tuple[str, int, int]] = set()

        sentence_methods = {
            "delete_sentence_retokenize",
            "pad_sentence_mask",
        }
        clause_methods = {
            "delete_clause_retokenize",
            "pad_clause_mask",
        }
        for sentence in group.get("window_sentences", []):
            for method in self.config.alignment.methods:
                if method not in sentence_methods:
                    continue
                key = (method, int(sentence["start_token_index"]), int(sentence["end_token_index"]))
                if key in seen:
                    continue
                seen.add(key)
                interventions.append(
                    {
                        "method": method,
                        "span_type": "sentence",
                        "start_token_position": int(sentence["start_token_index"]),
                        "end_token_position": int(sentence["end_token_index"]),
                        "token_ids": group["window_token_ids"][
                            int(sentence["start_token_index"]) : int(sentence["end_token_index"]) + 1
                        ],
                        "tokens": group["window_tokens"][
                            int(sentence["start_token_index"]) : int(sentence["end_token_index"]) + 1
                        ],
                    }
                )

        for clause in group.get("window_clauses", []):
            for method in self.config.alignment.methods:
                if method not in clause_methods:
                    continue
                key = (method, int(clause["start_token_index"]), int(clause["end_token_index"]))
                if key in seen:
                    continue
                seen.add(key)
                interventions.append(
                    {
                        "method": method,
                        "span_type": "clause",
                        "start_token_position": int(clause["start_token_index"]),
                        "end_token_position": int(clause["end_token_index"]),
                        "token_ids": group["window_token_ids"][
                            int(clause["start_token_index"]) : int(clause["end_token_index"]) + 1
                        ],
                        "tokens": group["window_tokens"][
                            int(clause["start_token_index"]) : int(clause["end_token_index"]) + 1
                        ],
                    }
                )

        return interventions

    @staticmethod
    def _logit_changes(
        group: dict[str, Any],
        original_logits: torch.Tensor | None,
        ablated_logits: torch.Tensor | None,
        anchor_position: int,
    ) -> dict[str, float | None]:
        if original_logits is None or ablated_logits is None:
            return {"positive_logit_change": None, "negative_logit_change": None}

        original_position = min(anchor_position, original_logits.shape[0] - 1)
        ablated_position = min(anchor_position, ablated_logits.shape[0] - 1)
        positive_token_id = group["positive_token_ids"][original_position]
        negative_token_id = group["negative_token_ids"][original_position]

        positive_change = None
        if positive_token_id is not None:
            positive_change = float(
                original_logits[original_position, int(positive_token_id)].item()
                - ablated_logits[ablated_position, int(positive_token_id)].item()
            )

        negative_change = None
        if negative_token_id is not None:
            negative_change = float(
                original_logits[original_position, int(negative_token_id)].item()
                - ablated_logits[ablated_position, int(negative_token_id)].item()
            )

        return {
            "positive_logit_change": positive_change,
            "negative_logit_change": negative_change,
        }

