"""Streaming, resume-safe feature ranking, correlation, judge-input, and selection stages."""

from __future__ import annotations

import asyncio
import heapq
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from thesis_neuro.config import AppConfig
from thesis_neuro.storage import JsonlArtifactStore

ANALYSIS_STAGES = (
    "build-feature-relevance",
    "build-feature-correlations",
    "build-feature-judge-input",
    "run-feature-judge",
    "select-alignment-features",
)


class FeatureConceptAnalysisPipeline:
    def __init__(
        self,
        config: AppConfig,
        transcript_output_dir: str | None = None,
        dolma_output_dir: str | None = None,
        alignment_path: str | None = None,
        run_judge: bool = False,
        judge_missing_only: bool = False,
        from_stage: str | None = None,
        until_stage: str | None = None,
    ) -> None:
        self.config = config
        self.output_store = JsonlArtifactStore(config.output_dir)
        self.transcript_store = JsonlArtifactStore(
            transcript_output_dir or config.analysis.transcript_output_dir or config.output.dir
        )
        self.dolma_store = JsonlArtifactStore(
            dolma_output_dir or config.analysis.dolma_output_dir or config.output.dir
        )
        self.alignment_path = (
            Path(alignment_path)
            if alignment_path is not None
            else (
                Path(config.analysis.alignment_path)
                if config.analysis.alignment_path
                else self.output_store.alignment_path
            )
        )
        self.run_judge = bool(run_judge or config.judge.enabled)
        self.judge_missing_only = bool(judge_missing_only)
        self.from_stage = from_stage
        self.until_stage = until_stage

    def run(self) -> dict[str, Any]:
        stages = self._selected_stages()
        self._log(
            "analysis_start",
            transcript_output_dir=str(self.transcript_store.output_dir),
            dolma_output_dir=str(self.dolma_store.output_dir),
            analysis_output_dir=str(self.output_store.output_dir),
            stages=stages,
            run_judge=self.run_judge,
            judge_missing_only=self.judge_missing_only,
        )

        if "build-feature-relevance" in stages:
            if self._has_rows(self.output_store.feature_relevance_path):
                self._log("skip_stage", stage="build-feature-relevance", reason="artifact_exists")
            else:
                self._build_feature_relevance()

        if "build-feature-correlations" in stages:
            if self._has_rows(self.output_store.feature_correlations_path):
                self._log("skip_stage", stage="build-feature-correlations", reason="artifact_exists")
            else:
                self._build_feature_correlations()

        if "build-feature-judge-input" in stages:
            if self._has_rows(self.output_store.feature_judge_input_path):
                self._log("skip_stage", stage="build-feature-judge-input", reason="artifact_exists")
            else:
                self._build_feature_judge_input()

        judged_rows = 0
        if "run-feature-judge" in stages and self.run_judge:
            judged_rows = self._run_feature_judge()

        if "select-alignment-features" in stages:
            if self._has_rows(self.output_store.selected_features_path):
                self._log("skip_stage", stage="select-alignment-features", reason="artifact_exists")
            else:
                self._write_selected_features()

        result = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "transcript_output_dir": str(self.transcript_store.output_dir),
            "dolma_output_dir": str(self.dolma_store.output_dir),
            "analysis_output_dir": str(self.output_store.output_dir),
            "judge_requested": self.run_judge,
            "judge_missing_only": self.judge_missing_only,
            "judge_model": self.config.judge.model if self.run_judge else None,
            "stages_requested": stages,
            "features_ranked": self._count_rows(self.output_store.feature_relevance_path),
            "features_selected_for_alignment": self._count_rows(self.output_store.selected_features_path),
            "features_prepared_for_judge": self._count_rows(self.output_store.feature_judge_input_path),
            "features_judged": self._count_rows(self.output_store.feature_concepts_path),
            "features_judged_this_run": judged_rows,
            "artifacts": {
                "feature_relevance": str(self.output_store.feature_relevance_path),
                "selected_features": str(self.output_store.selected_features_path),
                "feature_correlations": str(self.output_store.feature_correlations_path),
                "feature_judge_input": str(self.output_store.feature_judge_input_path),
                "feature_concepts": (
                    str(self.output_store.feature_concepts_path)
                    if self._has_rows(self.output_store.feature_concepts_path)
                    else None
                ),
            },
        }
        if self.config.output.write_manifest:
            self.output_store.write_manifest(result)
        return result

    def _selected_stages(self) -> list[str]:
        stages = list(ANALYSIS_STAGES)
        if not self.run_judge:
            stages.remove("run-feature-judge")
        if self.from_stage is not None:
            if self.from_stage not in ANALYSIS_STAGES:
                raise ValueError(f"Unknown analysis stage: {self.from_stage}")
            stages = stages[ANALYSIS_STAGES.index(self.from_stage) :]
        if self.until_stage is not None:
            if self.until_stage not in ANALYSIS_STAGES:
                raise ValueError(f"Unknown analysis stage: {self.until_stage}")
            allowed = ANALYSIS_STAGES[: ANALYSIS_STAGES.index(self.until_stage) + 1]
            stages = [stage for stage in stages if stage in allowed]
        return stages

    def _build_feature_relevance(self) -> None:
        stage = "build-feature-relevance"
        self._log(stage, status="start")
        shortlist_rows = list(self.transcript_store.iter_records(self.transcript_store.transcript_feature_shortlist_path))
        if not shortlist_rows:
            raise ValueError("Transcript shortlist is empty.")
        shortlist_by_key = {
            (int(row["layer"]), int(row["feature_id"])): row
            for row in shortlist_rows
        }
        stats_by_key = self._load_shortlist_stats(set(shortlist_by_key))
        if not stats_by_key:
            raise ValueError("Transcript feature stats are empty.")
        dolma_support = self._collect_dolma_support(set(shortlist_by_key))

        relevance_rows: list[dict[str, Any]] = []
        for shortlist_row in shortlist_rows:
            layer = int(shortlist_row["layer"])
            feature_id = int(shortlist_row["feature_id"])
            key = (layer, feature_id)
            stats_row = stats_by_key.get(key)
            if stats_row is None:
                continue
            support = dolma_support.get(key, self._empty_dolma_support())
            transcript_examples = self._select_transcript_examples(
                shortlist_row.get("top_transcript_examples", []),
                self.config.analysis.transcript_examples_per_feature,
            )
            selected_dolma_contexts = self._select_dolma_contexts(
                support["top_total"],
                support["top_peak"],
                support["top_fraction"],
                support["top_document"],
                self.config.analysis.dolma_contexts_per_feature,
            )
            rank_positions = shortlist_row.get("rank_positions", {})
            shortlist_rank = int(shortlist_row["shortlist_rank"])
            transcript_relevance_score = (
                (1.0 / shortlist_rank)
                + self._reciprocal_rank(rank_positions.get("peak"))
                + self._reciprocal_rank(rank_positions.get("total"))
                + self._reciprocal_rank(rank_positions.get("persistence"))
                + self._reciprocal_rank(rank_positions.get("sentence_pool"))
            )
            relevance_rows.append(
                {
                    "layer": layer,
                    "feature_id": feature_id,
                    "shortlist_rank": shortlist_rank,
                    "transcript_relevance_score": transcript_relevance_score,
                    "transcript_metrics": shortlist_row["selection_metrics"],
                    "transcript_support": {
                        "active_window_count": int(stats_row.get("active_window_count", 0)),
                        "active_sentence_count": int(stats_row.get("active_sentence_count", 0)),
                        "active_token_count": int(stats_row.get("active_token_count", 0)),
                        "active_token_fraction": float(stats_row.get("active_token_fraction", 0.0)),
                        "sentence_pooled_total_activation": float(
                            stats_row.get("sentence_pooled_total_activation", 0.0)
                        ),
                        "distinct_stimulus_ids_in_examples": len(
                            {
                                example.get("stimulus_id")
                                for example in transcript_examples
                                if example.get("stimulus_id")
                            }
                        ),
                    },
                    "top_transcript_examples": transcript_examples,
                    "dolma_support": {
                        "context_count": int(support["context_count"]),
                        "max_context_total_activation": float(support["max_total"]),
                        "mean_context_total_activation": float(
                            support["total_sum"] / support["context_count"] if support["context_count"] else 0.0
                        ),
                        "max_context_peak_activation": float(support["max_peak"]),
                        "distinct_documents_in_selected_contexts": len(
                            {self._context_document_key(item) for item in selected_dolma_contexts}
                        ),
                    },
                    "top_dolma_contexts": selected_dolma_contexts,
                }
            )

        ranked_rows = sorted(
            relevance_rows,
            key=lambda item: (-float(item["transcript_relevance_score"]), int(item["shortlist_rank"])),
        )
        for index, row in enumerate(ranked_rows, start=1):
            row["transcript_relevance_rank"] = index
        self.output_store.write_jsonl(self.output_store.feature_relevance_path, ranked_rows)
        self._log(
            stage,
            status="done",
            shortlist_features=len(shortlist_rows),
            stats_rows=len(stats_by_key),
            ranked_rows=len(ranked_rows),
        )

    def _build_feature_correlations(self) -> None:
        stage = "build-feature-correlations"
        self._log(stage, status="start")
        relevance_rows = list(self.output_store.iter_records(self.output_store.feature_relevance_path))
        rows_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in relevance_rows[: self.config.analysis.top_features_for_correlation]:
            rows_by_layer[int(row["layer"])].append(row)

        self.output_store.write_jsonl(self.output_store.feature_correlations_path, [])
        output_rows: list[dict[str, Any]] = []
        for layer in sorted(rows_by_layer):
            layer_rows = rows_by_layer[layer]
            feature_ids = [int(row["feature_id"]) for row in layer_rows]
            sentence_vectors = self._sentence_feature_maps_for_layer(layer, feature_ids)
            self._log(
                stage,
                status="layer",
                layer=layer,
                feature_count=len(feature_ids),
                sentence_count=len(sentence_vectors),
            )
            if len(sentence_vectors) < 3 or len(feature_ids) < 2:
                for feature_id in feature_ids:
                    output_rows.append(
                        {
                            "layer": layer,
                            "feature_id": feature_id,
                            "top_correlated_features": [],
                        }
                    )
                continue

            matrix = np.array(
                [[feature_map.get(feature_id, 0.0) for feature_id in feature_ids] for feature_map in sentence_vectors],
                dtype=float,
            )
            correlations = np.corrcoef(matrix, rowvar=False)
            for column_index, feature_id in enumerate(feature_ids):
                related: list[dict[str, Any]] = []
                for other_index, other_feature_id in enumerate(feature_ids):
                    if other_feature_id == feature_id:
                        continue
                    score = correlations[column_index, other_index]
                    if not np.isfinite(score):
                        continue
                    related.append(
                        {
                            "feature_id": int(other_feature_id),
                            "correlation": float(score),
                        }
                    )
                related = sorted(related, key=lambda item: item["correlation"], reverse=True)
                output_rows.append(
                    {
                        "layer": layer,
                        "feature_id": feature_id,
                        "top_correlated_features": related[
                            : self.config.analysis.correlated_features_per_feature
                        ],
                    }
                )
        self.output_store.write_jsonl(self.output_store.feature_correlations_path, output_rows)
        self._log(stage, status="done", rows=len(output_rows))

    def _build_feature_judge_input(self) -> None:
        stage = "build-feature-judge-input"
        self._log(stage, status="start")
        correlation_by_key = {
            (int(row["layer"]), int(row["feature_id"])): row
            for row in self.output_store.iter_records(self.output_store.feature_correlations_path)
        }
        alignment_rows = self._load_alignment_rows()
        judge_rows: list[dict[str, Any]] = []
        for row in self.output_store.iter_records(self.output_store.feature_relevance_path):
            if len(judge_rows) >= self.config.analysis.top_features_for_judge:
                break
            key = (int(row["layer"]), int(row["feature_id"]))
            evidence = {
                "layer": row["layer"],
                "feature_id": row["feature_id"],
                "transcript_relevance_rank": row["transcript_relevance_rank"],
                "transcript_relevance_score": row["transcript_relevance_score"],
                "transcript_metrics": row["transcript_metrics"],
                "transcript_support": row.get("transcript_support", {}),
                "top_transcript_examples": self._judge_transcript_examples(row["top_transcript_examples"]),
                "dolma_support": row["dolma_support"],
                "top_dolma_contexts": self._judge_dolma_contexts(row["top_dolma_contexts"]),
                "top_correlated_features": correlation_by_key.get(key, {}).get("top_correlated_features", []),
                "alignment_summary": self._judge_alignment_summary(alignment_rows.get(key)),
            }
            judge_rows.append(
                {
                    "layer": row["layer"],
                    "feature_id": row["feature_id"],
                    "transcript_relevance_rank": row["transcript_relevance_rank"],
                    "transcript_relevance_score": row["transcript_relevance_score"],
                    "evidence": evidence,
                }
            )
        self.output_store.write_jsonl(self.output_store.feature_judge_input_path, judge_rows)
        self._log(stage, status="done", rows=len(judge_rows))

    def _run_feature_judge(self) -> int:
        stage = "run-feature-judge"
        if not self.config.env.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when judge analysis is enabled.")
        self._log(stage, status="start")
        existing_rows = self._load_existing_judged_rows() if self.judge_missing_only else []
        if not self.judge_missing_only and self._has_rows(self.output_store.feature_concepts_path):
            self._log(stage, status="skip", reason="artifact_exists")
            return 0
        judge_input_rows = list(self.output_store.iter_records(self.output_store.feature_judge_input_path))
        judge_input_rows = self._filter_judge_input_rows(judge_input_rows, existing_rows)
        if not self.judge_missing_only:
            self.output_store.write_jsonl(self.output_store.feature_concepts_path, [])
        new_rows = asyncio.run(self._run_judge_async(judge_input_rows))
        if self.judge_missing_only:
            judged_rows = self._merge_judged_rows(existing_rows, new_rows)
            self.output_store.write_jsonl(self.output_store.feature_concepts_path, judged_rows)
        self._log(stage, status="done", rows=len(new_rows))
        return len(new_rows)

    def _write_selected_features(self) -> None:
        stage = "select-alignment-features"
        self._log(stage, status="start")
        relevance_rows = list(self.output_store.iter_records(self.output_store.feature_relevance_path))
        selected_rows = relevance_rows[: self.config.analysis.top_features_for_alignment]
        self.output_store.write_jsonl(self.output_store.selected_features_path, selected_rows)
        self._log(stage, status="done", rows=len(selected_rows))

    async def _run_judge_async(self, judge_input_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from thesis_neuro.judge import AsyncOpenAIJudge

        judge = AsyncOpenAIJudge(
            api_key=self.config.env.openai_api_key or "",
            model=self.config.judge.model,
            timeout_seconds=self.config.judge.timeout_seconds,
            max_retries=self.config.judge.max_retries,
        )
        semaphore = asyncio.Semaphore(self.config.judge.max_concurrency)

        async def evaluate(row: dict[str, Any]) -> dict[str, Any]:
            judged_row = {
                "layer": row["layer"],
                "feature_id": row["feature_id"],
                "transcript_relevance_rank": row["transcript_relevance_rank"],
                "transcript_relevance_score": row["transcript_relevance_score"],
                "judge_input": row["evidence"],
            }
            try:
                async with semaphore:
                    judged_row["judge_status"] = "ok"
                    judged_row["judge_output"] = await judge.judge_feature(row["evidence"])
            except Exception as exc:
                judged_row["judge_status"] = "error"
                judged_row["judge_output"] = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            self.output_store.append_jsonl(self.output_store.feature_concepts_path, judged_row)
            return judged_row

        judged_rows: list[dict[str, Any]] = []
        tasks = [asyncio.create_task(evaluate(row)) for row in judge_input_rows]
        for task in asyncio.as_completed(tasks):
            judged_rows.append(await task)
        judged_rows.sort(
            key=lambda row: (
                int(row["transcript_relevance_rank"]),
                int(row["layer"]),
                int(row["feature_id"]),
            )
        )
        return judged_rows

    def _load_shortlist_stats(self, shortlist_keys: set[tuple[int, int]]) -> dict[tuple[int, int], dict[str, Any]]:
        stats_by_key: dict[tuple[int, int], dict[str, Any]] = {}
        for row in self.transcript_store.iter_records(self.transcript_store.transcript_feature_stats_path):
            key = (int(row["layer"]), int(row["feature_id"]))
            if key in shortlist_keys:
                stats_by_key[key] = row
        return stats_by_key

    def _collect_dolma_support(self, shortlist_keys: set[tuple[int, int]]) -> dict[tuple[int, int], dict[str, Any]]:
        context_limit = max(self.config.analysis.dolma_contexts_per_feature * 4, 32)
        support: dict[tuple[int, int], dict[str, Any]] = defaultdict(self._empty_dolma_support)
        for row in self.dolma_store.iter_records(self.dolma_store.dolma_contexts_path):
            key = (int(row["layer"]), int(row["feature_id"]))
            if key not in shortlist_keys:
                continue
            item = self._compact_dolma_context(row)
            entry = support[key]
            entry["context_count"] += 1
            total = float(item["feature_activation_total"])
            peak = float(item["feature_activation_peak"])
            fraction = float(item["active_token_fraction"])
            entry["total_sum"] += total
            entry["max_total"] = max(entry["max_total"], total)
            entry["max_peak"] = max(entry["max_peak"], peak)
            self._push_top_context(entry["top_total"], total, item, context_limit)
            self._push_top_context(entry["top_peak"], peak, item, context_limit)
            self._push_top_context(entry["top_fraction"], fraction, item, context_limit)
            doc_key = self._context_document_key(item)
            best_doc = entry["top_document"].get(doc_key)
            if best_doc is None or total > float(best_doc["feature_activation_total"]):
                entry["top_document"][doc_key] = item
        self._log(
            "build-feature-relevance",
            status="dolma_support",
            shortlisted_features=len(shortlist_keys),
            retained_features=len(support),
        )
        return dict(support)

    def _sentence_feature_maps_for_layer(
        self,
        layer: int,
        feature_ids: list[int],
    ) -> list[dict[int, float]]:
        target_features = set(feature_ids)
        seen: set[tuple[str, int, int]] = set()
        sentence_vectors: list[dict[int, float]] = []
        for row in self.transcript_store.iter_records(self.transcript_store.transcript_paired_path):
            if int(row["layer"]) != layer:
                continue
            sentence_id = row.get("sentence_id")
            if sentence_id is None:
                continue
            dedupe_key = (str(row["sample_id"]), layer, int(sentence_id))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            summary_lookup = {
                int(feature["feature_id"]): float(feature["total_activation"])
                for feature in row.get("sentence_feature_summaries", [])
                if int(feature["feature_id"]) in target_features
            }
            sentence_vectors.append(summary_lookup)
        return sentence_vectors

    def _load_alignment_rows(self) -> dict[tuple[int, int], dict[str, Any]]:
        if not self.alignment_path.exists():
            return {}
        rows: dict[tuple[int, int], dict[str, Any]] = {}
        for row in self.output_store.iter_records(self.alignment_path):
            key = (int(row["layer"]), int(row["feature_id"]))
            rows[key] = {
                "feature_total_activation": row.get("feature_total_activation"),
                "top_token_alignments": row.get("top_token_alignments", [])[:3],
                "top_span_alignments": row.get("top_span_alignments", [])[:3],
            }
        return rows

    def _load_existing_judged_rows(self) -> list[dict[str, Any]]:
        path = self.output_store.feature_concepts_path
        if not path.exists():
            return []
        return list(self.output_store.iter_records(path))

    def _filter_judge_input_rows(
        self,
        judge_input_rows: list[dict[str, Any]],
        existing_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not existing_rows:
            return judge_input_rows
        existing_by_key = {
            (int(row["layer"]), int(row["feature_id"])): row
            for row in existing_rows
        }
        filtered: list[dict[str, Any]] = []
        for row in judge_input_rows:
            key = (int(row["layer"]), int(row["feature_id"]))
            existing = existing_by_key.get(key)
            if not existing:
                filtered.append(row)
                continue
            if str(existing.get("judge_status")) != "ok":
                filtered.append(row)
                continue
            judge_output = existing.get("judge_output", {})
            if not judge_output or not judge_output.get("conceptual_label"):
                filtered.append(row)
        return filtered

    @staticmethod
    def _merge_judged_rows(
        existing_rows: list[dict[str, Any]],
        new_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged = {
            (int(row["layer"]), int(row["feature_id"])): row
            for row in existing_rows
        }
        for row in new_rows:
            merged[(int(row["layer"]), int(row["feature_id"]))] = row
        return sorted(
            merged.values(),
            key=lambda row: (
                int(row.get("transcript_relevance_rank", 10**9)),
                int(row["layer"]),
                int(row["feature_id"]),
            ),
        )

    @staticmethod
    def _empty_dolma_support() -> dict[str, Any]:
        return {
            "context_count": 0,
            "total_sum": 0.0,
            "max_total": 0.0,
            "max_peak": 0.0,
            "top_total": [],
            "top_peak": [],
            "top_fraction": [],
            "top_document": {},
        }

    @staticmethod
    def _compact_dolma_context(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "sample_id": row.get("sample_id"),
            "feature_activation_total": float(row.get("feature_activation_total", 0.0)),
            "feature_activation_peak": float(row.get("feature_activation_peak", 0.0)),
            "active_token_fraction": float(row.get("active_token_fraction", 0.0)),
            "dominant_scale": row.get("dominant_scale"),
            "scale_scores": row.get("scale_scores", {}),
            "window_text": row.get("window_text"),
            "top_token_snippets": row.get("top_token_snippets", []),
            "top_span_snippets": row.get("top_span_snippets", []),
            "top_sentence_snippets": row.get("top_sentence_snippets", []),
            "document_snippet": row.get("document_snippet"),
            "provenance": row.get("provenance", {}),
        }

    @staticmethod
    def _push_top_context(
        heap: list[tuple[float, str, dict[str, Any]]],
        score: float,
        item: dict[str, Any],
        limit: int,
    ) -> None:
        # Ties break on the sample id string so context selection is reproducible across processes.
        entry = (float(score), str(item.get("sample_id", "")), item)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
            return
        if entry > heap[0]:
            heapq.heapreplace(heap, entry)

    def _select_dolma_contexts(
        self,
        top_total_heap: list[tuple[float, str, dict[str, Any]]],
        top_peak_heap: list[tuple[float, str, dict[str, Any]]],
        top_fraction_heap: list[tuple[float, str, dict[str, Any]]],
        top_document: dict[str, dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        by_total = [item for _, _, item in sorted(top_total_heap, reverse=True)]
        by_peak = [item for _, _, item in sorted(top_peak_heap, reverse=True)]
        by_fraction = [item for _, _, item in sorted(top_fraction_heap, reverse=True)]
        by_document = sorted(
            top_document.values(),
            key=lambda item: float(item.get("feature_activation_total", 0.0)),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        seen_samples: set[str] = set()
        seen_documents: set[str] = set()

        def add(context: dict[str, Any], reason: str) -> bool:
            sample_id = str(context.get("sample_id", ""))
            if sample_id in seen_samples:
                return False
            seen_samples.add(sample_id)
            seen_documents.add(self._context_document_key(context))
            selected.append({**context, "selection_reason": reason})
            return True

        for reason, ranked in (
            ("max_total_activation", by_total),
            ("max_peak_activation", by_peak),
            ("broad_activation_fraction", by_fraction),
        ):
            for context in ranked:
                if add(context, reason):
                    break
            if len(selected) >= limit:
                return selected

        for context in by_document:
            doc_key = self._context_document_key(context)
            if doc_key not in seen_documents and add(context, "distinct_document"):
                if len(selected) >= limit:
                    return selected

        for context in by_total:
            if add(context, "high_total_fill") and len(selected) >= limit:
                return selected
        return selected

    @staticmethod
    def _reciprocal_rank(value: Any) -> float:
        if value is None:
            return 0.0
        rank = int(value)
        if rank <= 0:
            return 0.0
        return 1.0 / rank

    def _select_transcript_examples(
        self,
        examples: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if not examples or limit <= 0:
            return []
        ranked = sorted(examples, key=lambda item: float(item.get("activation", 0.0)), reverse=True)
        selected: list[dict[str, Any]] = []
        seen_positions: set[tuple[str, int]] = set()
        seen_stimuli: set[str] = set()
        seen_tokens: set[str] = set()

        def add(example: dict[str, Any], reason: str) -> bool:
            sample_id = str(example.get("sample_id", ""))
            token_position = int(example.get("token_position", -1))
            dedupe_key = (sample_id, token_position)
            if dedupe_key in seen_positions:
                return False
            seen_positions.add(dedupe_key)
            stimulus_id = example.get("stimulus_id")
            if stimulus_id:
                seen_stimuli.add(str(stimulus_id))
            token = example.get("token")
            if token:
                seen_tokens.add(str(token))
            selected.append(
                {
                    "sample_id": example.get("sample_id"),
                    "stimulus_id": example.get("stimulus_id"),
                    "token_position": token_position,
                    "token": example.get("token"),
                    "activation": float(example.get("activation", 0.0)),
                    "snippet_tokens": example.get("snippet_tokens", []),
                    "text": example.get("text"),
                    "selection_reason": reason,
                }
            )
            return True

        if ranked and len(selected) < limit:
            add(ranked[0], "max_activation")
        for example in ranked:
            stimulus_id = example.get("stimulus_id")
            if stimulus_id and str(stimulus_id) not in seen_stimuli and add(example, "distinct_stimulus"):
                if len(selected) >= limit:
                    return selected
        for example in ranked:
            token = example.get("token")
            if token and str(token) not in seen_tokens and add(example, "distinct_token"):
                if len(selected) >= limit:
                    return selected
        for example in ranked:
            if add(example, "high_activation_fill") and len(selected) >= limit:
                return selected
        return selected

    @staticmethod
    def _context_document_key(context: dict[str, Any]) -> str:
        provenance = context.get("provenance", {})
        return str(
            provenance.get("doc_id")
            or provenance.get("id")
            or provenance.get("path")
            or provenance.get("source")
            or context.get("sample_id")
        )

    @staticmethod
    def _judge_transcript_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for example in examples:
            compact.append(
                {
                    "stimulus_id": example.get("stimulus_id"),
                    "token": example.get("token"),
                    "token_position": example.get("token_position"),
                    "activation": example.get("activation"),
                    "snippet_tokens": example.get("snippet_tokens", []),
                    "selection_reason": example.get("selection_reason"),
                }
            )
        return compact

    @staticmethod
    def _judge_dolma_contexts(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for context in contexts:
            compact.append(
                {
                    "feature_activation_total": context.get("feature_activation_total"),
                    "feature_activation_peak": context.get("feature_activation_peak"),
                    "active_token_fraction": context.get("active_token_fraction"),
                    "dominant_scale": context.get("dominant_scale"),
                    "scale_scores": context.get("scale_scores", {}),
                    "selection_reason": context.get("selection_reason"),
                    "token_snippets": (context.get("top_token_snippets") or [])[:2],
                    "span_snippets": (context.get("top_span_snippets") or [])[:2],
                    "sentence_snippets": (context.get("top_sentence_snippets") or [])[:2],
                    "provenance": {
                        "doc_id": (context.get("provenance") or {}).get("doc_id"),
                        "id": (context.get("provenance") or {}).get("id"),
                        "source": (context.get("provenance") or {}).get("source"),
                        "path": (context.get("provenance") or {}).get("path"),
                    },
                }
            )
        return compact

    @staticmethod
    def _judge_alignment_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
        if not summary:
            return None
        return {
            "feature_total_activation": summary.get("feature_total_activation"),
            "top_token_alignments": (summary.get("top_token_alignments") or [])[:2],
            "top_span_alignments": (summary.get("top_span_alignments") or [])[:2],
        }

    def _has_rows(self, path: Path) -> bool:
        if not path.exists():
            return False
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    return True
        return False

    def _count_rows(self, path: Path) -> int:
        if not path.exists():
            return 0
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count

    def _log(self, event: str, **payload: Any) -> None:
        try:
            import resource

            # ru_maxrss is bytes on macOS and kilobytes on Linux.
            divisor = 1024.0**2 if sys.platform == "darwin" else 1024.0
            payload["rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor, 1)
        except Exception:
            pass
        print(f"[analyze-features] {event}: {payload}")
