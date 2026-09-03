from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis_neuro.dashboard_data import DashboardPaths, resolve_dashboard_paths


@dataclass(slots=True)
class AuditBundle:
    paths: DashboardPaths
    manifest: dict[str, Any]
    scripts: list[dict[str, Any]]
    relevance_rows: list[dict[str, Any]]
    relevance_by_key: dict[tuple[int, int], dict[str, Any]]
    correlations_by_key: dict[tuple[int, int], dict[str, Any]]
    concepts_by_key: dict[tuple[int, int], dict[str, Any]]
    judge_inputs_by_key: dict[tuple[int, int], dict[str, Any]]
    alignments_by_key: dict[tuple[int, int], list[dict[str, Any]]]
    shortlisted_feature_ids_by_layer: dict[int, set[int]]


def resolve_audit_paths(
    repo_root: str | Path | None = None,
    analysis_dir: str | Path = "outputs/remote_experiment_l4_l8_l13_l17_l22_l25",
    transcript_dir: str | Path | None = "outputs/remote_experiment_l4_l8_l13_l17_l22_l25",
    dolma_dir: str | Path | None = "outputs/remote_experiment_l4_l8_l13_l17_l22_l25",
) -> DashboardPaths:
    return resolve_dashboard_paths(
        repo_root=repo_root,
        analysis_dir=analysis_dir,
        transcript_dir=transcript_dir,
        dolma_dir=dolma_dir,
    )


def load_audit_bundle(paths: DashboardPaths) -> AuditBundle:
    manifest = _load_json(paths.manifest_path) if paths.manifest_path.exists() else {}
    relevance_rows = _load_jsonl_required(paths.relevance_path)
    correlations = _load_jsonl_optional(paths.correlations_path)
    concepts = _load_jsonl_optional(paths.concepts_path)
    judge_inputs = _load_jsonl_optional(paths.judge_input_path)
    alignments = _load_jsonl_optional(paths.alignment_path)
    scripts = list_transcript_scripts(paths)

    alignments_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in alignments:
        alignments_by_key[_feature_key(row)].append(row)

    shortlist_by_layer: dict[int, set[int]] = defaultdict(set)
    for row in relevance_rows:
        shortlist_by_layer[int(row["layer"])].add(int(row["feature_id"]))

    return AuditBundle(
        paths=paths,
        manifest=manifest,
        scripts=scripts,
        relevance_rows=relevance_rows,
        relevance_by_key={_feature_key(row): row for row in relevance_rows},
        correlations_by_key={_feature_key(row): row for row in correlations},
        concepts_by_key={_feature_key(row): row for row in concepts},
        judge_inputs_by_key={_feature_key(row): row for row in judge_inputs},
        alignments_by_key=alignments_by_key,
        shortlisted_feature_ids_by_layer=dict(shortlist_by_layer),
    )


def list_transcript_scripts(paths: DashboardPaths) -> list[dict[str, Any]]:
    transcript_path = paths.transcript_paired_path
    if transcript_path is None or not transcript_path.exists():
        return []

    scripts: dict[str, dict[str, Any]] = {}
    seen_windows: set[tuple[str, int]] = set()
    for row in _iter_jsonl(transcript_path):
        script_id = _script_id_from_row(row)
        script = scripts.setdefault(
            script_id,
            {
                "script_id": script_id,
                "stimulus_id": (row.get("provenance") or {}).get("stimulus_id"),
                "filename": (row.get("provenance") or {}).get("filename"),
                "path": (row.get("provenance") or {}).get("path"),
                "layers": set(),
                "window_count": 0,
            },
        )
        layer = int(row.get("layer", -1))
        script["layers"].add(layer)
        window_key = (_row_sample_id(row), layer)
        if window_key not in seen_windows:
            seen_windows.add(window_key)
            script["window_count"] += 1

    return sorted(
        [{**item, "layers": sorted(item["layers"])} for item in scripts.values()],
        key=lambda item: str(item["script_id"]),
    )


def build_script_audit_view(
    bundle: AuditBundle,
    script_id: str,
    layer: int,
) -> dict[str, Any]:
    transcript_path = bundle.paths.transcript_paired_path
    if transcript_path is None or not transcript_path.exists():
        return {"script": None, "top_script_features": [], "windows": [], "script_feature_reference": {}}

    shortlisted_ids = bundle.shortlisted_feature_ids_by_layer.get(int(layer), set())
    script_info: dict[str, Any] | None = None
    windows: dict[str, dict[str, Any]] = {}
    script_feature_totals: dict[int, dict[str, Any]] = {}
    script_feature_reference: dict[int, dict[str, float | int]] = {}

    for row in _iter_jsonl(transcript_path):
        if _script_id_from_row(row) != script_id or int(row.get("layer", -1)) != int(layer):
            continue

        provenance = row.get("provenance") or {}
        if script_info is None:
            script_info = {
                "script_id": script_id,
                "stimulus_id": provenance.get("stimulus_id"),
                "filename": provenance.get("filename"),
                "path": provenance.get("path"),
            }

        sample_id = _row_sample_id(row)
        window = windows.setdefault(
            sample_id,
            {
                "sample_id": sample_id,
                "script_id": script_id,
                "text": row.get("text", ""),
                "window_start": int(row.get("window_start", 0)),
                "window_end": int(row.get("window_end", row.get("window_start", 0))),
                "window_tokens": list(row.get("window_tokens", []) or []),
                "token_sentence_ids": list(row.get("token_sentence_ids", []) or []),
                "provenance": provenance,
                "sentences": {},
                "token_details": [],
                "window_features": [],
            },
        )

        if not window["window_features"]:
            feature_rows = _filter_feature_summaries(row.get("window_feature_summaries", []), shortlisted_ids)
            window["window_features"] = feature_rows[:25]

        sentence_id = int(row.get("sentence_id", 0) if row.get("sentence_id") is not None else 0)
        if sentence_id not in window["sentences"]:
            start = int(row.get("sentence_start_token_index", 0))
            end = int(row.get("sentence_end_token_index", row.get("token_position", 0)))
            sentence_features = _filter_feature_summaries(
                row.get("sentence_feature_summaries", []), shortlisted_ids
            )[:25]
            window["sentences"][sentence_id] = {
                "sentence_id": sentence_id,
                "sentence_start_token_index": start,
                "sentence_end_token_index": end,
                "sentence_text": _tokens_to_text(row.get("window_tokens", [])[start : end + 1]),
                "feature_summaries": sentence_features,
            }

        token_features = _filter_latent_activations(row.get("latent_activations", []), shortlisted_ids)
        window["token_details"].append(
            {
                "token_position": int(row["token_position"]),
                "token_id": int(row.get("token_id", row.get("token_position", 0))),
                "token": row["token"],
                "display_token": _display_token(row["token"], token_position=int(row["token_position"])),
                "sentence_id": sentence_id,
                "top_logits": row.get("top_logits", []),
                "latent_activations": token_features,
                "sentence_feature_summaries": window["sentences"][sentence_id]["feature_summaries"],
                "window_feature_summaries": window["window_features"],
            }
        )

    for window in windows.values():
        window["token_details"] = sorted(window["token_details"], key=lambda item: item["token_position"])

        if not window["window_tokens"]:
            window["window_tokens"] = [str(item.get("token", "")) for item in window["token_details"]]
        if not window["token_sentence_ids"]:
            window["token_sentence_ids"] = [int(item.get("sentence_id", 0)) for item in window["token_details"]]
        if not window["text"]:
            window["text"] = _tokens_to_text(window["window_tokens"])
        if window["token_details"]:
            window["window_end"] = max(int(window["window_end"]), int(window["token_details"][-1]["token_position"]))

        if not window["window_features"]:
            window["window_features"] = _summaries_from_token_details(window["token_details"])[:25]

        if _should_infer_sentences(window):
            inferred_sentences = _infer_sentences_from_tokens(window)
            window["sentences"] = {int(sentence["sentence_id"]): sentence for sentence in inferred_sentences}
            sentence_lookup = {int(sentence["sentence_id"]): sentence for sentence in inferred_sentences}
            token_to_sentence = _build_token_sentence_map(inferred_sentences)
            for token in window["token_details"]:
                token_position = int(token.get("token_position", 0))
                token["sentence_id"] = int(token_to_sentence.get(token_position, 0))
                token["sentence_feature_summaries"] = sentence_lookup[token["sentence_id"]]["feature_summaries"]
            window["token_sentence_ids"] = [int(token.get("sentence_id", 0)) for token in window["token_details"]]

        for sentence in window["sentences"].values():
            if not sentence.get("feature_summaries"):
                sentence_tokens = [
                    item for item in window["token_details"] if int(item.get("sentence_id", 0)) == int(sentence["sentence_id"])
                ]
                sentence["feature_summaries"] = _summaries_from_token_details(sentence_tokens)[:25]
            if not sentence.get("sentence_text"):
                start = int(sentence.get("sentence_start_token_index", 0))
                end = int(sentence.get("sentence_end_token_index", start))
                sentence["sentence_text"] = _tokens_to_text(window["window_tokens"][start : end + 1])

        for feature in window["window_features"]:
            feature_id = int(feature["feature_id"])
            aggregate = script_feature_totals.setdefault(
                feature_id,
                {
                    "feature_id": feature_id,
                    "total_activation": 0.0,
                    "max_activation": 0.0,
                    "window_count": 0,
                },
            )
            aggregate["total_activation"] += float(feature.get("total_activation", 0.0))
            aggregate["max_activation"] = max(
                float(aggregate["max_activation"]),
                float(feature.get("max_activation", 0.0)),
            )
            aggregate["window_count"] += 1
            reference = script_feature_reference.setdefault(
                feature_id,
                {
                    "window_total_sum": 0.0,
                    "window_count": 0,
                    "sentence_total_sum": 0.0,
                    "sentence_count": 0,
                },
            )
            reference["window_total_sum"] += float(feature.get("total_activation", 0.0))
            reference["window_count"] += 1

        for sentence in window["sentences"].values():
            for feature in sentence.get("feature_summaries", []):
                feature_id = int(feature["feature_id"])
                reference = script_feature_reference.setdefault(
                    feature_id,
                    {
                        "window_total_sum": 0.0,
                        "window_count": 0,
                        "sentence_total_sum": 0.0,
                        "sentence_count": 0,
                    },
                )
                reference["sentence_total_sum"] += float(feature.get("total_activation", 0.0))
                reference["sentence_count"] += 1

    top_script_features = sorted(
        script_feature_totals.values(),
        key=lambda item: (float(item["total_activation"]), float(item["max_activation"])),
        reverse=True,
    )[:50]

    output_windows = []
    for window in sorted(windows.values(), key=lambda item: int(item["window_start"])):
        window["sentences"] = sorted(window["sentences"].values(), key=lambda item: item["sentence_id"])
        output_windows.append(window)

    return {
        "script": script_info,
        "top_script_features": [_build_script_feature_row(bundle, int(layer), row) for row in top_script_features],
        "windows": output_windows,
        "script_feature_reference": script_feature_reference,
    }


def build_focus_features(
    bundle: AuditBundle,
    layer: int,
    script_view: dict[str, Any],
    window: dict[str, Any],
    mode: str,
    lens: str = "strongest",
    token_position: int | None = None,
    sentence_id: int | None = None,
    span_start: int | None = None,
    span_end: int | None = None,
    feature_filter: int | None = None,
) -> list[dict[str, Any]]:
    shortlisted_ids = bundle.shortlisted_feature_ids_by_layer.get(int(layer), set())
    token_details = window.get("token_details", [])
    sentence_map = {int(item["sentence_id"]): item for item in window.get("sentences", [])}
    script_feature_reference = script_view.get("script_feature_reference", {})
    feature_metrics: dict[int, dict[str, Any]] = {}
    coverage_map: dict[int, set[str]] = defaultdict(set)
    source_note = ""

    def add_metric(feature_id: int, payload: dict[str, Any], coverage: str) -> None:
        metrics = feature_metrics.setdefault(
            feature_id,
            {
                "feature_id": int(feature_id),
                "token_total_activation": 0.0,
                "token_max_activation": 0.0,
                "active_token_count": 0,
                "token_positions": [],
                "span_total_activation": 0.0,
                "span_max_activation": 0.0,
                "active_token_fraction": 0.0,
                "sentence_total_activation": None,
                "sentence_max_activation": None,
                "window_total_activation": None,
                "window_max_activation": None,
                "source": coverage,
            },
        )
        coverage_map[feature_id].add(coverage)
        for key, value in payload.items():
            if key == "token_positions":
                for position in value:
                    if position not in metrics["token_positions"]:
                        metrics["token_positions"].append(position)
                continue
            if key in {"token_total_activation", "span_total_activation"}:
                metrics[key] += float(value)
            elif key in {"token_max_activation", "span_max_activation"}:
                metrics[key] = max(float(metrics[key]), float(value))
            elif key == "active_token_count":
                metrics[key] += int(value)
            else:
                metrics[key] = value

    if mode == "token" and token_position is not None:
        token = token_details[token_position]
        for latent in token.get("latent_activations", []):
            feature_id = int(latent["latent_id"])
            add_metric(
                feature_id,
                {
                    "token_total_activation": float(latent["activation"]),
                    "token_max_activation": float(latent["activation"]),
                    "span_total_activation": float(latent["activation"]),
                    "span_max_activation": float(latent["activation"]),
                    "active_token_count": 1,
                    "active_token_fraction": 1.0,
                    "token_positions": [token_position],
                },
                "token",
            )
        if not feature_metrics:
            source_note = "No shortlisted token-level features were stored here. Showing sentence/window fallback."
            for feature in token.get("sentence_feature_summaries", []):
                feature_id = int(feature["feature_id"])
                add_metric(
                    feature_id,
                    {
                        "sentence_total_activation": float(feature.get("total_activation", 0.0)),
                        "sentence_max_activation": float(feature.get("max_activation", 0.0)),
                        "active_token_fraction": float(feature.get("active_token_fraction", 0.0)),
                    },
                    "sentence",
                )
            if not feature_metrics:
                for feature in token.get("window_feature_summaries", []):
                    feature_id = int(feature["feature_id"])
                    add_metric(
                        feature_id,
                        {
                            "window_total_activation": float(feature.get("total_activation", 0.0)),
                            "window_max_activation": float(feature.get("max_activation", 0.0)),
                            "active_token_fraction": float(feature.get("active_token_fraction", 0.0)),
                        },
                        "window",
                    )

    elif mode == "sentence" and sentence_id is not None and sentence_id in sentence_map:
        sentence = sentence_map[sentence_id]
        source_note = "Showing sentence-level pooled feature summaries."
        for feature in sentence.get("feature_summaries", []):
            feature_id = int(feature["feature_id"])
            add_metric(
                feature_id,
                {
                    "sentence_total_activation": float(feature.get("total_activation", 0.0)),
                    "sentence_max_activation": float(feature.get("max_activation", 0.0)),
                    "active_token_fraction": float(feature.get("active_token_fraction", 0.0)),
                },
                "sentence",
            )

    elif mode == "span" and span_start is not None and span_end is not None:
        span_tokens = [token for token in token_details if span_start <= int(token["token_position"]) <= span_end]
        for token in span_tokens:
            for latent in token.get("latent_activations", []):
                feature_id = int(latent["latent_id"])
                add_metric(
                    feature_id,
                    {
                        "token_total_activation": float(latent["activation"]),
                        "token_max_activation": float(latent["activation"]),
                        "span_total_activation": float(latent["activation"]),
                        "span_max_activation": float(latent["activation"]),
                        "active_token_count": 1,
                        "token_positions": [int(token["token_position"])],
                    },
                    "token",
                )
        span_length = max(1, int(span_end) - int(span_start) + 1)
        for metrics in feature_metrics.values():
            metrics["active_token_fraction"] = float(metrics["active_token_count"]) / float(span_length)
        if not feature_metrics:
            source_note = "No shortlisted token-level features were stored inside this span. Showing pooled sentence/window fallback."
            touched_sentences = {
                int(token["sentence_id"])
                for token in token_details
                if span_start <= int(token["token_position"]) <= span_end
            }
            pooled_sentence_features: dict[int, dict[str, Any]] = {}
            for sid in touched_sentences:
                for feature in sentence_map.get(sid, {}).get("feature_summaries", []):
                    feature_id = int(feature["feature_id"])
                    aggregate = pooled_sentence_features.setdefault(
                        feature_id,
                        {
                            "total_activation": 0.0,
                            "max_activation": 0.0,
                            "active_token_fraction": 0.0,
                        },
                    )
                    aggregate["total_activation"] += float(feature.get("total_activation", 0.0))
                    aggregate["max_activation"] = max(
                        float(aggregate["max_activation"]),
                        float(feature.get("max_activation", 0.0)),
                    )
                    aggregate["active_token_fraction"] = max(
                        float(aggregate["active_token_fraction"]),
                        float(feature.get("active_token_fraction", 0.0)),
                    )
            for feature_id, feature in pooled_sentence_features.items():
                add_metric(
                    feature_id,
                    {
                        "sentence_total_activation": feature["total_activation"],
                        "sentence_max_activation": feature["max_activation"],
                        "active_token_fraction": feature["active_token_fraction"],
                    },
                    "sentence",
                )
            if not feature_metrics:
                for feature in window.get("window_features", []):
                    feature_id = int(feature["feature_id"])
                    add_metric(
                        feature_id,
                        {
                            "window_total_activation": float(feature.get("total_activation", 0.0)),
                            "window_max_activation": float(feature.get("max_activation", 0.0)),
                            "active_token_fraction": float(feature.get("active_token_fraction", 0.0)),
                        },
                        "window",
                    )

    rows = []
    for feature_id, metrics in feature_metrics.items():
        if feature_id not in shortlisted_ids:
            continue
        detail = build_feature_evidence(bundle, int(layer), feature_id)
        detail["local_metrics"] = metrics
        detail["coverage"] = sorted(coverage_map.get(feature_id, set()))
        detail["source_note"] = source_note or None
        detail["distinctiveness"] = _feature_distinctiveness(
            detail=detail,
            local_metrics=metrics,
            script_reference=script_feature_reference.get(feature_id, {}),
        )
        rows.append(detail)

    if feature_filter is not None:
        rows = [row for row in rows if int(row["feature_id"]) == int(feature_filter)]
    return sorted(rows, key=lambda row: _feature_card_sort_key(row, lens=lens), reverse=True)


def build_feature_lookup(
    bundle: AuditBundle,
    layer: int,
    script_view: dict[str, Any],
    feature_id: int,
) -> dict[str, Any]:
    detail = build_feature_evidence(bundle, int(layer), int(feature_id))
    token_hits: list[dict[str, Any]] = []
    sentence_hits: list[dict[str, Any]] = []
    window_hits: list[dict[str, Any]] = []

    for window in script_view.get("windows", []):
        token_matches = []
        for token in window.get("token_details", []):
            for latent in token.get("latent_activations", []):
                if int(latent.get("latent_id", -1)) != int(feature_id):
                    continue
                token_matches.append(
                    {
                        "sample_id": window.get("sample_id"),
                        "token_position": int(token.get("token_position", 0)),
                        "token": token.get("token"),
                        "display_token": token.get("display_token"),
                        "activation": float(latent.get("activation", 0.0)),
                        "sentence_id": int(token.get("sentence_id", -1)),
                        "window_start": int(window.get("window_start", 0)),
                        "window_end": int(window.get("window_end", 0)),
                        "snippet": _token_context_snippet(window, int(token.get("token_position", 0))),
                    }
                )
        token_hits.extend(
            sorted(token_matches, key=lambda item: float(item["activation"]), reverse=True)[:8]
        )

        sentence_matches = []
        for sentence in window.get("sentences", []):
            for feature in sentence.get("feature_summaries", []):
                if int(feature.get("feature_id", -1)) != int(feature_id):
                    continue
                sentence_matches.append(
                    {
                        "sample_id": window.get("sample_id"),
                        "sentence_id": int(sentence.get("sentence_id", -1)),
                        "sentence_text": sentence.get("sentence_text", ""),
                        "total_activation": float(feature.get("total_activation", 0.0)),
                        "max_activation": float(feature.get("max_activation", 0.0)),
                        "active_token_fraction": float(feature.get("active_token_fraction", 0.0)),
                    }
                )
        sentence_hits.extend(
            sorted(
                sentence_matches,
                key=lambda item: (float(item["total_activation"]), float(item["max_activation"])),
                reverse=True,
            )[:6]
        )

        for feature in window.get("window_features", []):
            if int(feature.get("feature_id", -1)) != int(feature_id):
                continue
            window_hits.append(
                {
                    "sample_id": window.get("sample_id"),
                    "window_start": int(window.get("window_start", 0)),
                    "window_end": int(window.get("window_end", 0)),
                    "text": window.get("text", ""),
                    "total_activation": float(feature.get("total_activation", 0.0)),
                    "max_activation": float(feature.get("max_activation", 0.0)),
                    "active_token_fraction": float(feature.get("active_token_fraction", 0.0)),
                }
            )

    detail["script_token_hits"] = sorted(
        token_hits,
        key=lambda item: float(item["activation"]),
        reverse=True,
    )[:10]
    detail["script_sentence_hits"] = sorted(
        sentence_hits,
        key=lambda item: (float(item["total_activation"]), float(item["max_activation"])),
        reverse=True,
    )[:8]
    detail["script_window_hits"] = sorted(
        window_hits,
        key=lambda item: (float(item["total_activation"]), float(item["max_activation"])),
        reverse=True,
    )[:6]
    detail["lookup_feature_id"] = int(feature_id)
    return detail


def build_feature_evidence(bundle: AuditBundle, layer: int, feature_id: int) -> dict[str, Any]:
    key = (int(layer), int(feature_id))
    relevance = bundle.relevance_by_key.get(key, {})
    concept = bundle.concepts_by_key.get(key, {})
    judge_output = concept.get("judge_output", {})
    judge_status = concept.get("judge_status")
    alignments = sorted(
        bundle.alignments_by_key.get(key, []),
        key=lambda row: float(row.get("feature_total_activation", 0.0)),
        reverse=True,
    )
    correlations = bundle.correlations_by_key.get(key, {}).get("top_correlated_features", [])
    judge_input = bundle.judge_inputs_by_key.get(key, {})
    return {
        "layer": int(layer),
        "feature_id": int(feature_id),
        "transcript_relevance_rank": relevance.get("transcript_relevance_rank"),
        "transcript_relevance_score": relevance.get("transcript_relevance_score"),
        "transcript_metrics": relevance.get("transcript_metrics", {}),
        "transcript_support": relevance.get("transcript_support", {}),
        "label": judge_output.get("conceptual_label"),
        "feature_type": judge_output.get("feature_type"),
        "confidence": judge_output.get("confidence"),
        "summary": judge_output.get("summary"),
        "transcript_rationale": judge_output.get("transcript_relevance_rationale"),
        "judge_label": judge_output.get("conceptual_label"),
        "judge_summary": judge_output.get("summary"),
        "judge_evidence_for": judge_output.get("evidence_for", []),
        "judge_evidence_against": judge_output.get("evidence_against", []),
        "judge_uncertainty": judge_output.get("uncertainty"),
        "judge_follow_up": judge_output.get("follow_up", []),
        "judge_status": judge_status,
        "judge_coverage_status": "ok" if judge_output else (str(judge_status) if judge_status else "missing"),
        "has_judge": bool(judge_output),
        "has_alignment": bool(alignments),
        "has_dolma": bool(relevance.get("top_dolma_contexts")),
        "top_transcript_examples": relevance.get("top_transcript_examples", [])[:5],
        "top_dolma_contexts": relevance.get("top_dolma_contexts", [])[:5],
        "top_correlated_features": correlations[:5],
        "alignment_summary": [
            {
                "sample_id": row.get("sample_id"),
                "feature_total_activation": row.get("feature_total_activation"),
                "top_token_alignments": row.get("top_token_alignments", [])[:2],
                "top_span_alignments": row.get("top_span_alignments", [])[:2],
            }
            for row in alignments[:2]
        ],
        "judge_input": judge_input.get("evidence"),
        "raw": {
            "relevance": relevance,
            "concept": concept,
            "alignments": alignments[:2],
            "correlations": correlations,
        },
    }


def build_token_tooltip(
    bundle: AuditBundle,
    layer: int,
    token_detail: dict[str, Any],
) -> str:
    token_features = [
        build_feature_evidence(bundle, layer, int(item["latent_id"]))
        | {"activation": float(item["activation"])}
        for item in token_detail.get("latent_activations", [])[:3]
    ]
    lines = [
        f"token {token_detail['token_position']}",
        f"sentence {token_detail['sentence_id']}",
    ]
    if token_features:
        for feature in token_features:
            label = feature.get("label") or f"feature {feature['feature_id']}"
            lines.append(f"{label}: {feature['activation']:.2f}")
    else:
        sentence_features = token_detail.get("sentence_feature_summaries", [])[:2]
        if sentence_features:
            lines.append("no stored token-level shortlist features")
            for feature in sentence_features:
                label = _feature_label(bundle, int(layer), int(feature["feature_id"]))
                lines.append(f"{label}: sent {float(feature.get('max_activation', 0.0)):.2f}")
        else:
            lines.append("no shortlisted features at token level")
    return "\n".join(lines)


def sentence_options(script_view: dict[str, Any], search: str = "") -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    needle = search.strip().lower()
    for window in script_view.get("windows", []):
        for sentence in window.get("sentences", []):
            sentence_text = str(sentence.get("sentence_text", ""))
            if needle and needle not in sentence_text.lower():
                continue
            options.append(
                {
                    "sample_id": window["sample_id"],
                    "sentence_id": int(sentence["sentence_id"]),
                    "label": f"{sentence['sentence_id']} | {_compact_text(sentence_text, 120)}",
                }
            )
    return options


def window_options(script_view: dict[str, Any], search: str = "") -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    needle = search.strip().lower()
    for window in script_view.get("windows", []):
        text = str(window.get("text", ""))
        if needle and needle not in text.lower():
            continue
        options.append(
            {
                "sample_id": window["sample_id"],
                "label": f"{window['window_start']}:{window['window_end']} | {_compact_text(text, 120)}",
            }
        )
    return options


def find_script_default(bundle: AuditBundle, preferred_script: str = "shapesphysical") -> dict[str, Any] | None:
    scripts = bundle.scripts
    if not scripts:
        return None
    for script in scripts:
        name = str(script.get("script_id", "")).lower()
        stimulus = str(script.get("stimulus_id", "")).lower()
        if preferred_script.lower() in name or preferred_script.lower() in stimulus:
            return script
    return scripts[0]


def render_token_text(token: str, token_position: int) -> str:
    return _display_token(token, token_position)


def _build_script_feature_row(bundle: AuditBundle, layer: int, row: dict[str, Any]) -> dict[str, Any]:
    detail = build_feature_evidence(bundle, layer, int(row["feature_id"]))
    return {
        **row,
        "label": detail.get("label"),
        "feature_type": detail.get("feature_type"),
        "confidence": detail.get("confidence"),
        "transcript_relevance_rank": detail.get("transcript_relevance_rank"),
        "has_judge": detail.get("has_judge"),
        "has_alignment": detail.get("has_alignment"),
        "has_dolma": detail.get("has_dolma"),
    }


def _feature_label(bundle: AuditBundle, layer: int, feature_id: int) -> str:
    detail = build_feature_evidence(bundle, layer, feature_id)
    return detail.get("label") or f"feature {feature_id}"


def _feature_card_sort_key(row: dict[str, Any], lens: str = "strongest") -> tuple[float, float, float]:
    metrics = row.get("local_metrics", {})
    if lens == "distinctive":
        return (
            float(row.get("distinctiveness", 0.0)),
            float(metrics.get("span_total_activation") or metrics.get("sentence_total_activation") or metrics.get("window_total_activation") or 0.0),
            -float(row.get("transcript_relevance_rank") or 10**9),
        )
    return (
        float(metrics.get("span_total_activation") or metrics.get("sentence_total_activation") or metrics.get("window_total_activation") or 0.0),
        float(metrics.get("token_max_activation") or metrics.get("sentence_max_activation") or metrics.get("window_max_activation") or 0.0),
        -float(row.get("transcript_relevance_rank") or 10**9),
    )


def _feature_distinctiveness(
    detail: dict[str, Any],
    local_metrics: dict[str, Any],
    script_reference: dict[str, float | int],
) -> float:
    local_primary = float(
        local_metrics.get("span_total_activation")
        or local_metrics.get("sentence_total_activation")
        or local_metrics.get("window_total_activation")
        or local_metrics.get("token_max_activation")
        or 0.0
    )
    sentence_count = max(1, int(script_reference.get("sentence_count", 0) or 0))
    window_count = max(1, int(script_reference.get("window_count", 0) or 0))
    sentence_mean = float(script_reference.get("sentence_total_sum", 0.0)) / float(sentence_count)
    window_mean = float(script_reference.get("window_total_sum", 0.0)) / float(window_count)
    transcript_baseline = max(sentence_mean, window_mean, 1e-6)
    transcript_support = detail.get("transcript_support", {})
    active_fraction = float(transcript_support.get("active_token_fraction", 0.0) or 0.0)
    rarity_bonus = 1.0 / max(active_fraction, 1e-3)
    return float(local_primary / transcript_baseline) * min(rarity_bonus, 25.0)


def _filter_latent_activations(latents: list[dict[str, Any]], shortlisted_ids: set[int]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "latent_id": int(latent.get("latent_id", -1)),
                "activation": float(latent.get("activation", 0.0)),
            }
            for latent in latents
            if int(latent.get("latent_id", -1)) in shortlisted_ids
        ],
        key=lambda item: float(item["activation"]),
        reverse=True,
    )


def _filter_feature_summaries(feature_rows: list[dict[str, Any]], shortlisted_ids: set[int]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "feature_id": int(feature.get("feature_id", -1)),
                "total_activation": float(feature.get("total_activation", 0.0)),
                "max_activation": float(feature.get("max_activation", 0.0)),
                "mean_activation": float(feature.get("mean_activation", 0.0)),
                "active_token_fraction": float(feature.get("active_token_fraction", 0.0)),
            }
            for feature in feature_rows
            if int(feature.get("feature_id", -1)) in shortlisted_ids
        ],
        key=lambda item: (item["total_activation"], item["max_activation"]),
        reverse=True,
    )


def _summaries_from_token_details(token_details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: dict[int, dict[str, float | int]] = {}
    token_count = max(1, len(token_details))
    for token in token_details:
        for latent in token.get("latent_activations", []):
            feature_id = int(latent.get("latent_id", -1))
            if feature_id < 0:
                continue
            activation = float(latent.get("activation", 0.0))
            aggregate = aggregates.setdefault(
                feature_id,
                {
                    "feature_id": feature_id,
                    "total_activation": 0.0,
                    "max_activation": 0.0,
                    "active_token_count": 0,
                },
            )
            aggregate["total_activation"] += activation
            aggregate["max_activation"] = max(float(aggregate["max_activation"]), activation)
            aggregate["active_token_count"] += 1

    rows = []
    for feature_id, aggregate in aggregates.items():
        total_activation = float(aggregate["total_activation"])
        active_token_count = int(aggregate["active_token_count"])
        rows.append(
            {
                "feature_id": int(feature_id),
                "total_activation": total_activation,
                "max_activation": float(aggregate["max_activation"]),
                "mean_activation": total_activation / max(1, active_token_count),
                "active_token_fraction": float(active_token_count) / float(token_count),
            }
        )
    return sorted(rows, key=lambda item: (item["total_activation"], item["max_activation"]), reverse=True)


def _should_infer_sentences(window: dict[str, Any]) -> bool:
    sentences = list(window.get("sentences", {}).values())
    if not sentences:
        return True
    if len(sentences) != 1:
        return False
    sentence = sentences[0]
    token_details = list(window.get("token_details", []))
    if not token_details:
        return False
    only_sentence_zero = all(int(token.get("sentence_id", 0)) == 0 for token in token_details)
    sentence_has_features = bool(sentence.get("feature_summaries"))
    window_has_multiple_boundaries = _count_sentence_boundaries(window.get("window_tokens", [])) > 1
    return only_sentence_zero and not sentence_has_features and window_has_multiple_boundaries


def _infer_sentences_from_tokens(window: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = list(window.get("window_tokens", []))
    token_details = list(window.get("token_details", []))
    if not tokens:
        return []

    sentence_ranges: list[tuple[int, int]] = []
    start = 0
    for idx, token in enumerate(tokens):
        if _is_sentence_boundary_token(str(token)):
            sentence_ranges.append((start, idx))
            start = idx + 1
    if start < len(tokens):
        sentence_ranges.append((start, len(tokens) - 1))
    if not sentence_ranges:
        sentence_ranges = [(0, len(tokens) - 1)]

    sentences: list[dict[str, Any]] = []
    for sentence_id, (start_idx, end_idx) in enumerate(sentence_ranges):
        sentence_tokens = [
            item for item in token_details if start_idx <= int(item.get("token_position", 0)) <= end_idx
        ]
        sentences.append(
            {
                "sentence_id": int(sentence_id),
                "sentence_start_token_index": int(start_idx),
                "sentence_end_token_index": int(end_idx),
                "sentence_text": _tokens_to_text(tokens[start_idx : end_idx + 1]),
                "feature_summaries": _summaries_from_token_details(sentence_tokens)[:25],
            }
        )
    return sentences


def _build_token_sentence_map(sentences: list[dict[str, Any]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for sentence in sentences:
        sentence_id = int(sentence["sentence_id"])
        start = int(sentence["sentence_start_token_index"])
        end = int(sentence["sentence_end_token_index"])
        for token_position in range(start, end + 1):
            mapping[token_position] = sentence_id
    return mapping


def _count_sentence_boundaries(tokens: list[str]) -> int:
    return sum(1 for token in tokens if _is_sentence_boundary_token(str(token)))


def _is_sentence_boundary_token(token: str) -> bool:
    normalized = token.strip()
    if normalized in {".", "!", "?", "<0x0A>"}:
        return True
    return normalized.endswith((".", "!", "?"))


def _script_id_from_row(row: dict[str, Any]) -> str:
    provenance = row.get("provenance") or {}
    return str(
        provenance.get("path")
        or provenance.get("filename")
        or provenance.get("stimulus_id")
        or str(row.get("sample_id", "")).split(":", 1)[0]
    )


def _row_sample_id(row: dict[str, Any]) -> str:
    sample_id = row.get("sample_id")
    if sample_id is not None:
        return str(sample_id)
    script_id = _script_id_from_row(row)
    window_start = int(row.get("window_start", 0))
    return f"{script_id}:{window_start}"


def _display_token(token: str, token_position: int) -> str:
    text = str(token)
    if text.startswith("▁"):
        core = text[1:] or "_"
        return (" " if token_position > 0 else "") + core
    if text == "<0x0A>":
        return " ↵ "
    return text


def _tokens_to_text(tokens: list[str]) -> str:
    return "".join(_display_token(token, idx) for idx, token in enumerate(tokens)).strip()


def _token_context_snippet(window: dict[str, Any], token_position: int, radius: int = 4) -> str:
    tokens = window.get("window_tokens", [])
    start = max(0, int(token_position) - int(radius))
    end = min(len(tokens), int(token_position) + int(radius) + 1)
    return _tokens_to_text(tokens[start:end])


def _compact_text(text: str, width: int) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= width:
        return normalized
    return normalized[: max(0, width - 3)] + "..."


def _feature_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["layer"]), int(row["feature_id"])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_required(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return list(_iter_jsonl(path))


def _load_jsonl_optional(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(_iter_jsonl(path))


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)
