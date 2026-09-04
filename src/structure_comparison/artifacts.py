"""TR-level predictor and target bundles built from feature-run artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from structure_comparison.alignment import (
    TrBin,
    assign_words_to_trs,
    build_token_to_word_index,
    extract_global_tokens,
    group_tokens_with_model_tokenizer,
    load_tr_bins,
    load_word_rows,
    resolve_transcript_paths,
    validate_tr_alignment,
    validate_word_alignment,
)
from structure_comparison.utils import FeatureKey, feature_name, load_jsonl, safe_divide_rows, write_json


@dataclass(frozen=True, slots=True)
class FeatureArtifacts:
    predictor_presence: np.ndarray
    predictor_mass: np.ndarray
    predictor_average: np.ndarray
    predictor_peak: np.ndarray
    predictor_count: np.ndarray
    predictor_feature_names: np.ndarray
    predictor_layers: np.ndarray
    predictor_keys: tuple[FeatureKey, ...]
    lm_target_mass: np.ndarray
    lm_target_average: np.ndarray
    lm_target_count: np.ndarray
    lm_target_names: np.ndarray
    lm_target_layers: np.ndarray
    lm_target_keys: tuple[FeatureKey, ...]
    tr_indices: np.ndarray
    start_s: np.ndarray
    end_s: np.ndarray
    sample_ids: np.ndarray
    stimulus_id: str


@dataclass(frozen=True, slots=True)
class FamilyMatrices:
    family_name: str
    predictor_values: np.ndarray
    predictor_feature_names: np.ndarray
    predictor_base_feature_names: np.ndarray
    predictor_layers: np.ndarray
    lm_target_values: np.ndarray
    lm_target_names: np.ndarray
    lm_target_layers: np.ndarray
    tr_indices: np.ndarray
    sample_ids: np.ndarray


def build_tr_feature_artifacts(
    feature_run_dir: Path,
    transcript_root: Path,
    stimulus_id: str,
    output_dir: Path,
    lm_targets_per_layer: int,
    predictor_top_k: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    required_feature_paths = {
        "transcript_paired_records": resolve_transcript_paired_path(feature_run_dir),
        "selected_features_for_alignment": feature_run_dir / "selected_features_for_alignment.jsonl",
        "feature_concepts": feature_run_dir / "feature_concepts.jsonl",
        "manifest": feature_run_dir / "manifest.json",
    }
    for label, path in required_feature_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required feature-run artifact {label}: {path}")

    transcript_paths = resolve_transcript_paths(transcript_root, stimulus_id)
    transcript_txt = transcript_paths.transcript_txt.read_text(encoding="utf-8").strip()
    word_rows = load_word_rows(transcript_paths.words_tsv)
    tr_bins = load_tr_bins(transcript_paths.tr_aligned_tsv)
    metadata = json.loads(transcript_paths.metadata_json.read_text(encoding="utf-8"))

    predictor_keys = load_predictor_feature_keys(
        required_feature_paths["selected_features_for_alignment"],
        top_k=predictor_top_k,
    )
    family_layers = tuple(sorted({layer for layer, _feature_id in predictor_keys}))
    lm_target_keys = select_lm_target_feature_keys(
        feature_concepts_path=required_feature_paths["feature_concepts"],
        predictor_keys=predictor_keys,
        targets_per_layer=lm_targets_per_layer,
        target_layers=family_layers,
    )

    token_stream = extract_global_tokens(
        transcript_paired_path=required_feature_paths["transcript_paired_records"],
        stimulus_id=stimulus_id,
        layer=family_layers[0],
    )
    token_word_groups = group_tokens_with_model_tokenizer(
        model_id=token_stream.model_id,
        tokens=token_stream.tokens,
        transcript_text=transcript_txt,
    )
    validate_word_alignment(token_word_groups, word_rows, transcript_txt)
    token_to_word_index = build_token_to_word_index(token_word_groups)
    word_to_tr_index = assign_words_to_trs(
        word_rows=word_rows,
        tr_bins=tr_bins,
        stimulus_onset_s=float(metadata.get("stimulus_onset_s", 0.0)),
    )
    validate_tr_alignment(word_rows, word_to_tr_index, tr_bins, float(metadata.get("stimulus_onset_s", 0.0)))

    feature_artifacts = aggregate_tr_feature_views(
        transcript_paired_path=required_feature_paths["transcript_paired_records"],
        stimulus_id=stimulus_id,
        token_to_word_index=token_to_word_index,
        word_to_tr_index=word_to_tr_index,
        tr_bins=tr_bins,
        predictor_keys=predictor_keys,
        lm_target_keys=lm_target_keys,
    )
    summary = write_feature_artifact_bundle(
        feature_artifacts=feature_artifacts,
        output_dir=output_dir,
        feature_run_dir=feature_run_dir,
        predictor_top_k=predictor_top_k,
    )
    return {"artifacts": feature_artifacts, "summary": summary}


def build_pooled_tr_feature_artifacts(
    feature_run_dir: Path,
    transcript_root: Path,
    stimulus_ids: list[str],
    output_dir: Path,
    lm_targets_per_layer: int,
    predictor_top_k: int | None = None,
) -> dict[str, Any]:
    if not stimulus_ids:
        raise ValueError("stimulus_ids must not be empty.")
    output_dir.mkdir(parents=True, exist_ok=True)
    required_feature_paths = {
        "transcript_paired_records": resolve_transcript_paired_path(feature_run_dir),
        "selected_features_for_alignment": feature_run_dir / "selected_features_for_alignment.jsonl",
        "feature_concepts": feature_run_dir / "feature_concepts.jsonl",
        "manifest": feature_run_dir / "manifest.json",
    }
    for label, path in required_feature_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required feature-run artifact {label}: {path}")

    predictor_keys = load_predictor_feature_keys(
        required_feature_paths["selected_features_for_alignment"],
        top_k=predictor_top_k,
    )
    family_layers = tuple(sorted({layer for layer, _feature_id in predictor_keys}))
    lm_target_keys = select_lm_target_feature_keys(
        feature_concepts_path=required_feature_paths["feature_concepts"],
        predictor_keys=predictor_keys,
        targets_per_layer=lm_targets_per_layer,
        target_layers=family_layers,
    )

    per_story_artifacts: list[FeatureArtifacts] = []
    for stimulus_id in stimulus_ids:
        transcript_paths = resolve_transcript_paths(transcript_root, stimulus_id)
        transcript_txt = transcript_paths.transcript_txt.read_text(encoding="utf-8").strip()
        word_rows = load_word_rows(transcript_paths.words_tsv)
        tr_bins = load_tr_bins(transcript_paths.tr_aligned_tsv)
        metadata = json.loads(transcript_paths.metadata_json.read_text(encoding="utf-8"))

        token_stream = extract_global_tokens(
            transcript_paired_path=required_feature_paths["transcript_paired_records"],
            stimulus_id=stimulus_id,
            layer=family_layers[0],
        )
        token_word_groups = group_tokens_with_model_tokenizer(
            model_id=token_stream.model_id,
            tokens=token_stream.tokens,
            transcript_text=transcript_txt,
        )
        validate_word_alignment(token_word_groups, word_rows, transcript_txt)
        token_to_word_index = build_token_to_word_index(token_word_groups)
        word_to_tr_index = assign_words_to_trs(
            word_rows=word_rows,
            tr_bins=tr_bins,
            stimulus_onset_s=float(metadata.get("stimulus_onset_s", 0.0)),
        )
        validate_tr_alignment(word_rows, word_to_tr_index, tr_bins, float(metadata.get("stimulus_onset_s", 0.0)))

        per_story_artifacts.append(
            aggregate_tr_feature_views(
                transcript_paired_path=required_feature_paths["transcript_paired_records"],
                stimulus_id=stimulus_id,
                token_to_word_index=token_to_word_index,
                word_to_tr_index=word_to_tr_index,
                tr_bins=tr_bins,
                predictor_keys=predictor_keys,
                lm_target_keys=lm_target_keys,
            )
        )

    feature_artifacts = concatenate_feature_artifacts(per_story_artifacts)
    summary = write_feature_artifact_bundle(
        feature_artifacts=feature_artifacts,
        output_dir=output_dir,
        feature_run_dir=feature_run_dir,
        predictor_top_k=predictor_top_k,
    )
    summary["stimulus_ids"] = [str(stimulus_id) for stimulus_id in stimulus_ids]
    summary["tr_count_by_stimulus"] = {
        artifact.stimulus_id: int(artifact.predictor_mass.shape[0]) for artifact in per_story_artifacts
    }
    write_json(output_dir / "intermediates" / "tr_feature_summary.json", summary)
    return {"artifacts": feature_artifacts, "summary": summary}


def write_feature_artifact_bundle(
    feature_artifacts: FeatureArtifacts,
    output_dir: Path,
    feature_run_dir: Path,
    predictor_top_k: int | None = None,
) -> dict[str, Any]:
    intermediates_dir = output_dir / "intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = feature_artifacts.sample_ids
    save_feature_view_npz(
        path=intermediates_dir / "tr_feature_presence.npz",
        values=feature_artifacts.predictor_presence.astype(np.uint8),
        sample_ids=sample_ids,
        feature_names=feature_artifacts.predictor_feature_names,
        layers=feature_artifacts.predictor_layers,
        tr_indices=feature_artifacts.tr_indices,
        start_s=feature_artifacts.start_s,
        end_s=feature_artifacts.end_s,
        stimulus_id=feature_artifacts.stimulus_id,
    )
    save_feature_view_npz(
        path=intermediates_dir / "tr_feature_mass.npz",
        values=feature_artifacts.predictor_mass,
        sample_ids=sample_ids,
        feature_names=feature_artifacts.predictor_feature_names,
        layers=feature_artifacts.predictor_layers,
        tr_indices=feature_artifacts.tr_indices,
        start_s=feature_artifacts.start_s,
        end_s=feature_artifacts.end_s,
        stimulus_id=feature_artifacts.stimulus_id,
    )
    save_feature_view_npz(
        path=intermediates_dir / "tr_feature_average.npz",
        values=feature_artifacts.predictor_average,
        sample_ids=sample_ids,
        feature_names=feature_artifacts.predictor_feature_names,
        layers=feature_artifacts.predictor_layers,
        tr_indices=feature_artifacts.tr_indices,
        start_s=feature_artifacts.start_s,
        end_s=feature_artifacts.end_s,
        stimulus_id=feature_artifacts.stimulus_id,
    )
    save_feature_view_npz(
        path=intermediates_dir / "tr_feature_peak.npz",
        values=feature_artifacts.predictor_peak,
        sample_ids=sample_ids,
        feature_names=feature_artifacts.predictor_feature_names,
        layers=feature_artifacts.predictor_layers,
        tr_indices=feature_artifacts.tr_indices,
        start_s=feature_artifacts.start_s,
        end_s=feature_artifacts.end_s,
        stimulus_id=feature_artifacts.stimulus_id,
    )
    save_feature_view_npz(
        path=intermediates_dir / "tr_feature_count.npz",
        values=feature_artifacts.predictor_count,
        sample_ids=sample_ids,
        feature_names=feature_artifacts.predictor_feature_names,
        layers=feature_artifacts.predictor_layers,
        tr_indices=feature_artifacts.tr_indices,
        start_s=feature_artifacts.start_s,
        end_s=feature_artifacts.end_s,
        stimulus_id=feature_artifacts.stimulus_id,
    )
    save_target_view_npz(
        path=intermediates_dir / "tr_lm_target_mass.npz",
        values=feature_artifacts.lm_target_mass,
        sample_ids=sample_ids,
        target_names=feature_artifacts.lm_target_names,
        layers=feature_artifacts.lm_target_layers,
        tr_indices=feature_artifacts.tr_indices,
        start_s=feature_artifacts.start_s,
        end_s=feature_artifacts.end_s,
        stimulus_id=feature_artifacts.stimulus_id,
    )
    save_target_view_npz(
        path=intermediates_dir / "tr_lm_target_average.npz",
        values=feature_artifacts.lm_target_average,
        sample_ids=sample_ids,
        target_names=feature_artifacts.lm_target_names,
        layers=feature_artifacts.lm_target_layers,
        tr_indices=feature_artifacts.tr_indices,
        start_s=feature_artifacts.start_s,
        end_s=feature_artifacts.end_s,
        stimulus_id=feature_artifacts.stimulus_id,
    )
    summary = {
        "stimulus_id": feature_artifacts.stimulus_id,
        "feature_run_dir": str(feature_run_dir),
        "predictor_feature_count": int(len(feature_artifacts.predictor_keys)),
        "predictor_top_k": int(predictor_top_k) if predictor_top_k is not None else None,
        "lm_target_count": int(len(feature_artifacts.lm_target_keys)),
        "predictor_counts_by_layer": counts_by_layer(feature_artifacts.predictor_keys),
        "lm_target_counts_by_layer": counts_by_layer(feature_artifacts.lm_target_keys),
        "tr_count": int(feature_artifacts.predictor_mass.shape[0]),
        "artifacts": {
            "tr_feature_presence": str(intermediates_dir / "tr_feature_presence.npz"),
            "tr_feature_mass": str(intermediates_dir / "tr_feature_mass.npz"),
            "tr_feature_average": str(intermediates_dir / "tr_feature_average.npz"),
            "tr_feature_peak": str(intermediates_dir / "tr_feature_peak.npz"),
            "tr_feature_count": str(intermediates_dir / "tr_feature_count.npz"),
            "tr_lm_target_mass": str(intermediates_dir / "tr_lm_target_mass.npz"),
            "tr_lm_target_average": str(intermediates_dir / "tr_lm_target_average.npz"),
        },
    }
    write_json(intermediates_dir / "tr_feature_summary.json", summary)
    return summary


def concatenate_feature_artifacts(artifacts_list: list[FeatureArtifacts]) -> FeatureArtifacts:
    if not artifacts_list:
        raise ValueError("artifacts_list must not be empty.")
    reference = artifacts_list[0]
    for artifact in artifacts_list[1:]:
        if reference.predictor_keys != artifact.predictor_keys:
            raise ValueError("Cannot concatenate FeatureArtifacts with different predictor_keys.")
        if reference.lm_target_keys != artifact.lm_target_keys:
            raise ValueError("Cannot concatenate FeatureArtifacts with different lm_target_keys.")
        for label, left, right in (
            ("predictor_feature_names", reference.predictor_feature_names, artifact.predictor_feature_names),
            ("predictor_layers", reference.predictor_layers, artifact.predictor_layers),
            ("lm_target_names", reference.lm_target_names, artifact.lm_target_names),
            ("lm_target_layers", reference.lm_target_layers, artifact.lm_target_layers),
        ):
            if left.shape != right.shape or not np.array_equal(left, right):
                raise ValueError(f"Cannot concatenate FeatureArtifacts with different {label}.")

    pooled_stimulus_id = "pooled:" + "+".join(artifact.stimulus_id for artifact in artifacts_list)
    return FeatureArtifacts(
        predictor_presence=np.vstack([artifact.predictor_presence for artifact in artifacts_list]).astype(np.uint8, copy=False),
        predictor_mass=np.vstack([artifact.predictor_mass for artifact in artifacts_list]).astype(np.float32, copy=False),
        predictor_average=np.vstack([artifact.predictor_average for artifact in artifacts_list]).astype(np.float32, copy=False),
        predictor_peak=np.vstack([artifact.predictor_peak for artifact in artifacts_list]).astype(np.float32, copy=False),
        predictor_count=np.vstack([artifact.predictor_count for artifact in artifacts_list]).astype(np.float32, copy=False),
        predictor_feature_names=reference.predictor_feature_names,
        predictor_layers=reference.predictor_layers,
        predictor_keys=reference.predictor_keys,
        lm_target_mass=np.vstack([artifact.lm_target_mass for artifact in artifacts_list]).astype(np.float32, copy=False),
        lm_target_average=np.vstack([artifact.lm_target_average for artifact in artifacts_list]).astype(np.float32, copy=False),
        lm_target_count=np.vstack([artifact.lm_target_count for artifact in artifacts_list]).astype(np.float32, copy=False),
        lm_target_names=reference.lm_target_names,
        lm_target_layers=reference.lm_target_layers,
        lm_target_keys=reference.lm_target_keys,
        tr_indices=np.concatenate([artifact.tr_indices for artifact in artifacts_list]).astype(int, copy=False),
        start_s=np.concatenate([artifact.start_s for artifact in artifacts_list]).astype(float, copy=False),
        end_s=np.concatenate([artifact.end_s for artifact in artifacts_list]).astype(float, copy=False),
        sample_ids=np.concatenate([artifact.sample_ids for artifact in artifacts_list]).astype(str, copy=False),
        stimulus_id=pooled_stimulus_id,
    )


def load_predictor_feature_keys(path: Path, top_k: int | None = None) -> tuple[FeatureKey, ...]:
    rows = load_jsonl(path)
    ordered = [
        (int(row["layer"]), int(row["feature_id"]))
        for row in sorted(
            rows,
            key=lambda row: (
                int(row["transcript_relevance_rank"]),
                int(row["layer"]),
                int(row["feature_id"]),
            ),
        )
    ]
    if top_k is not None:
        if top_k <= 0:
            raise ValueError(f"predictor top-k must be positive; got {top_k}.")
        ordered = ordered[:top_k]
    if len(set(ordered)) != len(ordered):
        raise ValueError("Predictor feature list contains duplicates.")
    return tuple(ordered)


def select_lm_target_feature_keys(
    feature_concepts_path: Path,
    predictor_keys: tuple[FeatureKey, ...],
    targets_per_layer: int,
    target_layers: tuple[int, ...] | None = None,
) -> tuple[FeatureKey, ...]:
    predictor_set = set(predictor_keys)
    rows = load_jsonl(feature_concepts_path)
    selected: list[FeatureKey] = []
    layers = target_layers or tuple(sorted({layer for layer, _feature_id in predictor_keys}))
    for layer in layers:
        candidates = [
            row
            for row in rows
            if int(row["layer"]) == layer
            and (int(row["layer"]), int(row["feature_id"])) not in predictor_set
            and str(row.get("judge_status")) == "ok"
        ]
        candidates.sort(
            key=lambda row: (
                -extract_confidence(row),
                int(row.get("transcript_relevance_rank", 10**9)),
                int(row["feature_id"]),
            )
        )
        chosen = [
            (int(row["layer"]), int(row["feature_id"]))
            for row in candidates[:targets_per_layer]
        ]
        if len(chosen) < targets_per_layer:
            raise ValueError(
                f"Layer {layer} only has {len(chosen)} disjoint judged LM targets; "
                f"expected {targets_per_layer}."
            )
        selected.extend(chosen)
    return tuple(selected)


def extract_confidence(row: dict[str, Any]) -> float:
    judge_output = row.get("judge_output") or {}
    if "confidence" in judge_output:
        return float(judge_output["confidence"])
    if "confidence" in row:
        return float(row["confidence"])
    return 0.0


def resolve_transcript_paired_path(feature_run_dir: Path) -> Path:
    """Return the paired-record artifact, preferring the slim ``.minimal`` variant when both exist.

    The minimal file (see scripts/analysis/build_minimal_transcript_paired_records.py) carries the same
    rows for the selected stimuli without the fields the structure workflow never reads.
    """

    candidates = (
        feature_run_dir / "transcript_paired_records.minimal.jsonl",
        feature_run_dir / "transcript_paired_records.jsonl",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def aggregate_tr_feature_views(
    transcript_paired_path: Path,
    stimulus_id: str,
    token_to_word_index: dict[int, int],
    word_to_tr_index: list[int],
    tr_bins: list[TrBin],
    predictor_keys: tuple[FeatureKey, ...],
    lm_target_keys: tuple[FeatureKey, ...],
) -> FeatureArtifacts:
    predictor_names = np.asarray([feature_name(key) for key in predictor_keys], dtype=str)
    predictor_layers = np.asarray([int(layer) for layer, _ in predictor_keys], dtype=int)
    lm_target_names = np.asarray([feature_name(key) for key in lm_target_keys], dtype=str)
    lm_target_layers = np.asarray([int(layer) for layer, _ in lm_target_keys], dtype=int)

    all_keys = tuple(dict.fromkeys((*predictor_keys, *lm_target_keys)))
    all_index = {key: index for index, key in enumerate(all_keys)}
    tr_count = len(tr_bins)
    mass = np.zeros((tr_count, len(all_keys)), dtype=np.float32)
    peak = np.zeros((tr_count, len(all_keys)), dtype=np.float32)
    count = np.zeros((tr_count, len(all_keys)), dtype=np.int32)
    presence = np.zeros((tr_count, len(all_keys)), dtype=np.uint8)

    relevant_by_layer: dict[int, set[int]] = defaultdict(set)
    for layer, feature_id in all_keys:
        relevant_by_layer[int(layer)].add(int(feature_id))

    with transcript_paired_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            provenance = row.get("provenance") or {}
            if provenance.get("stimulus_id") != stimulus_id:
                continue
            layer = int(row["layer"])
            if layer not in relevant_by_layer:
                continue
            global_token_index = int(row["window_start"]) + int(row["token_position"])
            word_index = token_to_word_index.get(global_token_index)
            if word_index is None:
                continue
            tr_index = int(word_to_tr_index[word_index])
            for latent in row.get("latent_activations", []):
                feature_id = int(latent["latent_id"])
                if feature_id not in relevant_by_layer[layer]:
                    continue
                key = (layer, feature_id)
                column_index = all_index[key]
                activation = float(latent["activation"])
                mass[tr_index, column_index] += activation
                count[tr_index, column_index] += 1
                if activation > peak[tr_index, column_index]:
                    peak[tr_index, column_index] = activation
                if activation > 0:
                    presence[tr_index, column_index] = 1

    tr_indices = np.asarray([int(item.tr_index) for item in tr_bins], dtype=int)
    start_s = np.asarray([float(item.start_s) for item in tr_bins], dtype=float)
    end_s = np.asarray([float(item.end_s) for item in tr_bins], dtype=float)
    sample_ids = np.asarray([f"{stimulus_id}:tr:{int(item.tr_index)}" for item in tr_bins], dtype=str)

    predictor_columns = np.asarray([all_index[key] for key in predictor_keys], dtype=int)
    lm_columns = np.asarray([all_index[key] for key in lm_target_keys], dtype=int)
    predictor_counts = count[:, predictor_columns].astype(np.float32)
    lm_target_counts = count[:, lm_columns].astype(np.float32)
    return FeatureArtifacts(
        predictor_presence=presence[:, predictor_columns],
        predictor_mass=mass[:, predictor_columns],
        predictor_average=safe_divide_rows(mass[:, predictor_columns], predictor_counts),
        predictor_peak=peak[:, predictor_columns],
        predictor_count=predictor_counts,
        predictor_feature_names=predictor_names,
        predictor_layers=predictor_layers,
        predictor_keys=predictor_keys,
        lm_target_mass=mass[:, lm_columns],
        lm_target_average=safe_divide_rows(mass[:, lm_columns], lm_target_counts),
        lm_target_count=lm_target_counts,
        lm_target_names=lm_target_names,
        lm_target_layers=lm_target_layers,
        lm_target_keys=lm_target_keys,
        tr_indices=tr_indices,
        start_s=start_s,
        end_s=end_s,
        sample_ids=sample_ids,
        stimulus_id=stimulus_id,
    )


def build_family_matrices(
    artifacts: FeatureArtifacts,
    predictor_matrix: np.ndarray,
    lm_target_matrix: np.ndarray,
) -> list[FamilyMatrices]:
    families: list[FamilyMatrices] = []
    for layer in sorted(np.unique(artifacts.predictor_layers).tolist()):
        predictor_mask = artifacts.predictor_layers == layer
        target_mask = artifacts.lm_target_layers == layer
        families.append(
            FamilyMatrices(
                family_name=f"layer{layer}",
                predictor_values=predictor_matrix[:, predictor_mask],
                predictor_feature_names=artifacts.predictor_feature_names[predictor_mask],
                predictor_base_feature_names=artifacts.predictor_feature_names[predictor_mask],
                predictor_layers=artifacts.predictor_layers[predictor_mask],
                lm_target_values=lm_target_matrix[:, target_mask],
                lm_target_names=artifacts.lm_target_names[target_mask],
                lm_target_layers=artifacts.lm_target_layers[target_mask],
                tr_indices=artifacts.tr_indices,
                sample_ids=artifacts.sample_ids,
            )
        )
    families.append(
        FamilyMatrices(
            family_name="all_layers",
            predictor_values=predictor_matrix,
            predictor_feature_names=artifacts.predictor_feature_names,
            predictor_base_feature_names=artifacts.predictor_feature_names,
            predictor_layers=artifacts.predictor_layers,
            lm_target_values=lm_target_matrix,
            lm_target_names=artifacts.lm_target_names,
            lm_target_layers=artifacts.lm_target_layers,
            tr_indices=artifacts.tr_indices,
            sample_ids=artifacts.sample_ids,
        )
    )
    return families


def parse_predictor_sample_id(sample_id: str) -> tuple[str, int]:
    parts = str(sample_id).split(":")
    if len(parts) != 3 or parts[1] != "tr":
        raise ValueError(f"Expected predictor sample_id in stimulus:tr:index format, got: {sample_id}")
    return parts[0], int(parts[2])


def parse_stimulus_from_pooled_run_id(run_id: str) -> str:
    if ":" not in run_id:
        raise ValueError(f"Expected pooled run_id in stimulus:run format, got: {run_id}")
    return run_id.split(":", 1)[0]


def save_feature_view_npz(
    path: Path,
    values: np.ndarray,
    sample_ids: np.ndarray,
    feature_names: np.ndarray,
    layers: np.ndarray,
    tr_indices: np.ndarray,
    start_s: np.ndarray,
    end_s: np.ndarray,
    stimulus_id: str,
) -> None:
    np.savez_compressed(
        path,
        values=values,
        sample_ids=sample_ids,
        feature_names=feature_names,
        layers=layers,
        tr_indices=tr_indices,
        start_s=start_s,
        end_s=end_s,
        stimulus_id=np.asarray([stimulus_id], dtype=str),
    )


def save_target_view_npz(
    path: Path,
    values: np.ndarray,
    sample_ids: np.ndarray,
    target_names: np.ndarray,
    layers: np.ndarray,
    tr_indices: np.ndarray,
    start_s: np.ndarray,
    end_s: np.ndarray,
    stimulus_id: str,
) -> None:
    np.savez_compressed(
        path,
        values=values,
        sample_ids=sample_ids,
        target_names=target_names,
        layers=layers,
        tr_indices=tr_indices,
        start_s=start_s,
        end_s=end_s,
        stimulus_id=np.asarray([stimulus_id], dtype=str),
    )


def counts_by_layer(keys: Iterable[FeatureKey]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for layer, _feature_id in keys:
        key = str(int(layer))
        counts[key] = counts.get(key, 0) + 1
    return counts
