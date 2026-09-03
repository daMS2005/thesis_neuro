#!/usr/bin/env python3
"""Pooled multi-story all-layers analysis against final-hidden-state LM targets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from structure_comparison.alignment import (
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
from structure_comparison.artifacts import (
    FamilyMatrices,
    build_pooled_tr_feature_artifacts,
    build_tr_feature_artifacts,
    parse_predictor_sample_id,
    resolve_transcript_paired_path,
)
from structure_comparison.brain import (
    BrainTargets,
    build_brain_design_matrix,
    build_pooled_brain_design_matrix,
    load_brain_targets,
)
from structure_comparison.modeling import (
    average_predictions_by_tr,
    collapse_lagged_weights,
    compare_feature_importance,
    compare_sample_geometry,
    consensus_alpha,
    contiguous_block_groups,
    cosine_similarity_matrix,
    fit_final_model,
    run_grouped_ridge_cv,
    top_metric_rows,
)
from structure_comparison.utils import write_json
from thesis_neuro.config import load_app_config
from thesis_neuro.models import GemmaModelAdapter
from thesis_neuro.paths import default_config_path

ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
BRAIN_LAGS = [0, 1, 2, 3, 4]
LM_FOLDS = 5
LM_TARGETS_PER_LAYER = 8


def build_hidden_state_family_matrices(
    artifacts,
    predictor_matrix: np.ndarray,
    hidden_target_values: np.ndarray,
    hidden_target_names: np.ndarray,
    family_mode: str,
) -> list[FamilyMatrices]:
    layer_families = []
    for layer in sorted(np.unique(artifacts.predictor_layers).tolist()):
        predictor_mask = artifacts.predictor_layers == layer
        layer_families.append(
            FamilyMatrices(
                family_name=f"layer{layer}",
                predictor_values=predictor_matrix[:, predictor_mask],
                predictor_feature_names=artifacts.predictor_feature_names[predictor_mask],
                predictor_base_feature_names=artifacts.predictor_feature_names[predictor_mask],
                predictor_layers=artifacts.predictor_layers[predictor_mask],
                lm_target_values=hidden_target_values,
                lm_target_names=hidden_target_names,
                lm_target_layers=np.full(hidden_target_names.shape, -1, dtype=int),
                tr_indices=artifacts.tr_indices,
                sample_ids=artifacts.sample_ids,
            )
        )
    all_layers_family = FamilyMatrices(
        family_name="all_layers",
        predictor_values=predictor_matrix,
        predictor_feature_names=artifacts.predictor_feature_names,
        predictor_base_feature_names=artifacts.predictor_feature_names,
        predictor_layers=artifacts.predictor_layers,
        lm_target_values=hidden_target_values,
        lm_target_names=hidden_target_names,
        lm_target_layers=np.full(hidden_target_names.shape, -1, dtype=int),
        tr_indices=artifacts.tr_indices,
        sample_ids=artifacts.sample_ids,
    )
    if family_mode == "all_layers":
        return [all_layers_family]
    if family_mode == "per_layer":
        return layer_families
    if family_mode == "both":
        return [*layer_families, all_layers_family]
    raise ValueError(f"Unsupported family_mode: {family_mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pooled all-layers final-hidden-state analysis.")
    parser.add_argument("--feature-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--brain-targets-npz", required=True)
    parser.add_argument("--transcript-root", required=True)
    parser.add_argument("--stimulus-ids", nargs="+", required=True)
    parser.add_argument("--config-path", default=str(default_config_path().parent / "examples" / "full-extraction.yaml"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--scope-release", required=True)
    parser.add_argument("--scope-width", default="width_16k")
    parser.add_argument("--token-layer", type=int, required=True)
    parser.add_argument("--layer-selection", nargs="+", type=int, required=True)
    parser.add_argument("--predictor-top-k", type=int, default=None)
    parser.add_argument("--lm-folds", type=int, default=LM_FOLDS)
    parser.add_argument("--brain-lags", nargs="+", type=int, default=BRAIN_LAGS)
    parser.add_argument("--family-mode", choices=["all_layers", "per_layer", "both"], default="all_layers")
    return parser


def build_single_story_final_hidden_state_targets(
    feature_run_dir: Path,
    transcript_root: Path,
    stimulus_id: str,
    config_path: Path,
    model_id: str,
    scope_release: str,
    scope_width: str,
    layer_selection: list[int],
    token_layer: int,
) -> dict[str, np.ndarray]:
    transcript_paths = resolve_transcript_paths(transcript_root, stimulus_id)
    transcript_text = transcript_paths.transcript_txt.read_text(encoding="utf-8").strip()
    word_rows = load_word_rows(transcript_paths.words_tsv)
    tr_bins = load_tr_bins(transcript_paths.tr_aligned_tsv)
    metadata = json.loads(transcript_paths.metadata_json.read_text(encoding="utf-8"))

    token_stream = extract_global_tokens(
        transcript_paired_path=resolve_transcript_paired_path(feature_run_dir),
        stimulus_id=stimulus_id,
        layer=token_layer,
    )
    token_word_groups = group_tokens_with_model_tokenizer(
        model_id=token_stream.model_id,
        tokens=token_stream.tokens,
        transcript_text=transcript_text,
    )
    validate_word_alignment(token_word_groups, word_rows, transcript_text)
    token_to_word_index = build_token_to_word_index(token_word_groups)
    word_to_tr_index = assign_words_to_trs(
        word_rows=word_rows,
        tr_bins=tr_bins,
        stimulus_onset_s=float(metadata.get("stimulus_onset_s", 0.0)),
    )
    validate_tr_alignment(
        word_rows=word_rows,
        word_to_tr_index=word_to_tr_index,
        tr_bins=tr_bins,
        stimulus_onset_s=float(metadata.get("stimulus_onset_s", 0.0)),
    )

    os.environ.setdefault("HF_LOCAL_FILES_ONLY", "1")
    config = load_app_config(config_path)
    config.model.base_model_id = model_id
    config.model.scope_release = scope_release
    config.model.scope_width = scope_width
    config.model.layer_selection = layer_selection
    model = GemmaModelAdapter(config)

    token_ids = model.tokenize_document(transcript_text)
    windows = model.make_windows(
        token_ids=token_ids,
        window_len=min(model.max_context_window_tokens(), config.tokenization.seq_len),
        metadata_mode="heuristic",
    )

    all_tokens: list[str] = []
    hidden_rows: list[np.ndarray] = []
    for window in windows:
        outputs, _ = model.forward_outputs(window.input_ids, require_grad=False)
        final_hidden = outputs.hidden_states[-1].squeeze(0).detach().to("cpu", dtype=torch.float32)
        all_tokens.extend(window.tokens)
        hidden_rows.append(final_hidden.numpy())

    if all_tokens != token_stream.tokens:
        raise ValueError(f"Final hidden-state tokenization does not match transcript artifacts for {stimulus_id}.")

    token_hidden = np.vstack(hidden_rows).astype(np.float32, copy=False)
    tr_count = len(tr_bins)
    hidden_size = token_hidden.shape[1]
    sums = np.zeros((tr_count, hidden_size), dtype=np.float32)
    counts = np.zeros((tr_count, 1), dtype=np.float32)

    for token_index in range(token_hidden.shape[0]):
        word_index = token_to_word_index.get(token_index)
        if word_index is None:
            continue
        tr_index = int(word_to_tr_index[word_index])
        sums[tr_index] += token_hidden[token_index]
        counts[tr_index, 0] += 1.0

    averages = np.zeros_like(sums)
    np.divide(sums, counts, out=averages, where=counts > 0)

    target_names = np.asarray([f"final_hidden_{idx:04d}" for idx in range(hidden_size)], dtype=str)
    tr_indices = np.asarray([int(item.tr_index) for item in tr_bins], dtype=int)
    sample_ids = np.asarray([f"{stimulus_id}:tr:{int(tr)}" for tr in tr_indices], dtype=str)
    start_s = np.asarray([float(item.start_s) for item in tr_bins], dtype=float)
    end_s = np.asarray([float(item.end_s) for item in tr_bins], dtype=float)
    return {
        "values": averages,
        "target_names": target_names,
        "tr_indices": tr_indices,
        "sample_ids": sample_ids,
        "start_s": start_s,
        "end_s": end_s,
        "model_id": np.asarray([model_id], dtype=str),
        "hidden_size": np.asarray([hidden_size], dtype=int),
    }


def concatenate_hidden_target_dicts(targets_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not targets_list:
        raise ValueError("targets_list must not be empty.")
    reference_names = targets_list[0]["target_names"]
    for target_dict in targets_list[1:]:
        if target_dict["target_names"].shape != reference_names.shape or not np.array_equal(
            target_dict["target_names"], reference_names
        ):
            raise ValueError("Hidden-state target names do not match across stimuli.")
    return {
        "values": np.vstack([target_dict["values"] for target_dict in targets_list]).astype(np.float32, copy=False),
        "target_names": reference_names,
        "tr_indices": np.concatenate([target_dict["tr_indices"] for target_dict in targets_list]).astype(int, copy=False),
        "sample_ids": np.concatenate([target_dict["sample_ids"] for target_dict in targets_list]).astype(str, copy=False),
        "start_s": np.concatenate([target_dict["start_s"] for target_dict in targets_list]).astype(float, copy=False),
        "end_s": np.concatenate([target_dict["end_s"] for target_dict in targets_list]).astype(float, copy=False),
        "model_id": targets_list[0]["model_id"],
        "hidden_size": targets_list[0]["hidden_size"],
    }


def align_hidden_targets_to_predictor_sample_ids(
    predictor_sample_ids: np.ndarray,
    hidden_targets: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    lookup = {
        str(sample_id): row_index
        for row_index, sample_id in enumerate(hidden_targets["sample_ids"].tolist())
    }
    missing = [str(sample_id) for sample_id in predictor_sample_ids.tolist() if str(sample_id) not in lookup]
    if missing:
        raise ValueError(f"Hidden-state targets are missing predictor sample ids, including {missing[:5]}")
    ordered_indices = np.asarray([lookup[str(sample_id)] for sample_id in predictor_sample_ids.tolist()], dtype=int)
    return {
        "values": hidden_targets["values"][ordered_indices],
        "target_names": hidden_targets["target_names"],
        "tr_indices": hidden_targets["tr_indices"][ordered_indices],
        "sample_ids": hidden_targets["sample_ids"][ordered_indices],
    }


def lm_groups_from_predictor_sample_ids(sample_ids: np.ndarray, lm_folds: int) -> np.ndarray:
    stimuli = np.asarray([parse_predictor_sample_id(str(sample_id))[0] for sample_id in sample_ids.tolist()], dtype=str)
    if np.unique(stimuli).size >= 2:
        return stimuli
    return contiguous_block_groups(sample_ids.shape[0], lm_folds)


def average_predictions_by_story_tr_key(
    predictions: np.ndarray,
    story_tr_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ordered_keys = list(dict.fromkeys(story_tr_keys.tolist()))
    averaged_rows = []
    for key in ordered_keys:
        averaged_rows.append(predictions[story_tr_keys == key].mean(axis=0))
    return np.vstack(averaged_rows), np.asarray(ordered_keys, dtype=str)


def analyze_hidden_state_family(
    family: FamilyMatrices,
    brain_targets: BrainTargets,
    aligned_hidden_targets: dict[str, np.ndarray],
    family_output_dir: Path,
    stimulus_ids: list[str],
    brain_lags: list[int],
    lm_folds: int,
) -> dict[str, object]:
    if len(stimulus_ids) == 1:
        brain_design = build_brain_design_matrix(
            tr_predictors=family.predictor_values,
            tr_indices=family.tr_indices,
            feature_names=family.predictor_feature_names,
            brain_targets=brain_targets,
            lags=brain_lags,
        )
    else:
        brain_design = build_pooled_brain_design_matrix(
            tr_predictors=family.predictor_values,
            predictor_sample_ids=family.sample_ids,
            feature_names=family.predictor_feature_names,
            brain_targets=brain_targets,
            lags=brain_lags,
        )

    brain_cv = run_grouped_ridge_cv(
        x=brain_design["x"],
        y=brain_design["y"],
        groups=brain_design["groups"],
        sample_ids=brain_design["sample_ids"],
        target_names=brain_targets.target_names,
        alpha_grid=ALPHA_GRID,
        family_name=f"{family.family_name}:brain",
    )
    brain_alpha = consensus_alpha(brain_cv["outer_folds"])
    brain_final = fit_final_model(brain_design["x"], brain_design["y"], brain_alpha, f"{family.family_name}:brain")
    brain_collapsed_weights = collapse_lagged_weights(
        weights=brain_final["weights"],
        base_feature_count=family.predictor_values.shape[1],
        lags=brain_lags,
    )

    lm_groups = lm_groups_from_predictor_sample_ids(family.sample_ids, lm_folds)
    lm_cv = run_grouped_ridge_cv(
        x=family.predictor_values,
        y=aligned_hidden_targets["values"],
        groups=lm_groups,
        sample_ids=aligned_hidden_targets["sample_ids"],
        target_names=aligned_hidden_targets["target_names"],
        alpha_grid=ALPHA_GRID,
        family_name=f"{family.family_name}:lm",
    )
    lm_alpha = consensus_alpha(lm_cv["outer_folds"])
    lm_final = fit_final_model(
        x=family.predictor_values,
        y=aligned_hidden_targets["values"],
        alpha=lm_alpha,
        family_name=f"{family.family_name}:lm",
    )

    if len(stimulus_ids) == 1:
        brain_predicted, shared_tr_indices = average_predictions_by_tr(
            predictions=brain_final["predictions"],
            tr_indices=brain_design["tr_indices"],
        )
        lm_lookup = {int(tr_index): idx for idx, tr_index in enumerate(aligned_hidden_targets["tr_indices"].tolist())}
        common_lm_indices = np.asarray([lm_lookup[int(tr)] for tr in shared_tr_indices.tolist()], dtype=int)
        lm_predictions_aligned = lm_final["predictions"][common_lm_indices]
    else:
        brain_predicted, shared_keys = average_predictions_by_story_tr_key(
            predictions=brain_final["predictions"],
            story_tr_keys=brain_design["story_tr_keys"],
        )
        lm_lookup = {
            str(sample_id): idx for idx, sample_id in enumerate(aligned_hidden_targets["sample_ids"].tolist())
        }
        keep_positions = [idx for idx, key in enumerate(shared_keys.tolist()) if str(key) in lm_lookup]
        if not keep_positions:
            raise ValueError(f"{family.family_name}: no pooled LM predictions match the averaged brain story/TR keys.")
        brain_predicted = brain_predicted[np.asarray(keep_positions, dtype=int)]
        ordered_keys = [str(shared_keys[idx]) for idx in keep_positions]
        common_lm_indices = np.asarray([lm_lookup[key] for key in ordered_keys], dtype=int)
        lm_predictions_aligned = lm_final["predictions"][common_lm_indices]

    weight_similarity = cosine_similarity_matrix(brain_collapsed_weights, lm_final["weights"])
    sample_rsa = compare_sample_geometry(brain_predicted, lm_predictions_aligned)
    feature_importance = compare_feature_importance(
        brain_collapsed_weights,
        lm_final["weights"],
        family.predictor_feature_names,
    )

    write_json(family_output_dir / "brain_cv_summary.json", brain_cv)
    write_json(family_output_dir / "lm_cv_summary.json", lm_cv)
    np.savez_compressed(
        family_output_dir / "brain_final_model.npz",
        weights=brain_final["weights"],
        collapsed_weights=brain_collapsed_weights,
        predictions=brain_final["predictions"],
        sample_ids=brain_design["sample_ids"],
        tr_indices=brain_design["tr_indices"],
        feature_names=brain_design["feature_names"],
        base_feature_names=family.predictor_feature_names,
        target_names=brain_targets.target_names,
        alpha=np.asarray([brain_alpha], dtype=float),
    )
    np.savez_compressed(
        family_output_dir / "lm_final_model.npz",
        weights=lm_final["weights"],
        predictions=lm_final["predictions"],
        sample_ids=aligned_hidden_targets["sample_ids"],
        tr_indices=aligned_hidden_targets["tr_indices"],
        feature_names=family.predictor_feature_names,
        target_names=aligned_hidden_targets["target_names"],
        alpha=np.asarray([lm_alpha], dtype=float),
    )
    np.savez_compressed(
        family_output_dir / "brain_lm_weight_similarity.npz",
        similarity=weight_similarity,
        brain_target_names=brain_targets.target_names,
        lm_target_names=aligned_hidden_targets["target_names"],
        feature_names=family.predictor_feature_names,
    )
    write_json(family_output_dir / "sample_rsa.json", sample_rsa)
    write_json(family_output_dir / "feature_importance_summary.json", feature_importance)

    top_parcels = top_metric_rows(
        target_names=brain_targets.target_names,
        correlations=np.asarray(brain_cv["aggregate"]["per_target_mean_correlation"], dtype=float),
        r2_scores=np.asarray(brain_cv["aggregate"]["per_target_mean_r2"], dtype=float),
        top_k=10,
    )
    top_lm_targets = top_metric_rows(
        target_names=aligned_hidden_targets["target_names"],
        correlations=np.asarray(lm_cv["aggregate"]["per_target_mean_correlation"], dtype=float),
        r2_scores=np.asarray(lm_cv["aggregate"]["per_target_mean_r2"], dtype=float),
        top_k=10,
    )
    summary = {
        "family_name": family.family_name,
        "predictor_view": "average",
        "predictor_count": int(family.predictor_values.shape[1]),
        "lm_target_count": int(aligned_hidden_targets["values"].shape[1]),
        "brain_sample_count": int(brain_design["x"].shape[0]),
        "brain_censored_sample_count": int(brain_design["censored_sample_count"]),
        "brain_retained_sample_fraction": float(brain_design["retained_sample_fraction"]),
        "brain_run_count": int(len(np.unique(brain_design["groups"]))),
        "lm_sample_count": int(family.predictor_values.shape[0]),
        "brain_consensus_alpha": float(brain_alpha),
        "lm_consensus_alpha": float(lm_alpha),
        "brain_mean_test_correlation": float(brain_cv["aggregate"]["mean_test_correlation"]),
        "brain_mean_test_r2": float(brain_cv["aggregate"]["mean_test_r2"]),
        "lm_mean_test_correlation": float(lm_cv["aggregate"]["mean_test_correlation"]),
        "lm_mean_test_r2": float(lm_cv["aggregate"]["mean_test_r2"]),
        "sample_rsa_correlation": float(sample_rsa["sample_rsa_correlation"]),
        "feature_importance_correlation": float(feature_importance["importance_correlation"]),
        "top_parcels": top_parcels,
        "top_lm_targets": top_lm_targets,
        "top_shared_predictors": feature_importance["top_features"][:10],
        "artifacts": {
            "brain_cv_summary": str(family_output_dir / "brain_cv_summary.json"),
            "lm_cv_summary": str(family_output_dir / "lm_cv_summary.json"),
            "brain_final_model": str(family_output_dir / "brain_final_model.npz"),
            "lm_final_model": str(family_output_dir / "lm_final_model.npz"),
            "brain_lm_weight_similarity": str(family_output_dir / "brain_lm_weight_similarity.npz"),
            "sample_rsa": str(family_output_dir / "sample_rsa.json"),
            "feature_importance_summary": str(family_output_dir / "feature_importance_summary.json"),
        },
    }
    write_json(family_output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    stimulus_ids = [str(value) for value in args.stimulus_ids]
    if not stimulus_ids:
        raise ValueError("stimulus_ids must not be empty.")

    feature_run_dir = Path(args.feature_run_dir)
    output_dir = Path(args.output_dir)
    brain_targets_npz = Path(args.brain_targets_npz)
    transcript_root = Path(args.transcript_root)
    config_path = Path(args.config_path)
    brain_lags = [int(value) for value in args.brain_lags]

    output_dir.mkdir(parents=True, exist_ok=True)
    intermediates_dir = output_dir / "intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    if len(stimulus_ids) == 1:
        feature_bundle = build_tr_feature_artifacts(
            feature_run_dir=feature_run_dir,
            transcript_root=transcript_root,
            stimulus_id=stimulus_ids[0],
            output_dir=output_dir,
            lm_targets_per_layer=LM_TARGETS_PER_LAYER,
            predictor_top_k=args.predictor_top_k,
        )
    else:
        feature_bundle = build_pooled_tr_feature_artifacts(
            feature_run_dir=feature_run_dir,
            transcript_root=transcript_root,
            stimulus_ids=stimulus_ids,
            output_dir=output_dir,
            lm_targets_per_layer=LM_TARGETS_PER_LAYER,
            predictor_top_k=args.predictor_top_k,
        )
    artifacts = feature_bundle["artifacts"]
    predictor_matrix = artifacts.predictor_average
    hidden_targets = concatenate_hidden_target_dicts(
        [
            build_single_story_final_hidden_state_targets(
                feature_run_dir=feature_run_dir,
                transcript_root=transcript_root,
                stimulus_id=stimulus_id,
                config_path=config_path,
                model_id=str(args.model_id),
                scope_release=str(args.scope_release),
                scope_width=str(args.scope_width),
                layer_selection=[int(v) for v in args.layer_selection],
                token_layer=int(args.token_layer),
            )
            for stimulus_id in stimulus_ids
        ]
    )
    np.savez_compressed(intermediates_dir / "tr_lm_target_final_hidden_average.npz", **hidden_targets)
    aligned_hidden_targets = align_hidden_targets_to_predictor_sample_ids(artifacts.sample_ids, hidden_targets)
    brain_targets = load_brain_targets(brain_targets_npz)
    families = build_hidden_state_family_matrices(
        artifacts=artifacts,
        predictor_matrix=predictor_matrix,
        hidden_target_values=aligned_hidden_targets["values"],
        hidden_target_names=aligned_hidden_targets["target_names"],
        family_mode=str(args.family_mode),
    )
    family_summaries = []
    for family in families:
        family_output_dir = output_dir / family.family_name
        family_output_dir.mkdir(parents=True, exist_ok=True)
        family_summaries.append(
            analyze_hidden_state_family(
                family=family,
                brain_targets=brain_targets,
                aligned_hidden_targets=aligned_hidden_targets,
                family_output_dir=family_output_dir,
                stimulus_ids=stimulus_ids,
                brain_lags=brain_lags,
                lm_folds=int(args.lm_folds),
            )
        )

    analysis_summary = {
        "stimulus_ids": stimulus_ids,
        "feature_run_dir": str(feature_run_dir),
        "brain_targets_npz": str(brain_targets_npz),
        "predictor_view": "average",
        "predictor_top_k": int(args.predictor_top_k) if args.predictor_top_k is not None else None,
        "brain_lags": brain_lags,
        "lm_folds": int(args.lm_folds),
        "family_mode": str(args.family_mode),
        "families": family_summaries,
        "artifacts": {
            "tr_lm_target_final_hidden_average": str(intermediates_dir / "tr_lm_target_final_hidden_average.npz"),
        },
        "model": {
            "model_id": str(args.model_id),
            "scope_release": str(args.scope_release),
            "scope_width": str(args.scope_width),
            "token_layer": int(args.token_layer),
            "layer_selection": [int(v) for v in args.layer_selection],
        },
    }
    write_json(output_dir / "analysis_summary.json", analysis_summary)
    print(json.dumps(analysis_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
