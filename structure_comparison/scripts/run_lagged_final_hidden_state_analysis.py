#!/usr/bin/env python3
"""All-layers analysis with lagged predictors against final-hidden-state LM targets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from structure_comparison.workflow import (
    BrainTargets,
    assign_words_to_trs,
    average_predictions_by_tr,
    build_brain_design_matrix,
    build_token_to_word_index,
    build_tr_feature_artifacts,
    collapse_lagged_weights,
    compare_feature_importance,
    compare_sample_geometry,
    consensus_alpha,
    contiguous_block_groups,
    extract_global_tokens,
    fit_final_model,
    group_tokens_with_model_tokenizer,
    load_brain_targets,
    load_tr_bins,
    load_word_rows,
    resolve_transcript_paths,
    run_grouped_ridge_cv,
    top_metric_rows,
    validate_tr_alignment,
    validate_word_alignment,
    write_json,
)
from thesis_neuro.config import load_app_config
from thesis_neuro.models import GemmaModelAdapter
from thesis_neuro.paths import default_config_path

ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
BRAIN_LAGS = [0, 1, 2, 3, 4]
LM_FOLDS = 5
LM_TARGETS_PER_LAYER = 8
PREDICTOR_VIEW = "average"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all-layers final-hidden-state analysis.")
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--brain-targets-npz", required=True)
    parser.add_argument("--transcript-root", required=True)
    parser.add_argument("--config-path", default=str(default_config_path().parent / "experiments" / "final-output.yaml"))
    parser.add_argument("--stimulus-id", default="shapesphysical")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--scope-release", required=True)
    parser.add_argument("--scope-width", default="width_16k")
    parser.add_argument("--token-layer", type=int, required=True)
    parser.add_argument("--layer-selection", nargs="+", type=int, required=True)
    parser.add_argument("--predictor-top-k", type=int, default=None)
    parser.add_argument("--lm-folds", type=int, default=LM_FOLDS)
    parser.add_argument("--brain-lags", nargs="+", type=int, default=BRAIN_LAGS)
    parser.add_argument("--lm-lags", nargs="+", type=int, default=None)
    return parser


def build_lm_design_matrix(
    tr_predictors: np.ndarray,
    tr_indices: np.ndarray,
    feature_names: np.ndarray,
    hidden_targets: dict[str, np.ndarray],
    lags: list[int],
    lm_folds: int,
    stimulus_id: str,
) -> dict[str, np.ndarray]:
    lm_targets = BrainTargets(
        values=hidden_targets["values"],
        sample_ids=hidden_targets["sample_ids"],
        subject_ids=np.asarray([stimulus_id] * len(hidden_targets["sample_ids"]), dtype=str),
        run_ids=np.asarray(["run-1"] * len(hidden_targets["sample_ids"]), dtype=str),
        tr_indices=hidden_targets["tr_indices"],
        target_names=hidden_targets["target_names"],
        censor_mask=None,
    )
    design = build_brain_design_matrix(
        tr_predictors=tr_predictors,
        tr_indices=tr_indices,
        feature_names=feature_names,
        brain_targets=lm_targets,
        lags=lags,
    )
    design["groups"] = contiguous_block_groups(design["x"].shape[0], lm_folds)
    return design


def build_final_hidden_state_targets(
    remote_run_dir: Path,
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
        transcript_paired_path=remote_run_dir / "transcript_paired_records.jsonl",
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
    os.environ.setdefault("TORCH_DEVICE", "mps")
    os.environ.setdefault("TORCH_DTYPE", "float16")
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
        raise ValueError("Final hidden-state tokenization does not match transcript artifacts.")

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


def main() -> int:
    args = build_parser().parse_args()

    remote_run_dir = Path(args.remote_run_dir)
    output_dir = Path(args.output_dir)
    brain_targets_npz = Path(args.brain_targets_npz)
    transcript_root = Path(args.transcript_root)
    config_path = Path(args.config_path)
    brain_lags = [int(value) for value in args.brain_lags]
    lm_lags = brain_lags if args.lm_lags is None else [int(value) for value in args.lm_lags]

    output_dir.mkdir(parents=True, exist_ok=True)
    intermediates_dir = output_dir / "intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    family_output_dir = output_dir / "all_layers"
    family_output_dir.mkdir(parents=True, exist_ok=True)

    feature_bundle = build_tr_feature_artifacts(
        remote_run_dir=remote_run_dir,
        transcript_root=transcript_root,
        stimulus_id=str(args.stimulus_id),
        output_dir=output_dir,
        lm_targets_per_layer=LM_TARGETS_PER_LAYER,
        predictor_top_k=args.predictor_top_k,
    )
    artifacts = feature_bundle["artifacts"]
    predictor_matrix = artifacts.predictor_average

    hidden_targets = build_final_hidden_state_targets(
        remote_run_dir=remote_run_dir,
        transcript_root=transcript_root,
        stimulus_id=str(args.stimulus_id),
        config_path=config_path,
        model_id=str(args.model_id),
        scope_release=str(args.scope_release),
        scope_width=str(args.scope_width),
        layer_selection=[int(v) for v in args.layer_selection],
        token_layer=int(args.token_layer),
    )
    np.savez_compressed(intermediates_dir / "tr_lm_target_final_hidden_average.npz", **hidden_targets)

    brain_targets = load_brain_targets(brain_targets_npz)
    brain_design = build_brain_design_matrix(
        tr_predictors=predictor_matrix,
        tr_indices=artifacts.tr_indices,
        feature_names=artifacts.predictor_feature_names,
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
        family_name="all_layers:brain",
    )
    brain_alpha = consensus_alpha(brain_cv["outer_folds"])
    brain_final = fit_final_model(brain_design["x"], brain_design["y"], brain_alpha, "all_layers:brain")
    brain_collapsed_weights = collapse_lagged_weights(
        weights=brain_final["weights"],
        base_feature_count=predictor_matrix.shape[1],
        lags=brain_lags,
    )
    brain_predicted_by_tr, shared_tr_indices = average_predictions_by_tr(
        predictions=brain_final["predictions"],
        tr_indices=brain_design["tr_indices"],
    )

    lm_design = build_lm_design_matrix(
        tr_predictors=predictor_matrix,
        tr_indices=artifacts.tr_indices,
        feature_names=artifacts.predictor_feature_names,
        hidden_targets=hidden_targets,
        lags=lm_lags,
        lm_folds=int(args.lm_folds),
        stimulus_id=str(args.stimulus_id),
    )
    lm_cv = run_grouped_ridge_cv(
        x=lm_design["x"],
        y=lm_design["y"],
        groups=lm_design["groups"],
        sample_ids=lm_design["sample_ids"],
        target_names=hidden_targets["target_names"],
        alpha_grid=ALPHA_GRID,
        family_name="all_layers:lm_final_hidden_lagged",
    )
    lm_alpha = consensus_alpha(lm_cv["outer_folds"])
    lm_final = fit_final_model(
        x=lm_design["x"],
        y=lm_design["y"],
        alpha=lm_alpha,
        family_name="all_layers:lm_final_hidden_lagged",
    )
    lm_collapsed_weights = collapse_lagged_weights(
        weights=lm_final["weights"],
        base_feature_count=predictor_matrix.shape[1],
        lags=lm_lags,
    )

    common_lm_indices = np.asarray([np.where(lm_design["tr_indices"] == tr)[0][0] for tr in shared_tr_indices], dtype=int)
    lm_predictions_aligned = lm_final["predictions"][common_lm_indices]
    sample_rsa = compare_sample_geometry(brain_predicted_by_tr, lm_predictions_aligned)
    feature_importance = compare_feature_importance(
        brain_collapsed_weights,
        lm_collapsed_weights,
        artifacts.predictor_feature_names,
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
        base_feature_names=artifacts.predictor_feature_names,
        target_names=brain_targets.target_names,
        alpha=np.asarray([brain_alpha], dtype=float),
    )
    np.savez_compressed(
        family_output_dir / "lm_final_model.npz",
        weights=lm_final["weights"],
        collapsed_weights=lm_collapsed_weights,
        predictions=lm_final["predictions"],
        sample_ids=lm_design["sample_ids"],
        tr_indices=lm_design["tr_indices"],
        feature_names=lm_design["feature_names"],
        base_feature_names=artifacts.predictor_feature_names,
        target_names=hidden_targets["target_names"],
        alpha=np.asarray([lm_alpha], dtype=float),
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
        target_names=hidden_targets["target_names"],
        correlations=np.asarray(lm_cv["aggregate"]["per_target_mean_correlation"], dtype=float),
        r2_scores=np.asarray(lm_cv["aggregate"]["per_target_mean_r2"], dtype=float),
        top_k=10,
    )
    summary = {
        "stimulus_id": str(args.stimulus_id),
        "remote_run_dir": str(remote_run_dir),
        "brain_targets_npz": str(brain_targets_npz),
        "predictor_view": PREDICTOR_VIEW,
        "predictor_top_k": int(args.predictor_top_k) if args.predictor_top_k is not None else None,
        "lm_target_mode": "final_hidden_state",
        "brain_lags": brain_lags,
        "lm_lags": lm_lags,
        "lm_folds": int(args.lm_folds),
        "model_id": str(args.model_id),
        "scope_release": str(args.scope_release),
        "scope_width": str(args.scope_width),
        "layer_selection": [int(v) for v in args.layer_selection],
        "token_layer": int(args.token_layer),
        "families": [
            {
                "family_name": "all_layers",
                "predictor_view": PREDICTOR_VIEW,
                "predictor_count": int(predictor_matrix.shape[1]),
                "lm_design_predictor_count": int(lm_design["x"].shape[1]),
                "lm_target_count": int(hidden_targets["values"].shape[1]),
                "brain_sample_count": int(brain_design["x"].shape[0]),
                "brain_censored_sample_count": int(brain_design["censored_sample_count"]),
                "brain_retained_sample_fraction": float(brain_design["retained_sample_fraction"]),
                "brain_run_count": int(len(np.unique(brain_design["groups"]))),
                "lm_sample_count": int(lm_design["x"].shape[0]),
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
                    "sample_rsa": str(family_output_dir / "sample_rsa.json"),
                    "feature_importance_summary": str(family_output_dir / "feature_importance_summary.json"),
                    "tr_lm_target_final_hidden_average": str(intermediates_dir / "tr_lm_target_final_hidden_average.npz"),
                },
            }
        ],
    }
    write_json(family_output_dir / "summary.json", summary["families"][0])
    write_json(output_dir / "analysis_summary.json", summary)
    print(str(output_dir / "analysis_summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
