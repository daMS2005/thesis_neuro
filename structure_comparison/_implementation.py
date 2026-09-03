"""Established structure-comparison implementation behind the focused facade modules."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from thesis_neuro.paths import data_root, output_root

DEFAULT_REMOTE_HOST = os.environ.get("THESIS_NEURO_REMOTE_HOST", "")
DEFAULT_REMOTE_RUN_DIR = os.environ.get(
    "THESIS_NEURO_REMOTE_RUN_DIR",
    "",
)
DEFAULT_LOCAL_REMOTE_RUN_DIR = str(data_root() / "feature-runs" / "default")
DEFAULT_TRANSCRIPT_ROOT = str(data_root() / "transcripts")
DEFAULT_DATASET_DIR = str(data_root() / "openneuro" / "ds002345")
DEFAULT_ATLAS_PATH = str(data_root() / "atlases" / "schaefer200.nii.gz")
DEFAULT_ATLAS_LABELS_CSV = str(data_root() / "atlases" / "schaefer200_labels.csv")
DEFAULT_STIMULUS_ID = "shapessocial"
DEFAULT_OUTPUT_DIR = str(output_root() / "structure-comparison")
DEFAULT_FMRIPREP_DERIVATIVES_DIR = str(data_root() / "derivatives" / "ds002345-fmriprep")
DEFAULT_ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
DEFAULT_BRAIN_LAGS = [0, 1, 2, 3, 4]
DEFAULT_LM_FOLDS = 5
DEFAULT_LM_TARGETS_PER_LAYER = 8
DEFAULT_PREDICTOR_TOP_K: int | None = None
DEFAULT_FD_THRESHOLD = 0.5
DEFAULT_STD_DVARS_THRESHOLD = 1.5
DEFAULT_HIGH_PASS_HZ = 0.008
DEFAULT_ACOMPCOR_COUNT = 6
LAYER_FAMILIES = (8, 13, 22)


FeatureKey = tuple[int, int]


@dataclass(frozen=True, slots=True)
class TranscriptPaths:
    stimulus_id: str
    transcript_txt: Path
    words_tsv: Path
    tr_aligned_tsv: Path
    metadata_json: Path


@dataclass(frozen=True, slots=True)
class WordTiming:
    word: str
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class TrBin:
    tr_index: int
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True, slots=True)
class TokenWordGroup:
    text: str
    token_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TranscriptTokenStream:
    model_id: str
    word_start_marker: str | None
    tokens: list[str]


@dataclass(frozen=True, slots=True)
class BrainTargets:
    values: np.ndarray
    sample_ids: np.ndarray
    subject_ids: np.ndarray
    run_ids: np.ndarray
    tr_indices: np.ndarray
    target_names: np.ndarray
    censor_mask: np.ndarray | None = None
    framewise_displacement: np.ndarray | None = None
    std_dvars: np.ndarray | None = None


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="structure-comparison",
        description="Standalone structure-comparison workflow over synced remote thesis-neuro artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync-remote-run", help="Copy the full remote run locally with scp.")
    sync.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    sync.add_argument("--remote-run-dir", default=DEFAULT_REMOTE_RUN_DIR)
    sync.add_argument("--local-run-dir", default=DEFAULT_LOCAL_REMOTE_RUN_DIR)

    build = subparsers.add_parser(
        "build-tr-artifacts",
        help="Build TR-level predictor and LM target matrices from synced transcript artifacts.",
    )
    _add_common_build_args(build)

    pooled_build = subparsers.add_parser(
        "build-pooled-tr-artifacts",
        help="Build pooled TR-level predictor and LM target matrices across multiple stimuli.",
    )
    pooled_build.add_argument("--remote-run-dir", default=DEFAULT_LOCAL_REMOTE_RUN_DIR)
    pooled_build.add_argument("--transcript-root", default=DEFAULT_TRANSCRIPT_ROOT)
    pooled_build.add_argument("--stimulus-ids", nargs="+", required=True)
    pooled_build.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    pooled_build.add_argument("--lm-targets-per-layer", type=int, default=DEFAULT_LM_TARGETS_PER_LAYER)
    pooled_build.add_argument(
        "--predictor-top-k",
        type=int,
        default=DEFAULT_PREDICTOR_TOP_K,
        help="Keep only the top-k transcript-ranked predictor features before family construction.",
    )

    brain = subparsers.add_parser(
        "build-brain-targets",
        help="Build a parcel-level brain target matrix from local shapessocial BOLD runs using the atlas image in assets/roi.",
    )
    brain.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    brain.add_argument("--stimulus-id", default=DEFAULT_STIMULUS_ID)
    brain.add_argument("--atlas-path", default=DEFAULT_ATLAS_PATH)
    brain.add_argument("--atlas-labels-csv", default=DEFAULT_ATLAS_LABELS_CSV)
    brain.add_argument("--transcript-root", default=DEFAULT_TRANSCRIPT_ROOT)
    brain.add_argument("--output-path", default="structure_comparison/brain_targets/shapessocial_schaefer200.npz")

    clean_brain = subparsers.add_parser(
        "build-clean-brain-targets",
        help="Build a confound-cleaned parcel bundle from existing fMRIPrep derivatives with censor-mask support.",
    )
    clean_brain.add_argument("--fmriprep-dir", default=DEFAULT_FMRIPREP_DERIVATIVES_DIR)
    clean_brain.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    clean_brain.add_argument("--stimulus-id", default="shapesphysical")
    clean_brain.add_argument("--atlas-path", default=DEFAULT_ATLAS_PATH)
    clean_brain.add_argument("--atlas-labels-csv", default=DEFAULT_ATLAS_LABELS_CSV)
    clean_brain.add_argument("--transcript-root", default=DEFAULT_TRANSCRIPT_ROOT)
    clean_brain.add_argument("--fd-threshold", type=float, default=DEFAULT_FD_THRESHOLD)
    clean_brain.add_argument("--std-dvars-threshold", type=float, default=DEFAULT_STD_DVARS_THRESHOLD)
    clean_brain.add_argument("--high-pass-hz", type=float, default=DEFAULT_HIGH_PASS_HZ)
    clean_brain.add_argument("--acompcor-count", type=int, default=DEFAULT_ACOMPCOR_COUNT)
    clean_brain.add_argument(
        "--allow-partial-runs",
        action="store_true",
        help="Allow building a cleaned parcel bundle from a partial set of completed fMRIPrep runs.",
    )
    clean_brain.add_argument(
        "--output-path",
        default="structure_comparison/brain_targets/shapesphysical_schaefer200_cleaned.npz",
    )

    combine_brain = subparsers.add_parser(
        "combine-brain-targets",
        help="Combine multiple cleaned brain target bundles into one pooled bundle.",
    )
    combine_brain.add_argument(
        "--bundle",
        action="append",
        required=True,
        help="Bundle spec of the form stimulus_id=/path/to/bundle.npz . Repeat for each story.",
    )
    combine_brain.add_argument("--output-path", required=True)

    analyze = subparsers.add_parser(
        "run-analysis",
        help="Build TR artifacts and run the full structure comparison against parcel targets.",
    )
    _add_common_build_args(analyze)
    analyze.add_argument("--brain-targets-npz", required=True)
    analyze.add_argument("--alpha-grid", nargs="+", type=float, default=DEFAULT_ALPHA_GRID)
    analyze.add_argument("--brain-lags", nargs="+", type=int, default=DEFAULT_BRAIN_LAGS)
    analyze.add_argument("--lm-folds", type=int, default=DEFAULT_LM_FOLDS)
    analyze.add_argument(
        "--predictor-view",
        choices=("mass", "presence", "average"),
        default="mass",
        help="Analysis view. 'average' divides TR activation mass by active token-event count.",
    )

    run_all = subparsers.add_parser(
        "run-all",
        help="Sync the remote run first, then build artifacts and run the full analysis.",
    )
    run_all.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    run_all.add_argument("--remote-run-dir", default=DEFAULT_REMOTE_RUN_DIR)
    run_all.add_argument("--local-run-dir", default=DEFAULT_LOCAL_REMOTE_RUN_DIR)
    _add_common_build_args(run_all, include_remote_default=False, include_remote_arg=False)
    run_all.add_argument("--brain-targets-npz", required=True)
    run_all.add_argument("--alpha-grid", nargs="+", type=float, default=DEFAULT_ALPHA_GRID)
    run_all.add_argument("--brain-lags", nargs="+", type=int, default=DEFAULT_BRAIN_LAGS)
    run_all.add_argument("--lm-folds", type=int, default=DEFAULT_LM_FOLDS)
    run_all.add_argument(
        "--predictor-view",
        choices=("mass", "presence", "average"),
        default="mass",
        help="Analysis view. 'average' divides TR activation mass by active token-event count.",
    )
    return parser


def _add_common_build_args(
    parser: argparse.ArgumentParser,
    include_remote_default: bool = True,
    include_remote_arg: bool = True,
) -> None:
    if include_remote_arg:
        parser.add_argument(
            "--remote-run-dir",
            default=DEFAULT_LOCAL_REMOTE_RUN_DIR if include_remote_default else None,
            required=not include_remote_default,
        )
    parser.add_argument("--transcript-root", default=DEFAULT_TRANSCRIPT_ROOT)
    parser.add_argument("--stimulus-id", default=DEFAULT_STIMULUS_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lm-targets-per-layer", type=int, default=DEFAULT_LM_TARGETS_PER_LAYER)
    parser.add_argument(
        "--predictor-top-k",
        type=int,
        default=DEFAULT_PREDICTOR_TOP_K,
        help="Keep only the top-k transcript-ranked predictor features before family construction.",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "sync-remote-run":
        local_run_dir = sync_remote_run(
            remote_host=args.remote_host,
            remote_run_dir=args.remote_run_dir,
            local_run_dir=args.local_run_dir,
        )
        print(json.dumps({"local_run_dir": str(local_run_dir)}, indent=2))
        return

    if args.command == "build-tr-artifacts":
        output_dir = Path(args.output_dir)
        artifacts = build_tr_feature_artifacts(
            remote_run_dir=Path(args.remote_run_dir),
            transcript_root=Path(args.transcript_root),
            stimulus_id=args.stimulus_id,
            output_dir=output_dir,
            lm_targets_per_layer=int(args.lm_targets_per_layer),
            predictor_top_k=args.predictor_top_k,
        )
        print(json.dumps(artifacts["summary"], indent=2))
        return

    if args.command == "build-pooled-tr-artifacts":
        output_dir = Path(args.output_dir)
        artifacts = build_pooled_tr_feature_artifacts(
            remote_run_dir=Path(args.remote_run_dir),
            transcript_root=Path(args.transcript_root),
            stimulus_ids=[str(value) for value in args.stimulus_ids],
            output_dir=output_dir,
            lm_targets_per_layer=int(args.lm_targets_per_layer),
            predictor_top_k=args.predictor_top_k,
        )
        print(json.dumps(artifacts["summary"], indent=2))
        return

    if args.command == "build-brain-targets":
        summary = build_brain_targets_from_dataset(
            dataset_dir=Path(args.dataset_dir),
            transcript_root=Path(args.transcript_root),
            stimulus_id=args.stimulus_id,
            atlas_path=Path(args.atlas_path),
            atlas_labels_csv=Path(args.atlas_labels_csv),
            output_path=Path(args.output_path),
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "build-clean-brain-targets":
        summary = build_clean_brain_targets_from_fmriprep(
            fmriprep_dir=Path(args.fmriprep_dir),
            dataset_dir=Path(args.dataset_dir),
            transcript_root=Path(args.transcript_root),
            stimulus_id=args.stimulus_id,
            atlas_path=Path(args.atlas_path),
            atlas_labels_csv=Path(args.atlas_labels_csv),
            fd_threshold=float(args.fd_threshold),
            std_dvars_threshold=float(args.std_dvars_threshold),
            high_pass_hz=float(args.high_pass_hz),
            acompcor_count=int(args.acompcor_count),
            allow_partial_runs=bool(args.allow_partial_runs),
            output_path=Path(args.output_path),
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "combine-brain-targets":
        bundle_specs: list[tuple[str, Path]] = []
        for item in args.bundle:
            if "=" not in item:
                raise ValueError(f"Bundle spec must be stimulus_id=/path/to/bundle.npz, got: {item}")
            stimulus_id, raw_path = item.split("=", 1)
            bundle_specs.append((str(stimulus_id), Path(raw_path)))
        summary = combine_brain_target_bundles(bundle_specs=bundle_specs, output_path=Path(args.output_path))
        print(json.dumps(summary, indent=2))
        return

    if args.command == "run-analysis":
        summary = run_structure_comparison(
            remote_run_dir=Path(args.remote_run_dir),
            transcript_root=Path(args.transcript_root),
            stimulus_id=args.stimulus_id,
            output_dir=Path(args.output_dir),
            brain_targets_npz=Path(args.brain_targets_npz),
            alpha_grid=[float(value) for value in args.alpha_grid],
            brain_lags=[int(value) for value in args.brain_lags],
            lm_folds=int(args.lm_folds),
            lm_targets_per_layer=int(args.lm_targets_per_layer),
            predictor_view=str(args.predictor_view),
            predictor_top_k=args.predictor_top_k,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "run-all":
        local_run_dir = sync_remote_run(
            remote_host=args.remote_host,
            remote_run_dir=args.remote_run_dir,
            local_run_dir=args.local_run_dir,
        )
        summary = run_structure_comparison(
            remote_run_dir=local_run_dir,
            transcript_root=Path(args.transcript_root),
            stimulus_id=args.stimulus_id,
            output_dir=Path(args.output_dir),
            brain_targets_npz=Path(args.brain_targets_npz),
            alpha_grid=[float(value) for value in args.alpha_grid],
            brain_lags=[int(value) for value in args.brain_lags],
            lm_folds=int(args.lm_folds),
            lm_targets_per_layer=int(args.lm_targets_per_layer),
            predictor_view=str(args.predictor_view),
            predictor_top_k=args.predictor_top_k,
        )
        print(json.dumps(summary, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


def sync_remote_run(remote_host: str, remote_run_dir: str, local_run_dir: str | Path) -> Path:
    if not remote_host or not remote_run_dir:
        raise ValueError(
            "Remote sync requires --remote-host and --remote-run-dir; "
            "machine-specific defaults are intentionally not stored in the repository."
        )
    local_path = Path(local_run_dir)
    if local_path.exists() and any(local_path.iterdir()):
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "scp",
            "-r",
            f"{remote_host}:{remote_run_dir}",
            str(local_path.parent),
        ],
        check=True,
    )
    if not local_path.exists():
        raise FileNotFoundError(f"Sync completed without creating {local_path}")
    return local_path


def run_structure_comparison(
    remote_run_dir: Path,
    transcript_root: Path,
    stimulus_id: str,
    output_dir: Path,
    brain_targets_npz: Path,
    alpha_grid: list[float],
    brain_lags: list[int],
    lm_folds: int,
    lm_targets_per_layer: int,
    predictor_view: str,
    predictor_top_k: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_result = build_tr_feature_artifacts(
        remote_run_dir=remote_run_dir,
        transcript_root=transcript_root,
        stimulus_id=stimulus_id,
        output_dir=output_dir,
        lm_targets_per_layer=lm_targets_per_layer,
        predictor_top_k=predictor_top_k,
    )
    artifacts = artifact_result["artifacts"]
    brain_targets = load_brain_targets(brain_targets_npz)

    if predictor_view == "mass":
        predictor_matrix = artifacts.predictor_mass
        lm_target_matrix = artifacts.lm_target_mass
    elif predictor_view == "average":
        predictor_matrix = artifacts.predictor_average
        lm_target_matrix = artifacts.lm_target_average
    else:
        predictor_matrix = artifacts.predictor_presence.astype(float)
        lm_target_matrix = artifacts.lm_target_mass
    families = build_family_matrices(artifacts, predictor_matrix, lm_target_matrix)
    family_summaries: list[dict[str, Any]] = []
    for family in families:
        family_output_dir = output_dir / family.family_name
        family_output_dir.mkdir(parents=True, exist_ok=True)
        family_summary = run_family_analysis(
            family=family,
            brain_targets=brain_targets,
            family_output_dir=family_output_dir,
            alpha_grid=alpha_grid,
            brain_lags=brain_lags,
            lm_folds=lm_folds,
            predictor_view=predictor_view,
        )
        family_summaries.append(family_summary)

    summary = {
        "stimulus_id": stimulus_id,
        "remote_run_dir": str(remote_run_dir),
        "brain_targets_npz": str(brain_targets_npz),
        "predictor_view": predictor_view,
        "predictor_top_k": int(predictor_top_k) if predictor_top_k is not None else None,
        "brain_lags": [int(value) for value in brain_lags],
        "lm_folds": int(lm_folds),
        "lm_targets_per_layer": int(lm_targets_per_layer),
        "families": family_summaries,
        "artifacts": artifact_result["summary"]["artifacts"],
    }
    write_json(output_dir / "analysis_summary.json", summary)
    return summary


def build_tr_feature_artifacts(
    remote_run_dir: Path,
    transcript_root: Path,
    stimulus_id: str,
    output_dir: Path,
    lm_targets_per_layer: int,
    predictor_top_k: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    required_remote_paths = {
        "transcript_paired_records": resolve_transcript_paired_path(remote_run_dir),
        "selected_features_for_alignment": remote_run_dir / "selected_features_for_alignment.jsonl",
        "feature_alignment": remote_run_dir / "feature_alignment.jsonl",
        "feature_concepts": remote_run_dir / "feature_concepts.jsonl",
        "manifest": remote_run_dir / "manifest.json",
    }
    for label, path in required_remote_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required remote artifact {label}: {path}")

    transcript_paths = resolve_transcript_paths(transcript_root, stimulus_id)
    transcript_txt = transcript_paths.transcript_txt.read_text(encoding="utf-8").strip()
    word_rows = load_word_rows(transcript_paths.words_tsv)
    tr_bins = load_tr_bins(transcript_paths.tr_aligned_tsv)
    metadata = json.loads(transcript_paths.metadata_json.read_text(encoding="utf-8"))

    predictor_keys = load_predictor_feature_keys(
        required_remote_paths["selected_features_for_alignment"],
        top_k=predictor_top_k,
    )
    family_layers = tuple(sorted({layer for layer, _feature_id in predictor_keys}))
    lm_target_keys = select_lm_target_feature_keys(
        feature_concepts_path=required_remote_paths["feature_concepts"],
        predictor_keys=predictor_keys,
        targets_per_layer=lm_targets_per_layer,
        target_layers=family_layers,
    )

    token_stream = extract_global_tokens(
        transcript_paired_path=required_remote_paths["transcript_paired_records"],
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
        transcript_paired_path=required_remote_paths["transcript_paired_records"],
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
        remote_run_dir=remote_run_dir,
        predictor_top_k=predictor_top_k,
    )
    return {"artifacts": feature_artifacts, "summary": summary}


def build_pooled_tr_feature_artifacts(
    remote_run_dir: Path,
    transcript_root: Path,
    stimulus_ids: list[str],
    output_dir: Path,
    lm_targets_per_layer: int,
    predictor_top_k: int | None = None,
) -> dict[str, Any]:
    if not stimulus_ids:
        raise ValueError("stimulus_ids must not be empty.")
    output_dir.mkdir(parents=True, exist_ok=True)
    required_remote_paths = {
        "transcript_paired_records": resolve_transcript_paired_path(remote_run_dir),
        "selected_features_for_alignment": remote_run_dir / "selected_features_for_alignment.jsonl",
        "feature_alignment": remote_run_dir / "feature_alignment.jsonl",
        "feature_concepts": remote_run_dir / "feature_concepts.jsonl",
        "manifest": remote_run_dir / "manifest.json",
    }
    for label, path in required_remote_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required remote artifact {label}: {path}")

    predictor_keys = load_predictor_feature_keys(
        required_remote_paths["selected_features_for_alignment"],
        top_k=predictor_top_k,
    )
    family_layers = tuple(sorted({layer for layer, _feature_id in predictor_keys}))
    lm_target_keys = select_lm_target_feature_keys(
        feature_concepts_path=required_remote_paths["feature_concepts"],
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
            transcript_paired_path=required_remote_paths["transcript_paired_records"],
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
                transcript_paired_path=required_remote_paths["transcript_paired_records"],
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
        remote_run_dir=remote_run_dir,
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
    remote_run_dir: Path,
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
        "remote_run_dir": str(remote_run_dir),
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


def resolve_transcript_paths(transcript_root: Path, stimulus_id: str) -> TranscriptPaths:
    base = transcript_root / stimulus_id
    paths = TranscriptPaths(
        stimulus_id=stimulus_id,
        transcript_txt=base / f"{stimulus_id}_transcript.txt",
        words_tsv=base / f"{stimulus_id}_words.tsv",
        tr_aligned_tsv=base / f"{stimulus_id}_tr_aligned.tsv",
        metadata_json=base / "metadata.json",
    )
    for path in (paths.transcript_txt, paths.words_tsv, paths.tr_aligned_tsv, paths.metadata_json):
        if not path.exists():
            raise FileNotFoundError(path)
    return paths


def build_brain_targets_from_dataset(
    dataset_dir: Path,
    transcript_root: Path,
    stimulus_id: str,
    atlas_path: Path,
    atlas_labels_csv: Path,
    output_path: Path,
) -> dict[str, Any]:
    import nibabel as nib
    from nilearn.maskers import NiftiLabelsMasker

    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    if not atlas_path.exists():
        raise FileNotFoundError(atlas_path)
    if not atlas_labels_csv.exists():
        raise FileNotFoundError(atlas_labels_csv)

    transcript_paths = resolve_transcript_paths(transcript_root, stimulus_id)
    tr_bins = load_tr_bins(transcript_paths.tr_aligned_tsv)
    expected_trs = len(tr_bins)
    parcel_names = load_atlas_label_names(atlas_labels_csv)
    bold_paths = sorted(dataset_dir.glob(f"sub-*/func/*task-{stimulus_id}_bold.nii.gz"))
    if not bold_paths:
        raise ValueError(f"No local BOLD paths found for stimulus {stimulus_id} under {dataset_dir}")

    resolved_paths = [path for path in bold_paths if path.exists()]
    missing_paths = [path for path in bold_paths if not path.exists()]
    if not resolved_paths:
        raise ValueError(
            f"Found {len(bold_paths)} BOLD entries for {stimulus_id}, but none have local annex payloads materialized."
        )

    atlas_img = nib.load(str(atlas_path))
    atlas_name_by_label = load_atlas_label_map(atlas_labels_csv)
    run_entries: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    subject_ids: list[str] = []
    run_ids: list[str] = []
    tr_indices: list[int] = []
    used_paths: list[str] = []
    truncated_runs: list[str] = []
    common_label_ids: set[int] | None = None
    for bold_path in resolved_paths:
        subject_id = bold_path.parts[-3]
        run_id = parse_run_id(bold_path)
        masker = NiftiLabelsMasker(
            labels_img=str(atlas_path),
            standardize=False,
            detrend=False,
            smoothing_fwhm=None,
            low_pass=None,
            high_pass=None,
            t_r=None,
            verbose=0,
        )
        time_series = np.asarray(masker.fit_transform(str(bold_path)), dtype=np.float32)
        if time_series.ndim != 2:
            raise ValueError(f"Expected a 2D parcel matrix for {bold_path}, got {time_series.shape}")
        if time_series.shape[0] < expected_trs:
            raise ValueError(
                f"{bold_path} has {time_series.shape[0]} TRs, but {expected_trs} are required to match the transcript bins."
            )
        if time_series.shape[0] > expected_trs:
            time_series = time_series[:expected_trs]
            truncated_runs.append(str(bold_path))
        label_ids = present_label_ids_for_run(atlas_img, bold_path)
        if time_series.shape[1] != len(label_ids):
            raise ValueError(
                f"{bold_path} produced {time_series.shape[1]} parcel columns, but {len(label_ids)} labels survived "
                "after atlas resampling."
            )
        label_set = set(label_ids)
        common_label_ids = label_set if common_label_ids is None else common_label_ids.intersection(label_set)
        used_paths.append(str(bold_path))
        run_entries.append(
            {
                "bold_path": str(bold_path),
                "subject_id": subject_id,
                "run_id": run_id,
                "time_series": time_series,
                "label_ids": label_ids,
            }
        )

    if not run_entries:
        raise ValueError("No resolved shapessocial runs were available for parcel extraction.")
    if not common_label_ids:
        raise ValueError("No common atlas parcels survived resampling across the resolved runs.")

    common_label_ids_sorted = sorted(common_label_ids)
    parcel_names = [atlas_name_by_label[label_id] for label_id in common_label_ids_sorted]
    rows: list[np.ndarray] = []
    for entry in run_entries:
        label_index = {label_id: index for index, label_id in enumerate(entry["label_ids"])}
        aligned_series = entry["time_series"][:, [label_index[label_id] for label_id in common_label_ids_sorted]]
        for tr_index in range(expected_trs):
            rows.append(aligned_series[tr_index])
            sample_ids.append(f"{entry['subject_id']}:{entry['run_id']}:{tr_index}")
            subject_ids.append(entry["subject_id"])
            run_ids.append(entry["run_id"])
            tr_indices.append(tr_index)

    values = np.vstack(rows).astype(np.float32, copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        values=values,
        sample_ids=np.asarray(sample_ids, dtype=str),
        subject_ids=np.asarray(subject_ids, dtype=str),
        run_ids=np.asarray(run_ids, dtype=str),
        tr_indices=np.asarray(tr_indices, dtype=int),
        target_names=np.asarray(parcel_names, dtype=str),
    )
    summary = {
        "stimulus_id": stimulus_id,
        "dataset_dir": str(dataset_dir),
        "atlas_path": str(atlas_path),
        "atlas_labels_csv": str(atlas_labels_csv),
        "output_path": str(output_path),
        "resolved_bold_runs": len(resolved_paths),
        "missing_annex_payloads": len(missing_paths),
        "expected_trs_per_run": expected_trs,
        "total_samples": int(values.shape[0]),
        "parcel_count": int(values.shape[1]),
        "common_label_ids": common_label_ids_sorted,
        "used_bold_paths": used_paths,
        "missing_bold_paths_preview": [str(path) for path in missing_paths[:10]],
        "truncated_runs": truncated_runs,
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def build_clean_brain_targets_from_fmriprep(
    fmriprep_dir: Path,
    dataset_dir: Path,
    transcript_root: Path,
    stimulus_id: str,
    atlas_path: Path,
    atlas_labels_csv: Path,
    fd_threshold: float,
    std_dvars_threshold: float,
    high_pass_hz: float,
    acompcor_count: int,
    allow_partial_runs: bool,
    output_path: Path,
) -> dict[str, Any]:
    import nibabel as nib
    from nilearn.maskers import NiftiLabelsMasker

    if not fmriprep_dir.exists():
        raise FileNotFoundError(fmriprep_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    if not atlas_path.exists():
        raise FileNotFoundError(atlas_path)
    if not atlas_labels_csv.exists():
        raise FileNotFoundError(atlas_labels_csv)

    transcript_paths = resolve_transcript_paths(transcript_root, stimulus_id)
    expected_trs = len(load_tr_bins(transcript_paths.tr_aligned_tsv))
    atlas_img = nib.load(str(atlas_path))
    atlas_name_by_label = load_atlas_label_map(atlas_labels_csv)
    run_artifacts = find_fmriprep_run_artifacts(fmriprep_dir=fmriprep_dir, stimulus_id=stimulus_id)
    expected_raw_runs = sorted(dataset_dir.glob(f"sub-*/func/*task-{stimulus_id}_bold.nii.gz"))
    expected_keys = {(path.parts[-3], parse_run_id(path)) for path in expected_raw_runs if path.exists()}
    observed_keys = {(str(item["subject_id"]), str(item["run_id"])) for item in run_artifacts}
    missing_runs = sorted(expected_keys - observed_keys)
    if not run_artifacts:
        raise ValueError(f"No fMRIPrep preprocessed runs found for stimulus {stimulus_id} under {fmriprep_dir}")
    if missing_runs and not allow_partial_runs:
        raise ValueError(
            "Missing fMRIPrep derivatives for some raw runs, including "
            + ", ".join(f"{subject}:{run}" for subject, run in missing_runs[:10])
        )

    cleaned_runs: list[dict[str, Any]] = []
    common_label_ids: set[int] | None = None
    run_summaries: list[dict[str, Any]] = []
    confound_columns_union: list[str] = []
    sample_ids: list[str] = []
    subject_ids: list[str] = []
    run_ids: list[str] = []
    tr_indices: list[int] = []
    framewise_displacement: list[float] = []
    std_dvars: list[float] = []
    censor_mask: list[bool] = []
    rows: list[np.ndarray] = []

    for artifact in run_artifacts:
        subject_id = str(artifact["subject_id"])
        run_id = str(artifact["run_id"])
        bold_path = Path(artifact["bold_path"])
        confounds_path = Path(artifact["confounds_path"])
        metadata_path = Path(artifact["metadata_path"])
        if not confounds_path.exists():
            raise FileNotFoundError(confounds_path)
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        confounds_frame = pd.read_csv(confounds_path, sep="\t")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        tr_value = extract_repetition_time(metadata)
        label_ids = present_label_ids_for_run(atlas_img, bold_path)
        masker = NiftiLabelsMasker(
            labels_img=str(atlas_path),
            standardize=False,
            detrend=False,
            smoothing_fwhm=None,
            low_pass=None,
            high_pass=None,
            t_r=None,
            verbose=0,
        )
        parcel_matrix = np.asarray(masker.fit_transform(str(bold_path)), dtype=np.float32)
        if parcel_matrix.ndim != 2:
            raise ValueError(f"Expected a 2D parcel matrix for {bold_path}, got {parcel_matrix.shape}")
        if parcel_matrix.shape[0] < expected_trs:
            raise ValueError(
                f"{bold_path} has {parcel_matrix.shape[0]} TRs, but {expected_trs} are required to match the transcript bins."
            )
        if parcel_matrix.shape[1] != len(label_ids):
            raise ValueError(
                f"{bold_path} produced {parcel_matrix.shape[1]} parcel columns, but {len(label_ids)} labels survived "
                "after atlas resampling."
            )

        parcel_matrix = parcel_matrix[:expected_trs]
        confounds_frame = confounds_frame.iloc[:expected_trs].copy()
        confound_matrix, confound_columns, fd_values, dvars_values, run_censor_mask = build_confounds_for_cleaning(
            confounds_frame=confounds_frame,
            fd_threshold=fd_threshold,
            std_dvars_threshold=std_dvars_threshold,
            acompcor_count=acompcor_count,
        )
        cleaned_series = clean_parcel_matrix(
            parcel_matrix=parcel_matrix,
            confound_matrix=confound_matrix,
            tr_value=tr_value,
            high_pass_hz=high_pass_hz,
        )
        cleaned_tsnr = compute_tsnr(cleaned_series)
        standardized_series = zscore_with_reference_mask(cleaned_series, ~run_censor_mask)
        label_set = set(label_ids)
        common_label_ids = label_set if common_label_ids is None else common_label_ids.intersection(label_set)
        confound_columns_union.extend(column for column in confound_columns if column not in confound_columns_union)
        cleaned_runs.append(
            {
                "subject_id": subject_id,
                "run_id": run_id,
                "bold_path": str(bold_path),
                "confounds_path": str(confounds_path),
                "metadata_path": str(metadata_path),
                "label_ids": label_ids,
                "values": standardized_series,
                "fd_values": fd_values,
                "dvars_values": dvars_values,
                "censor_mask": run_censor_mask,
                "cleaned_tsnr": cleaned_tsnr,
            }
        )
        run_summaries.append(
            {
                "subject_id": subject_id,
                "run_id": run_id,
                "bold_path": str(bold_path),
                "confounds_path": str(confounds_path),
                "tr_count": int(standardized_series.shape[0]),
                "parcel_count_before_intersection": int(standardized_series.shape[1]),
                "censor_fraction": float(run_censor_mask.mean()),
                "retained_tr_count": int((~run_censor_mask).sum()),
                "mean_cleaned_tsnr": float(np.nanmean(cleaned_tsnr)),
            }
        )

    if not cleaned_runs:
        raise ValueError("No cleaned runs were built from the fMRIPrep derivatives.")
    if not common_label_ids:
        raise ValueError("No common atlas parcels survived resampling across the cleaned runs.")

    common_label_ids_sorted = sorted(common_label_ids)
    parcel_names = [atlas_name_by_label[label_id] for label_id in common_label_ids_sorted]
    for entry in cleaned_runs:
        label_index = {label_id: index for index, label_id in enumerate(entry["label_ids"])}
        aligned_values = entry["values"][:, [label_index[label_id] for label_id in common_label_ids_sorted]]
        for tr_index in range(expected_trs):
            rows.append(aligned_values[tr_index])
            sample_ids.append(f"{entry['subject_id']}:{entry['run_id']}:{tr_index}")
            subject_ids.append(entry["subject_id"])
            run_ids.append(entry["run_id"])
            tr_indices.append(tr_index)
            framewise_displacement.append(float(entry["fd_values"][tr_index]))
            std_dvars.append(float(entry["dvars_values"][tr_index]))
            censor_mask.append(bool(entry["censor_mask"][tr_index]))

    values = np.vstack(rows).astype(np.float32, copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        values=values,
        sample_ids=np.asarray(sample_ids, dtype=str),
        subject_ids=np.asarray(subject_ids, dtype=str),
        run_ids=np.asarray(run_ids, dtype=str),
        tr_indices=np.asarray(tr_indices, dtype=int),
        target_names=np.asarray(parcel_names, dtype=str),
        censor_mask=np.asarray(censor_mask, dtype=bool),
        framewise_displacement=np.asarray(framewise_displacement, dtype=np.float32),
        std_dvars=np.asarray(std_dvars, dtype=np.float32),
    )
    summary = {
        "stimulus_id": stimulus_id,
        "dataset_dir": str(dataset_dir),
        "fmriprep_dir": str(fmriprep_dir),
        "atlas_path": str(atlas_path),
        "atlas_labels_csv": str(atlas_labels_csv),
        "output_path": str(output_path),
        "run_count": len(cleaned_runs),
        "expected_run_count": len(expected_keys),
        "observed_run_count": len(run_artifacts),
        "allow_partial_runs": bool(allow_partial_runs),
        "missing_runs": [f"{subject}:{run}" for subject, run in missing_runs],
        "expected_trs_per_run": expected_trs,
        "total_samples": int(values.shape[0]),
        "retained_samples_after_censoring": int((~np.asarray(censor_mask, dtype=bool)).sum()),
        "parcel_count": int(values.shape[1]),
        "common_label_ids": common_label_ids_sorted,
        "mean_censor_fraction_overall": float(np.mean(censor_mask)),
        "censor_fraction_by_run": {
            f"{run['subject_id']}:{run['run_id']}": float(run["censor_fraction"]) for run in run_summaries
        },
        "mean_cleaned_tsnr_by_run": {
            f"{run['subject_id']}:{run['run_id']}": float(run["mean_cleaned_tsnr"]) for run in run_summaries
        },
        "confound_columns_used": confound_columns_union,
        "preprocessing": {
            "high_pass_hz": float(high_pass_hz),
            "fd_threshold": float(fd_threshold),
            "std_dvars_threshold": float(std_dvars_threshold),
            "acompcor_count": int(acompcor_count),
            "detrend": True,
            "standardize_within_run": True,
            "standardize_reference": "uncensored_trs",
            "global_signal_regression": False,
            "preserve_full_tr_grid": True,
        },
        "run_summaries": run_summaries,
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def find_fmriprep_run_artifacts(fmriprep_dir: Path, stimulus_id: str) -> list[dict[str, str]]:
    bold_paths = sorted(
        fmriprep_dir.glob(
            f"sub-*/func/*task-{stimulus_id}*space-MNI152NLin6Asym*_desc-preproc_bold.nii.gz"
        )
    )
    if not bold_paths:
        # Capstone batch outputs are nested as story/batch_xx/out/sub-*/func/... rather than a flat
        # fMRIPrep derivatives tree, so fall back to a recursive search and ignore archived failed runs.
        bold_paths = sorted(
            path
            for path in fmriprep_dir.rglob(f"*task-{stimulus_id}*space-MNI152NLin6Asym*_desc-preproc_bold.nii.gz")
            if not any("out_failed" in part for part in path.parts)
        )
    run_artifacts: list[dict[str, str]] = []
    for bold_path in bold_paths:
        stem = bold_path.name.replace("_desc-preproc_bold.nii.gz", "")
        confounds_candidates = [
            bold_path.with_name(f"{stem}_desc-confounds_timeseries.tsv"),
            bold_path.with_name(
                bold_path.name.replace(
                    "_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz",
                    "_desc-confounds_timeseries.tsv",
                )
            ),
        ]
        confounds_path = next((path for path in confounds_candidates if path.exists()), None)
        metadata_path = bold_path.with_name(f"{stem}_desc-preproc_bold.json")
        if confounds_path is None:
            raise FileNotFoundError(
                "Missing confounds TSV for "
                f"{bold_path}: tried {', '.join(str(path) for path in confounds_candidates)}"
            )
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing preprocessed BOLD JSON sidecar for {bold_path}: {metadata_path}")
        run_artifacts.append(
            {
                "subject_id": bold_path.parts[-3],
                "run_id": parse_run_id(bold_path),
                "bold_path": str(bold_path),
                "confounds_path": str(confounds_path),
                "metadata_path": str(metadata_path),
            }
        )
    return run_artifacts


def extract_repetition_time(metadata: dict[str, Any]) -> float:
    tr_value = metadata.get("RepetitionTime")
    if tr_value is None:
        raise ValueError("Preprocessed BOLD sidecar is missing RepetitionTime.")
    return float(tr_value)


def build_confounds_for_cleaning(
    confounds_frame: pd.DataFrame,
    fd_threshold: float,
    std_dvars_threshold: float,
    acompcor_count: int,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    motion_columns = [
        "trans_x",
        "trans_y",
        "trans_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "trans_x_derivative1",
        "trans_y_derivative1",
        "trans_z_derivative1",
        "rot_x_derivative1",
        "rot_y_derivative1",
        "rot_z_derivative1",
        "trans_x_power2",
        "trans_y_power2",
        "trans_z_power2",
        "rot_x_power2",
        "rot_y_power2",
        "rot_z_power2",
        "trans_x_derivative1_power2",
        "trans_y_derivative1_power2",
        "trans_z_derivative1_power2",
        "rot_x_derivative1_power2",
        "rot_y_derivative1_power2",
        "rot_z_derivative1_power2",
    ]
    missing_motion = [column for column in motion_columns if column not in confounds_frame.columns]
    if missing_motion:
        raise ValueError(f"Confounds file is missing required 24-parameter motion columns: {missing_motion}")

    acompcor_columns = sorted(
        [
            column
            for column in confounds_frame.columns
            if re.fullmatch(r"a_comp_cor_\d+", str(column))
        ]
    )
    if len(acompcor_columns) < acompcor_count:
        raise ValueError(
            f"Confounds file only has {len(acompcor_columns)} aCompCor columns; expected at least {acompcor_count}."
        )
    selected_columns = [*motion_columns, *acompcor_columns[:acompcor_count]]
    confounds = confounds_frame[selected_columns].copy()
    confounds = confounds.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    fd_values = (
        confounds_frame["framewise_displacement"].to_numpy(dtype=float)
        if "framewise_displacement" in confounds_frame.columns
        else np.zeros(confounds_frame.shape[0], dtype=float)
    )
    dvars_values = (
        confounds_frame["std_dvars"].to_numpy(dtype=float)
        if "std_dvars" in confounds_frame.columns
        else np.zeros(confounds_frame.shape[0], dtype=float)
    )
    fd_values = np.nan_to_num(fd_values, nan=0.0, posinf=0.0, neginf=0.0)
    dvars_values = np.nan_to_num(dvars_values, nan=0.0, posinf=0.0, neginf=0.0)

    non_steady_columns = [
        column for column in confounds_frame.columns if str(column).startswith("non_steady_state_outlier")
    ]
    non_steady_mask = np.zeros(confounds_frame.shape[0], dtype=bool)
    if non_steady_columns:
        non_steady_values = confounds_frame[non_steady_columns].fillna(0.0).to_numpy(dtype=float)
        non_steady_mask = non_steady_values.any(axis=1)

    censor_mask = (
        (fd_values > float(fd_threshold))
        | (dvars_values > float(std_dvars_threshold))
        | non_steady_mask
    )
    return confounds.to_numpy(dtype=float), selected_columns, fd_values, dvars_values, censor_mask.astype(bool)


def clean_parcel_matrix(
    parcel_matrix: np.ndarray,
    confound_matrix: np.ndarray,
    tr_value: float,
    high_pass_hz: float,
) -> np.ndarray:
    from nilearn.signal import clean as nilearn_clean

    cleaned = nilearn_clean(
        signals=np.asarray(parcel_matrix, dtype=float),
        confounds=np.asarray(confound_matrix, dtype=float),
        detrend=True,
        standardize=False,
        standardize_confounds=True,
        t_r=float(tr_value),
        low_pass=None,
        high_pass=float(high_pass_hz),
    )
    return np.asarray(cleaned, dtype=np.float32)


def zscore_with_reference_mask(values: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D array for z-scoring, got {values.shape}")
    reference_mask = np.asarray(reference_mask, dtype=bool)
    if reference_mask.shape[0] != values.shape[0]:
        raise ValueError("Reference mask length does not match values rows.")
    if not np.any(reference_mask):
        reference_mask = np.ones(values.shape[0], dtype=bool)
    reference_values = values[reference_mask]
    mean = reference_values.mean(axis=0)
    scale = reference_values.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    return ((values - mean) / scale).astype(np.float32)


def compute_tsnr(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    std[std == 0] = np.nan
    return np.abs(mean) / std


def load_atlas_label_names(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if "ROI Name" not in frame.columns:
        raise ValueError(f"{path} does not contain an 'ROI Name' column.")
    return frame["ROI Name"].astype(str).tolist()


def load_atlas_label_map(path: Path) -> dict[int, str]:
    frame = pd.read_csv(path)
    if "ROI Label" not in frame.columns or "ROI Name" not in frame.columns:
        raise ValueError(f"{path} must contain 'ROI Label' and 'ROI Name' columns.")
    return {int(row["ROI Label"]): str(row["ROI Name"]) for _, row in frame.iterrows()}


def present_label_ids_for_run(atlas_img: Any, bold_path: Path) -> list[int]:
    import nibabel as nib
    from nilearn.image import resample_to_img

    bold_img = nib.load(str(bold_path))
    reference_img = bold_img.slicer[..., 0]
    resampled = resample_to_img(atlas_img, reference_img, interpolation="nearest", force_resample=True, copy_header=True)
    data = np.rint(resampled.get_fdata()).astype(int)
    return [int(value) for value in sorted(np.unique(data[data > 0]).tolist())]


def parse_run_id(path: Path) -> str:
    match = re.search(r"_run-([A-Za-z0-9]+)_", path.name)
    return f"run-{match.group(1)}" if match else "run-1"


def load_word_rows(path: Path) -> list[WordTiming]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows: list[WordTiming] = []
        for row in reader:
            word = str(row["word"])
            # Some timing files include punctuation-only rows (for elongated pauses like ". . . .")
            # that do not correspond to token-derived lexical items, so we skip them here.
            if not normalize_text(word):
                continue
            rows.append(
                WordTiming(
                    word=word,
                    start_s=float(row["start_s"]),
                    end_s=float(row["end_s"]),
                )
            )
        return rows


def load_tr_bins(path: Path) -> list[TrBin]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [
            TrBin(
                tr_index=int(row["tr_index"]),
                start_s=float(row["start_s"]),
                end_s=float(row["end_s"]),
                text=str(row["text"]),
            )
            for row in reader
        ]


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


def extract_global_tokens(transcript_paired_path: Path, stimulus_id: str, layer: int) -> TranscriptTokenStream:
    by_index: dict[int, str] = {}
    model_ids: set[str] = set()
    with transcript_paired_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["layer"]) != int(layer):
                continue
            provenance = row.get("provenance") or {}
            if provenance.get("stimulus_id") != stimulus_id:
                continue
            global_token_index = int(row["window_start"]) + int(row["token_position"])
            token = str(row["token"])
            existing = by_index.get(global_token_index)
            if existing is not None and existing != token:
                raise ValueError(
                    f"Inconsistent token reconstruction at index {global_token_index}: {existing!r} vs {token!r}"
                )
            by_index[global_token_index] = token
            model_ids.add(str(row.get("model_id", "")))
    if not by_index:
        raise ValueError(f"No transcript rows found for stimulus={stimulus_id} layer={layer}")
    cleaned_model_ids = {model_id for model_id in model_ids if model_id}
    if len(cleaned_model_ids) > 1:
        raise ValueError(f"Expected a single model_id for {stimulus_id} layer {layer}, found {sorted(cleaned_model_ids)}")
    ordered_indices = sorted(by_index)
    expected_indices = list(range(ordered_indices[0], ordered_indices[-1] + 1))
    if ordered_indices != expected_indices:
        raise ValueError("Global token indices are not contiguous.")
    if ordered_indices[0] != 0:
        raise ValueError(f"Expected transcript tokens to start at global index 0, found {ordered_indices[0]}")
    tokens = [by_index[index] for index in ordered_indices]
    model_id = next(iter(cleaned_model_ids), "")
    return TranscriptTokenStream(
        model_id=model_id,
        word_start_marker=infer_word_start_marker(tokens, model_id=model_id),
        tokens=tokens,
    )


def resolve_transcript_paired_path(remote_run_dir: Path) -> Path:
    candidates = (
        remote_run_dir / "transcript_paired_records.minimal.jsonl",
        remote_run_dir / "transcript_paired_records.jsonl",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_cached_tokenizer(model_id: str) -> Any:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ImportError(
            "The tokenizers package is required to align transcript tokens with the model tokenizer."
        ) from exc
    model_stub = f"models--{model_id.replace('/', '--')}"
    search_roots: list[Path] = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        search_roots.append(Path(hf_home) / "hub")
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        search_roots.append(Path(hub_cache))
    search_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    candidates: list[Path] = []
    searched_paths: list[Path] = []
    for root in search_roots:
        snapshot_root = root / model_stub / "snapshots"
        searched_paths.append(snapshot_root)
        candidates.extend(sorted(snapshot_root.glob("*/tokenizer.json")))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find cached tokenizer.json for model_id={model_id!r} under "
            + ", ".join(str(path) for path in searched_paths)
        )
    return Tokenizer.from_file(str(candidates[-1]))


def group_tokens_with_model_tokenizer(
    model_id: str,
    tokens: list[str],
    transcript_text: str,
) -> list[TokenWordGroup]:
    tokenizer = load_cached_tokenizer(model_id)
    encoding = tokenizer.encode(transcript_text, add_special_tokens=False)
    encoded_tokens = list(encoding.tokens)
    if encoded_tokens != tokens:
        for index, (expected, observed) in enumerate(zip(tokens, encoded_tokens)):
            if expected != observed:
                raise ValueError(
                    f"Tokenizer mismatch for {model_id} at token {index}: "
                    f"artifact={expected!r} tokenizer={observed!r}"
                )
        if len(encoded_tokens) != len(tokens):
            raise ValueError(
                f"Tokenizer mismatch for {model_id}: artifact has {len(tokens)} tokens, "
                f"tokenizer produced {len(encoded_tokens)}"
            )
    return group_tokens_into_words(
        encoded_tokens,
        word_start_marker=infer_word_start_marker(encoded_tokens, model_id=model_id),
    )


def infer_word_start_marker(tokens: list[str], model_id: str = "") -> str | None:
    normalized_model_id = model_id.lower()
    if "gemma" in normalized_model_id:
        return "▁"
    if "llama" in normalized_model_id:
        return "Ġ"
    if any(token.startswith("Ġ") for token in tokens):
        return "Ġ"
    if any(token.startswith("▁") for token in tokens):
        return "▁"
    return None


def _is_tokenizer_whitespace_token(token: str) -> bool:
    return token in {"Ċ", "<0x0A>", "\n", "\r", "\t"}


def group_tokens_into_words(tokens: list[str], word_start_marker: str | None = None) -> list[TokenWordGroup]:
    groups: list[TokenWordGroup] = []
    current_text: list[str] = []
    current_indices: list[int] = []
    start_marker = word_start_marker if word_start_marker is not None else infer_word_start_marker(tokens)
    for index, token in enumerate(tokens):
        if _is_tokenizer_whitespace_token(token):
            if current_text:
                groups.append(TokenWordGroup(text="".join(current_text), token_indices=tuple(current_indices)))
                current_text = []
                current_indices = []
            continue
        is_word_start = bool(start_marker) and token.startswith(start_marker)
        if is_word_start and current_text:
            groups.append(TokenWordGroup(text="".join(current_text), token_indices=tuple(current_indices)))
            current_text = []
            current_indices = []
        piece = token[len(start_marker) :] if is_word_start and start_marker is not None else token
        if not piece and not normalize_text(token):
            continue
        current_text.append(piece)
        current_indices.append(index)
    if current_text:
        groups.append(TokenWordGroup(text="".join(current_text), token_indices=tuple(current_indices)))
    return groups


def validate_word_alignment(
    token_word_groups: list[TokenWordGroup],
    word_rows: list[WordTiming],
    transcript_text: str,
) -> None:
    if len(token_word_groups) != len(word_rows):
        raise ValueError(
            f"Token-derived word count {len(token_word_groups)} does not match timed words {len(word_rows)}."
        )
    mismatches: list[str] = []
    for index, (token_word, timed_word) in enumerate(zip(token_word_groups, word_rows)):
        if normalize_text(token_word.text) != normalize_text(timed_word.word):
            mismatches.append(f"{index}:{token_word.text!r}!={timed_word.word!r}")
            if len(mismatches) >= 5:
                break
    if mismatches:
        raise ValueError("Token-to-word alignment mismatches: " + ", ".join(mismatches))
    reconstructed = " ".join(word.word for word in word_rows).strip()
    if normalize_text(reconstructed) != normalize_text(transcript_text):
        raise ValueError("Timed words do not reconstruct the transcript text.")


def build_token_to_word_index(token_word_groups: list[TokenWordGroup]) -> dict[int, int]:
    token_to_word: dict[int, int] = {}
    for word_index, group in enumerate(token_word_groups):
        for token_index in group.token_indices:
            token_to_word[token_index] = word_index
    return token_to_word


def assign_words_to_trs(
    word_rows: list[WordTiming],
    tr_bins: list[TrBin],
    stimulus_onset_s: float,
) -> list[int]:
    assignments: list[int] = []
    for word in word_rows:
        midpoint = stimulus_onset_s + ((word.start_s + word.end_s) / 2.0)
        assigned = None
        for tr_index, tr_bin in enumerate(tr_bins):
            if tr_bin.start_s <= midpoint < tr_bin.end_s:
                assigned = tr_index
                break
        if assigned is None and math.isclose(midpoint, tr_bins[-1].end_s):
            assigned = len(tr_bins) - 1
        if assigned is None:
            raise ValueError(f"Could not assign word {word.word!r} at midpoint={midpoint} to a TR bin.")
        assignments.append(int(assigned))
    return assignments


def validate_tr_alignment(
    word_rows: list[WordTiming],
    word_to_tr_index: list[int],
    tr_bins: list[TrBin],
    stimulus_onset_s: float,
) -> None:
    by_tr: dict[int, list[str]] = defaultdict(list)
    for word, tr_index in zip(word_rows, word_to_tr_index):
        by_tr[int(tr_index)].append(word.word)
    mismatches: list[str] = []
    for tr_index, tr_bin in enumerate(tr_bins):
        expected = normalize_text(tr_bin.text)
        observed = normalize_text(" ".join(by_tr.get(tr_index, [])))
        if expected != observed:
            mismatches.append(f"TR {tr_index}: {observed!r}!={expected!r}")
            if len(mismatches) >= 5:
                break
    if mismatches:
        raise ValueError("Word-to-TR alignment mismatches: " + ", ".join(mismatches))
    if stimulus_onset_s > 0 and tr_bins and tr_bins[0].start_s != 0.0:
        raise ValueError("TR bins are expected to start at 0.0 seconds.")


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


def load_brain_targets(path: Path) -> BrainTargets:
    if not path.exists():
        raise FileNotFoundError(path)
    source = np.load(path, allow_pickle=False)
    required = ("values", "sample_ids", "subject_ids", "run_ids", "tr_indices", "target_names")
    missing = [key for key in required if key not in source]
    if missing:
        raise ValueError(f"Brain target bundle is missing keys: {', '.join(missing)}")
    values = np.asarray(source["values"], dtype=float)
    sample_ids = _string_array(source["sample_ids"])
    subject_ids = _string_array(source["subject_ids"])
    run_ids = _string_array(source["run_ids"])
    tr_indices = np.asarray(source["tr_indices"], dtype=int)
    target_names = _string_array(source["target_names"])
    if values.ndim != 2:
        raise ValueError(f"Brain target values must be 2D, got {values.shape}")
    n_samples = values.shape[0]
    for label, array in (
        ("sample_ids", sample_ids),
        ("subject_ids", subject_ids),
        ("run_ids", run_ids),
        ("tr_indices", tr_indices),
    ):
        if array.shape[0] != n_samples:
            raise ValueError(f"Brain target {label} has {array.shape[0]} rows, expected {n_samples}")
    if target_names.shape[0] != values.shape[1]:
        raise ValueError("Brain target_names length does not match values columns.")
    _ensure_unique(sample_ids, "brain sample_ids")
    censor_mask = None
    framewise_displacement = None
    std_dvars = None
    optional_vector_keys = (
        ("censor_mask", bool),
        ("framewise_displacement", float),
        ("std_dvars", float),
    )
    for key, dtype in optional_vector_keys:
        if key not in source:
            continue
        vector = np.asarray(source[key], dtype=dtype)
        if vector.shape[0] != n_samples:
            raise ValueError(f"Brain target {key} has {vector.shape[0]} rows, expected {n_samples}")
        if key == "censor_mask":
            censor_mask = vector.astype(bool, copy=False)
        elif key == "framewise_displacement":
            framewise_displacement = vector.astype(float, copy=False)
        elif key == "std_dvars":
            std_dvars = vector.astype(float, copy=False)
    return BrainTargets(
        values=values,
        sample_ids=sample_ids,
        subject_ids=subject_ids,
        run_ids=run_ids,
        tr_indices=tr_indices,
        target_names=target_names,
        censor_mask=censor_mask,
        framewise_displacement=framewise_displacement,
        std_dvars=std_dvars,
    )


def combine_brain_target_bundles(
    bundle_specs: list[tuple[str, Path]],
    output_path: Path,
) -> dict[str, Any]:
    if not bundle_specs:
        raise ValueError("bundle_specs must not be empty.")

    pooled_values: list[np.ndarray] = []
    pooled_sample_ids: list[np.ndarray] = []
    pooled_subject_ids: list[np.ndarray] = []
    pooled_run_ids: list[np.ndarray] = []
    pooled_tr_indices: list[np.ndarray] = []
    pooled_censor_mask: list[np.ndarray] = []
    pooled_fd: list[np.ndarray] = []
    pooled_dvars: list[np.ndarray] = []
    reference_target_names: np.ndarray | None = None
    summaries: list[dict[str, Any]] = []
    has_censor = False
    has_fd = False
    has_dvars = False

    for stimulus_id, bundle_path in bundle_specs:
        targets = load_brain_targets(bundle_path)
        if reference_target_names is None:
            reference_target_names = targets.target_names
        elif (
            reference_target_names.shape != targets.target_names.shape
            or not np.array_equal(reference_target_names, targets.target_names)
        ):
            raise ValueError(f"Target names do not match across pooled bundles: {bundle_path}")

        run_ids = np.asarray([f"{stimulus_id}:{run_id}" for run_id in targets.run_ids.tolist()], dtype=str)
        sample_ids = np.asarray(
            [
                f"{subject_id}:{stimulus_id}:{run_id}:{int(tr_index)}"
                for subject_id, run_id, tr_index in zip(
                    targets.subject_ids.tolist(),
                    targets.run_ids.tolist(),
                    targets.tr_indices.tolist(),
                )
            ],
            dtype=str,
        )
        pooled_values.append(targets.values.astype(np.float32, copy=False))
        pooled_sample_ids.append(sample_ids)
        pooled_subject_ids.append(targets.subject_ids.astype(str, copy=False))
        pooled_run_ids.append(run_ids)
        pooled_tr_indices.append(targets.tr_indices.astype(int, copy=False))
        if targets.censor_mask is not None:
            has_censor = True
            pooled_censor_mask.append(targets.censor_mask.astype(bool, copy=False))
        if targets.framewise_displacement is not None:
            has_fd = True
            pooled_fd.append(targets.framewise_displacement.astype(np.float32, copy=False))
        if targets.std_dvars is not None:
            has_dvars = True
            pooled_dvars.append(targets.std_dvars.astype(np.float32, copy=False))
        summaries.append(
            {
                "stimulus_id": str(stimulus_id),
                "bundle_path": str(bundle_path),
                "sample_count": int(targets.values.shape[0]),
                "run_count": int(len(np.unique(targets.run_ids))),
            }
        )

    values = np.vstack(pooled_values).astype(np.float32, copy=False)
    sample_ids = np.concatenate(pooled_sample_ids).astype(str, copy=False)
    subject_ids = np.concatenate(pooled_subject_ids).astype(str, copy=False)
    run_ids = np.concatenate(pooled_run_ids).astype(str, copy=False)
    tr_indices = np.concatenate(pooled_tr_indices).astype(int, copy=False)
    _ensure_unique(sample_ids, "pooled brain sample_ids")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {
        "values": values,
        "sample_ids": sample_ids,
        "subject_ids": subject_ids,
        "run_ids": run_ids,
        "tr_indices": tr_indices,
        "target_names": reference_target_names,
    }
    if has_censor:
        save_kwargs["censor_mask"] = np.concatenate(pooled_censor_mask).astype(bool, copy=False)
    if has_fd:
        save_kwargs["framewise_displacement"] = np.concatenate(pooled_fd).astype(np.float32, copy=False)
    if has_dvars:
        save_kwargs["std_dvars"] = np.concatenate(pooled_dvars).astype(np.float32, copy=False)
    np.savez_compressed(output_path, **save_kwargs)

    summary = {
        "output_path": str(output_path),
        "stimulus_ids": [stimulus_id for stimulus_id, _ in bundle_specs],
        "sample_count": int(values.shape[0]),
        "run_count": int(len(np.unique(run_ids))),
        "target_count": int(reference_target_names.shape[0]),
        "bundles": summaries,
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def run_family_analysis(
    family: FamilyMatrices,
    brain_targets: BrainTargets,
    family_output_dir: Path,
    alpha_grid: list[float],
    brain_lags: list[int],
    lm_folds: int,
    predictor_view: str,
) -> dict[str, Any]:
    brain_design = build_brain_design_matrix(
        tr_predictors=family.predictor_values,
        tr_indices=family.tr_indices,
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
        alpha_grid=alpha_grid,
        family_name=f"{family.family_name}:brain",
    )
    brain_alpha = consensus_alpha(brain_cv["outer_folds"])
    brain_final = fit_final_model(brain_design["x"], brain_design["y"], brain_alpha, family.family_name)
    brain_collapsed_weights = collapse_lagged_weights(
        weights=brain_final["weights"],
        base_feature_count=family.predictor_values.shape[1],
        lags=brain_lags,
    )
    brain_predicted_by_tr, shared_tr_indices = average_predictions_by_tr(
        predictions=brain_final["predictions"],
        tr_indices=brain_design["tr_indices"],
    )

    lm_groups = contiguous_block_groups(len(family.tr_indices), lm_folds)
    lm_cv = run_grouped_ridge_cv(
        x=family.predictor_values,
        y=family.lm_target_values,
        groups=lm_groups,
        sample_ids=family.sample_ids,
        target_names=family.lm_target_names,
        alpha_grid=alpha_grid,
        family_name=f"{family.family_name}:lm",
    )
    lm_alpha = consensus_alpha(lm_cv["outer_folds"])
    lm_final = fit_final_model(
        x=family.predictor_values,
        y=family.lm_target_values,
        alpha=lm_alpha,
        family_name=family.family_name,
    )

    common_lm_indices = np.asarray([np.where(family.tr_indices == tr)[0][0] for tr in shared_tr_indices], dtype=int)
    lm_predictions_aligned = lm_final["predictions"][common_lm_indices]

    weight_similarity = cosine_similarity_matrix(brain_collapsed_weights, lm_final["weights"])
    sample_rsa = compare_sample_geometry(brain_predicted_by_tr, lm_predictions_aligned)
    feature_importance = compare_feature_importance(
        brain_collapsed_weights,
        lm_final["weights"],
        family.predictor_base_feature_names,
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
        base_feature_names=family.predictor_base_feature_names,
        target_names=brain_targets.target_names,
        alpha=np.asarray([brain_alpha], dtype=float),
    )
    np.savez_compressed(
        family_output_dir / "lm_final_model.npz",
        weights=lm_final["weights"],
        predictions=lm_final["predictions"],
        sample_ids=family.sample_ids,
        tr_indices=family.tr_indices,
        feature_names=family.predictor_feature_names,
        target_names=family.lm_target_names,
        alpha=np.asarray([lm_alpha], dtype=float),
    )
    np.savez_compressed(
        family_output_dir / "brain_lm_weight_similarity.npz",
        similarity=weight_similarity,
        brain_target_names=brain_targets.target_names,
        lm_target_names=family.lm_target_names,
        feature_names=family.predictor_base_feature_names,
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
        target_names=family.lm_target_names,
        correlations=np.asarray(lm_cv["aggregate"]["per_target_mean_correlation"], dtype=float),
        r2_scores=np.asarray(lm_cv["aggregate"]["per_target_mean_r2"], dtype=float),
        top_k=10,
    )
    shared_predictors = feature_importance["top_features"][:10]

    summary = {
        "family_name": family.family_name,
        "predictor_view": predictor_view,
        "predictor_count": int(family.predictor_values.shape[1]),
        "lm_target_count": int(family.lm_target_values.shape[1]),
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
        "top_shared_predictors": shared_predictors,
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


def build_brain_design_matrix(
    tr_predictors: np.ndarray,
    tr_indices: np.ndarray,
    feature_names: np.ndarray,
    brain_targets: BrainTargets,
    lags: list[int],
) -> dict[str, np.ndarray]:
    if tr_predictors.ndim != 2:
        raise ValueError(f"TR predictor matrix must be 2D, got {tr_predictors.shape}")
    if sorted(lags) != list(lags):
        raise ValueError("Brain lags must be sorted in ascending order.")
    tr_lookup = {int(tr_index): index for index, tr_index in enumerate(tr_indices.tolist())}
    design_rows: list[np.ndarray] = []
    kept_indices: list[int] = []
    for sample_index, tr_index in enumerate(brain_targets.tr_indices.tolist()):
        if int(tr_index) not in tr_lookup:
            continue
        if brain_targets.censor_mask is not None and bool(brain_targets.censor_mask[sample_index]):
            continue
        base_tr_position = tr_lookup[int(tr_index)]
        lagged_parts: list[np.ndarray] = []
        for feature_index in range(tr_predictors.shape[1]):
            for lag in lags:
                source_position = base_tr_position - int(lag)
                if source_position < 0:
                    lagged_parts.append(np.zeros(1, dtype=float))
                else:
                    lagged_parts.append(np.asarray([tr_predictors[source_position, feature_index]], dtype=float))
        design_rows.append(np.concatenate(lagged_parts))
        kept_indices.append(sample_index)
    if not design_rows:
        raise ValueError("No brain samples share TR indices with the transcript predictors.")
    x = np.vstack(design_rows).astype(float, copy=False)
    y = brain_targets.values[np.asarray(kept_indices, dtype=int)]
    sample_ids = brain_targets.sample_ids[np.asarray(kept_indices, dtype=int)]
    subject_ids = brain_targets.subject_ids[np.asarray(kept_indices, dtype=int)]
    run_ids = brain_targets.run_ids[np.asarray(kept_indices, dtype=int)]
    matched_tr_indices = brain_targets.tr_indices[np.asarray(kept_indices, dtype=int)]
    censored_sample_count = 0
    if brain_targets.censor_mask is not None:
        matched_tr_lookup = np.isin(brain_targets.tr_indices, tr_indices)
        censored_sample_count = int(np.sum(brain_targets.censor_mask & matched_tr_lookup))
    groups = np.asarray(
        [f"{subject}:{run}" for subject, run in zip(subject_ids.tolist(), run_ids.tolist())],
        dtype=str,
    )
    expanded_feature_names = np.asarray(
        [
            f"{feature_name}@lag{lag}"
            for feature_name in feature_names.tolist()
            for lag in lags
        ],
        dtype=str,
    )
    return {
        "x": x,
        "y": y,
        "groups": groups,
        "sample_ids": sample_ids,
        "tr_indices": matched_tr_indices,
        "feature_names": expanded_feature_names,
        "censored_sample_count": int(censored_sample_count),
        "retained_sample_fraction": float(x.shape[0] / (x.shape[0] + censored_sample_count)),
    }


def build_pooled_brain_design_matrix(
    tr_predictors: np.ndarray,
    predictor_sample_ids: np.ndarray,
    feature_names: np.ndarray,
    brain_targets: BrainTargets,
    lags: list[int],
) -> dict[str, np.ndarray]:
    if tr_predictors.ndim != 2:
        raise ValueError(f"TR predictor matrix must be 2D, got {tr_predictors.shape}")
    if sorted(lags) != list(lags):
        raise ValueError("Brain lags must be sorted in ascending order.")

    predictor_lookup: dict[tuple[str, int], int] = {}
    for row_index, sample_id in enumerate(_string_array(predictor_sample_ids).tolist()):
        stimulus_id, tr_index = parse_predictor_sample_id(sample_id)
        predictor_lookup[(stimulus_id, int(tr_index))] = row_index

    design_rows: list[np.ndarray] = []
    kept_indices: list[int] = []
    kept_story_tr_keys: list[str] = []
    for sample_index, (run_id, tr_index) in enumerate(zip(brain_targets.run_ids.tolist(), brain_targets.tr_indices.tolist())):
        if brain_targets.censor_mask is not None and bool(brain_targets.censor_mask[sample_index]):
            continue
        stimulus_id = parse_stimulus_from_pooled_run_id(str(run_id))
        key = (stimulus_id, int(tr_index))
        if key not in predictor_lookup:
            continue
        lagged_parts: list[np.ndarray] = []
        for feature_index in range(tr_predictors.shape[1]):
            for lag in lags:
                source_key = (stimulus_id, int(tr_index) - int(lag))
                source_position = predictor_lookup.get(source_key)
                if source_position is None:
                    lagged_parts.append(np.zeros(1, dtype=float))
                else:
                    lagged_parts.append(np.asarray([tr_predictors[source_position, feature_index]], dtype=float))
        design_rows.append(np.concatenate(lagged_parts))
        kept_indices.append(sample_index)
        kept_story_tr_keys.append(f"{stimulus_id}:tr:{int(tr_index)}")

    if not design_rows:
        raise ValueError("No pooled brain samples share keyed TR indices with the transcript predictors.")
    x = np.vstack(design_rows).astype(float, copy=False)
    kept = np.asarray(kept_indices, dtype=int)
    y = brain_targets.values[kept]
    sample_ids = brain_targets.sample_ids[kept]
    subject_ids = brain_targets.subject_ids[kept]
    run_ids = brain_targets.run_ids[kept]
    matched_tr_indices = brain_targets.tr_indices[kept]
    censored_sample_count = 0
    if brain_targets.censor_mask is not None:
        matched_keys = {
            key for key in predictor_lookup
        }
        target_keys = np.asarray(
            [
                (parse_stimulus_from_pooled_run_id(str(run_id)), int(tr_index)) in matched_keys
                for run_id, tr_index in zip(brain_targets.run_ids.tolist(), brain_targets.tr_indices.tolist())
            ],
            dtype=bool,
        )
        censored_sample_count = int(np.sum(brain_targets.censor_mask & target_keys))
    groups = np.asarray(
        [f"{subject}:{run}" for subject, run in zip(subject_ids.tolist(), run_ids.tolist())],
        dtype=str,
    )
    expanded_feature_names = np.asarray(
        [
            f"{feature_name}@lag{lag}"
            for feature_name in feature_names.tolist()
            for lag in lags
        ],
        dtype=str,
    )
    return {
        "x": x,
        "y": y,
        "groups": groups,
        "sample_ids": sample_ids,
        "tr_indices": matched_tr_indices,
        "story_tr_keys": np.asarray(kept_story_tr_keys, dtype=str),
        "feature_names": expanded_feature_names,
        "censored_sample_count": int(censored_sample_count),
        "retained_sample_fraction": float(x.shape[0] / (x.shape[0] + censored_sample_count)),
    }


def parse_predictor_sample_id(sample_id: str) -> tuple[str, int]:
    parts = str(sample_id).split(":")
    if len(parts) != 3 or parts[1] != "tr":
        raise ValueError(f"Expected predictor sample_id in stimulus:tr:index format, got: {sample_id}")
    return parts[0], int(parts[2])


def parse_stimulus_from_pooled_run_id(run_id: str) -> str:
    if ":" not in run_id:
        raise ValueError(f"Expected pooled run_id in stimulus:run format, got: {run_id}")
    return run_id.split(":", 1)[0]


def run_grouped_ridge_cv(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_ids: np.ndarray,
    target_names: np.ndarray,
    alpha_grid: Iterable[float],
    family_name: str,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError(f"{family_name}: need at least two groups for evaluation.")
    alpha_values = [float(value) for value in alpha_grid]
    per_target_correlations: list[np.ndarray] = []
    per_target_r2_scores: list[np.ndarray] = []
    outer_folds: list[dict[str, Any]] = []

    for test_group in unique_groups.tolist():
        test_mask = groups == test_group
        train_mask = ~test_mask
        train_groups = groups[train_mask]
        best_alpha = select_alpha_with_inner_cv(
            x=x[train_mask],
            y=y[train_mask],
            groups=train_groups,
            alpha_grid=alpha_values,
        )
        model = fit_ridge_with_standardization(
            x_train=x[train_mask],
            y_train=y[train_mask],
            x_eval=x[test_mask],
            alpha=best_alpha,
        )
        metrics = regression_metrics(y[test_mask], model["predictions"])
        per_target_correlations.append(metrics["per_target_correlation"])
        per_target_r2_scores.append(metrics["per_target_r2"])
        outer_folds.append(
            {
                "test_group": str(test_group),
                "n_train_samples": int(train_mask.sum()),
                "n_test_samples": int(test_mask.sum()),
                "test_sample_ids": sample_ids[test_mask].tolist(),
                "selected_alpha": float(best_alpha),
                "mean_test_correlation": float(metrics["mean_correlation"]),
                "mean_test_r2": float(metrics["mean_r2"]),
                "top_targets_by_correlation": top_metric_rows(
                    target_names=target_names,
                    correlations=metrics["per_target_correlation"],
                    r2_scores=metrics["per_target_r2"],
                    top_k=10,
                ),
            }
        )

    stacked_correlations = np.vstack(per_target_correlations)
    stacked_r2 = np.vstack(per_target_r2_scores)
    aggregate = {
        "mean_test_correlation": float(np.nanmean([fold["mean_test_correlation"] for fold in outer_folds])),
        "mean_test_r2": float(np.nanmean([fold["mean_test_r2"] for fold in outer_folds])),
        "per_target_mean_correlation": safe_nanmean(stacked_correlations, axis=0).tolist(),
        "per_target_mean_r2": safe_nanmean(stacked_r2, axis=0).tolist(),
    }
    return {
        "family": family_name,
        "alpha_grid": alpha_values,
        "aggregate": aggregate,
        "outer_folds": outer_folds,
    }


def select_alpha_with_inner_cv(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alpha_grid: list[float],
) -> float:
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return float(alpha_grid[0])
    best_alpha = float(alpha_grid[0])
    best_score = -np.inf
    for alpha in alpha_grid:
        fold_scores: list[float] = []
        for validation_group in unique_groups.tolist():
            validation_mask = groups == validation_group
            train_mask = ~validation_mask
            if train_mask.sum() == 0 or validation_mask.sum() == 0:
                continue
            model = fit_ridge_with_standardization(
                x_train=x[train_mask],
                y_train=y[train_mask],
                x_eval=x[validation_mask],
                alpha=float(alpha),
            )
            metrics = regression_metrics(y[validation_mask], model["predictions"])
            fold_scores.append(float(metrics["mean_correlation"]))
        score = float(np.nanmean(fold_scores)) if fold_scores else -np.inf
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha


def fit_final_model(x: np.ndarray, y: np.ndarray, alpha: float, family_name: str) -> dict[str, np.ndarray]:
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{family_name}: X and Y have different sample counts.")
    return fit_ridge_with_standardization(x_train=x, y_train=y, x_eval=x, alpha=alpha)


def fit_ridge_with_standardization(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    x_train_scaled, x_mean, x_scale = zscore_fit_transform(x_train)
    y_train_scaled, y_mean, y_scale = zscore_fit_transform(y_train)
    x_eval_scaled = zscore_apply(x_eval, x_mean, x_scale)
    xtx = x_train_scaled.T @ x_train_scaled
    xty = x_train_scaled.T @ y_train_scaled
    penalty = alpha * np.eye(xtx.shape[0], dtype=x_train_scaled.dtype)
    weights_scaled = np.linalg.solve(xtx + penalty, xty)
    predictions_scaled = x_eval_scaled @ weights_scaled
    predictions = unscale(predictions_scaled, y_mean, y_scale)
    weights = weights_scaled * (y_scale[np.newaxis, :] / x_scale[:, np.newaxis])
    return {"weights": weights, "predictions": predictions}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    correlations = np.asarray(
        [_pearson(y_true[:, index], y_pred[:, index]) for index in range(y_true.shape[1])],
        dtype=float,
    )
    r2_scores = np.asarray(
        [_r2(y_true[:, index], y_pred[:, index]) for index in range(y_true.shape[1])],
        dtype=float,
    )
    return {
        "mean_correlation": float(np.nanmean(correlations)),
        "mean_r2": float(np.nanmean(r2_scores)),
        "per_target_correlation": correlations,
        "per_target_r2": r2_scores,
    }


def top_metric_rows(
    target_names: np.ndarray,
    correlations: np.ndarray,
    r2_scores: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    rows = [
        {
            "target_name": str(target_names[index]),
            "correlation": float(correlations[index]),
            "r2": float(r2_scores[index]),
        }
        for index in range(target_names.shape[0])
    ]
    rows.sort(key=lambda row: np.nan_to_num(row["correlation"], nan=-np.inf), reverse=True)
    return rows[:top_k]


def consensus_alpha(outer_folds: list[dict[str, Any]]) -> float:
    counts: dict[float, int] = {}
    for fold in outer_folds:
        alpha = float(fold["selected_alpha"])
        counts[alpha] = counts.get(alpha, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def contiguous_block_groups(sample_count: int, n_folds: int) -> np.ndarray:
    if n_folds <= 1:
        raise ValueError("n_folds must be at least 2.")
    groups = np.empty(sample_count, dtype=int)
    for fold_index, indices in enumerate(np.array_split(np.arange(sample_count), n_folds)):
        groups[indices] = fold_index
    return groups


def collapse_lagged_weights(weights: np.ndarray, base_feature_count: int, lags: list[int]) -> np.ndarray:
    if weights.shape[0] != base_feature_count * len(lags):
        raise ValueError("Lagged weight matrix shape does not match base feature count and lag count.")
    reshaped = weights.reshape(base_feature_count, len(lags), weights.shape[1])
    return reshaped.sum(axis=1)


def average_predictions_by_tr(predictions: np.ndarray, tr_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_trs = np.unique(tr_indices)
    averaged = []
    for tr_index in unique_trs.tolist():
        averaged.append(predictions[tr_indices == tr_index].mean(axis=0))
    return np.vstack(averaged), unique_trs.astype(int)


def cosine_similarity_matrix(left_weights: np.ndarray, right_weights: np.ndarray) -> np.ndarray:
    left_norm = _normalize_columns(left_weights)
    right_norm = _normalize_columns(right_weights)
    return left_norm.T @ right_norm


def compare_sample_geometry(brain_predictions: np.ndarray, lm_predictions: np.ndarray) -> dict[str, Any]:
    if brain_predictions.shape[0] != lm_predictions.shape[0]:
        raise ValueError("Brain and LM predictions must have the same number of shared TR samples.")
    brain_similarity = sample_similarity_matrix(brain_predictions)
    lm_similarity = sample_similarity_matrix(lm_predictions)
    upper = np.triu_indices(brain_similarity.shape[0], k=1)
    brain_vector = brain_similarity[upper]
    lm_vector = lm_similarity[upper]
    return {
        "sample_rsa_correlation": float(_pearson(brain_vector, lm_vector)),
        "n_sample_pairs": int(brain_vector.shape[0]),
    }


def compare_feature_importance(
    brain_weights: np.ndarray,
    lm_weights: np.ndarray,
    feature_names: np.ndarray,
    top_k: int = 25,
) -> dict[str, Any]:
    brain_importance = np.linalg.norm(brain_weights, axis=1)
    lm_importance = np.linalg.norm(lm_weights, axis=1)
    ranking = [
        {
            "feature_name": str(feature_names[index]),
            "brain_importance": float(brain_importance[index]),
            "lm_importance": float(lm_importance[index]),
        }
        for index in range(feature_names.shape[0])
    ]
    ranking.sort(key=lambda row: row["brain_importance"] + row["lm_importance"], reverse=True)
    return {
        "importance_correlation": float(_pearson(brain_importance, lm_importance)),
        "top_features": ranking[:top_k],
    }


def sample_similarity_matrix(values: np.ndarray) -> np.ndarray:
    row_norms = np.linalg.norm(values, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    normalized = values / row_norms
    return normalized @ normalized.T


def zscore_fit_transform(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    return (values - mean) / scale, mean, scale


def zscore_apply(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - mean) / scale


def unscale(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return values * scale + mean


def safe_divide_rows(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=np.float32)
    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator > 0,
    )
    return result


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def counts_by_layer(keys: Iterable[FeatureKey]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for layer, _feature_id in keys:
        key = str(int(layer))
        counts[key] = counts.get(key, 0) + 1
    return counts


def safe_nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    mask = ~np.isnan(values)
    totals = np.where(mask, values, 0.0).sum(axis=axis)
    counts = mask.sum(axis=axis)
    result = np.full(totals.shape, np.nan, dtype=float)
    valid = counts > 0
    result[valid] = totals[valid] / counts[valid]
    return result


def feature_name(key: FeatureKey) -> str:
    layer, feature_id = key
    return f"layer{int(layer)}:feature{int(feature_id)}"


def normalize_text(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _string_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=str)


def _ensure_unique(values: np.ndarray, label: str) -> None:
    unique_values, counts = np.unique(values, return_counts=True)
    if np.any(counts > 1):
        repeated = unique_values[counts > 1][:5].tolist()
        raise ValueError(f"{label} contains duplicates, including {repeated}")


def _normalize_columns(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size == 0 or right.size == 0:
        return float("nan")
    left_std = left.std(ddof=0)
    right_std = right.std(ddof=0)
    if left_std == 0 or right_std == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    baseline = y_true.mean()
    ss_tot = float(np.sum((y_true - baseline) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
