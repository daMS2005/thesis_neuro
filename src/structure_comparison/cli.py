"""Lightweight command line interface for structure-comparison workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from structure_comparison.defaults import DEFAULT_ALPHA_GRID, DEFAULT_BRAIN_LAGS
from thesis_neuro.paths import data_root, output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thesis-neuro-structure",
        description="Build transcript/TR artifacts and compare brain and LM ridge models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-tr-artifacts", help="Build one stimulus TR feature bundle.")
    _add_feature_args(build, pooled=False)

    pooled = subparsers.add_parser(
        "build-pooled-tr-artifacts",
        help="Build one TR feature bundle across multiple stimuli.",
    )
    _add_feature_args(pooled, pooled=True)

    brain = subparsers.add_parser("build-brain-targets", help="Extract parcel targets from BOLD runs.")
    _add_brain_args(brain, cleaned=False)

    clean_brain = subparsers.add_parser(
        "build-clean-brain-targets",
        help="Extract confound-cleaned parcel targets from fMRIPrep derivatives.",
    )
    _add_brain_args(clean_brain, cleaned=True)

    combine = subparsers.add_parser("combine-brain-targets", help="Combine stimulus brain bundles.")
    combine.add_argument("--bundle", action="append", required=True, metavar="STIMULUS=PATH")
    combine.add_argument("--output-path", required=True)

    analyze = subparsers.add_parser("run-analysis", help="Fit ridge models and compare learned structure.")
    _add_feature_args(analyze, pooled=False)
    analyze.add_argument("--brain-targets-npz", required=True)
    analyze.add_argument("--alpha-grid", nargs="+", type=float, default=list(DEFAULT_ALPHA_GRID))
    analyze.add_argument("--brain-lags", nargs="+", type=int, default=list(DEFAULT_BRAIN_LAGS))
    analyze.add_argument("--lm-folds", type=int, default=5)
    analyze.add_argument("--predictor-view", choices=("mass", "presence", "average"), default="mass")
    return parser


def _add_feature_args(parser: argparse.ArgumentParser, pooled: bool) -> None:
    parser.add_argument("--feature-run-dir", required=True)
    parser.add_argument("--transcript-root", default=str(data_root() / "transcripts"))
    if pooled:
        parser.add_argument("--stimulus-ids", nargs="+", required=True)
    else:
        parser.add_argument("--stimulus-id", required=True)
    parser.add_argument("--output-dir", default=str(output_root() / "structure-comparison"))
    parser.add_argument("--lm-targets-per-layer", type=int, default=8)
    parser.add_argument("--predictor-top-k", type=int, default=None)


def _add_brain_args(parser: argparse.ArgumentParser, cleaned: bool) -> None:
    parser.add_argument("--dataset-dir", default=str(data_root() / "openneuro" / "ds002345"))
    parser.add_argument("--transcript-root", default=str(data_root() / "transcripts"))
    parser.add_argument("--stimulus-id", required=True)
    parser.add_argument("--atlas-path", default=str(data_root() / "atlases" / "schaefer200.nii.gz"))
    parser.add_argument("--atlas-labels-csv", default=str(data_root() / "atlases" / "schaefer200_labels.csv"))
    parser.add_argument("--output-path", required=True)
    if cleaned:
        parser.add_argument("--fmriprep-dir", default=str(data_root() / "derivatives" / "ds002345-fmriprep"))
        parser.add_argument("--fd-threshold", type=float, default=0.5)
        parser.add_argument("--std-dvars-threshold", type=float, default=1.5)
        parser.add_argument("--high-pass-hz", type=float, default=0.008)
        parser.add_argument("--acompcor-count", type=int, default=6)
        parser.add_argument("--allow-partial-runs", action="store_true")


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch. Numerical modules are imported only once a command is chosen."""

    args = build_parser().parse_args(argv)

    if args.command in {"build-tr-artifacts", "build-pooled-tr-artifacts"}:
        from structure_comparison.artifacts import build_pooled_tr_feature_artifacts, build_tr_feature_artifacts

        common = {
            "feature_run_dir": Path(args.feature_run_dir),
            "transcript_root": Path(args.transcript_root),
            "output_dir": Path(args.output_dir),
            "lm_targets_per_layer": args.lm_targets_per_layer,
            "predictor_top_k": args.predictor_top_k,
        }
        if args.command == "build-tr-artifacts":
            result = build_tr_feature_artifacts(stimulus_id=args.stimulus_id, **common)
        else:
            result = build_pooled_tr_feature_artifacts(stimulus_ids=args.stimulus_ids, **common)
        _print(result["summary"])
        return

    if args.command in {"build-brain-targets", "build-clean-brain-targets"}:
        from structure_comparison.brain import build_brain_targets_from_dataset, build_clean_brain_targets_from_fmriprep

        common = {
            "dataset_dir": Path(args.dataset_dir),
            "transcript_root": Path(args.transcript_root),
            "stimulus_id": args.stimulus_id,
            "atlas_path": Path(args.atlas_path),
            "atlas_labels_csv": Path(args.atlas_labels_csv),
            "output_path": Path(args.output_path),
        }
        if args.command == "build-brain-targets":
            result = build_brain_targets_from_dataset(**common)
        else:
            result = build_clean_brain_targets_from_fmriprep(
                fmriprep_dir=Path(args.fmriprep_dir),
                fd_threshold=args.fd_threshold,
                std_dvars_threshold=args.std_dvars_threshold,
                high_pass_hz=args.high_pass_hz,
                acompcor_count=args.acompcor_count,
                allow_partial_runs=args.allow_partial_runs,
                **common,
            )
        _print(result)
        return

    if args.command == "combine-brain-targets":
        from structure_comparison.brain import combine_brain_target_bundles

        bundles = [_parse_bundle(value) for value in args.bundle]
        _print(combine_brain_target_bundles(bundles, Path(args.output_path)))
        return

    if args.command == "run-analysis":
        from structure_comparison.workflow import run_structure_comparison

        _print(
            run_structure_comparison(
                feature_run_dir=Path(args.feature_run_dir),
                transcript_root=Path(args.transcript_root),
                stimulus_id=args.stimulus_id,
                output_dir=Path(args.output_dir),
                brain_targets_npz=Path(args.brain_targets_npz),
                alpha_grid=args.alpha_grid,
                brain_lags=args.brain_lags,
                lm_folds=args.lm_folds,
                lm_targets_per_layer=args.lm_targets_per_layer,
                predictor_view=args.predictor_view,
                predictor_top_k=args.predictor_top_k,
            )
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


def _parse_bundle(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected STIMULUS=PATH, received: {value}")
    stimulus, path = value.split("=", 1)
    return stimulus, Path(path)


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
