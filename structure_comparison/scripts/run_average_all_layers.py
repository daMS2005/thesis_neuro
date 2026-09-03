from __future__ import annotations

import argparse
import json
from pathlib import Path

from structure_comparison.workflow import (
    build_family_matrices,
    build_tr_feature_artifacts,
    load_brain_targets,
    run_family_analysis,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an all-layers-only structure comparison using average TR activations."
    )
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--brain-targets-npz", required=True)
    parser.add_argument("--transcript-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stimulus-id", default="shapesphysical")
    parser.add_argument("--lm-targets-per-layer", type=int, default=8)
    parser.add_argument("--predictor-top-k", type=int, default=None)
    parser.add_argument("--lm-folds", type=int, default=5)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--brain-lags", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return parser


def main() -> None:
    args = build_parser().parse_args()

    remote_run_dir = Path(args.remote_run_dir)
    transcript_root = Path(args.transcript_root)
    output_dir = Path(args.output_dir)
    brain_targets_npz = Path(args.brain_targets_npz)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_result = build_tr_feature_artifacts(
        remote_run_dir=remote_run_dir,
        transcript_root=transcript_root,
        stimulus_id=str(args.stimulus_id),
        output_dir=output_dir,
        lm_targets_per_layer=int(args.lm_targets_per_layer),
        predictor_top_k=args.predictor_top_k,
    )
    artifacts = artifacts_result["artifacts"]
    families = build_family_matrices(
        artifacts=artifacts,
        predictor_matrix=artifacts.predictor_average,
        lm_target_matrix=artifacts.lm_target_average,
    )
    all_layers = next(family for family in families if family.family_name == "all_layers")

    family_output_dir = output_dir / "all_layers"
    family_output_dir.mkdir(parents=True, exist_ok=True)
    brain_targets = load_brain_targets(brain_targets_npz)
    family_summary = run_family_analysis(
        family=all_layers,
        brain_targets=brain_targets,
        family_output_dir=family_output_dir,
        alpha_grid=[float(value) for value in args.alpha_grid],
        brain_lags=[int(value) for value in args.brain_lags],
        lm_folds=int(args.lm_folds),
        predictor_view="average",
    )

    analysis_summary = {
        "stimulus_id": str(args.stimulus_id),
        "remote_run_dir": str(remote_run_dir),
        "brain_targets_npz": str(brain_targets_npz),
        "predictor_view": "average",
        "predictor_top_k": int(args.predictor_top_k) if args.predictor_top_k is not None else None,
        "brain_lags": [int(value) for value in args.brain_lags],
        "lm_folds": int(args.lm_folds),
        "families": [family_summary],
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(analysis_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(analysis_summary, indent=2))


if __name__ == "__main__":
    main()
