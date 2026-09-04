"""End-to-end structure comparison across predictor families."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from structure_comparison.artifacts import build_family_matrices, build_tr_feature_artifacts
from structure_comparison.brain import load_brain_targets
from structure_comparison.modeling import run_family_analysis
from structure_comparison.utils import write_json


def run_structure_comparison(
    feature_run_dir: Path,
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
        feature_run_dir=feature_run_dir,
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
        # Presence is a binary predictor view; LM targets stay continuous (mass) so the fit is still a regression.
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
        "feature_run_dir": str(feature_run_dir),
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
