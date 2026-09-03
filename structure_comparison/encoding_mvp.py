"""Minimal grouped-ridge encoding analysis over saved feature and target bundles."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(slots=True)
class FeatureBundle:
    sample_ids: np.ndarray
    values: np.ndarray
    feature_names: np.ndarray
    groups: np.ndarray


@dataclass(slots=True)
class TargetBundle:
    sample_ids: np.ndarray
    values: np.ndarray
    target_names: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an MVP structure-comparison analysis with grouped ridge encoding models."
    )
    parser.add_argument("--features", required=True, help="Path to feature bundle .npz")
    parser.add_argument("--brain-targets", required=True, help="Path to brain target bundle .npz")
    parser.add_argument("--lm-targets", required=True, help="Path to LM target bundle .npz")
    parser.add_argument("--output-dir", required=True, help="Directory for analysis artifacts")
    parser.add_argument(
        "--alpha-grid",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0, 100.0, 1000.0],
        help="Candidate ridge penalties for grouped CV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = load_feature_bundle(args.features)
    brain = load_target_bundle(args.brain_targets)
    lm = load_target_bundle(args.lm_targets)

    aligned = align_bundles(features, brain, lm)

    brain_cv = run_grouped_ridge_cv(
        x=aligned["x"],
        y=aligned["brain_y"],
        groups=aligned["groups"],
        target_names=aligned["brain_target_names"],
        alpha_grid=args.alpha_grid,
        family_name="brain",
    )
    lm_cv = run_grouped_ridge_cv(
        x=aligned["x"],
        y=aligned["lm_y"],
        groups=aligned["groups"],
        target_names=aligned["lm_target_names"],
        alpha_grid=args.alpha_grid,
        family_name="lm",
    )

    brain_alpha = consensus_alpha(brain_cv["outer_folds"])
    lm_alpha = consensus_alpha(lm_cv["outer_folds"])

    brain_final = fit_final_model(
        x=aligned["x"],
        y=aligned["brain_y"],
        alpha=brain_alpha,
        family_name="brain",
    )
    lm_final = fit_final_model(
        x=aligned["x"],
        y=aligned["lm_y"],
        alpha=lm_alpha,
        family_name="lm",
    )

    weight_similarity = cosine_similarity_matrix(
        brain_final["weights"],
        lm_final["weights"],
    )
    sample_rsa = compare_sample_geometry(
        brain_final["predictions"],
        lm_final["predictions"],
    )
    feature_importance = compare_feature_importance(
        brain_final["weights"],
        lm_final["weights"],
        aligned["feature_names"],
    )

    write_cv_outputs(output_dir / "brain_cv_summary.json", brain_cv)
    write_cv_outputs(output_dir / "lm_cv_summary.json", lm_cv)

    np.savez_compressed(
        output_dir / "brain_final_model.npz",
        weights=brain_final["weights"],
        predictions=brain_final["predictions"],
        feature_names=aligned["feature_names"],
        target_names=aligned["brain_target_names"],
        alpha=np.asarray([brain_alpha], dtype=float),
    )
    np.savez_compressed(
        output_dir / "lm_final_model.npz",
        weights=lm_final["weights"],
        predictions=lm_final["predictions"],
        feature_names=aligned["feature_names"],
        target_names=aligned["lm_target_names"],
        alpha=np.asarray([lm_alpha], dtype=float),
    )
    np.savez_compressed(
        output_dir / "brain_lm_weight_similarity.npz",
        similarity=weight_similarity,
        brain_target_names=aligned["brain_target_names"],
        lm_target_names=aligned["lm_target_names"],
    )
    write_json(
        output_dir / "sample_rsa.json",
        sample_rsa,
    )
    write_json(
        output_dir / "feature_importance_summary.json",
        feature_importance,
    )

    summary = {
        "n_samples": int(aligned["x"].shape[0]),
        "n_features": int(aligned["x"].shape[1]),
        "n_groups": int(len(np.unique(aligned["groups"]))),
        "brain_targets": int(aligned["brain_y"].shape[1]),
        "lm_targets": int(aligned["lm_y"].shape[1]),
        "brain_consensus_alpha": float(brain_alpha),
        "lm_consensus_alpha": float(lm_alpha),
        "brain_mean_test_correlation": float(brain_cv["aggregate"]["mean_test_correlation"]),
        "lm_mean_test_correlation": float(lm_cv["aggregate"]["mean_test_correlation"]),
        "sample_rsa_correlation": float(sample_rsa["sample_rsa_correlation"]),
        "feature_importance_correlation": float(feature_importance["importance_correlation"]),
        "artifacts": {
            "brain_cv_summary": str(output_dir / "brain_cv_summary.json"),
            "lm_cv_summary": str(output_dir / "lm_cv_summary.json"),
            "brain_final_model": str(output_dir / "brain_final_model.npz"),
            "lm_final_model": str(output_dir / "lm_final_model.npz"),
            "brain_lm_weight_similarity": str(output_dir / "brain_lm_weight_similarity.npz"),
            "sample_rsa": str(output_dir / "sample_rsa.json"),
            "feature_importance_summary": str(output_dir / "feature_importance_summary.json"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def load_feature_bundle(path: str | Path) -> FeatureBundle:
    source = np.load(path, allow_pickle=False)
    return FeatureBundle(
        sample_ids=_string_array(source["sample_ids"]),
        values=_float_matrix(source["values"]),
        feature_names=_string_array(source["feature_names"]),
        groups=_string_array(source["groups"]),
    )


def load_target_bundle(path: str | Path) -> TargetBundle:
    source = np.load(path, allow_pickle=False)
    return TargetBundle(
        sample_ids=_string_array(source["sample_ids"]),
        values=_float_matrix(source["values"]),
        target_names=_string_array(source["target_names"]),
    )


def align_bundles(
    features: FeatureBundle,
    brain: TargetBundle,
    lm: TargetBundle,
) -> dict[str, np.ndarray]:
    _ensure_unique(features.sample_ids, "feature sample_ids")
    _ensure_unique(brain.sample_ids, "brain sample_ids")
    _ensure_unique(lm.sample_ids, "lm sample_ids")

    brain_lookup = {sample_id: index for index, sample_id in enumerate(brain.sample_ids.tolist())}
    lm_lookup = {sample_id: index for index, sample_id in enumerate(lm.sample_ids.tolist())}

    keep_feature_indices: list[int] = []
    keep_brain_indices: list[int] = []
    keep_lm_indices: list[int] = []
    for feature_index, sample_id in enumerate(features.sample_ids.tolist()):
        if sample_id not in brain_lookup or sample_id not in lm_lookup:
            continue
        keep_feature_indices.append(feature_index)
        keep_brain_indices.append(brain_lookup[sample_id])
        keep_lm_indices.append(lm_lookup[sample_id])

    if not keep_feature_indices:
        raise ValueError("No shared sample_ids were found across features, brain targets, and LM targets.")

    return {
        "sample_ids": features.sample_ids[keep_feature_indices],
        "groups": features.groups[keep_feature_indices],
        "x": features.values[keep_feature_indices],
        "feature_names": features.feature_names,
        "brain_y": brain.values[keep_brain_indices],
        "brain_target_names": brain.target_names,
        "lm_y": lm.values[keep_lm_indices],
        "lm_target_names": lm.target_names,
    }


def run_grouped_ridge_cv(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    target_names: np.ndarray,
    alpha_grid: Iterable[float],
    family_name: str,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError(f"{family_name}: need at least two groups for grouped evaluation.")

    alpha_values = [float(value) for value in alpha_grid]
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

        outer_folds.append(
            {
                "test_group": str(test_group),
                "n_train_samples": int(train_mask.sum()),
                "n_test_samples": int(test_mask.sum()),
                "selected_alpha": float(best_alpha),
                "mean_test_correlation": float(metrics["mean_correlation"]),
                "mean_test_r2": float(metrics["mean_r2"]),
                "top_targets_by_correlation": top_metric_rows(
                    target_names=target_names,
                    correlations=metrics["per_target_correlation"],
                    r2_scores=metrics["per_target_r2"],
                ),
            }
        )

    return {
        "family": family_name,
        "alpha_grid": alpha_values,
        "aggregate": {
            "mean_test_correlation": float(
                np.nanmean([fold["mean_test_correlation"] for fold in outer_folds])
            ),
            "mean_test_r2": float(np.nanmean([fold["mean_test_r2"] for fold in outer_folds])),
        },
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
        if not fold_scores:
            score = -np.inf
        else:
            score = float(np.nanmean(fold_scores))
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha


def fit_final_model(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    family_name: str,
) -> dict[str, np.ndarray]:
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{family_name}: X and Y have different sample counts.")
    return fit_ridge_with_standardization(
        x_train=x,
        y_train=y,
        x_eval=x,
        alpha=float(alpha),
    )


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
    return {
        "weights": weights,
        "predictions": predictions,
    }


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
    top_k: int = 10,
) -> list[dict[str, Any]]:
    rows = [
        {
            "target_name": str(target_names[index]),
            "correlation": float(correlations[index]),
            "r2": float(r2_scores[index]),
        }
        for index in range(target_names.shape[0])
    ]
    rows.sort(key=lambda row: (np.nan_to_num(row["correlation"], nan=-np.inf)), reverse=True)
    return rows[:top_k]


def consensus_alpha(outer_folds: list[dict[str, Any]]) -> float:
    counts: dict[float, int] = {}
    for fold in outer_folds:
        alpha = float(fold["selected_alpha"])
        counts[alpha] = counts.get(alpha, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def cosine_similarity_matrix(left_weights: np.ndarray, right_weights: np.ndarray) -> np.ndarray:
    left_norm = _normalize_columns(left_weights)
    right_norm = _normalize_columns(right_weights)
    return left_norm.T @ right_norm


def compare_sample_geometry(brain_predictions: np.ndarray, lm_predictions: np.ndarray) -> dict[str, Any]:
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
    ranking.sort(
        key=lambda row: row["brain_importance"] + row["lm_importance"],
        reverse=True,
    )
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


def write_cv_outputs(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")


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


def _normalize_columns(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def _ensure_unique(values: np.ndarray, label: str) -> None:
    unique_values, counts = np.unique(values, return_counts=True)
    if np.any(counts > 1):
        repeated = unique_values[counts > 1][:5].tolist()
        raise ValueError(f"{label} contains duplicates, including {repeated}")


def _string_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=str)


def _float_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {matrix.shape}")
    return matrix


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


if __name__ == "__main__":
    main()
