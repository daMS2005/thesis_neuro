"""Ridge fitting, grouped cross-validation, and brain/LM structure comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from structure_comparison.artifacts import FamilyMatrices
from structure_comparison.brain import BrainTargets, build_brain_design_matrix
from structure_comparison.utils import _normalize_columns, write_json


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


def safe_nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    mask = ~np.isnan(values)
    totals = np.where(mask, values, 0.0).sum(axis=axis)
    counts = mask.sum(axis=axis)
    result = np.full(totals.shape, np.nan, dtype=float)
    valid = counts > 0
    result[valid] = totals[valid] / counts[valid]
    return result


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
