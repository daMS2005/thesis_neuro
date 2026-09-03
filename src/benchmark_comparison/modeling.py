"""Ridge fitting of benchmark targets and comparison against registered brain models."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from benchmark_comparison.items import read_jsonl
from benchmark_comparison.registry import ModelRegistryEntry, load_registry
from structure_comparison.modeling import (
    compare_feature_importance,
    consensus_alpha,
    cosine_similarity_matrix,
    fit_final_model,
    run_grouped_ridge_cv,
)

DEFAULT_TARGET_COLUMNS = ("correct", "gold_choice_avg_logprob", "margin")


def fit_benchmark_model(
    model_entry: ModelRegistryEntry,
    feature_npz: str | Path,
    score_jsonl: str | Path,
    output_dir: str | Path,
    target_columns: Iterable[str] = DEFAULT_TARGET_COLUMNS,
    alpha_grid: Iterable[float] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    feature_bundle = np.load(feature_npz, allow_pickle=False)
    score_rows = read_jsonl(score_jsonl)
    target_columns = tuple(target_columns)

    x, sample_ids, feature_names, task_names = _load_feature_matrix(feature_bundle)
    keep_indices, y, target_names, aligned_score_rows = _build_target_matrix(sample_ids, score_rows, target_columns)
    x = x[keep_indices]
    sample_ids = sample_ids[keep_indices]
    feature_names = feature_names
    task_names = task_names[keep_indices]
    groups = _build_task_balanced_groups(task_names=np.asarray([row["task"] for row in aligned_score_rows], dtype=str), folds=folds, seed=seed)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    benchmark_cv = run_grouped_ridge_cv(
        x=x,
        y=y,
        groups=groups,
        sample_ids=sample_ids,
        target_names=target_names,
        alpha_grid=list(float(value) for value in alpha_grid),
        family_name=f"{model_entry.name}:benchmark",
    )
    benchmark_alpha = consensus_alpha(benchmark_cv["outer_folds"])
    benchmark_final = fit_final_model(x=x, y=y, alpha=benchmark_alpha, family_name=f"{model_entry.name}:benchmark")

    brain_weights, brain_feature_names, brain_target_names = _load_brain_weights(model_entry.brain_final_model_path)
    aligned_brain_weights, aligned_benchmark_weights, aligned_feature_names = _align_feature_weights(
        left_weights=brain_weights,
        left_feature_names=brain_feature_names,
        right_weights=benchmark_final["weights"],
        right_feature_names=feature_names,
    )

    weight_similarity = cosine_similarity_matrix(aligned_brain_weights, aligned_benchmark_weights)
    feature_importance = compare_feature_importance(
        aligned_brain_weights,
        aligned_benchmark_weights,
        aligned_feature_names,
    )
    brain_metrics = _load_registered_brain_metrics(model_entry.analysis_summary_path)
    behavioral_summary = _summarize_behavior(aligned_score_rows)

    np.savez_compressed(
        output_root / "benchmark_target_bundle.npz",
        sample_ids=sample_ids,
        values=y,
        target_names=target_names,
        task_names=np.asarray([row["task"] for row in aligned_score_rows], dtype=str),
    )
    np.savez_compressed(
        output_root / "benchmark_final_model.npz",
        weights=benchmark_final["weights"],
        predictions=benchmark_final["predictions"],
        sample_ids=sample_ids,
        feature_names=feature_names,
        target_names=target_names,
        alpha=np.asarray([benchmark_alpha], dtype=float),
    )
    np.savez_compressed(
        output_root / "brain_benchmark_weight_similarity.npz",
        similarity=weight_similarity,
        brain_target_names=brain_target_names,
        benchmark_target_names=target_names,
        feature_names=aligned_feature_names,
    )
    _write_json(output_root / "benchmark_cv_summary.json", benchmark_cv)
    _write_json(output_root / "brain_benchmark_feature_importance.json", feature_importance)

    summary = {
        "model_name": model_entry.name,
        "model_id": model_entry.model_id,
        "feature_npz": str(Path(feature_npz)),
        "score_jsonl": str(Path(score_jsonl)),
        "item_count": int(x.shape[0]),
        "feature_count": int(x.shape[1]),
        "target_count": int(y.shape[1]),
        "target_columns": list(target_columns),
        "folds": int(folds),
        "benchmark_consensus_alpha": float(benchmark_alpha),
        "benchmark_mean_test_correlation": float(benchmark_cv["aggregate"]["mean_test_correlation"]),
        "benchmark_mean_test_r2": float(benchmark_cv["aggregate"]["mean_test_r2"]),
        "brain_benchmark_feature_importance_correlation": float(feature_importance["importance_correlation"]),
        "registered_brain_mean_test_correlation": float(brain_metrics["brain_mean_test_correlation"]),
        "registered_brain_mean_test_r2": float(brain_metrics["brain_mean_test_r2"]),
        "registered_brain_feature_importance_correlation": float(brain_metrics["feature_importance_correlation"]),
        "behavioral_summary": behavioral_summary,
        "top_shared_predictors": feature_importance["top_features"][:10],
        "artifacts": {
            "benchmark_target_bundle": str(output_root / "benchmark_target_bundle.npz"),
            "benchmark_cv_summary": str(output_root / "benchmark_cv_summary.json"),
            "benchmark_final_model": str(output_root / "benchmark_final_model.npz"),
            "brain_benchmark_weight_similarity": str(output_root / "brain_benchmark_weight_similarity.npz"),
            "brain_benchmark_feature_importance": str(output_root / "brain_benchmark_feature_importance.json"),
        },
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def summarize_model_runs(run_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    root = Path(run_root)
    summary_paths = sorted(root.rglob("summary.json"))
    rows: list[dict[str, Any]] = []
    for path in summary_paths:
        if path.name != "summary.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "brain_benchmark_feature_importance_correlation" not in payload:
            continue
        row = {
            "model_name": payload["model_name"],
            "model_id": payload["model_id"],
            "item_count": payload["item_count"],
            "feature_count": payload["feature_count"],
            "target_count": payload["target_count"],
            "brain_mean_test_correlation": payload["registered_brain_mean_test_correlation"],
            "brain_mean_test_r2": payload["registered_brain_mean_test_r2"],
            "benchmark_accuracy": payload["behavioral_summary"].get("overall_accuracy"),
            "benchmark_mean_test_correlation": payload["benchmark_mean_test_correlation"],
            "benchmark_mean_test_r2": payload["benchmark_mean_test_r2"],
            "brain_benchmark_feature_importance_correlation": payload["brain_benchmark_feature_importance_correlation"],
            "registered_brain_mean_test_correlation": payload["registered_brain_mean_test_correlation"],
            "registered_brain_mean_test_r2": payload["registered_brain_mean_test_r2"],
            "observed_choice_accuracy": payload["behavioral_summary"].get("overall_accuracy"),
            "path": str(path),
        }
        rows.append(row)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["path"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    correlation_summary = {
        "run_root": str(root),
        "rows_written": len(rows),
        "brain_vs_benchmark_accuracy_pearson_r": _safe_pearson_from_rows(
            rows=rows,
            x_key="brain_mean_test_correlation",
            y_key="benchmark_accuracy",
        ),
        "brain_r2_vs_benchmark_accuracy_pearson_r": _safe_pearson_from_rows(
            rows=rows,
            x_key="brain_mean_test_r2",
            y_key="benchmark_accuracy",
        ),
        "brain_vs_benchmark_fit_pearson_r": _safe_pearson_from_rows(
            rows=rows,
            x_key="brain_mean_test_correlation",
            y_key="benchmark_mean_test_correlation",
        ),
        "rows": rows,
    }
    _write_json(destination.with_suffix(".json"), correlation_summary)

    return {
        "run_root": str(root),
        "output_path": str(destination),
        "rows_written": len(rows),
    }


def registry_summary_rows(registry_path: str | Path | None = None) -> list[dict[str, Any]]:
    registry = load_registry(registry_path)
    rows: list[dict[str, Any]] = []
    for model_name, entry in sorted(registry.items()):
        brain_metrics = _load_registered_brain_metrics(entry.analysis_summary_path)
        rows.append(
            {
                "model_name": model_name,
                "model_id": entry.model_id,
                "brain_mean_test_correlation": brain_metrics["brain_mean_test_correlation"],
                "brain_mean_test_r2": brain_metrics["brain_mean_test_r2"],
                "feature_importance_correlation": brain_metrics["feature_importance_correlation"],
                "analysis_summary_path": str(entry.analysis_summary_path),
            }
        )
    return rows


def _load_feature_matrix(source: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(source["values"], dtype=float),
        np.asarray(source["sample_ids"], dtype=str),
        np.asarray(source["feature_names"], dtype=str),
        np.asarray(source["task_names"], dtype=str),
    )


def _build_target_matrix(
    feature_sample_ids: np.ndarray,
    score_rows: list[dict[str, Any]],
    target_columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    score_lookup = {str(row["item_id"]): row for row in score_rows}
    keep_indices: list[int] = []
    target_rows: list[list[float]] = []
    aligned_rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(feature_sample_ids.tolist()):
        row = score_lookup.get(sample_id)
        if row is None:
            continue
        values: list[float] = []
        missing_value = False
        for column in target_columns:
            value = row.get(column)
            if value is None:
                missing_value = True
                break
            values.append(float(value))
        if missing_value:
            continue
        keep_indices.append(index)
        target_rows.append(values)
        aligned_rows.append(row)
    if not keep_indices:
        columns = ", ".join(target_columns)
        raise ValueError(f"No benchmark rows had all requested target columns: {columns}")
    return (
        np.asarray(keep_indices, dtype=int),
        np.asarray(target_rows, dtype=float),
        np.asarray(list(target_columns), dtype=str),
        aligned_rows,
    )


def _build_task_balanced_groups(task_names: np.ndarray, folds: int, seed: int) -> np.ndarray:
    if folds <= 1:
        raise ValueError("folds must be at least 2.")
    rng = np.random.default_rng(seed)
    groups = np.empty(task_names.shape[0], dtype=object)
    for task in np.unique(task_names):
        indices = np.where(task_names == task)[0]
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        for offset, index in enumerate(shuffled.tolist()):
            groups[index] = f"fold_{offset % folds}"
    return np.asarray(groups, dtype=str)


def _load_brain_weights(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.load(path, allow_pickle=False)
    if "collapsed_weights" in source.files:
        return (
            np.asarray(source["collapsed_weights"], dtype=float),
            np.asarray(source["base_feature_names"], dtype=str),
            np.asarray(source["target_names"], dtype=str),
        )
    return (
        np.asarray(source["weights"], dtype=float),
        np.asarray(source["feature_names"], dtype=str),
        np.asarray(source["target_names"], dtype=str),
    )


def _align_feature_weights(
    left_weights: np.ndarray,
    left_feature_names: np.ndarray,
    right_weights: np.ndarray,
    right_feature_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_lookup = {name: index for index, name in enumerate(left_feature_names.tolist())}
    right_lookup = {name: index for index, name in enumerate(right_feature_names.tolist())}
    shared_names = [name for name in left_feature_names.tolist() if name in right_lookup]
    if not shared_names:
        raise ValueError("No shared feature names were found between brain and benchmark weights.")
    left_indices = np.asarray([left_lookup[name] for name in shared_names], dtype=int)
    right_indices = np.asarray([right_lookup[name] for name in shared_names], dtype=int)
    return (
        left_weights[left_indices],
        right_weights[right_indices],
        np.asarray(shared_names, dtype=str),
    )


def _load_registered_brain_metrics(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    family = next((row for row in payload["families"] if row["family_name"] == "all_layers"), None)
    if family is None:
        raise ValueError(f"No all_layers family found in {path}")
    return {
        "brain_mean_test_correlation": float(family["brain_mean_test_correlation"]),
        "brain_mean_test_r2": float(family["brain_mean_test_r2"]),
        "feature_importance_correlation": float(family["feature_importance_correlation"]),
        "top_parcels": [row["target_name"] for row in family.get("top_parcels", [])],
    }


def _summarize_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accuracy_values = [float(row["correct"]) for row in rows if row.get("correct") is not None]
    margin_values = [float(row["margin"]) for row in rows if row.get("margin") is not None]
    gold_avg_values = [float(row["gold_choice_avg_logprob"]) for row in rows if row.get("gold_choice_avg_logprob") is not None]
    per_task: dict[str, dict[str, float]] = {}
    for task in sorted({str(row["task"]) for row in rows}):
        task_rows = [row for row in rows if str(row["task"]) == task]
        task_accuracy = [float(row["correct"]) for row in task_rows if row.get("correct") is not None]
        task_margin = [float(row["margin"]) for row in task_rows if row.get("margin") is not None]
        per_task[task] = {
            "item_count": int(len(task_rows)),
            "accuracy": float(np.mean(task_accuracy)) if task_accuracy else float("nan"),
            "mean_margin": float(np.mean(task_margin)) if task_margin else float("nan"),
        }
    return {
        "overall_accuracy": float(np.mean(accuracy_values)) if accuracy_values else None,
        "mean_margin": float(np.mean(margin_values)) if margin_values else None,
        "mean_gold_choice_avg_logprob": float(np.mean(gold_avg_values)) if gold_avg_values else None,
        "per_task": per_task,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_pearson_from_rows(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float | None:
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        x_raw = row.get(x_key)
        y_raw = row.get(y_key)
        if x_raw is None or y_raw is None:
            continue
        x_value = float(x_raw)
        y_value = float(y_raw)
        if np.isnan(x_value) or np.isnan(y_value):
            continue
        x_values.append(x_value)
        y_values.append(y_value)
    if len(x_values) < 2:
        return None
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])
