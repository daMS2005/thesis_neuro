#!/usr/bin/env python3
"""Compare story-level feature activation summaries with regressor importance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def load_feature_rows(feature_csv: Path) -> list[dict[str, str]]:
    with feature_csv.open() as handle:
        return list(csv.DictReader(handle))


def activation_metrics(run_dir: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    average = np.load(run_dir / "intermediates" / "tr_feature_average.npz", allow_pickle=True)
    peak = np.load(run_dir / "intermediates" / "tr_feature_peak.npz", allow_pickle=True)
    mass = np.load(run_dir / "intermediates" / "tr_feature_mass.npz", allow_pickle=True)
    presence = np.load(run_dir / "intermediates" / "tr_feature_presence.npz", allow_pickle=True)

    feature_names = [str(name) for name in average["feature_names"]]
    metrics = {
        "mean_average": average["values"].mean(axis=0),
        "total_mass": mass["values"].sum(axis=0),
        "tr_presence": presence["values"].sum(axis=0),
        "peak_activation": peak["values"].max(axis=0),
    }
    return feature_names, metrics


def rank_from_vector(vec: np.ndarray, idx: int) -> int:
    return int(np.where(np.argsort(-vec) == idx)[0][0]) + 1


def analyze_story(run_dir: Path, feature_csv: Path) -> dict[str, object]:
    rows = load_feature_rows(feature_csv)
    feature_names, metrics = activation_metrics(run_dir)
    joint = np.array([float(row["joint_importance"]) for row in rows], dtype=float)
    brain = np.array([float(row["brain_importance"]) for row in rows], dtype=float)
    lm = np.array([float(row["lm_importance"]) for row in rows], dtype=float)

    metric_summary: dict[str, dict[str, float]] = {}
    for metric_name, values in metrics.items():
        metric_summary[metric_name] = {
            "joint_spearman": spearman(values, joint),
            "joint_pearson": float(np.corrcoef(values, joint)[0, 1]),
            "brain_spearman": spearman(values, brain),
            "brain_pearson": float(np.corrcoef(values, brain)[0, 1]),
            "lm_spearman": spearman(values, lm),
            "lm_pearson": float(np.corrcoef(values, lm)[0, 1]),
        }

    joint_order = sorted(rows, key=lambda row: -float(row["joint_importance"]))
    top_joint = [row["feature_name"] for row in joint_order[:10]]

    top_joint_ranks = []
    for row in joint_order[:5]:
        idx = feature_names.index(row["feature_name"])
        top_joint_ranks.append(
            {
                "feature_name": row["feature_name"],
                "joint_rank": int(joint_order.index(row)) + 1,
                "activation_ranks": {
                    metric_name: rank_from_vector(values, idx)
                    for metric_name, values in metrics.items()
                },
            }
        )

    activation_heads = {}
    for metric_name, values in metrics.items():
        order = np.argsort(-values)
        activation_heads[metric_name] = [feature_names[i] for i in order[:10]]

    overlaps = {
        metric_name: sorted(set(top_joint) & set(head))
        for metric_name, head in activation_heads.items()
    }

    return {
        "stimulus_id": rows[0]["stimulus_id"],
        "metric_summary": metric_summary,
        "top_joint_features": top_joint,
        "top_joint_activation_ranks": top_joint_ranks,
        "activation_heads": activation_heads,
        "top10_overlaps": overlaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    result = analyze_story(args.run_dir, args.feature_csv)
    args.output_json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
