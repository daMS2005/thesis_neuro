#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a readable RSA figure from saved final-model predictions.")
    parser.add_argument("--brain-final-model", required=True, type=Path)
    parser.add_argument("--lm-final-model", required=True, type=Path)
    parser.add_argument("--sample-rsa-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--story-label", default="shapesphysical")
    parser.add_argument("--model-label", default="Gemma 2 2B")
    parser.add_argument("--tr-s", type=float, default=1.5)
    parser.add_argument("--stimulus-onset-s", type=float, default=4.5)
    return parser.parse_args()


def average_predictions_by_tr(predictions: np.ndarray, tr_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_trs = np.unique(tr_indices)
    averaged = [predictions[tr_indices == tr].mean(axis=0) for tr in unique_trs.tolist()]
    return np.vstack(averaged), unique_trs.astype(int)


def sample_similarity_matrix(values: np.ndarray) -> np.ndarray:
    row_norms = np.linalg.norm(values, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    normalized = values / row_norms
    return normalized @ normalized.T


def standardize_off_diagonal(matrix: np.ndarray) -> np.ndarray:
    result = matrix.astype(float).copy()
    mask = ~np.eye(result.shape[0], dtype=bool)
    off = result[mask]
    mean = float(off.mean())
    std = float(off.std(ddof=0)) or 1.0
    result[mask] = (off - mean) / std
    np.fill_diagonal(result, np.nan)
    return result


def build_segment_edges(tr_count: int, onset_tr: int) -> list[int]:
    onset_tr = max(0, min(onset_tr, tr_count - 1))
    narrative = tr_count - onset_tr
    return [0, onset_tr, onset_tr + narrative // 3, onset_tr + (2 * narrative) // 3, tr_count]


def split_range(start: int, end: int, n_bins: int) -> list[tuple[int, int]]:
    if end <= start:
        return []
    indices = np.array_split(np.arange(start, end), n_bins)
    bins = []
    for idx in indices:
        if idx.size:
            bins.append((int(idx[0]), int(idx[-1]) + 1))
    return bins


def build_story_bins(segment_edges: list[int]) -> tuple[list[tuple[int, int]], list[str], list[int]]:
    intro_bins = split_range(segment_edges[0], segment_edges[1], 1)
    early_bins = split_range(segment_edges[1], segment_edges[2], 7)
    middle_bins = split_range(segment_edges[2], segment_edges[3], 7)
    late_bins = split_range(segment_edges[3], segment_edges[4], 7)
    bins = intro_bins + early_bins + middle_bins + late_bins
    labels = (
        ["Intro"]
        + [f"B{i}" for i in range(1, len(early_bins) + 1)]
        + [f"M{i}" for i in range(1, len(middle_bins) + 1)]
        + [f"E{i}" for i in range(1, len(late_bins) + 1)]
    )
    boundaries = [len(intro_bins), len(intro_bins) + len(early_bins), len(intro_bins) + len(early_bins) + len(middle_bins)]
    return bins, labels, boundaries


def aggregate_similarity(matrix: np.ndarray, bins: list[tuple[int, int]]) -> np.ndarray:
    out = np.full((len(bins), len(bins)), np.nan, dtype=float)
    for i, (s0, s1) in enumerate(bins):
        for j, (t0, t1) in enumerate(bins):
            block = matrix[s0:s1, t0:t1]
            if i == j:
                mask = ~np.eye(block.shape[0], dtype=bool)
                vals = block[mask]
            else:
                vals = block.ravel()
            if vals.size:
                out[i, j] = float(np.nanmean(vals))
    return out


def upper_triangle_only(matrix: np.ndarray) -> np.ndarray:
    out = matrix.astype(float).copy()
    out[np.tril_indices_from(out, k=0)] = np.nan
    return out


def decorate_binned_axis(ax: plt.Axes, boundaries: list[int], n_bins: int) -> None:
    for boundary in boundaries:
        if 0 < boundary < n_bins:
            ax.axvline(boundary - 0.5, color="black", linewidth=1.0, alpha=0.35)
            ax.axhline(boundary - 0.5, color="black", linewidth=1.0, alpha=0.35)
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.set_yticks([0, 4, 8, 12, 16, 20])
    ax.set_xticklabels(["I", "B", "B", "M", "E", "E"], fontsize=9)
    ax.set_yticklabels(["I", "B", "B", "M", "E", "E"], fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    args = parse_args()

    brain_npz = np.load(args.brain_final_model, allow_pickle=True)
    lm_npz = np.load(args.lm_final_model, allow_pickle=True)
    rsa_summary = json.loads(args.sample_rsa_json.read_text())

    brain_by_tr, brain_unique_trs = average_predictions_by_tr(brain_npz["predictions"], brain_npz["tr_indices"])
    lm_predictions = lm_npz["predictions"]
    lm_trs = lm_npz["tr_indices"].astype(int)

    common_trs = np.intersect1d(brain_unique_trs, lm_trs)
    brain_lookup = {int(tr): idx for idx, tr in enumerate(brain_unique_trs.tolist())}
    lm_lookup = {int(tr): idx for idx, tr in enumerate(lm_trs.tolist())}
    brain_aligned = np.vstack([brain_by_tr[brain_lookup[int(tr)]] for tr in common_trs])
    lm_aligned = np.vstack([lm_predictions[lm_lookup[int(tr)]] for tr in common_trs])

    brain_similarity = sample_similarity_matrix(brain_aligned)
    lm_similarity = sample_similarity_matrix(lm_aligned)

    onset_tr = int(round(args.stimulus_onset_s / args.tr_s))
    segment_edges = build_segment_edges(len(common_trs), onset_tr)
    bins, _, boundaries = build_story_bins(segment_edges)

    brain_binned = aggregate_similarity(brain_similarity, bins)
    lm_binned = aggregate_similarity(lm_similarity, bins)

    brain_plot = upper_triangle_only(standardize_off_diagonal(brain_binned))
    lm_plot = upper_triangle_only(standardize_off_diagonal(lm_binned))

    segment_labels = ["Intro", "Begin", "Middle", "End"]
    segment_bins = list(zip(segment_edges[:-1], segment_edges[1:]))
    brain_segment = aggregate_similarity(brain_similarity, segment_bins)
    lm_segment = aggregate_similarity(lm_similarity, segment_bins)
    segment_diff = standardize_off_diagonal(brain_segment) - standardize_off_diagonal(lm_segment)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#efefef")

    panel_abs = float(np.nanpercentile(np.abs(np.r_[brain_plot[np.isfinite(brain_plot)], lm_plot[np.isfinite(lm_plot)]]), 97.5))
    panel_abs = max(panel_abs, 1.5)
    diff_abs = float(np.nanpercentile(np.abs(segment_diff[np.isfinite(segment_diff)]), 97.5))
    diff_abs = max(diff_abs, 1.5)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), facecolor="white")

    axes[0].imshow(brain_plot, cmap=cmap, vmin=-panel_abs, vmax=panel_abs, interpolation="nearest")
    axes[0].set_title("Brain similarity", fontsize=12)
    axes[0].set_xlabel("Story-time bin")
    axes[0].set_ylabel("Story-time bin")
    decorate_binned_axis(axes[0], boundaries, len(bins))

    im1 = axes[1].imshow(lm_plot, cmap=cmap, vmin=-panel_abs, vmax=panel_abs, interpolation="nearest")
    axes[1].set_title("LM similarity", fontsize=12)
    axes[1].set_xlabel("Story-time bin")
    axes[1].set_ylabel("Story-time bin")
    decorate_binned_axis(axes[1], boundaries, len(bins))

    im2 = axes[2].imshow(segment_diff, cmap=cmap, vmin=-diff_abs, vmax=diff_abs, interpolation="nearest")
    axes[2].set_title("Segment-level difference", fontsize=12)
    axes[2].set_xlabel("Story segment")
    axes[2].set_ylabel("Story segment")
    axes[2].set_xticks(range(4))
    axes[2].set_yticks(range(4))
    axes[2].set_xticklabels(segment_labels, rotation=25, ha="right", fontsize=9)
    axes[2].set_yticklabels(segment_labels, fontsize=9)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    cbar0 = fig.colorbar(im1, ax=axes[:2], fraction=0.03, pad=0.04)
    cbar0.set_label("Within-matrix z-score", fontsize=10)
    cbar1 = fig.colorbar(im2, ax=axes[2], fraction=0.05, pad=0.06)
    cbar1.set_label("Brain - LM", fontsize=10)

    fig.text(
        0.5,
        0.01,
        f"Upper triangles are shown with the diagonal masked. Bins group adjacent TRs within intro, beginning, middle, and end segments. Reported sample RSA uses the raw TR-level cosine similarities (r = {rsa_summary['sample_rsa_correlation']:.3f}).",
        ha="center",
        fontsize=9.2,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.90, bottom=0.18, wspace=0.42)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
