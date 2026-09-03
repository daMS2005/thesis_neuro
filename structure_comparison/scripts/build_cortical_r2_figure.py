#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, plotting, surface
from PIL import Image, ImageChops, ImageOps

from thesis_neuro.paths import data_root

DEFAULT_ATLAS = data_root() / "atlases" / "schaefer200.nii.gz"
DEFAULT_LABELS = data_root() / "atlases" / "schaefer200_labels.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cortical Schaefer parcel R^2 map from a brain CV summary.")
    parser.add_argument("--brain-cv-summary", required=True, type=Path)
    parser.add_argument("--brain-targets-npz", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="")
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS)
    return parser.parse_args()


def load_parcel_r2(brain_cv_summary_path: Path) -> np.ndarray:
    summary = json.loads(brain_cv_summary_path.read_text())
    r2 = np.asarray(summary["aggregate"]["per_target_mean_r2"], dtype=float)
    if r2.shape[0] != 200:
        raise ValueError(f"Expected 200 parcel R^2 values, found {r2.shape[0]}")
    return r2


def build_value_volume(
    atlas_path: Path,
    labels_csv_path: Path,
    target_names: np.ndarray,
    parcel_values: np.ndarray,
) -> nib.Nifti1Image:
    atlas_img = nib.load(str(atlas_path))
    atlas_data = np.asanyarray(atlas_img.dataobj).astype(int)
    labels_df = pd.read_csv(labels_csv_path)
    label_id_by_name = dict(zip(labels_df["ROI Name"], labels_df["ROI Label"]))

    value_data = np.zeros(atlas_data.shape, dtype=np.float32)
    for target_name, value in zip(target_names, parcel_values):
        label_id = label_id_by_name.get(str(target_name))
        if label_id is None:
            continue
        value_data[atlas_data == int(label_id)] = max(float(value), 0.0)

    return nib.Nifti1Image(value_data, atlas_img.affine, atlas_img.header)


def trim_panel(image: Image.Image, border: int = 12) -> Image.Image:
    bg = Image.new(image.mode, image.size, "white")
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return image
    trimmed = image.crop(bbox)
    return ImageOps.expand(trimmed, border=border, fill="white")


def render_panel(
    surf_mesh: str,
    texture: np.ndarray,
    bg_map: str,
    hemi: str,
    view: str,
    panel_title: str,
    vmax: float,
    add_colorbar: bool,
) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    fig = plt.figure(figsize=(4.2, 3.6), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    plotting.plot_surf_stat_map(
        surf_mesh=surf_mesh,
        stat_map=texture,
        bg_map=bg_map,
        hemi=hemi,
        view=view,
        axes=ax,
        figure=fig,
        colorbar=add_colorbar,
        cmap="YlOrRd",
        threshold=1e-6,
        vmin=0.0,
        vmax=vmax,
        symmetric_cbar=False,
    )
    ax.set_title(panel_title, fontsize=12, pad=2)
    fig.savefig(tmp_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    image = Image.open(tmp_path).convert("RGB")
    tmp_path.unlink(missing_ok=True)
    return trim_panel(image)


def compose_grid(panels: list[Image.Image], output_path: Path, title: str) -> None:
    gap_x = 24
    gap_y = 30
    margin = 28

    panel_w = max(panel.width for panel in panels)
    panel_h = max(panel.height for panel in panels)
    title_h = 38 if title else 0
    caption_h = 32

    canvas_w = margin * 2 + panel_w * 2 + gap_x
    canvas_h = margin * 2 + title_h + panel_h * 2 + gap_y + caption_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    positions = [
        (margin, margin + title_h),
        (margin + panel_w + gap_x, margin + title_h),
        (margin, margin + title_h + panel_h + gap_y),
        (margin + panel_w + gap_x, margin + title_h + panel_h + gap_y),
    ]

    for panel, (x, y) in zip(panels, positions):
        x_off = x + (panel_w - panel.width) // 2
        y_off = y + (panel_h - panel.height) // 2
        canvas.paste(panel, (x_off, y_off))

    fig = plt.figure(figsize=(canvas_w / 180, canvas_h / 180), dpi=180, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas)
    ax.axis("off")

    if title:
        fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=14)
    fig.text(0.5, 0.02, "Held-out parcelwise $R^2$ projected from the Schaefer-200 atlas", ha="center", fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_surface_grid(value_img: nib.Nifti1Image, output_path: Path, title: str) -> None:
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    tex_left = surface.vol_to_surf(value_img, fsaverage.pial_left)
    tex_right = surface.vol_to_surf(value_img, fsaverage.pial_right)

    vmax = float(np.nanpercentile(np.r_[tex_left, tex_right], 99))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.r_[tex_left, tex_right]))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 0.1

    panel_specs = [
        (fsaverage.infl_left, tex_left, fsaverage.sulc_left, "left", "lateral", "Left lateral", False),
        (fsaverage.infl_left, tex_left, fsaverage.sulc_left, "left", "medial", "Left medial", False),
        (fsaverage.infl_right, tex_right, fsaverage.sulc_right, "right", "lateral", "Right lateral", False),
        (fsaverage.infl_right, tex_right, fsaverage.sulc_right, "right", "medial", "Right medial", True),
    ]

    panels = []
    for mesh, texture, bg_map, hemi, view, panel_title, add_colorbar in panel_specs:
        panels.append(
            render_panel(
                surf_mesh=mesh,
                texture=texture,
                bg_map=bg_map,
                hemi=hemi,
                view=view,
                panel_title=panel_title,
                vmax=vmax,
                add_colorbar=add_colorbar,
            )
        )

    compose_grid(panels, output_path, title)


def main() -> None:
    args = parse_args()
    parcel_r2 = load_parcel_r2(args.brain_cv_summary)
    target_names = np.load(args.brain_targets_npz, allow_pickle=True)["target_names"]
    value_img = build_value_volume(args.atlas, args.labels_csv, target_names, parcel_r2)
    plot_surface_grid(value_img, args.output, args.title)


if __name__ == "__main__":
    main()
