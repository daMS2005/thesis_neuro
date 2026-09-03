"""Parcel extraction, confound cleaning, and brain target bundles."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from structure_comparison.alignment import load_tr_bins, resolve_transcript_paths
from structure_comparison.artifacts import parse_predictor_sample_id, parse_stimulus_from_pooled_run_id
from structure_comparison.utils import _ensure_unique, _string_array, write_json


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
        # Batched fMRIPrep runs may be nested as <story>/<batch>/out/sub-*/func/... rather than a flat
        # derivatives tree, so fall back to a recursive search and ignore archived failed runs.
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
