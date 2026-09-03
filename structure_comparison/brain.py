"""Brain-target construction and confound-cleaning interfaces."""

from structure_comparison._implementation import (
    BrainTargets,
    build_brain_design_matrix,
    build_brain_targets_from_dataset,
    build_clean_brain_targets_from_fmriprep,
    build_confounds_for_cleaning,
    build_pooled_brain_design_matrix,
    clean_parcel_matrix,
    combine_brain_target_bundles,
    compute_tsnr,
    find_fmriprep_run_artifacts,
    load_brain_targets,
    zscore_with_reference_mask,
)

__all__ = [
    "BrainTargets",
    "build_brain_design_matrix",
    "build_brain_targets_from_dataset",
    "build_clean_brain_targets_from_fmriprep",
    "build_confounds_for_cleaning",
    "build_pooled_brain_design_matrix",
    "clean_parcel_matrix",
    "combine_brain_target_bundles",
    "compute_tsnr",
    "find_fmriprep_run_artifacts",
    "load_brain_targets",
    "zscore_with_reference_mask",
]
