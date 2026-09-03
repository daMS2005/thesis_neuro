"""Ridge fitting, validation, and representational comparison interfaces."""

from structure_comparison._implementation import (
    average_predictions_by_tr,
    collapse_lagged_weights,
    compare_feature_importance,
    compare_sample_geometry,
    consensus_alpha,
    contiguous_block_groups,
    cosine_similarity_matrix,
    fit_final_model,
    fit_ridge_with_standardization,
    regression_metrics,
    run_family_analysis,
    run_grouped_ridge_cv,
    sample_similarity_matrix,
    select_alpha_with_inner_cv,
)

__all__ = [
    "average_predictions_by_tr",
    "collapse_lagged_weights",
    "compare_feature_importance",
    "compare_sample_geometry",
    "consensus_alpha",
    "contiguous_block_groups",
    "cosine_similarity_matrix",
    "fit_final_model",
    "fit_ridge_with_standardization",
    "regression_metrics",
    "run_family_analysis",
    "run_grouped_ridge_cv",
    "sample_similarity_matrix",
    "select_alpha_with_inner_cv",
]
