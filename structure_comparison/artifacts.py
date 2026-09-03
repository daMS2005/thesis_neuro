"""Feature and target artifact construction interfaces."""

from structure_comparison._implementation import (
    FamilyMatrices,
    FeatureArtifacts,
    build_family_matrices,
    build_pooled_tr_feature_artifacts,
    build_tr_feature_artifacts,
    concatenate_feature_artifacts,
    counts_by_layer,
    load_predictor_feature_keys,
    save_feature_view_npz,
    save_target_view_npz,
    write_feature_artifact_bundle,
)

__all__ = [
    "FamilyMatrices",
    "FeatureArtifacts",
    "build_family_matrices",
    "build_pooled_tr_feature_artifacts",
    "build_tr_feature_artifacts",
    "concatenate_feature_artifacts",
    "counts_by_layer",
    "load_predictor_feature_keys",
    "save_feature_view_npz",
    "save_target_view_npz",
    "write_feature_artifact_bundle",
]
