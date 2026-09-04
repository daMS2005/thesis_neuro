"""Pins the lag ordering shared by the brain design matrix and the lag-collapse used for weight comparison."""

from __future__ import annotations

import numpy as np
import pytest

from structure_comparison.brain import BrainTargets, build_brain_design_matrix
from structure_comparison.modeling import collapse_lagged_weights


def _targets(n: int) -> BrainTargets:
    return BrainTargets(
        values=np.arange(n, dtype=float).reshape(n, 1),
        sample_ids=np.asarray([f"s{i}" for i in range(n)], dtype=str),
        subject_ids=np.asarray(["sub1"] * n, dtype=str),
        run_ids=np.asarray(["run1"] * n, dtype=str),
        tr_indices=np.arange(n, dtype=int),
        target_names=np.asarray(["parcel0"], dtype=str),
    )


def test_design_matrix_is_feature_major_and_lag_minor() -> None:
    predictors = np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    design = build_brain_design_matrix(
        tr_predictors=predictors,
        tr_indices=np.arange(3, dtype=int),
        feature_names=np.asarray(["f0", "f1"], dtype=str),
        brain_targets=_targets(3),
        lags=[0, 1],
    )
    assert design["feature_names"].tolist() == ["f0@lag0", "f0@lag1", "f1@lag0", "f1@lag1"]
    # Row for TR 2: f0 at lag 0 and 1, then f1 at lag 0 and 1.
    assert design["x"][2].tolist() == [3.0, 2.0, 30.0, 20.0]
    # The first TR has no earlier sample, so lag 1 is zero-padded.
    assert design["x"][0].tolist() == [1.0, 0.0, 10.0, 0.0]


def test_collapse_lagged_weights_matches_design_ordering() -> None:
    # Two base features, two lags, one target: weights laid out exactly as the design matrix columns.
    weights = np.asarray([[1.0], [2.0], [10.0], [20.0]])
    collapsed = collapse_lagged_weights(weights, base_feature_count=2, lags=[0, 1])
    assert collapsed.shape == (2, 1)
    assert collapsed[:, 0].tolist() == [3.0, 30.0]


def test_negative_lags_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_brain_design_matrix(
            tr_predictors=np.zeros((3, 1)),
            tr_indices=np.arange(3, dtype=int),
            feature_names=np.asarray(["f0"], dtype=str),
            brain_targets=_targets(3),
            lags=[-1, 0],
        )
