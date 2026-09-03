"""Unit tests for transcript alignment, brain design matrices, and grouped ridge CV."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from structure_comparison.alignment import (
    assign_words_to_trs,
    build_token_to_word_index,
    group_tokens_into_words,
    group_tokens_with_model_tokenizer,
    load_word_rows,
    validate_tr_alignment,
)
from structure_comparison.artifacts import load_predictor_feature_keys
from structure_comparison.brain import (
    BrainTargets,
    build_brain_design_matrix,
    build_confounds_for_cleaning,
    load_brain_targets,
    zscore_with_reference_mask,
)
from structure_comparison.modeling import collapse_lagged_weights, contiguous_block_groups, run_grouped_ridge_cv


class _TokenizerStub:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
        del text, add_special_tokens
        return SimpleNamespace(tokens=self.tokens)


class WorkflowTests(unittest.TestCase):
    def test_group_tokens_into_words_handles_subwords(self) -> None:
        tokens = ["Let", "'", "s", "▁all", "▁go", "▁home", "."]
        groups = group_tokens_into_words(tokens)
        self.assertEqual([group.text for group in groups], ["Let's", "all", "go", "home."])
        token_to_word = build_token_to_word_index(groups)
        self.assertEqual(token_to_word[0], 0)
        self.assertEqual(token_to_word[2], 0)
        self.assertEqual(token_to_word[6], 3)

    def test_group_tokens_with_model_tokenizer_handles_gemma(self) -> None:
        model_id = "google/gemma-2-2b"
        transcript_text = "Alice was beginning to get very tired."
        tokens = ["Alice", "▁was", "▁beginning", "▁to", "▁get", "▁very", "▁tired", "."]
        tokenizer = _TokenizerStub(tokens)
        with patch("structure_comparison.alignment.load_cached_tokenizer", return_value=tokenizer):
            groups = group_tokens_with_model_tokenizer(model_id, tokens, transcript_text)
        self.assertEqual([group.text for group in groups], ["Alice", "was", "beginning", "to", "get", "very", "tired."])

    def test_group_tokens_with_model_tokenizer_handles_llama(self) -> None:
        model_id = "meta-llama/Llama-3.1-8B"
        transcript_text = "Alice was beginning to get very tired."
        tokens = ["Alice", "Ġwas", "Ġbeginning", "Ġto", "Ġget", "Ġvery", "Ġtired", "."]
        tokenizer = _TokenizerStub(tokens)
        with patch("structure_comparison.alignment.load_cached_tokenizer", return_value=tokenizer):
            groups = group_tokens_with_model_tokenizer(model_id, tokens, transcript_text)
        self.assertEqual([group.text for group in groups], ["Alice", "was", "beginning", "to", "get", "very", "tired."])

    def test_midpoint_word_to_tr_assignment_matches_bins(self) -> None:
        from structure_comparison.alignment import TrBin, WordTiming

        words = [
            WordTiming(word="Let's", start_s=0.0, end_s=1.2),
            WordTiming(word="all", start_s=1.2, end_s=1.36),
            WordTiming(word="go", start_s=1.36, end_s=1.68),
            WordTiming(word="home.", start_s=1.68, end_s=2.4),
        ]
        tr_bins = [
            TrBin(tr_index=0, start_s=0.0, end_s=1.5, text=""),
            TrBin(tr_index=1, start_s=1.5, end_s=3.0, text=""),
            TrBin(tr_index=2, start_s=3.0, end_s=4.5, text=""),
            TrBin(tr_index=3, start_s=4.5, end_s=6.0, text="Let's all"),
            TrBin(tr_index=4, start_s=6.0, end_s=7.5, text="go home."),
        ]
        assignments = assign_words_to_trs(words, tr_bins, stimulus_onset_s=4.5)
        self.assertEqual(assignments, [3, 3, 4, 4])
        validate_tr_alignment(words, assignments, tr_bins, stimulus_onset_s=4.5)

    def test_contiguous_block_groups_cover_all_samples(self) -> None:
        groups = contiguous_block_groups(sample_count=13, n_folds=5)
        self.assertEqual(groups.shape[0], 13)
        self.assertEqual(set(groups.tolist()), {0, 1, 2, 3, 4})
        blocks = [np.where(groups == fold)[0].tolist() for fold in range(5)]
        for block in blocks:
            self.assertEqual(block, list(range(block[0], block[-1] + 1)))

    def test_collapse_lagged_weights(self) -> None:
        weights = np.asarray(
            [
                [1.0, 2.0],
                [0.5, 0.5],
                [3.0, 4.0],
                [1.0, 1.0],
            ]
        )
        collapsed = collapse_lagged_weights(weights, base_feature_count=2, lags=[0, 1])
        expected = np.asarray([[1.5, 2.5], [4.0, 5.0]])
        np.testing.assert_allclose(collapsed, expected)

    def test_grouped_ridge_cv_runs_on_synthetic_data(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(20, 3))
        weights = np.asarray([[1.2, 0.2], [-0.4, 0.7], [0.5, -1.0]])
        y = x @ weights + rng.normal(scale=0.01, size=(20, 2))
        groups = np.asarray(["a"] * 5 + ["b"] * 5 + ["c"] * 5 + ["d"] * 5, dtype=str)
        sample_ids = np.asarray([f"s{i}" for i in range(20)], dtype=str)
        target_names = np.asarray(["t0", "t1"], dtype=str)
        result = run_grouped_ridge_cv(
            x=x,
            y=y,
            groups=groups,
            sample_ids=sample_ids,
            target_names=target_names,
            alpha_grid=[0.1, 1.0, 10.0],
            family_name="synthetic",
        )
        self.assertIn("aggregate", result)
        self.assertEqual(len(result["outer_folds"]), 4)

    def test_brain_design_matrix_uses_subject_run_groups(self) -> None:
        tr_predictors = np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 1.0], [4.0, 2.0]])
        feature_names = np.asarray(["f0", "f1"], dtype=str)
        brain_targets = BrainTargets(
            values=np.asarray([[1.0], [2.0], [3.0], [4.0]]),
            sample_ids=np.asarray(["a", "b", "c", "d"], dtype=str),
            subject_ids=np.asarray(["sub1", "sub1", "sub2", "sub2"], dtype=str),
            run_ids=np.asarray(["run1", "run1", "run1", "run1"], dtype=str),
            tr_indices=np.asarray([0, 1, 2, 3], dtype=int),
            target_names=np.asarray(["parcel0"], dtype=str),
        )
        design = build_brain_design_matrix(
            tr_predictors=tr_predictors,
            tr_indices=np.asarray([0, 1, 2, 3], dtype=int),
            feature_names=feature_names,
            brain_targets=brain_targets,
            lags=[0, 1],
        )
        self.assertEqual(design["x"].shape, (4, 4))
        self.assertEqual(design["groups"].tolist(), ["sub1:run1", "sub1:run1", "sub2:run1", "sub2:run1"])

    def test_brain_design_matrix_drops_censored_samples(self) -> None:
        tr_predictors = np.asarray([[1.0], [2.0], [3.0], [4.0]])
        brain_targets = BrainTargets(
            values=np.asarray([[10.0], [20.0], [30.0], [40.0]]),
            sample_ids=np.asarray(["a", "b", "c", "d"], dtype=str),
            subject_ids=np.asarray(["sub1", "sub1", "sub1", "sub1"], dtype=str),
            run_ids=np.asarray(["run1", "run1", "run1", "run1"], dtype=str),
            tr_indices=np.asarray([0, 1, 2, 3], dtype=int),
            target_names=np.asarray(["parcel0"], dtype=str),
            censor_mask=np.asarray([False, True, False, False]),
        )
        design = build_brain_design_matrix(
            tr_predictors=tr_predictors,
            tr_indices=np.asarray([0, 1, 2, 3], dtype=int),
            feature_names=np.asarray(["f0"], dtype=str),
            brain_targets=brain_targets,
            lags=[0],
        )
        self.assertEqual(design["x"].shape[0], 3)
        self.assertEqual(design["sample_ids"].tolist(), ["a", "c", "d"])
        self.assertEqual(design["censored_sample_count"], 1)

    def test_load_brain_targets_reads_optional_cleaning_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "brain_targets.npz"
            np.savez_compressed(
                path,
                values=np.asarray([[1.0], [2.0]], dtype=np.float32),
                sample_ids=np.asarray(["s0", "s1"], dtype=str),
                subject_ids=np.asarray(["sub1", "sub1"], dtype=str),
                run_ids=np.asarray(["run1", "run1"], dtype=str),
                tr_indices=np.asarray([0, 1], dtype=int),
                target_names=np.asarray(["parcel0"], dtype=str),
                censor_mask=np.asarray([False, True]),
                framewise_displacement=np.asarray([0.1, 0.7], dtype=np.float32),
                std_dvars=np.asarray([0.5, 1.8], dtype=np.float32),
            )
            targets = load_brain_targets(path)
        self.assertIsNotNone(targets.censor_mask)
        self.assertEqual(targets.censor_mask.tolist(), [False, True])
        np.testing.assert_allclose(targets.framewise_displacement, [0.1, 0.7], atol=1e-6)
        np.testing.assert_allclose(targets.std_dvars, [0.5, 1.8], atol=1e-6)

    def test_build_confounds_for_cleaning_selects_motion_and_acompcor(self) -> None:
        frame = pd.DataFrame(
            {
                "trans_x": [0.1, 0.2, 0.3],
                "trans_y": [0.0, 0.1, 0.2],
                "trans_z": [0.0, 0.0, 0.1],
                "rot_x": [0.0, 0.0, 0.0],
                "rot_y": [0.1, 0.2, 0.1],
                "rot_z": [0.1, 0.1, 0.1],
                "trans_x_derivative1": [np.nan, 0.1, 0.1],
                "trans_y_derivative1": [np.nan, 0.1, 0.1],
                "trans_z_derivative1": [np.nan, 0.0, 0.1],
                "rot_x_derivative1": [np.nan, 0.0, 0.0],
                "rot_y_derivative1": [np.nan, 0.1, -0.1],
                "rot_z_derivative1": [np.nan, 0.0, 0.0],
                "trans_x_power2": [0.01, 0.04, 0.09],
                "trans_y_power2": [0.0, 0.01, 0.04],
                "trans_z_power2": [0.0, 0.0, 0.01],
                "rot_x_power2": [0.0, 0.0, 0.0],
                "rot_y_power2": [0.01, 0.04, 0.01],
                "rot_z_power2": [0.01, 0.01, 0.01],
                "trans_x_derivative1_power2": [0.0, 0.01, 0.01],
                "trans_y_derivative1_power2": [0.0, 0.01, 0.01],
                "trans_z_derivative1_power2": [0.0, 0.0, 0.01],
                "rot_x_derivative1_power2": [0.0, 0.0, 0.0],
                "rot_y_derivative1_power2": [0.0, 0.01, 0.01],
                "rot_z_derivative1_power2": [0.0, 0.0, 0.0],
                "a_comp_cor_00": [0.0, 0.1, 0.2],
                "a_comp_cor_01": [0.0, 0.0, 0.1],
                "framewise_displacement": [0.1, 0.6, 0.2],
                "std_dvars": [0.2, 0.4, 1.7],
            }
        )
        confounds, columns, fd_values, dvars_values, censor_mask = build_confounds_for_cleaning(
            confounds_frame=frame,
            fd_threshold=0.5,
            std_dvars_threshold=1.5,
            acompcor_count=2,
        )
        self.assertEqual(confounds.shape, (3, 26))
        self.assertEqual(len(columns), 26)
        np.testing.assert_allclose(fd_values, [0.1, 0.6, 0.2], atol=1e-6)
        np.testing.assert_allclose(dvars_values, [0.2, 0.4, 1.7], atol=1e-6)
        self.assertEqual(censor_mask.tolist(), [False, True, True])

    def test_zscore_with_reference_mask_uses_uncensored_rows(self) -> None:
        values = np.asarray(
            [
                [1.0, 10.0],
                [3.0, 12.0],
                [100.0, 1000.0],
            ],
            dtype=float,
        )
        standardized = zscore_with_reference_mask(values, np.asarray([True, True, False]))
        uncensored = standardized[:2]
        np.testing.assert_allclose(uncensored.mean(axis=0), [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(uncensored.std(axis=0, ddof=0), [1.0, 1.0], atol=1e-6)

    def test_load_word_rows_skips_punctuation_only_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "words.tsv"
            path.write_text(
                "word\tstart_s\tend_s\n"
                "hello\t0.0\t0.5\n"
                ".\t0.5\t0.6\n"
                "world\t0.6\t1.0\n"
                "...\t1.0\t1.2\n",
                encoding="utf-8",
            )
            rows = load_word_rows(path)
        self.assertEqual([row.word for row in rows], ["hello", "world"])

    def test_load_predictor_feature_keys_respects_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "selected_features_for_alignment.jsonl"
            rows = [
                {"layer": 8, "feature_id": 20, "transcript_relevance_rank": 3},
                {"layer": 4, "feature_id": 10, "transcript_relevance_rank": 1},
                {"layer": 13, "feature_id": 30, "transcript_relevance_rank": 2},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            keys = load_predictor_feature_keys(path, top_k=2)
        self.assertEqual(keys, ((4, 10), (13, 30)))


if __name__ == "__main__":
    unittest.main()
