"""Tests that the structure CLI parses arguments and dispatches to the right module functions."""

from __future__ import annotations

from pathlib import Path

import pytest

from structure_comparison import artifacts, brain, cli, workflow


def test_run_analysis_dispatches_with_parsed_arguments(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(workflow, "run_structure_comparison", fake_run)
    cli.main(
        [
            "run-analysis",
            "--feature-run-dir", "runs/example",
            "--stimulus-id", "story",
            "--brain-targets-npz", "brain.npz",
            "--output-dir", "out",
            "--alpha-grid", "1", "10",
            "--brain-lags", "0", "1",
            "--predictor-view", "average",
            "--predictor-top-k", "16",
        ]
    )
    assert captured["feature_run_dir"] == Path("runs/example")
    assert captured["stimulus_id"] == "story"
    assert captured["brain_targets_npz"] == Path("brain.npz")
    assert captured["alpha_grid"] == [1.0, 10.0]
    assert captured["brain_lags"] == [0, 1]
    assert captured["predictor_view"] == "average"
    assert captured["predictor_top_k"] == 16
    assert captured["lm_folds"] == 5
    assert captured["lm_targets_per_layer"] == 8
    assert '"status": "ok"' in capsys.readouterr().out


def test_build_tr_artifacts_forwards_feature_run_dir(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return {"summary": {"rows": 1}}

    monkeypatch.setattr(artifacts, "build_tr_feature_artifacts", fake_build)
    cli.main(["build-tr-artifacts", "--feature-run-dir", "runs/example", "--stimulus-id", "story"])
    assert captured["feature_run_dir"] == Path("runs/example")
    assert captured["stimulus_id"] == "story"
    assert captured["transcript_root"].name == "transcripts"


def test_clean_brain_targets_uses_documented_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_clean(**kwargs):
        captured.update(kwargs)
        return {"runs": 0}

    monkeypatch.setattr(brain, "build_clean_brain_targets_from_fmriprep", fake_clean)
    cli.main(["build-clean-brain-targets", "--stimulus-id", "story", "--output-path", "brain.npz"])
    assert captured["fd_threshold"] == 0.5
    assert captured["std_dvars_threshold"] == 1.5
    assert captured["high_pass_hz"] == 0.008
    assert captured["acompcor_count"] == 6
    assert captured["allow_partial_runs"] is False


def test_combine_brain_targets_rejects_malformed_bundle_spec() -> None:
    with pytest.raises(ValueError, match="STIMULUS=PATH"):
        cli.main(["combine-brain-targets", "--bundle", "no-equals-sign", "--output-path", "out.npz"])
