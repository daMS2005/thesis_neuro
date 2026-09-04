"""Tests for the repository hygiene gate that keeps data, secrets, and machine paths out of Git."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from thesis_neuro.paths import repository_root


def _load_hygiene_module():
    path = repository_root() / "scripts" / "quality" / "check_repo_hygiene.py"
    spec = importlib.util.spec_from_file_location("check_repo_hygiene", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_text_scan_flags_secrets_and_machine_paths() -> None:
    hygiene = _load_hygiene_module()
    token = "hf_" + "A" * 24
    assert hygiene._inspect_text(f"token={token}") == ["credential-like token"]
    # Built at runtime so the hygiene scan of this test file does not trip on the fixtures.
    home_path = "/" + "Users" + "/someone/data"
    private_ip = "10." + "0.0.12"
    assert hygiene._inspect_text(f"root = '{home_path}'") == ["machine-specific path or address"]
    assert hygiene._inspect_text(f"host = {private_ip}") == ["machine-specific path or address"]
    assert hygiene._inspect_text("nothing to see here") == []


def test_notebook_scan_flags_outputs_and_execution_counts(tmp_path: Path) -> None:
    hygiene = _load_hygiene_module()
    notebook = {
        "cells": [
            {"cell_type": "code", "execution_count": 3, "outputs": [{"text": "x"}], "source": ["print(1)\n"]},
            {"cell_type": "markdown", "source": ["# clean\n"]},
        ]
    }
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(notebook))
    issues = hygiene._inspect_notebook(path)
    assert "cell 0 has an execution count" in issues
    assert "cell 0 has stored outputs" in issues
    assert len(issues) == 2


def test_tracked_data_and_binary_paths_are_rejected(tmp_path: Path, monkeypatch) -> None:
    hygiene = _load_hygiene_module()
    monkeypatch.setattr(hygiene, "ROOT", tmp_path)
    data_file = tmp_path / "data" / "subject.tsv"
    data_file.parent.mkdir()
    data_file.write_text("a\tb\n")
    bundle = tmp_path / "bundle.npz"
    bundle.write_bytes(b"\x00")
    clean = tmp_path / "module.py"
    clean.write_text("x = 1\n")
    assert hygiene.inspect_file(data_file) == ["research data or generated output is tracked"]
    assert hygiene.inspect_file(bundle) == ["binary data or model artifact is tracked"]
    assert hygiene.inspect_file(clean) == []
