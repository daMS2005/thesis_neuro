"""Tests for lightweight CLI help, fixtures, schemas, storage, and dashboard packaging."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark_comparison.items import BenchmarkItem, load_items, validate_items, write_items
from benchmark_comparison.workflow import build_parser as build_benchmark_parser
from structure_comparison.cli import build_parser as build_structure_parser
from thesis_neuro.cli import build_parser as build_main_parser
from thesis_neuro.dashboard_data import resolve_dashboard_paths
from thesis_neuro.probes.schema import ProbeTarget, validate_probe_report
from thesis_neuro.storage import JsonlArtifactStore


def test_all_cli_parsers_support_lightweight_help() -> None:
    for parser in (build_main_parser(), build_structure_parser(), build_benchmark_parser()):
        with pytest.raises(SystemExit) as exit_info:
            parser.parse_args(["--help"])
        assert exit_info.value.code == 0


def test_benchmark_fixture_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    item = BenchmarkItem(
        item_id="fixture:0",
        benchmark="fixture",
        task="boolq",
        split="test",
        feature_text="A short passage.",
        score_prompt="Is the statement supported?",
        choices=(" yes", " no"),
        correct_choice=0,
        metadata={},
    )
    validate_items([item])
    write_items(path, [item])
    assert load_items(path) == [item]


def test_probe_schema_validation() -> None:
    assert ProbeTarget(layer=4, feature_id=464).feature_id == 464
    validate_probe_report(
        {
            "final_hypothesis": "past habitual construction",
            "summary": "Activates on used-to constructions.",
            "confidence": 0.8,
            "uncertainty": "Limited contrast examples.",
        }
    )
    with pytest.raises(ValueError):
        ProbeTarget(layer=-1, feature_id=0)


def test_probe_schema_import_does_not_require_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import thesis_neuro.probes.schema; assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_artifact_store_writes_manifest_and_rows(tmp_path: Path) -> None:
    store = JsonlArtifactStore(tmp_path / "run")
    store.append_record({"sample_id": "sample-0"})
    store.write_manifest({"status": "complete"})
    assert list(store.iter_records()) == [{"sample_id": "sample-0"}]
    assert json.loads(store.manifest_path.read_text()) == {"status": "complete"}


def test_dashboard_paths_resolve_explicit_root(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    paths = resolve_dashboard_paths(
        repo_root=root,
        analysis_dir="results/analysis",
        transcript_dir="results/transcripts",
        dolma_dir="results/dolma",
    )
    assert paths.analysis_dir == root / "results" / "analysis"
    assert paths.transcript_paired_path == root / "results" / "transcripts" / "transcript_paired_records.jsonl"


def test_dashboard_assets_are_packaged_separately() -> None:
    from importlib import resources

    static = resources.files("thesis_neuro.static")
    for name in ("audit.html", "audit.css", "audit.js"):
        assert static.joinpath(name).is_file(), name
    html = static.joinpath("audit.html").read_text(encoding="utf-8")
    assert "<!doctype html>" in html.lower()
    assert "/static/audit.css" in html and "/static/audit.js" in html
    assert "http://" not in html and "https://" not in html  # no external assets
