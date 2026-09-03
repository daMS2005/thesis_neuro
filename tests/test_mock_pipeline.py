"""Tests that the mock extraction pipeline writes the artifact schema without model dependencies."""

from __future__ import annotations

import json
from pathlib import Path

from thesis_neuro.config import load_app_config
from thesis_neuro.paths import repository_root
from thesis_neuro.pipelines.mock import MockExtractionPipeline


def test_mock_pipeline_writes_schema_without_model_dependencies(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("THESIS_NEURO_OUTPUT_ROOT", str(tmp_path))
    config = load_app_config(repository_root() / "configs" / "examples" / "mock.yaml")
    manifest = MockExtractionPipeline(config).run()
    records_path = Path(manifest["artifacts"]["paired_records"])
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert manifest["mode"] == "mock"
    assert rows
    assert {"sample_id", "layer", "token", "latent_activations", "top_logits"}.issubset(rows[0])
