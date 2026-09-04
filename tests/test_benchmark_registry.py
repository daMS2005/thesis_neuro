"""Tests for the packaged benchmark model registry."""

from __future__ import annotations

from pathlib import Path

from benchmark_comparison.registry import load_registry, registry_rows, resolve_registry_entry


def test_packaged_registry_lists_three_models_with_resolved_paths() -> None:
    registry = load_registry()
    assert set(registry) == {"gemma_2_2b", "gemma_2_9b", "llama_3_1_8b"}
    for entry in registry.values():
        assert isinstance(entry.feature_run_dir, Path) and entry.feature_run_dir.is_absolute()
        assert len(entry.layer_selection) == 6


def test_registry_rows_expose_feature_run_dir_and_presence_flag() -> None:
    rows = registry_rows()
    assert [row["name"] for row in rows] == ["gemma_2_2b", "gemma_2_9b", "llama_3_1_8b"]
    assert all("feature_run_dir" in row and "all_paths_present" in row for row in rows)


def test_unknown_model_lists_available_names() -> None:
    try:
        resolve_registry_entry("not-a-model")
    except KeyError as exc:
        assert "gemma_2_2b" in str(exc)
    else:
        raise AssertionError("expected KeyError")
