"""Tests for the pure layer-resolution helpers in scripts/analysis/render_run_config.py."""

from __future__ import annotations

import importlib.util

import pytest

from thesis_neuro.paths import repository_root


def _load_module():
    path = repository_root() / "scripts" / "analysis" / "render_run_config.py"
    spec = importlib.util.spec_from_file_location("render_run_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_relative_depth_layers_match_the_registered_model_layers() -> None:
    module = _load_module()
    fractions = list(module.DEFAULT_LAYER_FRACTIONS)
    assert module.resolve_target_layers(26, fractions) == [4, 8, 13, 17, 22, 25]
    assert module.resolve_target_layers(42, fractions) == [7, 13, 21, 27, 36, 40]
    assert module.resolve_target_layers(32, fractions) == [5, 10, 16, 21, 27, 31]


def test_layer_fraction_parsing_validates_range() -> None:
    module = _load_module()
    assert module.parse_layer_fractions("0.1, 0.5,0.9") == [0.1, 0.5, 0.9]
    with pytest.raises(ValueError):
        module.parse_layer_fractions("1.5")
    with pytest.raises(ValueError):
        module.parse_layer_fractions("")


def test_sae_id_parsing_handles_gemma_scope_and_llama_scope() -> None:
    module = _load_module()
    assert module.parse_registered_layer_sae_id("layer_12/width_16k/canonical", "width_16k") == 12
    assert module.parse_registered_layer_sae_id("layer_12/width_65k/canonical", "width_16k") is None
    assert module.parse_registered_layer_sae_id("l16r_8x", "8x") == 16
    assert module.normalize_scope_repo("gemma-scope-2b-pt-res-canonical") == "google/gemma-scope-2b-pt-res"
