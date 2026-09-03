from __future__ import annotations

from pathlib import Path

from thesis_neuro.config import load_app_config
from thesis_neuro.paths import data_root, output_root, repository_root, resolve_data_path, resolve_output_path


def test_default_roots_are_repository_relative(monkeypatch) -> None:
    monkeypatch.delenv("THESIS_NEURO_DATA_ROOT", raising=False)
    monkeypatch.delenv("THESIS_NEURO_OUTPUT_ROOT", raising=False)
    assert data_root() == repository_root() / "data"
    assert output_root() == repository_root() / "outputs"


def test_environment_roots_override_defaults(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "research-data"
    outputs = tmp_path / "research-outputs"
    monkeypatch.setenv("THESIS_NEURO_DATA_ROOT", str(data))
    monkeypatch.setenv("THESIS_NEURO_OUTPUT_ROOT", str(outputs))
    assert resolve_data_path("transcripts/story.txt") == data / "transcripts" / "story.txt"
    assert resolve_output_path("outputs/run-a") == outputs / "run-a"


def test_mock_config_resolves_packaged_fixture() -> None:
    config = load_app_config(repository_root() / "configs" / "examples" / "mock.yaml")
    assert Path(config.dataset.local_text_path or "").is_file()
    assert config.output_dir.name == "mock-schema"


def test_config_snapshot_redacts_credentials(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_value")
    monkeypatch.setenv("OPENAI_API_KEY", "provider_test_value")
    config = load_app_config(repository_root() / "configs" / "examples" / "mock.yaml")
    snapshot = config.to_dict()
    assert snapshot["env"]["hf_token"] == "***REDACTED***"
    assert snapshot["env"]["openai_api_key"] == "***REDACTED***"
