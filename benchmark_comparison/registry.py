"""Registry of model runs and the external artifact paths each benchmark fit expects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_comparison import PACKAGE_ROOT
from thesis_neuro.paths import resolve_output_path, resolve_repo_path


@dataclass(frozen=True, slots=True)
class ModelRegistryEntry:
    name: str
    model_id: str
    scope_release: str
    scope_width: str
    token_layer: int
    layer_selection: tuple[int, ...]
    remote_run_dir: Path
    selected_features_path: Path
    analysis_summary_path: Path
    brain_final_model_path: Path
    notes: str | None = None


def default_registry_path() -> Path:
    return PACKAGE_ROOT / "model_registry.json"


def load_registry(path: str | Path | None = None) -> dict[str, ModelRegistryEntry]:
    registry_path = Path(path) if path is not None else default_registry_path()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    entries: dict[str, ModelRegistryEntry] = {}
    for name, raw_entry in payload.items():
        entries[name] = _parse_entry(name=name, raw_entry=raw_entry)
    return entries


def resolve_registry_entry(
    model_name: str,
    registry_path: str | Path | None = None,
) -> ModelRegistryEntry:
    registry = load_registry(registry_path)
    if model_name not in registry:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Unknown model '{model_name}'. Available models: {available}")
    return registry[model_name]


def registry_rows(path: str | Path | None = None) -> list[dict[str, Any]]:
    registry = load_registry(path)
    rows: list[dict[str, Any]] = []
    for name, entry in sorted(registry.items()):
        rows.append(
            {
                "name": name,
                "model_id": entry.model_id,
                "scope_release": entry.scope_release,
                "scope_width": entry.scope_width,
                "token_layer": entry.token_layer,
                "layer_selection": list(entry.layer_selection),
                "remote_run_dir": str(entry.remote_run_dir),
                "selected_features_path": str(entry.selected_features_path),
                "analysis_summary_path": str(entry.analysis_summary_path),
                "brain_final_model_path": str(entry.brain_final_model_path),
                "notes": entry.notes,
                "all_paths_present": all(
                    path.exists()
                    for path in (
                        entry.remote_run_dir,
                        entry.selected_features_path,
                        entry.analysis_summary_path,
                        entry.brain_final_model_path,
                    )
                ),
            }
        )
    return rows


def _parse_entry(name: str, raw_entry: dict[str, Any]) -> ModelRegistryEntry:
    return ModelRegistryEntry(
        name=name,
        model_id=str(raw_entry["model_id"]),
        scope_release=str(raw_entry["scope_release"]),
        scope_width=str(raw_entry["scope_width"]),
        token_layer=int(raw_entry["token_layer"]),
        layer_selection=tuple(int(layer) for layer in raw_entry["layer_selection"]),
        remote_run_dir=_resolve_repo_path(raw_entry["remote_run_dir"]),
        selected_features_path=_resolve_repo_path(raw_entry["selected_features_path"]),
        analysis_summary_path=_resolve_repo_path(raw_entry["analysis_summary_path"]),
        brain_final_model_path=_resolve_repo_path(raw_entry["brain_final_model_path"]),
        notes=str(raw_entry["notes"]) if raw_entry.get("notes") is not None else None,
    )


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() and path.parts and path.parts[0] == "outputs":
        return resolve_output_path(path)
    return resolve_repo_path(value)
