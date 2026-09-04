"""Portable filesystem resolution for thesis-neuro workflows."""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "THESIS_NEURO_DATA_ROOT"
OUTPUT_ROOT_ENV = "THESIS_NEURO_OUTPUT_ROOT"
DEFAULT_RUN_DIR = "outputs/default-run"


def repository_root() -> Path:
    """Return the source checkout root, independent of the current directory."""

    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Return the configured data root, defaulting to ``<repo>/data``."""

    return _environment_root(DATA_ROOT_ENV, repository_root() / "data")


def output_root() -> Path:
    """Return the configured output root, defaulting to ``<repo>/outputs``."""

    return _environment_root(OUTPUT_ROOT_ENV, repository_root() / "outputs")


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve an explicit path relative to the repository when needed."""

    return _resolve(value, repository_root())


def resolve_data_path(value: str | Path) -> Path:
    """Resolve a data path under :envvar:`THESIS_NEURO_DATA_ROOT`.

    Paths that start with ``examples/`` are the repository's tracked offline fixtures and
    always resolve inside the checkout, so the mock config works under any data root.
    """

    path = Path(value).expanduser()
    if not path.is_absolute() and path.parts and path.parts[0] == "examples":
        return (repository_root() / path).resolve()
    return _resolve_without_prefix(value, data_root(), "data")


def resolve_output_path(value: str | Path) -> Path:
    """Resolve an output path under :envvar:`THESIS_NEURO_OUTPUT_ROOT`."""

    return _resolve_without_prefix(value, output_root(), "outputs")


def default_config_path() -> Path:
    return repository_root() / "configs" / "default.yaml"


def _environment_root(name: str, fallback: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else fallback.resolve()


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_without_prefix(value: str | Path, root: Path, prefix: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    parts = path.parts[1:] if path.parts and path.parts[0] == prefix else path.parts
    return root.joinpath(*parts).resolve()
