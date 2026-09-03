"""JSON/JSONL helpers and small array utilities shared across the package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

FeatureKey = tuple[int, int]


def safe_divide_rows(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=np.float32)
    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator > 0,
    )
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def feature_name(key: FeatureKey) -> str:
    layer, feature_id = key
    return f"layer{int(layer)}:feature{int(feature_id)}"


def _string_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=str)


def _ensure_unique(values: np.ndarray, label: str) -> None:
    unique_values, counts = np.unique(values, return_counts=True)
    if np.any(counts > 1):
        repeated = unique_values[counts > 1][:5].tolist()
        raise ValueError(f"{label} contains duplicates, including {repeated}")


def _normalize_columns(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
