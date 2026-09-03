"""Dependency-free data contracts for feature probing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ProbeTarget:
    layer: int
    feature_id: int
    script_id: str | None = None

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if self.feature_id < 0:
            raise ValueError("feature_id must be non-negative")


@dataclass(slots=True)
class ProbeRunPaths:
    root: Path
    evidence_path: Path
    rounds_path: Path
    tests_path: Path
    steering_path: Path
    report_path: Path
    manifest_path: Path


def validate_probe_report(report: dict[str, Any]) -> None:
    required = {"final_hypothesis", "summary", "confidence", "uncertainty"}
    missing = sorted(required.difference(report))
    if missing:
        raise ValueError(f"Probe report is missing required fields: {', '.join(missing)}")
    confidence = float(report["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Probe report confidence must be between 0 and 1")


__all__ = ["ProbeRunPaths", "ProbeTarget", "validate_probe_report"]
