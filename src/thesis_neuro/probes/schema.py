"""Dependency-free data contracts for feature probing."""

from __future__ import annotations

import re
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


def slugify(value: str) -> str:
    """Lower-case a label and collapse non-alphanumerics so it is safe as a directory name."""

    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "default"


def probe_run_paths(root: Path, bundle_id: str, script_id: str | None, layer: int, feature_id: int) -> ProbeRunPaths:
    """Build the artifact layout for one probe run.

    Both the probing runner and the audit dashboard use this so that a run started from
    the dashboard is found again under ``<root>/<bundle>/<script>/layer_<n>/feature_<id>``.
    """

    script_segment = slugify(script_id) if script_id else "all_scripts"
    run_root = root / slugify(bundle_id) / script_segment / f"layer_{int(layer)}" / f"feature_{int(feature_id)}"
    return ProbeRunPaths(
        root=run_root,
        evidence_path=run_root / "feature_probe_evidence.json",
        rounds_path=run_root / "feature_probe_rounds.jsonl",
        tests_path=run_root / "feature_probe_tests.jsonl",
        steering_path=run_root / "feature_probe_steering.jsonl",
        report_path=run_root / "feature_probe_report.json",
        manifest_path=run_root / "manifest.json",
    )
