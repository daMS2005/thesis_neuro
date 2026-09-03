"""Evidence, judge, schema, and runner components for feature probing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from thesis_neuro.probes.runner import FeatureProbingPipeline

__all__ = ["FeatureProbingPipeline"]


def __getattr__(name: str) -> Any:
    if name == "FeatureProbingPipeline":
        from thesis_neuro.probes.runner import FeatureProbingPipeline

        return FeatureProbingPipeline
    raise AttributeError(name)
