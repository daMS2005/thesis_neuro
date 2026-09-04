"""Shared defaults for the benchmark track (kept free of numerical imports for the CLI)."""

from __future__ import annotations

from structure_comparison.defaults import DEFAULT_ALPHA_GRID

DEFAULT_TARGET_COLUMNS = ("correct", "gold_choice_avg_logprob", "margin")

__all__ = ["DEFAULT_ALPHA_GRID", "DEFAULT_TARGET_COLUMNS"]
