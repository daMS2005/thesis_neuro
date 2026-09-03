"""Compatibility facade for the pre-0.2 probing import path."""

from thesis_neuro.probes.judge import OpenAIProbingAgent
from thesis_neuro.probes.runner import FeatureProbingPipeline
from thesis_neuro.probes.schema import ProbeRunPaths

__all__ = ["FeatureProbingPipeline", "OpenAIProbingAgent", "ProbeRunPaths"]
