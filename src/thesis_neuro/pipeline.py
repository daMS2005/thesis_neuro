"""Compatibility facade for the pre-0.2 pipeline import path."""

from thesis_neuro.pipelines.alignment import FeatureAlignmentPipeline
from thesis_neuro.pipelines.contexts import DolmaContextCollectionPipeline
from thesis_neuro.pipelines.extraction import ExtractionPipeline, PrefetchPipeline
from thesis_neuro.pipelines.mock import MockExtractionPipeline
from thesis_neuro.pipelines.summary import summarize_feature_records
from thesis_neuro.pipelines.transcripts import TranscriptFeatureDiscoveryPipeline

__all__ = [
    "DolmaContextCollectionPipeline",
    "ExtractionPipeline",
    "FeatureAlignmentPipeline",
    "MockExtractionPipeline",
    "PrefetchPipeline",
    "TranscriptFeatureDiscoveryPipeline",
    "summarize_feature_records",
]
