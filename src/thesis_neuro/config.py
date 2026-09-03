"""Typed run configuration assembled from YAML, environment variables, and CLI overrides."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from thesis_neuro.paths import (
    default_config_path,
    resolve_data_path,
    resolve_output_path,
)


@dataclass(slots=True)
class EnvironmentConfig:
    hf_token: str | None
    hf_home: str | None
    hf_local_files_only: bool
    openai_api_key: str | None
    torch_device: str
    torch_dtype: str


@dataclass(slots=True)
class ModelConfig:
    base_model_id: str
    scope_release: str
    scope_width: str
    layer_selection: str | list[int]


@dataclass(slots=True)
class DatasetConfig:
    id: str
    split: str
    streaming: bool
    text_field: str
    local_text_path: str | None
    shuffle_buffer_size: int
    max_documents: int
    max_windows: int


@dataclass(slots=True)
class TranscriptConfig:
    root_dir: str
    glob: str
    max_files: int


@dataclass(slots=True)
class TokenizationConfig:
    seq_len: int
    add_special_tokens: bool


@dataclass(slots=True)
class LatentConfig:
    token_top_k: int
    pooled_top_k: int
    top_n_logits: int

    @property
    def top_k(self) -> int:
        return self.token_top_k


@dataclass(slots=True)
class FeatureSelectionConfig:
    top_by_peak: int
    top_by_total: int
    top_by_persistence: int
    top_by_sentence_pool: int
    final_top_per_layer: int
    top_examples_per_feature: int


@dataclass(slots=True)
class AlignmentConfig:
    top_features_per_window: int
    top_windows_per_feature: int
    top_token_positions_per_feature_window: int
    top_token_alignments: int
    top_span_alignments: int
    methods: list[str]


@dataclass(slots=True)
class DolmaQueryConfig:
    enabled: bool
    max_windows: int
    top_contexts_per_feature: int
    context_window_tokens: int
    min_activation_threshold: float
    top_tokens_per_context: int
    top_spans_per_context: int
    top_sentences_per_context: int


@dataclass(slots=True)
class AnalysisConfig:
    transcript_output_dir: str | None
    dolma_output_dir: str | None
    alignment_path: str | None
    compute_distinctiveness: bool
    top_features_for_alignment: int
    top_features_for_correlation: int
    top_features_for_judge: int
    transcript_examples_per_feature: int
    dolma_contexts_per_feature: int
    correlated_features_per_feature: int


@dataclass(slots=True)
class JudgeConfig:
    enabled: bool
    model: str
    timeout_seconds: float
    max_retries: int
    max_concurrency: int


@dataclass(slots=True)
class ProbingConfig:
    model: str | None
    max_rounds: int
    synthetic_probes_per_round: int
    real_edits_per_round: int
    enable_steering: bool
    steering_strengths: list[float]
    stop_confidence: float
    max_batch_size: int


@dataclass(slots=True)
class OutputConfig:
    dir: str
    write_manifest: bool
    write_summary: bool


@dataclass(slots=True)
class RunSettings:
    seed: int


@dataclass(slots=True)
class AppConfig:
    env: EnvironmentConfig
    model: ModelConfig
    dataset: DatasetConfig
    transcripts: TranscriptConfig
    tokenization: TokenizationConfig
    latents: LatentConfig
    feature_selection: FeatureSelectionConfig
    alignment: AlignmentConfig
    dolma_query: DolmaQueryConfig
    analysis: AnalysisConfig
    judge: JudgeConfig
    probing: ProbingConfig
    output: OutputConfig
    run: RunSettings

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for field_name in ("hf_token", "openai_api_key"):
            if payload["env"].get(field_name):
                payload["env"][field_name] = "***REDACTED***"
        return payload

    @property
    def output_dir(self) -> Path:
        return Path(self.output.dir)


def load_app_config(
    config_path: str | Path | None = None,
    env_path: str | Path = ".env",
) -> AppConfig:
    load_dotenv(env_path, override=False)

    path = Path(config_path) if config_path is not None else default_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw_config = yaml.safe_load(path.read_text()) or {}

    env = EnvironmentConfig(
        hf_token=_optional_env("HF_TOKEN"),
        hf_home=_optional_env("HF_HOME"),
        hf_local_files_only=_bool_env("HF_LOCAL_FILES_ONLY", False),
        openai_api_key=_optional_env("OPENAI_API_KEY"),
        torch_device=os.getenv("TORCH_DEVICE", "auto"),
        torch_dtype=os.getenv("TORCH_DTYPE", "bfloat16"),
    )
    if env.hf_home:
        os.environ.setdefault("HF_HOME", env.hf_home)
    if env.hf_local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = AppConfig(
        env=env,
        model=_parse_model_config(raw_config.get("model", {})),
        dataset=_parse_dataset_config(raw_config.get("dataset", {})),
        transcripts=_parse_transcript_config(raw_config.get("transcripts", {})),
        tokenization=_parse_tokenization_config(raw_config.get("tokenization", {})),
        latents=_parse_latents_config(raw_config.get("latents", {})),
        feature_selection=_parse_feature_selection_config(raw_config.get("feature_selection", {})),
        alignment=_parse_alignment_config(raw_config.get("alignment", {})),
        dolma_query=_parse_dolma_query_config(raw_config.get("dolma_query", {})),
        analysis=_parse_analysis_config(raw_config.get("analysis", {})),
        judge=_parse_judge_config(raw_config.get("judge", {})),
        probing=_parse_probing_config(raw_config.get("probing", {})),
        output=_parse_output_config(raw_config.get("output", {})),
        run=_parse_run_config(raw_config.get("run", {})),
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    if config.tokenization.seq_len <= 0:
        raise ValueError("tokenization.seq_len must be positive")
    if config.latents.token_top_k <= 0:
        raise ValueError("latents.token_top_k must be positive")
    if config.latents.pooled_top_k <= 0:
        raise ValueError("latents.pooled_top_k must be positive")
    if config.latents.top_n_logits <= 0:
        raise ValueError("latents.top_n_logits must be positive")
    if config.feature_selection.top_by_peak <= 0:
        raise ValueError("feature_selection.top_by_peak must be positive")
    if config.feature_selection.top_by_total <= 0:
        raise ValueError("feature_selection.top_by_total must be positive")
    if config.feature_selection.top_by_persistence <= 0:
        raise ValueError("feature_selection.top_by_persistence must be positive")
    if config.feature_selection.top_by_sentence_pool <= 0:
        raise ValueError("feature_selection.top_by_sentence_pool must be positive")
    if config.feature_selection.final_top_per_layer <= 0:
        raise ValueError("feature_selection.final_top_per_layer must be positive")
    if config.feature_selection.top_examples_per_feature <= 0:
        raise ValueError("feature_selection.top_examples_per_feature must be positive")
    if config.alignment.top_features_per_window <= 0:
        raise ValueError("alignment.top_features_per_window must be positive")
    if config.alignment.top_windows_per_feature <= 0:
        raise ValueError("alignment.top_windows_per_feature must be positive")
    if config.alignment.top_token_positions_per_feature_window <= 0:
        raise ValueError("alignment.top_token_positions_per_feature_window must be positive")
    if config.alignment.top_token_alignments <= 0:
        raise ValueError("alignment.top_token_alignments must be positive")
    if config.alignment.top_span_alignments <= 0:
        raise ValueError("alignment.top_span_alignments must be positive")
    if not config.alignment.methods:
        raise ValueError("alignment.methods must not be empty")
    if config.judge.timeout_seconds <= 0:
        raise ValueError("judge.timeout_seconds must be positive")
    if config.judge.max_retries < 0:
        raise ValueError("judge.max_retries must be non-negative")
    if config.judge.max_concurrency <= 0:
        raise ValueError("judge.max_concurrency must be positive")
    if config.probing.max_rounds <= 0:
        raise ValueError("probing.max_rounds must be positive")
    if config.probing.synthetic_probes_per_round <= 0:
        raise ValueError("probing.synthetic_probes_per_round must be positive")
    if config.probing.real_edits_per_round <= 0:
        raise ValueError("probing.real_edits_per_round must be positive")
    if config.probing.max_batch_size <= 0:
        raise ValueError("probing.max_batch_size must be positive")
    if not 0.0 <= config.probing.stop_confidence <= 1.0:
        raise ValueError("probing.stop_confidence must be between 0 and 1")
    if config.probing.enable_steering and not config.probing.steering_strengths:
        raise ValueError("probing.steering_strengths must not be empty when steering is enabled")
    if config.dataset.max_documents <= 0:
        raise ValueError("dataset.max_documents must be positive")
    if config.dataset.max_windows <= 0:
        raise ValueError("dataset.max_windows must be positive")
    if config.dataset.local_text_path is None and config.dataset.shuffle_buffer_size <= 0:
        raise ValueError("dataset.shuffle_buffer_size must be positive")
    if not config.transcripts.root_dir:
        raise ValueError("transcripts.root_dir is required")
    if not config.transcripts.glob:
        raise ValueError("transcripts.glob is required")
    if config.transcripts.max_files <= 0:
        raise ValueError("transcripts.max_files must be positive")
    if config.dolma_query.max_windows <= 0:
        raise ValueError("dolma_query.max_windows must be positive")
    if config.dolma_query.top_contexts_per_feature <= 0:
        raise ValueError("dolma_query.top_contexts_per_feature must be positive")
    if config.dolma_query.context_window_tokens <= 0:
        raise ValueError("dolma_query.context_window_tokens must be positive")
    if config.dolma_query.top_tokens_per_context <= 0:
        raise ValueError("dolma_query.top_tokens_per_context must be positive")
    if config.dolma_query.top_spans_per_context <= 0:
        raise ValueError("dolma_query.top_spans_per_context must be positive")
    if config.dolma_query.top_sentences_per_context <= 0:
        raise ValueError("dolma_query.top_sentences_per_context must be positive")
    if config.analysis.top_features_for_alignment <= 0:
        raise ValueError("analysis.top_features_for_alignment must be positive")
    if config.analysis.top_features_for_correlation <= 0:
        raise ValueError("analysis.top_features_for_correlation must be positive")
    if config.analysis.top_features_for_judge <= 0:
        raise ValueError("analysis.top_features_for_judge must be positive")
    if config.analysis.transcript_examples_per_feature <= 0:
        raise ValueError("analysis.transcript_examples_per_feature must be positive")
    if config.analysis.dolma_contexts_per_feature <= 0:
        raise ValueError("analysis.dolma_contexts_per_feature must be positive")
    if config.analysis.correlated_features_per_feature <= 0:
        raise ValueError("analysis.correlated_features_per_feature must be positive")
    if not config.model.base_model_id:
        raise ValueError("model.base_model_id is required")
    if not config.model.scope_release:
        raise ValueError("model.scope_release is required")
    if not config.model.scope_width:
        raise ValueError("model.scope_width is required")


def _parse_model_config(section: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        base_model_id=section.get("base_model_id", "google/gemma-2-2b"),
        scope_release=section.get("scope_release", "gemma-scope-2b-pt-res-canonical"),
        scope_width=section.get("scope_width", "width_16k"),
        layer_selection=section.get("layer_selection", "all"),
    )


def _parse_dataset_config(section: dict[str, Any]) -> DatasetConfig:
    local_text_path = section.get("local_text_path")
    return DatasetConfig(
        id=section.get("id", "allenai/dolma"),
        split=section.get("split", "train"),
        streaming=bool(section.get("streaming", True)),
        text_field=section.get("text_field", "text"),
        local_text_path=str(resolve_data_path(local_text_path)) if local_text_path else None,
        shuffle_buffer_size=int(section.get("shuffle_buffer_size", 1000)),
        max_documents=int(section.get("max_documents", 25)),
        max_windows=int(section.get("max_windows", 50)),
    )


def _parse_transcript_config(section: dict[str, Any]) -> TranscriptConfig:
    root_dir = section.get("root_dir", "data/transcripts")
    return TranscriptConfig(
        root_dir=str(resolve_data_path(root_dir)),
        glob=section.get("glob", "*/*_transcript.txt"),
        max_files=int(section.get("max_files", 100)),
    )


def _parse_tokenization_config(section: dict[str, Any]) -> TokenizationConfig:
    return TokenizationConfig(
        seq_len=int(section.get("seq_len", 256)),
        add_special_tokens=bool(section.get("add_special_tokens", False)),
    )


def _parse_latents_config(section: dict[str, Any]) -> LatentConfig:
    fallback_top_k = int(section.get("top_k", 256))
    return LatentConfig(
        token_top_k=int(section.get("token_top_k", fallback_top_k)),
        pooled_top_k=int(section.get("pooled_top_k", fallback_top_k)),
        top_n_logits=int(section.get("top_n_logits", 10)),
    )


def _parse_feature_selection_config(section: dict[str, Any]) -> FeatureSelectionConfig:
    return FeatureSelectionConfig(
        top_by_peak=int(section.get("top_by_peak", 128)),
        top_by_total=int(section.get("top_by_total", 128)),
        top_by_persistence=int(section.get("top_by_persistence", 128)),
        top_by_sentence_pool=int(section.get("top_by_sentence_pool", 128)),
        final_top_per_layer=int(section.get("final_top_per_layer", 256)),
        top_examples_per_feature=int(section.get("top_examples_per_feature", 5)),
    )


def _parse_alignment_config(section: dict[str, Any]) -> AlignmentConfig:
    methods = section.get(
        "methods",
        [
            "deletion_retokenize",
            "pad_eos_mask",
            "delete_sentence_retokenize",
            "pad_sentence_mask",
            "delete_clause_retokenize",
            "pad_clause_mask",
        ],
    )
    return AlignmentConfig(
        top_features_per_window=int(section.get("top_features_per_window", 32)),
        top_windows_per_feature=int(section.get("top_windows_per_feature", 3)),
        top_token_positions_per_feature_window=int(
            section.get("top_token_positions_per_feature_window", 24)
        ),
        top_token_alignments=int(section.get("top_token_alignments", 5)),
        top_span_alignments=int(section.get("top_span_alignments", 5)),
        methods=[str(method) for method in methods],
    )


def _parse_dolma_query_config(section: dict[str, Any]) -> DolmaQueryConfig:
    return DolmaQueryConfig(
        enabled=bool(section.get("enabled", True)),
        max_windows=int(section.get("max_windows", 10000)),
        top_contexts_per_feature=int(section.get("top_contexts_per_feature", 25)),
        context_window_tokens=int(section.get("context_window_tokens", 4)),
        min_activation_threshold=float(section.get("min_activation_threshold", 0.0)),
        top_tokens_per_context=int(section.get("top_tokens_per_context", 5)),
        top_spans_per_context=int(section.get("top_spans_per_context", 3)),
        top_sentences_per_context=int(section.get("top_sentences_per_context", 3)),
    )


def _parse_analysis_config(section: dict[str, Any]) -> AnalysisConfig:
    transcript_output_dir = section.get("transcript_output_dir")
    dolma_output_dir = section.get("dolma_output_dir")
    alignment_path = section.get("alignment_path")
    return AnalysisConfig(
        transcript_output_dir=(
            str(resolve_output_path(transcript_output_dir)) if transcript_output_dir else None
        ),
        dolma_output_dir=str(resolve_output_path(dolma_output_dir)) if dolma_output_dir else None,
        alignment_path=str(resolve_output_path(alignment_path)) if alignment_path else None,
        compute_distinctiveness=bool(section.get("compute_distinctiveness", True)),
        top_features_for_alignment=int(section.get("top_features_for_alignment", 32)),
        top_features_for_correlation=int(section.get("top_features_for_correlation", 64)),
        top_features_for_judge=int(section.get("top_features_for_judge", 20)),
        transcript_examples_per_feature=int(section.get("transcript_examples_per_feature", 3)),
        dolma_contexts_per_feature=int(section.get("dolma_contexts_per_feature", 3)),
        correlated_features_per_feature=int(section.get("correlated_features_per_feature", 5)),
    )


def _parse_judge_config(section: dict[str, Any]) -> JudgeConfig:
    return JudgeConfig(
        enabled=bool(section.get("enabled", False)),
        model=str(section.get("model", "gpt-5-mini")),
        timeout_seconds=float(section.get("timeout_seconds", 60.0)),
        max_retries=int(section.get("max_retries", 2)),
        max_concurrency=int(section.get("max_concurrency", 5)),
    )


def _parse_probing_config(section: dict[str, Any]) -> ProbingConfig:
    strengths = section.get("steering_strengths", [0.5, 1.0, 2.0])
    return ProbingConfig(
        model=section.get("model"),
        max_rounds=int(section.get("max_rounds", 3)),
        synthetic_probes_per_round=int(section.get("synthetic_probes_per_round", 8)),
        real_edits_per_round=int(section.get("real_edits_per_round", 6)),
        enable_steering=bool(section.get("enable_steering", True)),
        steering_strengths=[float(value) for value in strengths],
        stop_confidence=float(section.get("stop_confidence", 0.85)),
        max_batch_size=int(section.get("max_batch_size", 8)),
    )


def _parse_output_config(section: dict[str, Any]) -> OutputConfig:
    return OutputConfig(
        dir=str(resolve_output_path(section.get("dir", "outputs/default_run"))),
        write_manifest=bool(section.get("write_manifest", True)),
        write_summary=bool(section.get("write_summary", False)),
    )


def _parse_run_config(section: dict[str, Any]) -> RunSettings:
    return RunSettings(seed=int(section.get("seed", 42)))


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
