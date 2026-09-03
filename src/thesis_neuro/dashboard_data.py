from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from thesis_neuro.paths import repository_root, resolve_output_path


@dataclass(slots=True)
class DashboardPaths:
    repo_root: Path
    analysis_dir: Path
    transcript_dir: Path | None
    dolma_dir: Path | None
    manifest_path: Path
    relevance_path: Path
    correlations_path: Path
    concepts_path: Path
    judge_input_path: Path
    alignment_path: Path
    transcript_paired_path: Path | None
    dolma_contexts_path: Path | None


def resolve_dashboard_paths(
    repo_root: str | Path | None = None,
    analysis_dir: str | Path = "outputs/default_run",
    transcript_dir: str | Path | None = None,
    dolma_dir: str | Path | None = None,
) -> DashboardPaths:
    root = _resolve_repo_root(repo_root)
    analysis = _resolve_path(root, analysis_dir)
    transcript = _resolve_path(root, transcript_dir) if transcript_dir is not None else None
    dolma = _resolve_path(root, dolma_dir) if dolma_dir is not None else None

    transcript_paired_path = None
    if transcript is not None:
        preferred = transcript / "transcript_paired_records.jsonl"
        fallback = transcript / "transcript_paired_records.minimal.jsonl"
        if preferred.exists():
            transcript_paired_path = preferred
        elif fallback.exists():
            transcript_paired_path = fallback
        else:
            transcript_paired_path = preferred

    return DashboardPaths(
        repo_root=root,
        analysis_dir=analysis,
        transcript_dir=transcript,
        dolma_dir=dolma,
        manifest_path=analysis / "manifest.json",
        relevance_path=analysis / "feature_relevance.jsonl",
        correlations_path=analysis / "feature_correlations.jsonl",
        concepts_path=analysis / "feature_concepts.jsonl",
        judge_input_path=analysis / "feature_judge_input.jsonl",
        alignment_path=analysis / "feature_alignment.jsonl",
        transcript_paired_path=transcript_paired_path,
        dolma_contexts_path=(dolma / "dolma_feature_contexts.jsonl") if dolma is not None else None,
    )


def _resolve_repo_root(repo_root: str | Path | None) -> Path:
    if repo_root is not None:
        root = Path(repo_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        return root

    return repository_root()


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and path.parts and path.parts[0] == "outputs":
        return resolve_output_path(path)
    if not path.is_absolute():
        path = root / path
    return path.resolve()
