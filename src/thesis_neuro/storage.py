from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class JsonlArtifactStore:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.paired_path = self.output_dir / "paired_records.jsonl"
        self.transcript_paired_path = self.output_dir / "transcript_paired_records.jsonl"
        self.transcript_feature_stats_path = self.output_dir / "transcript_feature_stats.jsonl"
        self.transcript_feature_shortlist_path = self.output_dir / "transcript_feature_shortlist.jsonl"
        self.dolma_contexts_path = self.output_dir / "dolma_feature_contexts.jsonl"
        self.alignment_path = self.output_dir / "feature_alignment.jsonl"
        self.feature_relevance_path = self.output_dir / "feature_relevance.jsonl"
        self.selected_features_path = self.output_dir / "selected_features_for_alignment.jsonl"
        self.feature_correlations_path = self.output_dir / "feature_correlations.jsonl"
        self.feature_judge_input_path = self.output_dir / "feature_judge_input.jsonl"
        self.feature_concepts_path = self.output_dir / "feature_concepts.jsonl"
        self.summary_path = self.output_dir / "feature_summary.jsonl"
        self.manifest_path = self.output_dir / "manifest.json"

    def append_record(self, record: dict[str, Any]) -> None:
        self.append_jsonl(self.paired_path, record)

    def append_transcript_record(self, record: dict[str, Any]) -> None:
        self.append_jsonl(self.transcript_paired_path, record)

    def reset_run_files(self, include_summary: bool) -> None:
        if self.paired_path.exists():
            self.paired_path.unlink()
        if self.manifest_path.exists():
            self.manifest_path.unlink()
        if include_summary and self.summary_path.exists():
            self.summary_path.unlink()

    def reset_alignment_file(self) -> None:
        if self.alignment_path.exists():
            self.alignment_path.unlink()

    def reset_transcript_files(self) -> None:
        for path in (
            self.transcript_paired_path,
            self.transcript_feature_stats_path,
            self.transcript_feature_shortlist_path,
        ):
            if path.exists():
                path.unlink()

    def reset_dolma_contexts_file(self) -> None:
        if self.dolma_contexts_path.exists():
            self.dolma_contexts_path.unlink()

    def reset_analysis_files(self, include_concepts: bool = True) -> None:
        paths = [
            self.feature_relevance_path,
            self.selected_features_path,
            self.feature_correlations_path,
            self.feature_judge_input_path,
        ]
        if include_concepts:
            paths.append(self.feature_concepts_path)
        for path in paths:
            if path.exists():
                path.unlink()

    def append_alignment_row(self, row: dict[str, Any]) -> None:
        self.append_jsonl(self.alignment_path, row)

    def append_dolma_context_row(self, row: dict[str, Any]) -> None:
        self.append_jsonl(self.dolma_contexts_path, row)

    def write_summary(self, rows: Iterable[dict[str, Any]]) -> None:
        with self.summary_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_jsonl(self, path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
        target = Path(path)
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_manifest(self, payload: dict[str, Any]) -> None:
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def iter_records(self, path: str | Path | None = None) -> Iterable[dict[str, Any]]:
        source = Path(path) if path is not None else self.paired_path
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)

    def append_jsonl(self, path: str | Path, row: dict[str, Any]) -> None:
        target = Path(path)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
