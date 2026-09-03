from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from thesis_neuro.config import AppConfig


@dataclass(slots=True)
class RawDocument:
    doc_id: str
    text: str
    provenance: dict[str, Any]


class DolmaStreamingAdapter:
    def __init__(self, config: AppConfig, allow_local_text: bool = True) -> None:
        self.config = config
        self.allow_local_text = allow_local_text

    def stream_documents(self) -> Iterable[RawDocument]:
        if self.allow_local_text and self.config.dataset.local_text_path:
            yield self._load_local_text(Path(self.config.dataset.local_text_path))
            return

        from datasets import load_dataset

        iterable = load_dataset(
            self.config.dataset.id,
            split=self.config.dataset.split,
            streaming=self.config.dataset.streaming,
            token=self.config.env.hf_token,
            trust_remote_code=self.config.dataset.id == "allenai/dolma",
        )
        iterable = iterable.shuffle(
            seed=self.config.run.seed,
            buffer_size=self.config.dataset.shuffle_buffer_size,
        )

        seen = 0
        for example in iterable:
            if seen >= self.config.dataset.max_documents:
                break

            text = example.get(self.config.dataset.text_field)
            if not isinstance(text, str) or not text.strip():
                continue

            doc_id = str(
                example.get("id")
                or example.get("doc_id")
                or example.get("source")
                or f"doc_{seen}"
            )
            provenance = {
                key: value
                for key, value in example.items()
                if key != self.config.dataset.text_field
            }

            yield RawDocument(doc_id=doc_id, text=text, provenance=provenance)
            seen += 1

    def _load_local_text(self, path: Path) -> RawDocument:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Local text file is empty: {path}")

        return RawDocument(
            doc_id=path.stem,
            text=text,
            provenance={"source": "local_text", "path": str(path)},
        )


class TranscriptDirectoryAdapter:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def stream_documents(self) -> Iterable[RawDocument]:
        root = Path(self.config.transcripts.root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Transcript root not found: {root}")

        files = sorted(root.glob(self.config.transcripts.glob))
        seen = 0
        for path in files:
            if seen >= self.config.transcripts.max_files:
                break
            if not path.is_file():
                continue

            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue

            stimulus_id = path.parent.name
            doc_id = f"{stimulus_id}:{path.stem}"
            yield RawDocument(
                doc_id=doc_id,
                text=text,
                provenance={
                    "source": "transcript_directory",
                    "stimulus_id": stimulus_id,
                    "path": str(path),
                    "filename": path.name,
                },
            )
            seen += 1
