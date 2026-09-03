"""Dependency-light mock extraction for schema and installation checks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from thesis_neuro.config import AppConfig
from thesis_neuro.datasets import DolmaStreamingAdapter
from thesis_neuro.storage import JsonlArtifactStore


class MockExtractionPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.dataset = DolmaStreamingAdapter(config)
        self.store = JsonlArtifactStore(config.output_dir)

    def run(self) -> dict[str, Any]:
        self.store.reset_run_files(include_summary=False)
        documents_seen = 0
        windows_seen = 0
        records_written = 0

        for document in self.dataset.stream_documents():
            if windows_seen >= self.config.dataset.max_windows:
                break
            documents_seen += 1
            tokens = document.text.split()
            if not tokens:
                continue
            token_ids = [1000 + index for index in range(len(tokens))]
            sample_id = f"{document.doc_id}:0:{len(tokens)}:0"
            for layer in self._layers():
                for position, (token, token_id) in enumerate(zip(tokens, token_ids)):
                    positive_id = token_ids[position + 1] if position + 1 < len(token_ids) else None
                    negative_id = token_ids[0] if positive_id != token_ids[0] else None
                    self.store.append_record(
                        {
                            "sample_id": sample_id,
                            "model_id": self.config.model.base_model_id,
                            "scope_release": self.config.model.scope_release,
                            "layer": layer,
                            "token_position": position,
                            "token_id": token_id,
                            "token": token,
                            "window_token_ids": token_ids,
                            "window_tokens": tokens,
                            "text": document.text,
                            "window_start": 0,
                            "window_end": len(tokens),
                            "latent_activations": self._latents(position),
                            "top_logits": self._logits(token_ids, tokens, position),
                            "chosen_positive_token_id": positive_id,
                            "chosen_negative_token_id": negative_id,
                            "provenance": {**document.provenance, "mock": True},
                        }
                    )
                    records_written += 1
            windows_seen += 1

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "mock",
            "config": self.config.to_dict(),
            "documents_seen": documents_seen,
            "windows_seen": windows_seen,
            "records_written": records_written,
            "artifacts": {
                "paired_records": str(self.store.paired_path),
                "manifest": str(self.store.manifest_path),
            },
        }
        if self.config.output.write_manifest:
            self.store.write_manifest(manifest)
        return manifest

    def _layers(self) -> list[int]:
        selection = self.config.model.layer_selection
        return [0] if selection == "all" else [int(layer) for layer in selection]

    def _logits(
        self,
        token_ids: list[int],
        tokens: list[str],
        position: int,
    ) -> list[dict[str, float | int | str]]:
        top_n = min(self.config.latents.top_n_logits, len(token_ids))
        return [
            {
                "token_id": token_ids[min(position + offset, len(token_ids) - 1)],
                "token": tokens[min(position + offset, len(tokens) - 1)],
                "logit": float(top_n - offset),
            }
            for offset in range(top_n)
        ]

    def _latents(self, position: int) -> list[dict[str, float | int]]:
        return [
            {
                "latent_id": position * 100 + offset,
                "activation": round(1.0 / (offset + 1), 4),
            }
            for offset in range(self.config.latents.token_top_k)
        ]
