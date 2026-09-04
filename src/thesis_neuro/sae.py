"""Sparse-autoencoder adapter that discovers and loads Gemma Scope and Llama Scope residual SAE releases."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from sae_lens import SAE
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

from thesis_neuro.config import AppConfig


class GemmaScopeAdapter:
    def __init__(self, config: AppConfig, device: str, dtype: torch.dtype) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        self._sae_cache: dict[int, Any] = {}
        self._available_layers: list[int] | None = None
        self._layer_sae_ids: dict[int, str] | None = None

    def available_layers(self, model_layer_count: int | None = None) -> list[int]:
        if self._available_layers is not None:
            return self._available_layers

        registered_layer_map = self._registered_release_layer_map()
        if registered_layer_map:
            self._layer_sae_ids = registered_layer_map
            self._available_layers = self._resolve_layer_selection(sorted(registered_layer_map))
            return self._available_layers

        repo_id = self._infer_repo_id(self.config.model.scope_release)
        files = self._list_repo_files(repo_id)

        pattern = re.compile(rf"(?:^|/)layer_(\d+)/{re.escape(self.config.model.scope_width)}/")
        discovered = sorted({int(match.group(1)) for path in files if (match := pattern.search(path))})

        if not discovered:
            if model_layer_count is None:
                raise RuntimeError(
                    f"Could not discover Gemma Scope layers in repo {repo_id} for width {self.config.model.scope_width}"
                )
            discovered = list(range(model_layer_count))

        self._available_layers = self._resolve_layer_selection(discovered)
        return self._available_layers

    def encode_layer(self, layer_idx: int, residual: torch.Tensor) -> torch.Tensor:
        sae = self._load_sae(layer_idx)
        return sae.encode(residual.to(self.device, dtype=self.dtype))

    def decoder_vector(self, layer_idx: int, feature_id: int) -> torch.Tensor:
        sae = self._load_sae(layer_idx)
        if feature_id < 0 or feature_id >= sae.W_dec.shape[0]:
            raise ValueError(f"Feature id {feature_id} is out of range for layer {layer_idx}.")
        return sae.W_dec[int(feature_id)].detach().to(self.device, dtype=self.dtype)

    def select_top_latents(
        self,
        token_latents: torch.Tensor,
        top_k: int,
    ) -> list[dict[str, float | int]]:
        k = min(top_k, token_latents.shape[-1])
        values, indices = torch.topk(token_latents, k=k, dim=-1)

        records: list[dict[str, float | int]] = []
        for activation, feature_idx in zip(values.tolist(), indices.tolist()):
            if activation <= 0:
                continue

            records.append(
                {
                    "latent_id": int(feature_idx),
                    "activation": float(activation),
                }
            )
        return records

    def _load_sae(self, layer_idx: int) -> Any:
        if layer_idx in self._sae_cache:
            return self._sae_cache[layer_idx]

        sae, _, _ = SAE.from_pretrained(
            release=self._pretrained_release_name(),
            sae_id=self._make_sae_id(layer_idx),
            device=self.device,
        )
        sae.to(dtype=self.dtype)
        sae.eval()
        self._sae_cache[layer_idx] = sae
        return sae

    def prefetch_layers(self, model_layer_count: int | None = None) -> list[str]:
        cached_sae_ids: list[str] = []
        for layer_idx in self.available_layers(model_layer_count=model_layer_count):
            self._load_sae(layer_idx)
            cached_sae_ids.append(self._make_sae_id(layer_idx))
        return cached_sae_ids

    def _make_sae_id(self, layer_idx: int) -> str:
        if self._layer_sae_ids is None:
            self.available_layers()
        if self._layer_sae_ids is not None and layer_idx in self._layer_sae_ids:
            return self._layer_sae_ids[layer_idx]
        return f"layer_{layer_idx}/{self.config.model.scope_width}/canonical"

    def _resolve_layer_selection(self, discovered_layers: list[int]) -> list[int]:
        selection = self.config.model.layer_selection
        if selection == "all":
            return discovered_layers
        if isinstance(selection, list):
            requested = set(int(layer) for layer in selection)
            missing = sorted(requested.difference(discovered_layers))
            if missing:
                raise ValueError(f"Requested layers not found in Gemma Scope release: {missing}")
            return [layer for layer in discovered_layers if layer in requested]
        raise ValueError(f"Unsupported layer_selection value: {selection}")

    @staticmethod
    def _infer_repo_id(scope_release: str) -> str:
        release_name = scope_release.removesuffix("-canonical")
        if "/" in release_name:
            return release_name
        return f"google/{release_name}"

    def _pretrained_release_name(self) -> str:
        directory = get_pretrained_saes_directory()
        for candidate in self._release_aliases():
            if candidate in directory:
                return candidate
        return self._infer_repo_id(self.config.model.scope_release)

    def _registered_release_layer_map(self) -> dict[int, str]:
        directory = get_pretrained_saes_directory()
        release = next((candidate for candidate in self._release_aliases() if candidate in directory), None)
        if release is None:
            return {}

        width_segment = f"/{self.config.model.scope_width}/"
        layer_map: dict[int, str] = {}
        for sae_id in directory[release].saes_map:
            match = self._parse_registered_layer_sae_id(sae_id, width_segment)
            if match is None:
                continue
            layer_idx, normalized_sae_id = match
            layer_map[layer_idx] = normalized_sae_id
        return dict(sorted(layer_map.items()))

    def _release_aliases(self) -> list[str]:
        release = self.config.model.scope_release
        aliases: list[str] = []
        for candidate in (release, release.split("/")[-1]):
            if candidate not in aliases:
                aliases.append(candidate)
        return aliases

    @staticmethod
    def _parse_registered_layer_sae_id(sae_id: str, width_segment: str) -> tuple[int, str] | None:
        canonical_match = re.match(r"layer_(\d+)/", sae_id)
        if canonical_match and width_segment in sae_id:
            return int(canonical_match.group(1)), sae_id

        llama_residual_match = re.match(r"l(\d+)r_[0-9]+x$", sae_id)
        if llama_residual_match:
            return int(llama_residual_match.group(1)), sae_id

        return None

    def _list_repo_files(self, repo_id: str) -> list[str]:
        if not self.config.env.hf_local_files_only:
            api = HfApi(token=self.config.env.hf_token)
            return api.list_repo_files(repo_id=repo_id, repo_type="model")

        cached_files = self._list_cached_repo_files(repo_id)
        if cached_files:
            return cached_files

        raise RuntimeError(
            f"No cached files found for {repo_id} while HF_LOCAL_FILES_ONLY=true. "
            "Run prefetch first."
        )

    def _list_cached_repo_files(self, repo_id: str) -> list[str]:
        snapshots_dir = self._hf_hub_root() / self._repo_cache_dir_name(repo_id) / "snapshots"
        if not snapshots_dir.exists():
            return []

        cached_paths: list[str] = []
        for snapshot_dir in snapshots_dir.iterdir():
            if not snapshot_dir.is_dir():
                continue
            for file_path in snapshot_dir.rglob("*"):
                if file_path.is_file():
                    cached_paths.append(str(file_path.relative_to(snapshot_dir)))
        return cached_paths

    def _hf_hub_root(self) -> Path:
        if self.config.env.hf_home:
            return Path(self.config.env.hf_home) / "hub"
        return Path.home() / ".cache" / "huggingface" / "hub"

    @staticmethod
    def _repo_cache_dir_name(repo_id: str) -> str:
        return "models--" + repo_id.replace("/", "--")
