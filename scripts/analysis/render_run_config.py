"""Render a model-specific run config by resolving SAE layers at matched relative depths."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

DEFAULT_LAYER_FRACTIONS = (0.16, 0.30, 0.50, 0.65, 0.85, 0.96)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--config-out", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--scope-release", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--scope-width", default="width_16k")
    parser.add_argument(
        "--layer-fractions",
        default=",".join(str(value) for value in DEFAULT_LAYER_FRACTIONS),
        help="Comma-separated relative depth targets in [0, 1].",
    )
    return parser.parse_args()


def normalize_scope_repo(scope_release: str) -> str:
    release_name = scope_release.removesuffix("-canonical")
    if "/" in release_name:
        return release_name
    return f"google/{release_name}"


def parse_layer_fractions(raw_value: str) -> list[float]:
    values = [float(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one layer fraction is required")
    for value in values:
        if value < 0 or value > 1:
            raise ValueError(f"Layer fractions must be within [0, 1], got {value}")
    return values


def resolve_target_layers(num_hidden_layers: int, fractions: list[float]) -> list[int]:
    max_index = max(0, int(num_hidden_layers) - 1)
    resolved: list[int] = []
    for fraction in fractions:
        candidate = int(round(fraction * num_hidden_layers))
        candidate = min(max(candidate, 0), max_index)
        if candidate not in resolved:
            resolved.append(candidate)
    return resolved


def parse_registered_layer_sae_id(sae_id: str, scope_width: str) -> int | None:
    width_segment = f"/{scope_width}/"
    canonical_match = re.match(r"layer_(\d+)/", sae_id)
    if canonical_match and width_segment in sae_id:
        return int(canonical_match.group(1))

    llama_residual_match = re.match(r"l(\d+)r_[0-9]+x$", sae_id)
    if llama_residual_match:
        return int(llama_residual_match.group(1))

    return None


def discover_sae_layers(scope_release: str, repo_id: str, scope_width: str, token: str | None) -> list[int]:
    from huggingface_hub import HfApi
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

    directory = get_pretrained_saes_directory()
    if scope_release in directory:
        layers: set[int] = set()
        for sae_id in directory[scope_release].saes_map:
            layer = parse_registered_layer_sae_id(sae_id, scope_width)
            if layer is not None:
                layers.add(layer)
        if layers:
            return sorted(layers)

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    pattern = re.compile(rf"(?:^|/)layer_(\d+)/{re.escape(scope_width)}/")
    discovered = sorted({int(match.group(1)) for path in files if (match := pattern.search(path))})
    if not discovered:
        raise RuntimeError(
            f"Could not discover SAE layers in {repo_id} for width {scope_width}"
        )
    return discovered


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file, override=False)

    base_path = Path(args.base_config)
    output_path = Path(args.config_out)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    raw.setdefault("model", {})
    raw.setdefault("alignment", {})
    raw.setdefault("analysis", {})
    raw.setdefault("output", {})

    hf_token = os.getenv("HF_TOKEN")
    local_only = os.getenv("HF_LOCAL_FILES_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}

    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(
        args.model_id,
        token=hf_token,
        local_files_only=local_only,
    )
    num_hidden_layers = int(getattr(model_config, "num_hidden_layers"))

    fractions = parse_layer_fractions(args.layer_fractions)
    requested_layers = resolve_target_layers(num_hidden_layers, fractions)

    scope_repo = normalize_scope_repo(args.scope_release)
    discovered_layers = discover_sae_layers(
        scope_release=args.scope_release,
        repo_id=scope_repo,
        scope_width=args.scope_width,
        token=hf_token,
    )
    missing_layers = [layer for layer in requested_layers if layer not in discovered_layers]
    if missing_layers:
        raise RuntimeError(
            f"Requested layers not found in {scope_repo}: {missing_layers}. "
            f"Discovered layers start/end: {discovered_layers[:5]} ... {discovered_layers[-5:]}"
        )

    raw["model"]["base_model_id"] = args.model_id
    raw["model"]["scope_release"] = args.scope_release
    raw["model"]["scope_width"] = args.scope_width
    raw["model"]["layer_selection"] = requested_layers
    raw["alignment"]["top_features_per_window"] = 128
    raw["analysis"]["top_features_for_alignment"] = 128
    raw["output"]["dir"] = str(output_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "config_out": str(output_path),
                "model_id": args.model_id,
                "scope_release": args.scope_release,
                "output_dir": str(output_dir),
                "num_hidden_layers": num_hidden_layers,
                "layer_selection": requested_layers,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
