"""SAE feature extraction and answer-choice scoring over normalized benchmark items."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from benchmark_comparison.items import BenchmarkItem, load_items, write_jsonl
from benchmark_comparison.registry import ModelRegistryEntry
from benchmark_comparison.scoring import score_item_choices
from structure_comparison.artifacts import counts_by_layer, load_predictor_feature_keys
from structure_comparison.utils import feature_name, write_json
from thesis_neuro.config import load_app_config
from thesis_neuro.models import GemmaModelAdapter
from thesis_neuro.paths import default_config_path, repository_root
from thesis_neuro.sae import GemmaScopeAdapter


def extract_benchmark_features(
    model_entry: ModelRegistryEntry,
    items_path: str | Path,
    output_dir: str | Path,
    top_k: int | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    items = load_items(items_path)
    selected_keys = load_predictor_feature_keys(model_entry.selected_features_path, top_k=top_k)
    selected_layers = tuple(sorted({layer for layer, _feature_id in selected_keys}))
    feature_names = np.asarray([feature_name((layer, feature_id)) for layer, feature_id in selected_keys], dtype=str)
    feature_layers = np.asarray([layer for layer, _feature_id in selected_keys], dtype=int)

    feature_index_by_layer: dict[int, list[tuple[int, int]]] = {}
    for feature_index, (layer, feature_id) in enumerate(selected_keys):
        feature_index_by_layer.setdefault(int(layer), []).append((feature_index, int(feature_id)))

    config = load_app_config(
        config_path=default_config_path(),
        env_path=repository_root() / ".env",
    )
    config.model.base_model_id = model_entry.model_id
    config.model.scope_release = model_entry.scope_release
    config.model.scope_width = model_entry.scope_width
    config.model.layer_selection = list(selected_layers)
    config.env.hf_local_files_only = bool(local_files_only)

    model = GemmaModelAdapter(config)
    sae = GemmaScopeAdapter(config, model.device, model.dtype)

    output_root = Path(output_dir)
    intermediates_dir = output_root / "intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)

    sample_ids: list[str] = []
    task_names: list[str] = []
    split_names: list[str] = []
    benchmark_names: list[str] = []
    presence_rows: list[np.ndarray] = []
    mass_rows: list[np.ndarray] = []
    average_rows: list[np.ndarray] = []
    peak_rows: list[np.ndarray] = []
    count_rows: list[np.ndarray] = []
    score_rows: list[dict[str, Any]] = []

    for item in items:
        aggregations = _extract_item_feature_views(
            model=model,
            sae=sae,
            item=item,
            feature_index_by_layer=feature_index_by_layer,
            feature_count=len(selected_keys),
        )
        score_rows.append(score_item_choices(model, item))
        sample_ids.append(item.item_id)
        task_names.append(item.task)
        split_names.append(item.split)
        benchmark_names.append(item.benchmark)
        presence_rows.append(aggregations["presence"])
        mass_rows.append(aggregations["mass"])
        average_rows.append(aggregations["average"])
        peak_rows.append(aggregations["peak"])
        count_rows.append(aggregations["count"])

    sample_ids_array = np.asarray(sample_ids, dtype=str)
    task_names_array = np.asarray(task_names, dtype=str)
    split_names_array = np.asarray(split_names, dtype=str)
    benchmark_names_array = np.asarray(benchmark_names, dtype=str)

    _save_feature_npz(
        intermediates_dir / "item_feature_presence.npz",
        np.vstack(presence_rows).astype(np.uint8, copy=False),
        sample_ids_array,
        feature_names,
        feature_layers,
        task_names_array,
        split_names_array,
        benchmark_names_array,
    )
    _save_feature_npz(
        intermediates_dir / "item_feature_mass.npz",
        np.vstack(mass_rows).astype(np.float32, copy=False),
        sample_ids_array,
        feature_names,
        feature_layers,
        task_names_array,
        split_names_array,
        benchmark_names_array,
    )
    _save_feature_npz(
        intermediates_dir / "item_feature_average.npz",
        np.vstack(average_rows).astype(np.float32, copy=False),
        sample_ids_array,
        feature_names,
        feature_layers,
        task_names_array,
        split_names_array,
        benchmark_names_array,
    )
    _save_feature_npz(
        intermediates_dir / "item_feature_peak.npz",
        np.vstack(peak_rows).astype(np.float32, copy=False),
        sample_ids_array,
        feature_names,
        feature_layers,
        task_names_array,
        split_names_array,
        benchmark_names_array,
    )
    _save_feature_npz(
        intermediates_dir / "item_feature_count.npz",
        np.vstack(count_rows).astype(np.float32, copy=False),
        sample_ids_array,
        feature_names,
        feature_layers,
        task_names_array,
        split_names_array,
        benchmark_names_array,
    )
    write_jsonl(intermediates_dir / "item_scores.jsonl", score_rows)

    accuracy_values = [row["correct"] for row in score_rows if row["correct"] is not None]
    summary = {
        "model_name": model_entry.name,
        "model_id": model_entry.model_id,
        "items_path": str(Path(items_path)),
        "item_count": len(items),
        "feature_count": int(len(selected_keys)),
        "predictor_top_k": int(top_k) if top_k is not None else None,
        "counts_by_layer": counts_by_layer(selected_keys),
        "task_counts": _counts(task_names),
        "observed_choice_accuracy": float(np.mean(accuracy_values)) if accuracy_values else None,
        "artifacts": {
            "item_feature_presence": str(intermediates_dir / "item_feature_presence.npz"),
            "item_feature_mass": str(intermediates_dir / "item_feature_mass.npz"),
            "item_feature_average": str(intermediates_dir / "item_feature_average.npz"),
            "item_feature_peak": str(intermediates_dir / "item_feature_peak.npz"),
            "item_feature_count": str(intermediates_dir / "item_feature_count.npz"),
            "item_scores": str(intermediates_dir / "item_scores.jsonl"),
        },
    }
    write_json(intermediates_dir / "item_feature_summary.json", summary)
    return summary


def _extract_item_feature_views(
    model: GemmaModelAdapter,
    sae: GemmaScopeAdapter,
    item: BenchmarkItem,
    feature_index_by_layer: dict[int, list[tuple[int, int]]],
    feature_count: int,
) -> dict[str, np.ndarray]:
    token_ids = model.tokenize_document(item.feature_text)
    if not token_ids:
        raise ValueError(f"{item.item_id}: feature_text tokenized to an empty sequence.")
    window_len = min(model.max_context_window_tokens(), model.config.tokenization.seq_len)
    windows = model.make_windows(token_ids, metadata_mode="heuristic", window_len=window_len)

    presence = np.zeros(feature_count, dtype=bool)
    mass = np.zeros(feature_count, dtype=np.float64)
    peak = np.zeros(feature_count, dtype=np.float64)
    count = np.zeros(feature_count, dtype=np.float64)

    for window in windows:
        outputs, model_info = model.forward_outputs(window.input_ids, require_grad=False)
        hidden_states = outputs.hidden_states
        for layer in sae.available_layers(model_info.get("num_hidden_layers")):
            if layer not in feature_index_by_layer:
                continue
            residual = hidden_states[layer + 1]
            latents = (
                sae.encode_layer(layer, residual)
                .squeeze(0)
                .detach()
                .to("cpu")
                .float()
                .clamp_min(0)
                .numpy()
            )
            feature_rows = feature_index_by_layer[layer]
            feature_ids = [feature_id for _feature_index, feature_id in feature_rows]
            selected_values = latents[:, feature_ids]
            active_mask = selected_values > 0
            for local_index, (global_index, _feature_id) in enumerate(feature_rows):
                column = selected_values[:, local_index]
                mass[global_index] += float(column.sum())
                peak_value = float(column.max()) if column.size else 0.0
                peak[global_index] = max(float(peak[global_index]), peak_value)
                count[global_index] += float(active_mask[:, local_index].sum())
                if bool(active_mask[:, local_index].any()):
                    presence[global_index] = True

    average = np.zeros_like(mass, dtype=np.float64)
    np.divide(mass, count, out=average, where=count > 0)
    return {
        "presence": presence.astype(np.float32, copy=False),
        "mass": mass.astype(np.float32, copy=False),
        "average": average.astype(np.float32, copy=False),
        "peak": peak.astype(np.float32, copy=False),
        "count": count.astype(np.float32, copy=False),
    }


def _save_feature_npz(
    path: Path,
    values: np.ndarray,
    sample_ids: np.ndarray,
    feature_names: np.ndarray,
    feature_layers: np.ndarray,
    task_names: np.ndarray,
    split_names: np.ndarray,
    benchmark_names: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        values=values,
        sample_ids=sample_ids,
        feature_names=feature_names,
        layers=feature_layers,
        task_names=task_names,
        split_names=split_names,
        benchmark_names=benchmark_names,
    )


def _counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts
