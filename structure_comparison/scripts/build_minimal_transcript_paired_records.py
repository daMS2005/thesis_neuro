#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from structure_comparison.workflow import load_predictor_feature_keys, select_lm_target_feature_keys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a slim transcript_paired_records artifact for hidden-state runs.")
    parser.add_argument("--remote-run-dir", required=True, help="Directory containing full remote-run JSONL artifacts.")
    parser.add_argument("--output-path", required=True, help="Output minimal JSONL path.")
    parser.add_argument("--stimulus-id", action="append", required=True, help="Stimulus id to retain. Repeatable.")
    parser.add_argument("--targets-per-layer", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    remote_run_dir = Path(args.remote_run_dir)
    output_path = Path(args.output_path)
    stimulus_ids = {str(value) for value in args.stimulus_id}

    selected_features_path = remote_run_dir / "selected_features_for_alignment.jsonl"
    feature_concepts_path = remote_run_dir / "feature_concepts.jsonl"
    full_transcript_path = remote_run_dir / "transcript_paired_records.jsonl"

    predictor_keys = load_predictor_feature_keys(selected_features_path, top_k=None)
    family_layers = tuple(sorted({layer for layer, _feature_id in predictor_keys}))
    lm_target_keys = select_lm_target_feature_keys(
        feature_concepts_path=feature_concepts_path,
        predictor_keys=predictor_keys,
        targets_per_layer=int(args.targets_per_layer),
        target_layers=family_layers,
    )

    relevant_by_layer: dict[int, set[int]] = {}
    for layer, feature_id in (*predictor_keys, *lm_target_keys):
        relevant_by_layer.setdefault(int(layer), set()).add(int(feature_id))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept_rows = 0
    with full_transcript_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            provenance = row.get("provenance") or {}
            stimulus_id = provenance.get("stimulus_id")
            if stimulus_id not in stimulus_ids:
                continue
            layer = int(row["layer"])
            if layer not in relevant_by_layer:
                continue
            filtered_latents = []
            for latent in row.get("latent_activations", []):
                latent_id = int(latent["latent_id"])
                if latent_id in relevant_by_layer[layer]:
                    filtered_latents.append(
                        {
                            "latent_id": latent_id,
                            "activation": float(latent["activation"]),
                        }
                    )
            out_row = {
                "model_id": row["model_id"],
                "layer": layer,
                "window_start": int(row["window_start"]),
                "token_position": int(row["token_position"]),
                "token": row["token"],
                "provenance": {"stimulus_id": stimulus_id},
                "latent_activations": filtered_latents,
            }
            dst.write(json.dumps(out_row, separators=(",", ":")) + "\n")
            kept_rows += 1

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "stimulus_ids": sorted(stimulus_ids),
                "row_count": kept_rows,
                "predictor_feature_count": len(predictor_keys),
                "lm_target_feature_count": len(lm_target_keys),
                "layers": list(family_layers),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
