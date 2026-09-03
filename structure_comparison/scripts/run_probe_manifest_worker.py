#!/usr/bin/env python3
"""Worker that executes a shard of probe jobs from a JSONL manifest and logs each outcome."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_neuro.config import load_app_config
from thesis_neuro.paths import output_root, repository_root
from thesis_neuro.probing import FeatureProbingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSONL probe manifest generated for the current experiment.",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--worker-log", default=None)
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _report_path(bundle: str, layer: int, feature_id: int) -> Path:
    return (
        output_root()
        / "probes"
        / bundle
        / "all_scripts"
        / f"layer_{int(layer)}"
        / f"feature_{int(feature_id)}"
        / "feature_probe_report.json"
    )


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    rows = _load_rows(manifest_path)
    if args.model_name:
        rows = [row for row in rows if row.get("model_name") == args.model_name]
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.shard_count > 1:
        rows = [row for idx, row in enumerate(rows) if idx % args.shard_count == args.shard_index]

    worker_log_path = Path(args.worker_log).expanduser().resolve() if args.worker_log else None
    if worker_log_path is not None:
        worker_log_path.parent.mkdir(parents=True, exist_ok=True)

    last_config_path: str | None = None
    config = None

    for index, row in enumerate(rows, start=1):
        bundle = Path(str(row["analysis_dir"])).name
        report_path = _report_path(bundle=bundle, layer=int(row["layer"]), feature_id=int(row["feature_id"]))
        status = {
            "index": index,
            "model_name": row["model_name"],
            "layer": int(row["layer"]),
            "feature_id": int(row["feature_id"]),
            "report_path": str(report_path),
        }

        if report_path.exists():
            status["status"] = "skipped_existing"
            _append_log(worker_log_path, status)
            continue

        config_path = str(row["config_path"])
        if config is None or config_path != last_config_path:
            config = load_app_config(config_path=config_path, env_path=repository_root() / ".env")
            last_config_path = config_path

        try:
            pipeline = FeatureProbingPipeline(
                config=config,
                analysis_dir=str(row["analysis_dir"]),
                transcript_dir=str(row["transcript_dir"]),
                dolma_dir=str(row["dolma_dir"]),
                alignment_path=str(row["alignment_path"]),
                layer=int(row["layer"]),
                feature_id=int(row["feature_id"]),
            )
            manifest = pipeline.run()
            status["status"] = "completed"
            status["manifest_path"] = str(pipeline.paths.manifest_path)
            status["rounds_written"] = manifest.get("rounds_written")
        except Exception as exc:  # noqa: BLE001
            status["status"] = "failed"
            status["error_type"] = type(exc).__name__
            status["error"] = str(exc)
        _append_log(worker_log_path, status)

    return 0


def _append_log(path: Path | None, row: dict) -> None:
    if path is None:
        print(json.dumps(row, ensure_ascii=False))
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
