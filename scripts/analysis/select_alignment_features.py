"""Select the top judged features from feature_concepts.jsonl for the alignment stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select the top judged features for the alignment stage.")
    parser.add_argument("--concepts-path", required=True, type=Path, help="feature_concepts.jsonl written by analyze-features.")
    parser.add_argument("--output-path", required=True, type=Path, help="Where to write selected_features_for_alignment.jsonl.")
    parser.add_argument("--top-n", required=True, type=int, help="Number of judged features to keep.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    concepts_path = args.concepts_path
    output_path = args.output_path
    top_n = args.top_n

    rows = load_jsonl(concepts_path)
    judged = [row for row in rows if row.get("judge_status") == "ok"]
    judged.sort(
        key=lambda row: (
            int(row.get("transcript_relevance_rank", 10**9)),
            -float(row.get("transcript_relevance_score", 0.0)),
            int(row.get("layer", 10**9)),
            int(row.get("feature_id", 10**9)),
        )
    )

    selected = []
    for row in judged[:top_n]:
        judge_output = row.get("judge_output") or {}
        selected.append(
            {
                "layer": int(row["layer"]),
                "feature_id": int(row["feature_id"]),
                "transcript_relevance_rank": int(row["transcript_relevance_rank"]),
                "transcript_relevance_score": float(row["transcript_relevance_score"]),
                "judge_status": row.get("judge_status"),
                "conceptual_label": judge_output.get("conceptual_label"),
                "feature_type": judge_output.get("feature_type"),
                "confidence": judge_output.get("confidence"),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input": str(concepts_path),
                "output": str(output_path),
                "selected": len(selected),
                "requested_top_n": top_n,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
