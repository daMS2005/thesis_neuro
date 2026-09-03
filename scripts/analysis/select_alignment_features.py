"""Select the top judged features from feature_concepts.jsonl for the alignment stage."""

from __future__ import annotations

import json
import sys
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


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: select_alignment_features.py <feature_concepts.jsonl> <output.jsonl> <top_n>")

    concepts_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    top_n = int(sys.argv[3])

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
