"""Feature-summary helpers that operate only on stored JSON records."""

from __future__ import annotations

import heapq
from typing import Any, Iterable


def summarize_feature_records(
    records: Iterable[dict[str, Any]],
    top_contexts: int = 5,
    context_radius: int = 4,
) -> list[dict[str, Any]]:
    leaders: dict[tuple[str, int], list[tuple[float, str, dict[str, Any]]]] = {}
    entry_counter = 0
    for record in records:
        tokens = record["window_tokens"]
        token_index = int(record["token_position"])
        start = max(0, token_index - context_radius)
        end = min(len(tokens), token_index + context_radius + 1)
        for feature in record["latent_activations"]:
            key = (f"layer_{record['layer']}", int(feature["latent_id"]))
            entry = {
                "sample_id": record["sample_id"],
                "token_index": token_index,
                "token": record["token"],
                "activation": float(feature["activation"]),
                "context_tokens": tokens[start:end],
                "text": record["text"],
            }
            heap = leaders.setdefault(key, [])
            heapq.heappush(heap, (entry["activation"], f"{record['sample_id']}:{entry_counter}", entry))
            if len(heap) > top_contexts:
                heapq.heappop(heap)
            entry_counter += 1

    return [
        {
            "layer": layer,
            "feature_id": feature_id,
            "top_contexts": [item[2] for item in sorted(heap, key=lambda item: item[0], reverse=True)],
        }
        for (layer, feature_id), heap in sorted(leaders.items())
    ]
