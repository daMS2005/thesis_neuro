"""Log-probability scoring of answer choices for a single benchmark item."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from benchmark_comparison.items import BenchmarkItem


def score_item_choices(model: Any, item: BenchmarkItem) -> dict[str, Any]:
    prompt_ids = model.tokenize_document(item.score_prompt)
    if not prompt_ids:
        raise ValueError(f"{item.item_id}: score prompt tokenized to an empty sequence.")

    choice_sum_logprobs: list[float] = []
    choice_avg_logprobs: list[float] = []
    choice_token_counts: list[int] = []

    for choice_text in item.choices:
        choice_ids = model.tokenize_document(choice_text)
        if not choice_ids:
            raise ValueError(f"{item.item_id}: choice tokenized to an empty sequence: {choice_text!r}")
        combined_ids = prompt_ids + choice_ids
        outputs, _ = model.forward_outputs(combined_ids, require_grad=False)
        logits = outputs.logits[0]
        log_probs = torch.log_softmax(logits[:-1], dim=-1)

        prompt_len = len(prompt_ids)
        token_sum = 0.0
        for offset, token_id in enumerate(choice_ids):
            source_position = prompt_len - 1 + offset
            token_sum += float(log_probs[source_position, int(token_id)].item())
        choice_sum_logprobs.append(token_sum)
        choice_avg_logprobs.append(token_sum / float(len(choice_ids)))
        choice_token_counts.append(len(choice_ids))

    avg_scores = np.asarray(choice_avg_logprobs, dtype=float)
    predicted_choice = int(avg_scores.argmax())
    sorted_scores = np.sort(avg_scores)
    margin = float(sorted_scores[-1] - sorted_scores[-2]) if avg_scores.size >= 2 else 0.0

    correct_choice = int(item.correct_choice) if item.correct_choice is not None else None
    correct = None
    gold_sum = None
    gold_avg = None
    if correct_choice is not None:
        correct = float(predicted_choice == correct_choice)
        gold_sum = float(choice_sum_logprobs[correct_choice])
        gold_avg = float(choice_avg_logprobs[correct_choice])

    return {
        "item_id": item.item_id,
        "benchmark": item.benchmark,
        "task": item.task,
        "split": item.split,
        "predicted_choice": predicted_choice,
        "correct_choice": correct_choice,
        "correct": correct,
        "margin": margin,
        "predicted_choice_sum_logprob": float(choice_sum_logprobs[predicted_choice]),
        "predicted_choice_avg_logprob": float(choice_avg_logprobs[predicted_choice]),
        "gold_choice_sum_logprob": gold_sum,
        "gold_choice_avg_logprob": gold_avg,
        "choice_sum_logprobs": [float(value) for value in choice_sum_logprobs],
        "choice_avg_logprobs": [float(value) for value in choice_avg_logprobs],
        "choice_token_counts": [int(value) for value in choice_token_counts],
        "choice_texts": list(item.choices),
    }

