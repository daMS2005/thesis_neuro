from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_SUPERGLUE_TASKS = ("boolq", "cb", "copa", "multirc", "rte", "wic", "wsc")


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    item_id: str
    benchmark: str
    task: str
    split: str
    feature_text: str
    score_prompt: str
    choices: tuple[str, ...]
    correct_choice: int | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "benchmark": self.benchmark,
            "task": self.task,
            "split": self.split,
            "feature_text": self.feature_text,
            "score_prompt": self.score_prompt,
            "choices": list(self.choices),
            "correct_choice": self.correct_choice,
            "metadata": self.metadata,
        }


def load_items(path: str | Path) -> list[BenchmarkItem]:
    rows = read_jsonl(path)
    items = [BenchmarkItem(**{**row, "choices": tuple(row["choices"])}) for row in rows]
    validate_items(items)
    return items


def write_items(path: str | Path, items: Iterable[BenchmarkItem]) -> None:
    rows = [item.to_dict() for item in items]
    write_jsonl(path, rows)


def validate_items(items: Iterable[BenchmarkItem]) -> None:
    seen_ids: set[str] = set()
    for item in items:
        if not item.item_id:
            raise ValueError("Every benchmark item needs a non-empty item_id.")
        if item.item_id in seen_ids:
            raise ValueError(f"Duplicate item_id found: {item.item_id}")
        seen_ids.add(item.item_id)
        if not item.feature_text.strip():
            raise ValueError(f"{item.item_id}: feature_text must not be empty.")
        if not item.score_prompt.strip():
            raise ValueError(f"{item.item_id}: score_prompt must not be empty.")
        if not item.choices:
            raise ValueError(f"{item.item_id}: choices must not be empty.")
        for choice in item.choices:
            if not isinstance(choice, str) or not choice.strip():
                raise ValueError(f"{item.item_id}: every choice must be a non-empty string.")
        if item.correct_choice is not None and not (0 <= int(item.correct_choice) < len(item.choices)):
            raise ValueError(f"{item.item_id}: correct_choice is out of range.")


def prepare_superglue_items(
    task: str,
    split: str,
    output_path: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    if task not in SUPPORTED_SUPERGLUE_TASKS:
        supported = ", ".join(SUPPORTED_SUPERGLUE_TASKS)
        raise ValueError(f"Unsupported SuperGLUE task '{task}'. Supported tasks: {supported}")

    from datasets import load_dataset

    try:
        dataset = load_dataset("super_glue", task, split=split)
    except (PermissionError, ValueError, OSError) as exc:
        dataset = _load_cached_superglue_split(task=task, split=split, original_error=exc)
    items: list[BenchmarkItem] = []
    for index, example in enumerate(dataset):
        items.extend(_normalize_superglue_example(task=task, split=split, index=index, example=example))
        if limit is not None and len(items) >= limit:
            items = items[:limit]
            break

    validate_items(items)
    write_items(output_path, items)
    return {
        "benchmark": "super_glue",
        "task": task,
        "split": split,
        "item_count": len(items),
        "output_path": str(Path(output_path)),
        "supported_tasks": list(SUPPORTED_SUPERGLUE_TASKS),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _normalize_superglue_example(
    task: str,
    split: str,
    index: int,
    example: dict[str, Any],
) -> list[BenchmarkItem]:
    if task == "boolq":
        label = _binary_true_false_to_yes_no_choice(example.get("label"))
        feature_text = (
            f"Passage:\n{example['passage']}\n\n"
            f"Question: {example['question']}\n"
            "Respond with yes or no."
        )
        score_prompt = (
            f"Passage:\n{example['passage']}\n\n"
            f"Question: {example['question']}\n"
            "Options:\nA. yes\nB. no\n"
            "Answer:"
        )
        return [
            BenchmarkItem(
                item_id=f"super_glue:{task}:{split}:{index}",
                benchmark="super_glue",
                task=task,
                split=split,
                feature_text=feature_text,
                score_prompt=score_prompt,
                choices=(" yes", " no"),
                correct_choice=label,
                metadata={"question": example["question"]},
            )
        ]

    if task == "rte":
        label = int(example["label"]) if example.get("label", -1) != -1 else None
        feature_text = (
            f"Premise:\n{example['premise']}\n\n"
            f"Hypothesis:\n{example['hypothesis']}\n"
            "Does the premise entail the hypothesis?"
        )
        score_prompt = (
            f"Premise:\n{example['premise']}\n\n"
            f"Hypothesis:\n{example['hypothesis']}\n"
            "Does the premise entail the hypothesis?\n"
            "Options:\nA. yes\nB. no\n"
            "Answer:"
        )
        return [
            BenchmarkItem(
                item_id=f"super_glue:{task}:{split}:{index}",
                benchmark="super_glue",
                task=task,
                split=split,
                feature_text=feature_text,
                score_prompt=score_prompt,
                choices=(" yes", " no"),
                correct_choice=label,
                metadata={"premise": example["premise"], "hypothesis": example["hypothesis"]},
            )
        ]

    if task == "cb":
        label = int(example["label"]) if example.get("label", -1) != -1 else None
        feature_text = (
            f"Premise:\n{example['premise']}\n\n"
            f"Hypothesis:\n{example['hypothesis']}\n"
            "What is the relation between the premise and the hypothesis?"
        )
        score_prompt = (
            f"Premise:\n{example['premise']}\n\n"
            f"Hypothesis:\n{example['hypothesis']}\n"
            "What is the relation?\n"
            "Options:\nA. entailment\nB. contradiction\nC. neutral\n"
            "Answer:"
        )
        return [
            BenchmarkItem(
                item_id=f"super_glue:{task}:{split}:{index}",
                benchmark="super_glue",
                task=task,
                split=split,
                feature_text=feature_text,
                score_prompt=score_prompt,
                choices=(" entailment", " contradiction", " neutral"),
                correct_choice=label,
                metadata={"premise": example["premise"], "hypothesis": example["hypothesis"]},
            )
        ]

    if task == "copa":
        label = int(example["label"]) if example.get("label", -1) != -1 else None
        question = "cause" if str(example["question"]).strip().lower() == "cause" else "effect"
        feature_text = (
            f"Premise:\n{example['premise']}\n\n"
            f"What was the most likely {question}?"
        )
        score_prompt = (
            f"Premise:\n{example['premise']}\n\n"
            f"What was the most likely {question}?\n"
            f"Options:\nA. {example['choice1']}\nB. {example['choice2']}\n"
            "Answer:"
        )
        return [
            BenchmarkItem(
                item_id=f"super_glue:{task}:{split}:{index}",
                benchmark="super_glue",
                task=task,
                split=split,
                feature_text=feature_text,
                score_prompt=score_prompt,
                choices=(f" {example['choice1']}", f" {example['choice2']}"),
                correct_choice=label,
                metadata={"premise": example["premise"], "question": question},
            )
        ]

    if task == "wic":
        label = _binary_true_false_to_yes_no_choice(example.get("label"))
        feature_text = (
            f"Word: {example['word']}\n"
            f"Sentence 1: {example['sentence1']}\n"
            f"Sentence 2: {example['sentence2']}\n"
            "Does the word have the same meaning in both sentences?"
        )
        score_prompt = (
            f"Word: {example['word']}\n"
            f"Sentence 1: {example['sentence1']}\n"
            f"Sentence 2: {example['sentence2']}\n"
            "Does the word have the same meaning in both sentences?\n"
            "Options:\nA. yes\nB. no\n"
            "Answer:"
        )
        return [
            BenchmarkItem(
                item_id=f"super_glue:{task}:{split}:{index}",
                benchmark="super_glue",
                task=task,
                split=split,
                feature_text=feature_text,
                score_prompt=score_prompt,
                choices=(" yes", " no"),
                correct_choice=label,
                metadata={"word": example["word"]},
            )
        ]

    if task == "wsc":
        label = _binary_true_false_to_yes_no_choice(example.get("label"))
        feature_text = (
            f"Text:\n{example['text']}\n\n"
            f"Candidate referent: {example['span1_text']}\n"
            f"Pronoun span: {example['span2_text']}\n"
            "Does the pronoun refer to the candidate referent?"
        )
        score_prompt = (
            f"Text:\n{example['text']}\n\n"
            f"Candidate referent: {example['span1_text']}\n"
            f"Pronoun span: {example['span2_text']}\n"
            "Does the pronoun refer to the candidate referent?\n"
            "Options:\nA. yes\nB. no\n"
            "Answer:"
        )
        return [
            BenchmarkItem(
                item_id=f"super_glue:{task}:{split}:{index}",
                benchmark="super_glue",
                task=task,
                split=split,
                feature_text=feature_text,
                score_prompt=score_prompt,
                choices=(" yes", " no"),
                correct_choice=label,
                metadata={"span1_text": example["span1_text"], "span2_text": example["span2_text"]},
            )
        ]

    if task == "multirc":
        label = _binary_true_false_to_yes_no_choice(example.get("label"))
        feature_text = (
            f"Paragraph:\n{example['paragraph']}\n\n"
            f"Question: {example['question']}\n"
            f"Candidate answer: {example['answer']}\n"
            "Is the candidate answer correct?"
        )
        score_prompt = (
            f"Paragraph:\n{example['paragraph']}\n\n"
            f"Question: {example['question']}\n"
            f"Candidate answer: {example['answer']}\n"
            "Is the candidate answer correct?\n"
            "Options:\nA. yes\nB. no\n"
            "Answer:"
        )
        paragraph_id = example.get("idx", {}).get("paragraph", index)
        question_id = example.get("idx", {}).get("question", 0)
        answer_id = example.get("idx", {}).get("answer", 0)
        return [
            BenchmarkItem(
                item_id=f"super_glue:{task}:{split}:{paragraph_id}:{question_id}:{answer_id}",
                benchmark="super_glue",
                task=task,
                split=split,
                feature_text=feature_text,
                score_prompt=score_prompt,
                choices=(" yes", " no"),
                correct_choice=label,
                metadata={
                    "paragraph_id": paragraph_id,
                    "question_id": question_id,
                    "answer_id": answer_id,
                },
            )
        ]

    raise ValueError(f"Unhandled SuperGLUE task: {task}")


def _binary_true_false_to_yes_no_choice(raw_label: Any) -> int | None:
    if raw_label is None:
        return None
    label = int(raw_label)
    if label == -1:
        return None
    if label not in (0, 1):
        raise ValueError(f"Expected a binary label in {{0, 1, -1}}, got {raw_label!r}")
    return 0 if label == 1 else 1


def _load_cached_superglue_split(task: str, split: str, original_error: Exception):
    from datasets import Dataset

    arrow_path = _resolve_cached_superglue_arrow(task=task, split=split)
    if arrow_path is None:
        raise original_error
    return Dataset.from_file(str(arrow_path))


def _resolve_cached_superglue_arrow(task: str, split: str) -> Path | None:
    default_root = Path.home() / ".cache" / "huggingface" / "datasets"
    task_root = default_root / "super_glue" / task / "0.0.0"
    if not task_root.exists():
        return None
    candidate_dirs = sorted(path for path in task_root.iterdir() if path.is_dir())
    for candidate_dir in reversed(candidate_dirs):
        arrow_path = candidate_dir / f"super_glue-{split}.arrow"
        if arrow_path.exists():
            return arrow_path
    return None
