#!/usr/bin/env python3
"""Build a task-specific BIDS subset with materialized files."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

DATASET_METADATA = [
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "README",
    "CHANGES",
]


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def subject_labels_for_task(dataset_root: Path, task: str) -> list[str]:
    return sorted({path.parts[-3] for path in dataset_root.glob(f"sub-*/func/*task-{task}_bold.nii.gz")})


def copy_subject(dataset_root: Path, subset_root: Path, subject: str, task: str) -> dict[str, int]:
    counts = {"bold": 0, "anat": 0}
    src_sub = dataset_root / subject
    dst_sub = subset_root / subject

    for pattern in [f"func/*task-{task}_bold.nii.gz", f"func/*task-{task}_bold.json"]:
        for src in sorted(src_sub.glob(pattern)):
            copy_file(src, dst_sub / src.relative_to(src_sub))
            if src.name.endswith(".nii.gz"):
                counts["bold"] += 1

    for pattern in ["anat/*T1w.nii.gz", "anat/*T1w.json", "anat/*T2w.nii.gz", "anat/*T2w.json"]:
        for src in sorted(src_sub.glob(pattern)):
            copy_file(src, dst_sub / src.relative_to(src_sub))
            if src.suffixes[-2:] == [".nii", ".gz"]:
                counts["anat"] += 1

    return counts


def filter_participants_tsv(src: Path, dst: Path, keep_subjects: set[str]) -> None:
    if not src.exists():
        return
    with src.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        rows = [row for row in reader if row.get("participant_id") in keep_subjects]
        fieldnames = reader.fieldnames or ["participant_id"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--subset-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    subset_root = args.subset_root.resolve()
    subjects = subject_labels_for_task(dataset_root, args.task)

    if args.clean and subset_root.exists():
        shutil.rmtree(subset_root)
    subset_root.mkdir(parents=True, exist_ok=True)

    for name in DATASET_METADATA:
        src = dataset_root / name
        dst = subset_root / name
        if name == "participants.tsv":
            filter_participants_tsv(src, dst, set(subjects))
        elif src.exists():
            copy_file(src, dst)

    bold_count = 0
    anat_count = 0
    for subject in subjects:
        counts = copy_subject(dataset_root, subset_root, subject, args.task)
        bold_count += counts["bold"]
        anat_count += counts["anat"]

    report = {
        "dataset_root": str(dataset_root),
        "subset_root": str(subset_root),
        "task": args.task,
        "subject_count": len(subjects),
        "subjects": subjects,
        "bold_nii_count": bold_count,
        "anat_nii_count": anat_count,
    }
    (subset_root / "subset_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
