#!/usr/bin/env python3
"""Audit BIDS inputs for a task-specific fMRIPrep launch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from thesis_neuro.paths import data_root


def _sorted_relpaths(paths: Iterable[Path], root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in paths)


def _existing(paths: Iterable[Path]) -> list[Path]:
    return sorted(path for path in paths if path.exists())


def _missing(paths: Iterable[Path]) -> list[Path]:
    return sorted(path for path in paths if not path.exists())


def build_report(dataset_root: Path, task: str) -> dict:
    subjects = sorted({path.parts[-3] for path in dataset_root.glob(f"sub-*/func/*task-{task}_bold.nii.gz")})
    report: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "task": task,
        "subject_count": len(subjects),
        "subjects": subjects,
        "missing": {
            "bold_nii": {},
            "bold_json": {},
            "t1w_nii": {},
            "t1w_json": {},
            "t2w_nii": {},
            "t2w_json": {},
        },
        "optional": {
            "subjects_with_t2w": [],
        },
        "unrelated_broken_symlinks_in_subject_dirs": [],
    }

    unrelated_broken: list[str] = []

    for subject in subjects:
        subject_root = dataset_root / subject
        func_dir = subject_root / "func"
        anat_dir = subject_root / "anat"

        bold_nii = sorted(func_dir.glob(f"*task-{task}_bold.nii.gz"))
        bold_json = sorted(func_dir.glob(f"*task-{task}_bold.json"))
        t1w_nii = sorted(anat_dir.glob("*T1w.nii.gz"))
        t1w_json = sorted(anat_dir.glob("*T1w.json"))
        t2w_nii = sorted(anat_dir.glob("*T2w.nii.gz"))
        t2w_json = sorted(anat_dir.glob("*T2w.json"))

        if not bold_nii or _missing(bold_nii):
            report["missing"]["bold_nii"][subject] = _sorted_relpaths(_missing(bold_nii) or [func_dir / f"NO_task-{task}_bold.nii.gz"], dataset_root)
        if not bold_json or _missing(bold_json):
            report["missing"]["bold_json"][subject] = _sorted_relpaths(_missing(bold_json) or [func_dir / f"NO_task-{task}_bold.json"], dataset_root)
        if not t1w_nii or _missing(t1w_nii):
            report["missing"]["t1w_nii"][subject] = _sorted_relpaths(_missing(t1w_nii) or [anat_dir / "NO_T1w.nii.gz"], dataset_root)
        if not t1w_json or _missing(t1w_json):
            report["missing"]["t1w_json"][subject] = _sorted_relpaths(_missing(t1w_json) or [anat_dir / "NO_T1w.json"], dataset_root)
        if t2w_nii:
            report["optional"]["subjects_with_t2w"].append(subject)
        if _missing(t2w_nii):
            report["missing"]["t2w_nii"][subject] = _sorted_relpaths(_missing(t2w_nii), dataset_root)
        if t2w_nii and _missing(t2w_json):
            report["missing"]["t2w_json"][subject] = _sorted_relpaths(_missing(t2w_json), dataset_root)

        for path in subject_root.rglob("*"):
            if path.is_symlink() and not path.exists():
                if f"task-{task}_" not in path.name and "/anat/" not in str(path):
                    unrelated_broken.append(str(path.relative_to(dataset_root)))

    report["unrelated_broken_symlinks_in_subject_dirs"] = sorted(unrelated_broken)
    missing_counts = {
        key: len(value) for key, value in report["missing"].items()
    }
    report["summary"] = {
        "missing_subject_counts": missing_counts,
        "relevant_inputs_complete": sum(missing_counts.values()) == 0,
        "unrelated_broken_symlink_count": len(unrelated_broken),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Root BIDS dataset directory (default: $THESIS_NEURO_DATA_ROOT/openneuro/ds002345).",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="BIDS task label to audit, for example shapesphysical.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for a JSON audit report.",
    )
    parser.add_argument(
        "--subject-list-out",
        type=Path,
        help="Optional path to write newline-delimited participant labels without the sub- prefix.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any relevant task input is missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dataset_root is None:
        args.dataset_root = data_root() / "openneuro" / "ds002345"
    report = build_report(args.dataset_root.resolve(), args.task)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.subject_list_out:
        args.subject_list_out.parent.mkdir(parents=True, exist_ok=True)
        lines = [subject.removeprefix("sub-") for subject in report["subjects"]]
        args.subject_list_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))

    if args.strict and not report["summary"]["relevant_inputs_complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
