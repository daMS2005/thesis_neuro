#!/usr/bin/env python3
"""Fail when public Git content contains machine-local or generated material."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PREFIXES = (
    ".local-backups/",
    "archive/",
    "data/",
    "eda_brain_data/assets/",
    "eda_brain_data/capstone_jobs/",
    "eda_brain_data/datasets/",
    "eda_brain_data/reports/",
    "ops/",
    "outputs/",
    "paper_revision/",
    "paper_rework/",
    "probe_runs/",
    "structure_comparison/brain_targets/",
    "structure_comparison/outputs/",
    "structure_comparison/remote_runs/",
)
FORBIDDEN_NAMES = {".DS_Store"}
FORBIDDEN_SUFFIXES = {".nii", ".nii.gz", ".npz", ".pyc", ".sbatch"}
MAX_FILE_BYTES = 2 * 1024 * 1024
SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
MACHINE_PATTERNS = (
    re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/"),
    re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"\b(?:capstone\d+|gcp-(?:gpu|sae))\b", re.IGNORECASE),
)


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def inspect_file(path: Path) -> list[str]:
    relative = path.relative_to(ROOT).as_posix()
    issues: list[str] = []
    if relative.startswith(FORBIDDEN_PREFIXES):
        issues.append("forbidden tracked path")
    if path.name in FORBIDDEN_NAMES:
        issues.append("forbidden filename")
    if any(relative.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
        issues.append("generated or binary artifact")
    if path.stat().st_size > MAX_FILE_BYTES:
        issues.append(f"oversized file ({path.stat().st_size} bytes)")
    if path.suffix == ".ipynb":
        issues.extend(_inspect_notebook(path))
    elif path.stat().st_size <= MAX_FILE_BYTES:
        text = path.read_text(encoding="utf-8", errors="ignore")
        issues.extend(_inspect_text(text, check_machine_paths=relative != "scripts/check_repo_hygiene.py"))
    return issues


def _inspect_notebook(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    issues: list[str] = []
    for index, cell in enumerate(payload.get("cells", [])):
        if cell.get("execution_count") is not None:
            issues.append(f"cell {index} has an execution count")
        if cell.get("outputs"):
            issues.append(f"cell {index} has stored outputs")
        source = "".join(cell.get("source", []))
        issues.extend(f"cell {index}: {issue}" for issue in _inspect_text(source))
    return issues


def _inspect_text(text: str, check_machine_paths: bool = True) -> list[str]:
    issues = ["credential-like token" for pattern in SECRET_PATTERNS if pattern.search(text)]
    if check_machine_paths:
        issues.extend("machine-specific path or host" for pattern in MACHINE_PATTERNS if pattern.search(text))
    return issues


def main() -> int:
    findings: list[str] = []
    for path in tracked_files():
        for issue in inspect_file(path):
            findings.append(f"{path.relative_to(ROOT)}: {issue}")
    if findings:
        print("Repository hygiene check failed:")
        print("\n".join(f"- {finding}" for finding in findings))
        return 1
    print(f"Repository hygiene check passed for {len(tracked_files())} tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
