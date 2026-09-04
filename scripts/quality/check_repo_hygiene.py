#!/usr/bin/env python3
"""Fail when tracked files contain data, secrets, machine-specific paths, or notebook outputs."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_PREFIXES = ("data/", "outputs/")
FORBIDDEN_SUFFIXES = (".nii", ".nii.gz", ".npz", ".pyc", ".pt", ".safetensors")
MAX_FILE_BYTES = 2 * 1024 * 1024
SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
MACHINE_PATTERNS = (
    re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/"),
    re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
)
SELF = Path(__file__).resolve()  # exempt from the machine-path scan because it defines those patterns


def tracked_files() -> list[Path]:
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True)
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def inspect_file(path: Path) -> list[str]:
    relative = path.relative_to(ROOT).as_posix()
    issues: list[str] = []
    if relative.startswith(FORBIDDEN_PREFIXES):
        issues.append("research data or generated output is tracked")
    if relative.endswith(FORBIDDEN_SUFFIXES):
        issues.append("binary data or model artifact is tracked")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        issues.append(f"oversized file ({size} bytes)")
    if path.suffix == ".ipynb":
        issues.extend(_inspect_notebook(path))
    elif size <= MAX_FILE_BYTES:
        text = path.read_text(encoding="utf-8", errors="ignore")
        issues.extend(_inspect_text(text, check_machine_paths=path.resolve() != SELF))
    return issues


def _inspect_notebook(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    issues: list[str] = []
    for index, cell in enumerate(payload.get("cells", [])):
        if cell.get("execution_count") is not None:
            issues.append(f"cell {index} has an execution count")
        if cell.get("outputs"):
            issues.append(f"cell {index} has stored outputs")
        issues.extend(f"cell {index}: {issue}" for issue in _inspect_text("".join(cell.get("source", []))))
    return issues


def _inspect_text(text: str, check_machine_paths: bool = True) -> list[str]:
    issues = ["credential-like token" for pattern in SECRET_PATTERNS if pattern.search(text)]
    if check_machine_paths:
        issues.extend("machine-specific path or address" for pattern in MACHINE_PATTERNS if pattern.search(text))
    return issues


def main() -> int:
    paths = tracked_files()
    findings = [f"{path.relative_to(ROOT)}: {issue}" for path in paths for issue in inspect_file(path)]
    if findings:
        print("Repository hygiene check failed:")
        print("\n".join(f"- {finding}" for finding in findings))
        return 1
    print(f"Repository hygiene check passed for {len(paths)} tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
