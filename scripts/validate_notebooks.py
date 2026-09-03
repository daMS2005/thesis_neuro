#!/usr/bin/env python3
"""Validate tracked notebooks and require clean execution state."""

from __future__ import annotations

import subprocess
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    result = subprocess.run(
        ["git", "ls-files", "*.ipynb"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [ROOT / line for line in result.stdout.splitlines() if line]
    failures: list[str] = []
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        for index, cell in enumerate(notebook.cells):
            if cell.get("execution_count") is not None:
                failures.append(f"{path.relative_to(ROOT)}: cell {index} has execution_count")
            if cell.get("outputs"):
                failures.append(f"{path.relative_to(ROOT)}: cell {index} has outputs")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Validated {len(paths)} clean notebooks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
