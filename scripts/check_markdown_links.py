#!/usr/bin/env python3
"""Check local links in tracked Markdown without making network requests."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def main() -> int:
    result = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    failures: list[str] = []
    for relative in result.stdout.splitlines():
        path = ROOT / relative
        for raw_target in LINK.findall(path.read_text(encoding="utf-8")):
            target = raw_target.split("#", 1)[0].strip()
            if not target or "://" in target or target.startswith(("mailto:", "#")):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                failures.append(f"{relative}: missing {raw_target}")
    if failures:
        print("\n".join(failures))
        return 1
    print("Tracked Markdown links resolve locally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
