from __future__ import annotations

import re
import subprocess

from thesis_neuro.paths import repository_root


def test_repository_root_is_checkout_root() -> None:
    root = repository_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "thesis_neuro").is_dir()


def test_public_source_has_no_personal_or_cluster_paths() -> None:
    root = repository_root()
    forbidden = re.compile(
        r"/(?:Users|home)/[A-Za-z0-9._-]+/|"
        r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|"
        r"\b(?:capstone\d+|gcp-(?:gpu|sae))\b",
        re.IGNORECASE,
    )
    offenders: list[str] = []
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    for raw_path in tracked:
        if not raw_path:
            continue
        path = root / raw_path.decode()
        if path.suffix not in {".py", ".json", ".md", ".sh", ".yaml", ".yml"}:
            continue
        if forbidden.search(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(str(path.relative_to(root)))
    assert offenders == []
