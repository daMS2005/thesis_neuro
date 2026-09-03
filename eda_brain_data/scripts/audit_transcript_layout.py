#!/usr/bin/env python3
"""Audit that each story directory contains the expected transcript artifacts."""

import argparse
from pathlib import Path

from thesis_neuro.paths import data_root

ACTIVE_STORIES = [
    "black",
    "bronx",
    "forgot",
    "piemanpni",
    "shapesphysical",
    "shapessocial",
]


def expected_files(slug: str) -> set[str]:
    return {
        f"{slug}.srt",
        f"{slug}_words.tsv",
        f"{slug}_transcript.txt",
        f"{slug}_tr_aligned.tsv",
        f"{slug}_phonemes.tsv",
        f"{slug}_phonemes_by_tr.tsv",
        "gentle_align.json",
        "metadata.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the expected transcript artifacts for each story.")
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=None,
        help="Transcript directory (default: $THESIS_NEURO_DATA_ROOT/transcripts).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    transcripts_dir = (args.transcripts_dir or data_root() / "transcripts").expanduser().resolve()

    failures = 0
    for slug in ACTIVE_STORIES:
        story_dir = transcripts_dir / slug
        if not story_dir.exists():
            print(f"[missing-dir] {slug}: {story_dir}")
            failures += 1
            continue

        files = {path.name for path in story_dir.iterdir()}
        expected = expected_files(slug)
        missing = sorted(expected - files)
        extra = sorted(files - expected)

        if missing or extra:
            failures += 1
            print(f"[mismatch] {slug}")
            if missing:
                print(f"  missing: {', '.join(missing)}")
            if extra:
                print(f"  extra: {', '.join(extra)}")
        else:
            print(f"[ok] {slug}")

    if failures:
        raise SystemExit(1)

    print("Active transcript layout is consistent.")


if __name__ == "__main__":
    main()
