#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path

import requests

from thesis_neuro.paths import data_root

PUNCT_RE = re.compile(r"\s+([.,!?;:])")


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(str(part) for part in cmd))
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True)


def normalize_text(text: str) -> str:
    return PUNCT_RE.sub(r"\1", re.sub(r"\s+", " ", text)).strip()


def load_metadata(story_dir: Path) -> dict:
    return json.loads((story_dir / "metadata.json").read_text())


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(rows: list[dict], path: Path, fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ensure_transcript_text(story_dir: Path, slug: str) -> Path:
    transcript_path = story_dir / f"{slug}_transcript.txt"
    if transcript_path.exists():
        return transcript_path

    if slug == "alice":
        original = story_dir / "alice_original_transcript.txt"
        if original.exists():
            text = original.read_text()
            transcript_path.write_text(text)
            return transcript_path

    words_path = story_dir / f"{slug}_words.tsv"
    if not words_path.exists():
        raise FileNotFoundError(f"Could not find a transcript source for {slug}")
    rows = read_tsv(words_path)
    text = normalize_text(" ".join(row["word"] for row in rows if row.get("word")))
    transcript_path.write_text(text + "\n")
    return transcript_path


def docker_container_state(name: str) -> tuple[bool, bool]:
    running = run(["docker", "ps", "--format", "{{.Names}}"], check=False)
    all_containers = run(["docker", "ps", "-a", "--format", "{{.Names}}"], check=False)
    running_names = set(running.stdout.splitlines())
    all_names = set(all_containers.stdout.splitlines())
    return name in running_names, name in all_names


def ensure_gentle_server(image: str, name: str, port: int):
    running, exists = docker_container_state(name)
    if running:
        return
    if exists:
        run(["docker", "start", name])
    else:
        run(["docker", "run", "-d", "--name", name, "-p", f"{port}:8765", image])


def wait_for_gentle(port: int, timeout_s: int = 120):
    url = f"http://127.0.0.1:{port}/"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code < 500:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("Timed out waiting for Gentle server")


def call_gentle(audio_path: Path, transcript_path: Path, port: int) -> dict:
    url = f"http://127.0.0.1:{port}/transcriptions?async=false"
    with audio_path.open("rb") as audio_f, transcript_path.open("rb") as transcript_f:
        resp = requests.post(
            url,
            files={
                "audio": (audio_path.name, audio_f, "audio/wav"),
                "transcript": (transcript_path.name, transcript_f, "text/plain"),
            },
            timeout=3600,
        )
    resp.raise_for_status()
    return resp.json()


def extract_phonemes(align_json: dict) -> list[dict]:
    rows = []
    word_index = -1
    for word in align_json.get("words", []):
        if word.get("case") != "success" or "phones" not in word or "start" not in word:
            continue
        word_index += 1
        t = float(word["start"])
        for phone_idx, phone in enumerate(word["phones"]):
            duration = float(phone["duration"])
            start = t
            end = t + duration
            raw_phone = str(phone["phone"])
            base_phone = raw_phone.split("_", 1)[0]
            rows.append(
                {
                    "word_index": word_index,
                    "word": word.get("alignedWord", word.get("word", "")),
                    "phone_index_in_word": phone_idx,
                    "phone": base_phone,
                    "phone_raw": raw_phone,
                    "start_s": round(start, 6),
                    "end_s": round(end, 6),
                    "duration_s": round(duration, 6),
                }
            )
            t = end
    return rows


def tr_bin_phonemes(phonemes: list[dict], tr_s: float, total_end_s: float, stimulus_onset_s: float) -> list[dict]:
    n_trs = int((total_end_s / tr_s) + 0.999999)
    rows = []
    for tr_index in range(n_trs):
        bin_start = tr_index * tr_s
        bin_end = bin_start + tr_s
        labels = []
        for phone in phonemes:
            midpoint = (float(phone["start_s"]) + float(phone["end_s"])) / 2
            global_midpoint = stimulus_onset_s + midpoint
            if bin_start <= global_midpoint < bin_end:
                labels.append(str(phone["phone"]))
        rows.append(
            {
                "tr_index": tr_index,
                "start_s": round(bin_start, 3),
                "end_s": round(bin_end, 3),
                "phonemes": " ".join(labels),
            }
        )
    return rows


def process_story(story_dir: Path, audio_dir: Path, port: int):
    slug = story_dir.name
    metadata = load_metadata(story_dir)
    audio_path = Path(metadata["audio_copy"])
    transcript_path = ensure_transcript_text(story_dir, slug)

    align_json = call_gentle(audio_path=audio_path, transcript_path=transcript_path, port=port)
    (story_dir / "gentle_align.json").write_text(json.dumps(align_json, indent=2))

    phonemes = extract_phonemes(align_json)
    write_tsv(
        phonemes,
        story_dir / f"{slug}_phonemes.tsv",
        ["word_index", "word", "phone_index_in_word", "phone", "phone_raw", "start_s", "end_s", "duration_s"],
    )

    tr_rows = tr_bin_phonemes(
        phonemes=phonemes,
        tr_s=float(metadata["tr_s"]),
        total_end_s=float(metadata["stimulus_end_s"]),
        stimulus_onset_s=float(metadata["stimulus_onset_s"]),
    )
    write_tsv(tr_rows, story_dir / f"{slug}_phonemes_by_tr.tsv", ["tr_index", "start_s", "end_s", "phonemes"])

    if metadata.get("subject_specific_tr_alignment"):
        onsets_path = story_dir / "alice_subject_onsets.tsv"
        if onsets_path.exists():
            subject_rows = read_tsv(onsets_path)
            for row in subject_rows:
                subject = row["subject"]
                subject_tr_rows = tr_bin_phonemes(
                    phonemes=phonemes,
                    tr_s=float(metadata["tr_s"]),
                    total_end_s=float(row["end_s"]),
                    stimulus_onset_s=float(row["onset_s"]),
                )
                write_tsv(
                    subject_tr_rows,
                    story_dir / "by_subject" / f"{subject}_alice_phonemes_by_tr.tsv",
                    ["tr_index", "start_s", "end_s", "phonemes"],
                )

    print(f"Wrote Gentle outputs for {slug}")


def main():
    parser = argparse.ArgumentParser(description="Run Gentle forced alignment on prepared transcript/audio assets.")
    parser.add_argument("--transcripts-dir", default=str(data_root() / "transcripts"))
    parser.add_argument("--targets", nargs="*", default=None, help="Optional subset of story slugs.")
    parser.add_argument("--image", default="lowerquality/gentle")
    parser.add_argument("--container-name", default="gentle-align")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    transcripts_dir = Path(args.transcripts_dir).expanduser().resolve()
    audio_dir = transcripts_dir / "audio"

    ensure_gentle_server(image=args.image, name=args.container_name, port=args.port)
    wait_for_gentle(port=args.port)

    story_dirs = sorted(
        path
        for path in transcripts_dir.iterdir()
        if path.is_dir() and path.name not in {"audio", ".whisper-cache"}
    )
    if args.targets:
        selected = set(args.targets)
        story_dirs = [path for path in story_dirs if path.name in selected]

    for story_dir in story_dirs:
        process_story(story_dir=story_dir, audio_dir=audio_dir, port=args.port)


if __name__ == "__main__":
    main()
