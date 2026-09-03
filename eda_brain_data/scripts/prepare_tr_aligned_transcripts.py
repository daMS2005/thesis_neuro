#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import srt
import whisper

from thesis_neuro.paths import data_root

PUNCT_RE = re.compile(r"\s+([.,!?;:])")


@dataclass(frozen=True)
class TranscriptTarget:
    slug: str
    dataset_dir: Path
    audio_relpath: Path
    task: str
    source_kind: str
    annotations_relpath: Path | None = None


def run(cmd: list[str], cwd: Path | None = None):
    print("$", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def git_output(cmd: list[str], cwd: Path) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def ensure_annex_ready(dataset_dir: Path):
    dbdir = Path("/tmp") / f"git-annex-db-{dataset_dir.name}"
    run(["git", "config", "annex.dbdir", str(dbdir)], cwd=dataset_dir)
    try:
        git_output(["git", "annex", "info", "--fast"], cwd=dataset_dir)
    except subprocess.CalledProcessError:
        run(["git", "annex", "init"], cwd=dataset_dir)


def ensure_annex_file(dataset_dir: Path, relpath: Path, fallback_paths: Iterable[Path] | None = None):
    target = dataset_dir / relpath
    try:
        target.resolve(strict=True)
        return target
    except FileNotFoundError:
        if fallback_paths:
            for fallback_path in fallback_paths:
                if fallback_path.exists():
                    return fallback_path
        try:
            run(["git", "annex", "get", str(relpath)], cwd=dataset_dir)
            target.resolve(strict=True)
            return target
        except (subprocess.CalledProcessError, FileNotFoundError):
            url = annex_public_url(dataset_dir, relpath)
            cache_dir = dataset_dir / ".audio-cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / relpath.name
            if cache_path.exists():
                return cache_path
            try:
                run(["curl", "-fL", url, "-o", str(cache_path)], cwd=dataset_dir)
                return cache_path
            except subprocess.CalledProcessError:
                if fallback_paths:
                    for fallback_path in fallback_paths:
                        if fallback_path.exists():
                            return fallback_path
                raise


def annex_public_url(dataset_dir: Path, relpath: Path) -> str:
    output = git_output(["git", "annex", "whereis", str(relpath)], cwd=dataset_dir)
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("s3-PUBLIC: https://"):
            return line.split("s3-PUBLIC: ", 1)[1]
    raise RuntimeError(f"Could not find public S3 URL for {relpath}")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def find_first(dataset_dir: Path, pattern: str) -> Path:
    matches = sorted(dataset_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern} in {dataset_dir}")
    return matches[0]


def normalize_text(text: str) -> str:
    return PUNCT_RE.sub(r"\1", re.sub(r"\s+", " ", text)).strip()


def get_task_timing(dataset_dir: Path, task: str) -> dict[str, float | str]:
    events_path = find_first(dataset_dir, f"sub-*/**/*task-{task}_events.tsv")
    bold_json_path = find_first(dataset_dir, f"sub-*/**/*task-{task}_bold.json")
    rows = read_tsv(events_path)
    numeric_rows = []
    for row in rows:
        try:
            onset = float(row["onset"])
            duration = float(row["duration"])
        except (TypeError, ValueError):
            continue
        numeric_rows.append(
            {
                "onset": onset,
                "duration": duration,
                "stim_file": row.get("stim_file", ""),
            }
        )
    if not numeric_rows:
        raise ValueError(f"No numeric timing rows found in {events_path}")
    bold_json = read_json(bold_json_path)
    onset = min(row["onset"] for row in numeric_rows)
    end = max(row["onset"] + row["duration"] for row in numeric_rows)
    return {
        "events_path": str(events_path),
        "bold_json_path": str(bold_json_path),
        "stimulus_onset_s": onset,
        "stimulus_end_s": end,
        "tr_s": float(bold_json["RepetitionTime"]),
        "stim_file": numeric_rows[0]["stim_file"],
    }


def get_subject_event_timings(dataset_dir: Path, task: str) -> list[dict[str, float | str]]:
    timings = []
    for events_path in sorted(dataset_dir.glob(f"sub-*/**/*task-{task}_events.tsv")):
        rows = read_tsv(events_path)
        numeric_rows = []
        for row in rows:
            try:
                onset = float(row["onset"])
                duration = float(row["duration"])
            except (TypeError, ValueError):
                continue
            numeric_rows.append({"onset": onset, "duration": duration})
        if not numeric_rows:
            continue
        subject = next(part for part in events_path.parts if part.startswith("sub-"))
        timings.append(
            {
                "subject": subject,
                "events_path": str(events_path),
                "onset_s": min(row["onset"] for row in numeric_rows),
                "end_s": max(row["onset"] + row["duration"] for row in numeric_rows),
            }
        )
    return timings


def load_alice_words(annotations_path: Path) -> list[dict[str, float | str]]:
    words = []
    for row in read_tsv(annotations_path):
        word = row["Words"].strip()
        try:
            start = float(row["Word Onset"])
            end = float(row["Word Offset"])
        except (TypeError, ValueError):
            continue
        if not word:
            continue
        words.append({"text": word, "start": start, "end": end})
    return words


def audio_duration_s(audio_path: Path) -> float:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        text=True,
    ).strip()
    return float(output)


def extract_audio_chunk(audio_path: Path, chunk_path: Path, start_s: float, duration_s: float):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_s:.3f}",
            "-t",
            f"{duration_s:.3f}",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(chunk_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def whisper_word_items(
    audio_path: Path,
    model_name: str,
    cache_dir: Path,
    chunk_duration_s: float = 60.0,
) -> list[dict[str, float | str]]:
    model = whisper.load_model(model_name, download_root=str(cache_dir))
    words = []
    total_duration_s = audio_duration_s(audio_path)
    with tempfile.TemporaryDirectory(prefix="whisper-chunks-") as tmpdir:
        chunk_dir = Path(tmpdir)
        chunk_start_s = 0.0
        chunk_index = 0
        while chunk_start_s < total_duration_s:
            current_duration_s = min(chunk_duration_s, total_duration_s - chunk_start_s)
            chunk_path = chunk_dir / f"chunk_{chunk_index:03d}.wav"
            extract_audio_chunk(audio_path, chunk_path, start_s=chunk_start_s, duration_s=current_duration_s)
            result = model.transcribe(
                str(chunk_path),
                language="en",
                word_timestamps=True,
                verbose=False,
                temperature=0,
                beam_size=5,
                best_of=5,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.0,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
            )
            for segment in result.get("segments", []):
                for word in segment.get("words", []):
                    word_text = word["word"].strip()
                    if not word_text:
                        continue
                    words.append(
                        {
                            "text": word_text,
                            "start": float(word["start"]) + chunk_start_s,
                            "end": float(word["end"]) + chunk_start_s,
                        }
                    )
            chunk_start_s += current_duration_s
            chunk_index += 1
    if not words:
        raise RuntimeError(f"Whisper returned no timed words for {audio_path}")
    return words


def words_to_segments(
    words: list[dict[str, float | str]],
    max_chars: int = 84,
    max_duration_s: float = 5.0,
    gap_s: float = 1.0,
) -> list[dict[str, float | str]]:
    segments = []
    current = []
    for word in words:
        if not current:
            current = [word]
            continue
        proposed = normalize_text(" ".join(str(item["text"]) for item in current + [word]))
        start = float(current[0]["start"])
        end = float(word["end"])
        gap = float(word["start"]) - float(current[-1]["end"])
        if len(proposed) > max_chars or (end - start) > max_duration_s or gap > gap_s:
            segments.append(
                {
                    "start": float(current[0]["start"]),
                    "end": float(current[-1]["end"]),
                    "text": normalize_text(" ".join(str(item["text"]) for item in current)),
                }
            )
            current = [word]
        else:
            current.append(word)
    if current:
        segments.append(
            {
                "start": float(current[0]["start"]),
                "end": float(current[-1]["end"]),
                "text": normalize_text(" ".join(str(item["text"]) for item in current)),
            }
        )
    return segments


def write_srt(segments: list[dict[str, float | str]], path: Path):
    subtitles = []
    for i, segment in enumerate(segments, start=1):
        subtitles.append(
            srt.Subtitle(
                index=i,
                start=srt.timedelta(seconds=float(segment["start"])),
                end=srt.timedelta(seconds=float(segment["end"])),
                content=str(segment["text"]),
            )
        )
    path.write_text(srt.compose(subtitles))


def write_word_tsv(words: list[dict[str, float | str]], path: Path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "start_s", "end_s"], delimiter="\t")
        writer.writeheader()
        for word in words:
            writer.writerow(
                {
                    "word": str(word["text"]),
                    "start_s": f"{float(word['start']):.3f}",
                    "end_s": f"{float(word['end']):.3f}",
                }
            )


def tr_align_words(
    words: list[dict[str, float | str]],
    tr_s: float,
    total_end_s: float,
    scan_onset_s: float,
) -> list[dict[str, float | int | str]]:
    n_trs = math.ceil(total_end_s / tr_s)
    rows = []
    for tr_index in range(n_trs):
        bin_start = tr_index * tr_s
        bin_end = bin_start + tr_s
        bin_words = []
        for word in words:
            midpoint = (float(word["start"]) + float(word["end"])) / 2
            global_midpoint = scan_onset_s + midpoint
            if bin_start <= global_midpoint < bin_end:
                bin_words.append(str(word["text"]))
        rows.append(
            {
                "tr_index": tr_index,
                "start_s": round(bin_start, 3),
                "end_s": round(bin_end, 3),
                "text": normalize_text(" ".join(bin_words)),
            }
        )
    return rows


def write_tr_tsv(rows: Iterable[dict[str, float | int | str]], path: Path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tr_index", "start_s", "end_s", "text"], delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_subject_onsets(rows: Iterable[dict[str, float | str]], path: Path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "onset_s", "end_s", "events_path"], delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def copy_audio(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.chmod(0o644)
        dest.unlink()
    shutil.copy2(src, dest)
    dest.chmod(0o644)


def write_plain_transcript(words: list[dict[str, float | str]], path: Path):
    path.write_text(normalize_text(" ".join(str(word["text"]) for word in words)) + "\n")


def process_target(target: TranscriptTarget, output_dir: Path, model_name: str, whisper_cache_dir: Path):
    ensure_annex_ready(target.dataset_dir)
    fallback_audio_paths = [output_dir / "audio" / target.audio_relpath.name]
    audio_path = ensure_annex_file(target.dataset_dir, target.audio_relpath, fallback_paths=fallback_audio_paths)
    timing = get_task_timing(target.dataset_dir, target.task)

    target_dir = output_dir / target.slug
    audio_out = output_dir / "audio" / audio_path.name
    target_dir.mkdir(parents=True, exist_ok=True)
    copy_audio(audio_path, audio_out)

    if target.source_kind == "alice-annotations":
        assert target.annotations_relpath is not None
        annotations_path = target.dataset_dir / target.annotations_relpath
        words = load_alice_words(annotations_path)
        subject_timings = get_subject_event_timings(target.dataset_dir, target.task)
        shutil.copy2(annotations_path, target_dir / "alice_original_annotations.tsv")
        write_plain_transcript(words, target_dir / "alice_original_transcript.txt")
        write_subject_onsets(subject_timings, target_dir / "alice_subject_onsets.tsv")
        by_subject_dir = target_dir / "by_subject"
        by_subject_dir.mkdir(parents=True, exist_ok=True)
        for subject_timing in subject_timings:
            subject_rows = tr_align_words(
                words=words,
                tr_s=float(timing["tr_s"]),
                total_end_s=float(subject_timing["end_s"]),
                scan_onset_s=float(subject_timing["onset_s"]),
            )
            write_tr_tsv(subject_rows, by_subject_dir / f"{subject_timing['subject']}_alice_tr_aligned.tsv")
        method = "annotations.tsv"
    elif target.source_kind == "whisper":
        words = whisper_word_items(audio_path, model_name=model_name, cache_dir=whisper_cache_dir)
        method = f"whisper:{model_name}"
    else:
        raise ValueError(f"Unsupported source kind: {target.source_kind}")

    segments = words_to_segments(words)
    tr_rows = tr_align_words(
        words=words,
        tr_s=float(timing["tr_s"]),
        total_end_s=float(timing["stimulus_end_s"]),
        scan_onset_s=0.0 if target.source_kind == "alice-annotations" else float(timing["stimulus_onset_s"]),
    )

    write_word_tsv(words, target_dir / f"{target.slug}_words.tsv")
    write_plain_transcript(words, target_dir / f"{target.slug}_transcript.txt")
    write_srt(segments, target_dir / f"{target.slug}.srt")
    write_tr_tsv(tr_rows, target_dir / f"{target.slug}_tr_aligned.tsv")

    metadata = {
        "slug": target.slug,
        "dataset_dir": str(target.dataset_dir),
        "audio_source": str(audio_path),
        "audio_copy": str(audio_out),
        "task": target.task,
        "transcript_method": method,
        "tr_s": timing["tr_s"],
        "stimulus_onset_s": 0.0 if target.source_kind == "alice-annotations" else timing["stimulus_onset_s"],
        "stimulus_end_s": timing["stimulus_end_s"],
        "stim_file": timing["stim_file"],
        "events_path": timing["events_path"],
        "bold_json_path": timing["bold_json_path"],
        "subject_specific_tr_alignment": target.source_kind == "alice-annotations",
    }
    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote outputs for {target.slug} to {target_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare WAV copies, subtitle files, and TR-aligned transcripts.")
    parser.add_argument(
        "--output-dir",
        default=str(data_root() / "transcripts"),
        help="Directory where transcript outputs should be written.",
    )
    parser.add_argument(
        "--whisper-model",
        default="base.en",
        help="Whisper model name to use for the Narratives stories.",
    )
    parser.add_argument(
        "--whisper-cache-dir",
        default=str(data_root() / "cache" / "whisper"),
        help="Local cache directory for Whisper models.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Optional subset of slugs to process, e.g. shapessocial alice.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    whisper_cache_dir = Path(args.whisper_cache_dir).expanduser().resolve()
    whisper_cache_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        TranscriptTarget(
            slug="bronx",
            dataset_dir=data_root() / "openneuro" / "ds002345",
            audio_relpath=Path("stimuli/bronx_audio.wav"),
            task="bronx",
            source_kind="whisper",
        ),
        TranscriptTarget(
            slug="black",
            dataset_dir=data_root() / "openneuro" / "ds002345",
            audio_relpath=Path("stimuli/black_audio.wav"),
            task="black",
            source_kind="whisper",
        ),
        TranscriptTarget(
            slug="forgot",
            dataset_dir=data_root() / "openneuro" / "ds002345",
            audio_relpath=Path("stimuli/forgot_audio.wav"),
            task="forgot",
            source_kind="whisper",
        ),
        TranscriptTarget(
            slug="piemanpni",
            dataset_dir=data_root() / "openneuro" / "ds002345",
            audio_relpath=Path("stimuli/piemanpni_audio.wav"),
            task="piemanpni",
            source_kind="whisper",
        ),
        TranscriptTarget(
            slug="shapesphysical",
            dataset_dir=data_root() / "openneuro" / "ds002345",
            audio_relpath=Path("stimuli/shapesphysical_audio.wav"),
            task="shapesphysical",
            source_kind="whisper",
        ),
        TranscriptTarget(
            slug="shapessocial",
            dataset_dir=data_root() / "openneuro" / "ds002345",
            audio_relpath=Path("stimuli/shapessocial_audio.wav"),
            task="shapessocial",
            source_kind="whisper",
        ),
        TranscriptTarget(
            slug="alice",
            dataset_dir=data_root() / "openneuro" / "ds002322",
            audio_relpath=Path("stimuli/DownTheRabbitHoleFinal_mono_exp120_NR16_pad.wav"),
            task="alice",
            source_kind="alice-annotations",
            annotations_relpath=Path("code/annotations.tsv"),
        ),
    ]

    if args.targets:
        selected = set(args.targets)
        targets = [target for target in targets if target.slug in selected]

    for target in targets:
        process_target(
            target=target,
            output_dir=output_dir,
            model_name=args.whisper_model,
            whisper_cache_dir=whisper_cache_dir,
        )


if __name__ == "__main__":
    main()
