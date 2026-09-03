from __future__ import annotations

import argparse
import json
import os

from thesis_neuro.config import load_app_config

ANALYSIS_STAGES = (
    "build-feature-relevance",
    "build-feature-correlations",
    "build-feature-judge-input",
    "run-feature-judge",
    "select-alignment-features",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thesis-neuro")
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config path. Defaults to configs/default.yaml.",
    )
    parser.add_argument("--env-file", default=".env", help="Path to the .env file.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Run Dolma -> Gemma -> Gemma Scope extraction.")
    extract.add_argument(
        "--write-summary",
        action="store_true",
        help="Override config and also emit feature_summary.jsonl.",
    )
    extract.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force Hugging Face loads to use cached local files only.",
    )

    prefetch = subparsers.add_parser(
        "prefetch",
        help="Download/cache the configured Gemma model and Gemma Scope SAE weights.",
    )
    prefetch.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force prefetch to use cached local files only.",
    )

    discover = subparsers.add_parser(
        "discover-transcript-features",
        help="Run transcript-only extraction and write aggregated transcript feature stats + shortlist.",
    )
    discover.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force Hugging Face loads to use cached local files only.",
    )

    dolma = subparsers.add_parser(
        "collect-dolma-contexts",
        help="Scan Dolma once for shortlisted transcript features and write multi-scale feature contexts.",
    )
    dolma.add_argument(
        "--shortlist",
        default=None,
        help="Path to a transcript feature shortlist JSONL. Defaults to <output.dir>/transcript_feature_shortlist.jsonl.",
    )
    dolma.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force Hugging Face model loads to use cached local files only.",
    )

    analyze = subparsers.add_parser(
        "analyze-features",
        help="Rank transcript-relevant features, compute correlations, and optionally run an LLM judge.",
    )
    analyze.add_argument(
        "--transcript-output-dir",
        default=None,
        help="Directory containing transcript_feature_stats.jsonl and transcript_feature_shortlist.jsonl.",
    )
    analyze.add_argument(
        "--dolma-output-dir",
        default=None,
        help="Directory containing dolma_feature_contexts.jsonl.",
    )
    analyze.add_argument(
        "--alignment-path",
        default=None,
        help="Optional path to feature_alignment.jsonl for adding ablation evidence to the judge input.",
    )
    analyze.add_argument(
        "--run-judge",
        action="store_true",
        help="Call the configured OpenAI judge model and write feature_concepts.jsonl.",
    )
    analyze.add_argument(
        "--judge-missing-only",
        action="store_true",
        help="When judging, keep successful prior judge rows and only fill missing or errored features.",
    )
    analyze.add_argument(
        "--from-stage",
        choices=ANALYSIS_STAGES,
        default=None,
        help="Optional analysis sub-stage to start from.",
    )
    analyze.add_argument(
        "--until-stage",
        choices=ANALYSIS_STAGES,
        default=None,
        help="Optional analysis sub-stage to stop after.",
    )

    align = subparsers.add_parser(
        "align-features",
        help="Read paired activation rows and write derived feature-to-token alignment summaries.",
    )
    align.add_argument(
        "--input",
        default=None,
        help="Path to an existing paired JSONL file. Defaults to <output.dir>/paired_records.jsonl.",
    )
    align.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force Hugging Face loads to use cached local files only.",
    )
    align.add_argument(
        "--wait-for-input-seconds",
        type=float,
        default=30.0,
        help="How long to wait for paired_records.jsonl to appear before failing.",
    )
    align.add_argument(
        "--focus-features",
        default=None,
        help="Optional JSONL file containing layer/feature_id rows to restrict ablation to a selected feature set.",
    )

    probe = subparsers.add_parser(
        "probe-feature",
        help="Run an LLM-guided feature probing loop with synthetic probes, real edits, and optional steering.",
    )
    probe.add_argument("--analysis-dir", default=None, help="Directory containing feature_relevance.jsonl and feature_concepts.jsonl.")
    probe.add_argument("--transcript-dir", default=None, help="Directory containing transcript artifacts.")
    probe.add_argument("--dolma-dir", default=None, help="Directory containing dolma_feature_contexts.jsonl.")
    probe.add_argument("--alignment-path", default=None, help="Optional path to feature_alignment.jsonl.")
    probe.add_argument("--layer", type=int, required=True, help="Target SAE layer.")
    probe.add_argument("--feature-id", type=int, required=True, help="Target feature id.")
    probe.add_argument("--script-id", default=None, help="Optional transcript script id to emphasize in evidence gathering.")
    probe.add_argument("--max-rounds", type=int, default=None, help="Override probing.max_rounds.")
    probe.add_argument("--run-steering", action="store_true", help="Enable steering experiments for this probing run.")
    probe.add_argument("--judge-model", default=None, help="Optional OpenAI model override for the probing agent.")

    subparsers.add_parser(
        "mock-extract",
        help="Write a mock paired artifact with the target schema for downstream validation.",
    )

    summarize = subparsers.add_parser(
        "summarize",
        help="Build a feature-centric summary from paired_records.jsonl.",
    )
    summarize.add_argument(
        "--input",
        default=None,
        help="Path to an existing paired JSONL file. Defaults to <output.dir>/paired_records.jsonl.",
    )
    summarize.add_argument(
        "--top-contexts",
        type=int,
        default=5,
        help="How many top activating contexts to keep per feature.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_app_config(config_path=args.config, env_path=args.env_file)

    if args.command == "extract":
        from thesis_neuro.pipelines.extraction import ExtractionPipeline

        if args.write_summary:
            config.output.write_summary = True
        if args.local_files_only:
            config.env.hf_local_files_only = True
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        manifest = ExtractionPipeline(config).run()
        print(json.dumps(manifest, indent=2))
        return

    if args.command == "prefetch":
        from thesis_neuro.pipelines.extraction import PrefetchPipeline

        if args.local_files_only:
            config.env.hf_local_files_only = True
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        result = PrefetchPipeline(config).run()
        print(json.dumps(result, indent=2))
        return

    if args.command == "discover-transcript-features":
        from thesis_neuro.pipelines.transcripts import TranscriptFeatureDiscoveryPipeline

        if args.local_files_only:
            config.env.hf_local_files_only = True
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        result = TranscriptFeatureDiscoveryPipeline(config).run()
        print(json.dumps(result, indent=2))
        return

    if args.command == "collect-dolma-contexts":
        from thesis_neuro.pipelines.contexts import DolmaContextCollectionPipeline

        if args.local_files_only:
            config.env.hf_local_files_only = True
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        result = DolmaContextCollectionPipeline(config, shortlist_path=args.shortlist).run()
        print(json.dumps(result, indent=2))
        return

    if args.command == "align-features":
        from thesis_neuro.pipelines.alignment import FeatureAlignmentPipeline

        if args.local_files_only:
            config.env.hf_local_files_only = True
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        result = FeatureAlignmentPipeline(
            config,
            input_path=args.input,
            wait_for_input_seconds=args.wait_for_input_seconds,
            focus_features_path=args.focus_features,
        ).run()
        print(json.dumps(result, indent=2))
        return

    if args.command == "analyze-features":
        from thesis_neuro.feature_analysis import FeatureConceptAnalysisPipeline

        result = FeatureConceptAnalysisPipeline(
            config,
            transcript_output_dir=args.transcript_output_dir,
            dolma_output_dir=args.dolma_output_dir,
            alignment_path=args.alignment_path,
            run_judge=args.run_judge,
            judge_missing_only=args.judge_missing_only,
            from_stage=args.from_stage,
            until_stage=args.until_stage,
        ).run()
        print(json.dumps(result, indent=2))
        return

    if args.command == "probe-feature":
        from thesis_neuro.probes.runner import FeatureProbingPipeline

        result = FeatureProbingPipeline(
            config,
            analysis_dir=args.analysis_dir,
            transcript_dir=args.transcript_dir,
            dolma_dir=args.dolma_dir,
            alignment_path=args.alignment_path,
            layer=args.layer,
            feature_id=args.feature_id,
            script_id=args.script_id,
            max_rounds=args.max_rounds,
            run_steering=args.run_steering,
            judge_model=args.judge_model,
        ).run()
        print(json.dumps(result, indent=2))
        return

    if args.command == "summarize":
        from thesis_neuro.pipelines.summary import summarize_feature_records
        from thesis_neuro.storage import JsonlArtifactStore

        store = JsonlArtifactStore(config.output_dir)
        summary_rows = summarize_feature_records(
            store.iter_records(args.input),
            top_contexts=args.top_contexts,
        )
        store.write_summary(summary_rows)
        print(
            json.dumps(
                {
                    "summary_path": str(store.summary_path),
                    "feature_rows": len(summary_rows),
                },
                indent=2,
            )
        )
        return

    if args.command == "mock-extract":
        from thesis_neuro.pipelines.mock import MockExtractionPipeline

        manifest = MockExtractionPipeline(config).run()
        print(json.dumps(manifest, indent=2))
        return

    raise ValueError(f"Unknown command: {args.command}")
