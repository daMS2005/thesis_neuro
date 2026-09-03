from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_TARGET_COLUMNS = ("correct", "gold_choice_avg_logprob", "margin")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thesis-neuro-benchmark",
        description="Exploratory behavioral benchmark extension for transcript-selected SAE features.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_registry = subparsers.add_parser("show-registry", help="List the registered latest model runs and brain summaries.")
    show_registry.add_argument("--registry-path", default=None)

    prepare = subparsers.add_parser("prepare-superglue", help="Normalize a SuperGLUE split into benchmark JSONL items.")
    prepare.add_argument("--task", required=True)
    prepare.add_argument("--split", default="validation")
    prepare.add_argument("--output-path", required=True)
    prepare.add_argument("--limit", type=int, default=None)

    validate = subparsers.add_parser("validate-items", help="Validate a normalized benchmark JSONL file.")
    validate.add_argument("--items-path", required=True)

    extract = subparsers.add_parser(
        "extract-features",
        help="Extract transcript-selected SAE features over benchmark items and score the model's choices.",
    )
    extract.add_argument("--model", required=True)
    extract.add_argument("--items-path", required=True)
    extract.add_argument("--output-dir", required=True)
    extract.add_argument("--top-k", type=int, default=None)
    extract.add_argument("--local-files-only", action="store_true")
    extract.add_argument("--registry-path", default=None)

    fit = subparsers.add_parser(
        "fit-benchmark",
        help="Fit a benchmark-side ridge model and compare feature importance against the registered brain model.",
    )
    fit.add_argument("--model", required=True)
    fit.add_argument("--feature-npz", required=True)
    fit.add_argument("--score-jsonl", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--target-columns", nargs="+", default=list(DEFAULT_TARGET_COLUMNS))
    fit.add_argument("--alpha-grid", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    fit.add_argument("--folds", type=int, default=5)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--registry-path", default=None)

    run_all = subparsers.add_parser(
        "run-all",
        help="Prepare a SuperGLUE split, extract features, and fit the benchmark model in one run.",
    )
    run_all.add_argument("--model", required=True)
    run_all.add_argument("--task", required=True)
    run_all.add_argument("--split", default="validation")
    run_all.add_argument("--output-dir", required=True)
    run_all.add_argument("--limit", type=int, default=None)
    run_all.add_argument("--top-k", type=int, default=None)
    run_all.add_argument("--target-columns", nargs="+", default=list(DEFAULT_TARGET_COLUMNS))
    run_all.add_argument("--alpha-grid", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    run_all.add_argument("--folds", type=int, default=5)
    run_all.add_argument("--seed", type=int, default=42)
    run_all.add_argument("--local-files-only", action="store_true")
    run_all.add_argument("--registry-path", default=None)

    run_suite = subparsers.add_parser(
        "run-model-suite",
        help="Run the same benchmark task for multiple registered models and write a comparison table.",
    )
    run_suite.add_argument("--task", required=True)
    run_suite.add_argument("--split", default="validation")
    run_suite.add_argument("--output-dir", required=True)
    run_suite.add_argument("--limit", type=int, default=None)
    run_suite.add_argument("--top-k", type=int, default=None)
    run_suite.add_argument("--target-columns", nargs="+", default=list(DEFAULT_TARGET_COLUMNS))
    run_suite.add_argument("--alpha-grid", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    run_suite.add_argument("--folds", type=int, default=5)
    run_suite.add_argument("--seed", type=int, default=42)
    run_suite.add_argument("--local-files-only", action="store_true")
    run_suite.add_argument("--registry-path", default=None)
    run_suite.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional subset of registered model names. Defaults to all registry entries.",
    )

    summarize = subparsers.add_parser(
        "summarize-models",
        help="Collect benchmark fit summaries under a run root into a single CSV.",
    )
    summarize.add_argument("--run-root", required=True)
    summarize.add_argument("--output-path", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "show-registry":
        from benchmark_comparison.registry import registry_rows

        print(json.dumps(registry_rows(args.registry_path), indent=2))
        return

    if args.command == "prepare-superglue":
        from benchmark_comparison.items import prepare_superglue_items

        summary = prepare_superglue_items(
            task=str(args.task),
            split=str(args.split),
            output_path=args.output_path,
            limit=args.limit,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "validate-items":
        from benchmark_comparison.items import load_items, validate_items

        items = load_items(args.items_path)
        validate_items(items)
        print(json.dumps({"items_path": str(Path(args.items_path)), "item_count": len(items)}, indent=2))
        return

    if args.command == "extract-features":
        from benchmark_comparison.extraction import extract_benchmark_features
        from benchmark_comparison.registry import resolve_registry_entry

        entry = resolve_registry_entry(args.model, args.registry_path)
        summary = extract_benchmark_features(
            model_entry=entry,
            items_path=args.items_path,
            output_dir=args.output_dir,
            top_k=args.top_k,
            local_files_only=bool(args.local_files_only),
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "fit-benchmark":
        from benchmark_comparison.modeling import fit_benchmark_model
        from benchmark_comparison.registry import resolve_registry_entry

        entry = resolve_registry_entry(args.model, args.registry_path)
        summary = fit_benchmark_model(
            model_entry=entry,
            feature_npz=args.feature_npz,
            score_jsonl=args.score_jsonl,
            output_dir=args.output_dir,
            target_columns=args.target_columns,
            alpha_grid=args.alpha_grid,
            folds=args.folds,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "run-all":
        from benchmark_comparison.extraction import extract_benchmark_features
        from benchmark_comparison.items import prepare_superglue_items
        from benchmark_comparison.modeling import fit_benchmark_model
        from benchmark_comparison.registry import resolve_registry_entry

        output_root = Path(args.output_dir)
        items_path = output_root / "items.jsonl"
        prepare_superglue_items(
            task=str(args.task),
            split=str(args.split),
            output_path=items_path,
            limit=args.limit,
        )
        entry = resolve_registry_entry(args.model, args.registry_path)
        extract_summary = extract_benchmark_features(
            model_entry=entry,
            items_path=items_path,
            output_dir=output_root / "extraction",
            top_k=args.top_k,
            local_files_only=bool(args.local_files_only),
        )
        fit_summary = fit_benchmark_model(
            model_entry=entry,
            feature_npz=Path(extract_summary["artifacts"]["item_feature_average"]),
            score_jsonl=Path(extract_summary["artifacts"]["item_scores"]),
            output_dir=output_root / "fit",
            target_columns=args.target_columns,
            alpha_grid=args.alpha_grid,
            folds=args.folds,
            seed=args.seed,
        )
        print(json.dumps({"extract": extract_summary, "fit": fit_summary}, indent=2))
        return

    if args.command == "run-model-suite":
        from benchmark_comparison.extraction import extract_benchmark_features
        from benchmark_comparison.items import prepare_superglue_items
        from benchmark_comparison.modeling import fit_benchmark_model, summarize_model_runs
        from benchmark_comparison.registry import load_registry, resolve_registry_entry

        output_root = Path(args.output_dir)
        items_path = output_root / "items.jsonl"
        prepare_superglue_items(
            task=str(args.task),
            split=str(args.split),
            output_path=items_path,
            limit=args.limit,
        )

        registry = load_registry(args.registry_path)
        model_names = args.models or sorted(registry)
        results: dict[str, dict[str, object]] = {}
        for model_name in model_names:
            entry = resolve_registry_entry(model_name, args.registry_path)
            model_root = output_root / model_name
            extract_summary = extract_benchmark_features(
                model_entry=entry,
                items_path=items_path,
                output_dir=model_root / "extraction",
                top_k=args.top_k,
                local_files_only=bool(args.local_files_only),
            )
            fit_summary = fit_benchmark_model(
                model_entry=entry,
                feature_npz=Path(extract_summary["artifacts"]["item_feature_average"]),
                score_jsonl=Path(extract_summary["artifacts"]["item_scores"]),
                output_dir=model_root / "fit",
                target_columns=args.target_columns,
                alpha_grid=args.alpha_grid,
                folds=args.folds,
                seed=args.seed,
            )
            results[model_name] = {
                "extract": extract_summary,
                "fit": fit_summary,
            }

        comparison = summarize_model_runs(
            run_root=output_root,
            output_path=output_root / "model_comparison.csv",
        )
        print(json.dumps({"results": results, "comparison": comparison}, indent=2))
        return

    if args.command == "summarize-models":
        from benchmark_comparison.modeling import summarize_model_runs

        summary = summarize_model_runs(run_root=args.run_root, output_path=args.output_path)
        print(json.dumps(summary, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")
