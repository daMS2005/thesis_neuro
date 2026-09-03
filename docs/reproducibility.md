# Reproducibility Guide

## 1. Install

Python 3.11 or 3.12 is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Use `python -m pip install -e ".[full]"` for model, fMRI, OpenAI judge, and notebook dependencies.

## 2. Configure Paths

```bash
cp .env.example .env
export THESIS_NEURO_DATA_ROOT=/path/to/research-data
export THESIS_NEURO_OUTPUT_ROOT=/path/to/generated-outputs
```

Expected data layout:

```text
$THESIS_NEURO_DATA_ROOT/
  transcripts/<stimulus>/...
  feature-runs/<run-name>/...
  openneuro/ds002345/...
  derivatives/ds002345-fmriprep/...
  atlases/schaefer200.nii.gz
  atlases/schaefer200_labels.csv
```

See [Data Contracts](data-contracts.md) for required fields. Remote hosts, private addresses, and cluster roots are intentionally not configured in this repository.

## 3. Offline Verification

These commands require no network, private data, or model cache:

```bash
thesis-neuro --help
thesis-neuro-audit --help
thesis-neuro-structure --help
thesis-neuro-benchmark --help
thesis-neuro --config configs/examples/mock.yaml mock-extract
thesis-neuro-benchmark validate-items --items-path examples/benchmark/mock_boolq.jsonl
pytest
ruff check src benchmark_comparison structure_comparison eda_brain_data/scripts tests scripts
python scripts/validate_notebooks.py
python scripts/check_markdown_links.py
python scripts/check_repo_hygiene.py
```

## 4. Feature Discovery And Probing

With model dependencies and local transcript assets available:

```bash
thesis-neuro --config configs/default.yaml discover-transcript-features --local-files-only
thesis-neuro --config configs/default.yaml collect-dolma-contexts --local-files-only
thesis-neuro --config configs/default.yaml analyze-features
thesis-neuro --config configs/default.yaml probe-feature --layer 4 --feature-id 464 --script-id shapessocial
```

Remove `--local-files-only` only when network downloads are intended. Set `OPENAI_API_KEY` only for judge/probe commands that call the provider.

## 5. TR And Brain Modeling

Build a TR feature bundle from an existing feature run:

```bash
thesis-neuro-structure build-tr-artifacts \
  --feature-run-dir "$THESIS_NEURO_DATA_ROOT/feature-runs/<run-name>" \
  --stimulus-id shapesphysical \
  --output-dir "$THESIS_NEURO_OUTPUT_ROOT/structure/shapesphysical"
```

Build cleaned brain targets:

```bash
thesis-neuro-structure build-clean-brain-targets \
  --stimulus-id shapesphysical \
  --output-path "$THESIS_NEURO_OUTPUT_ROOT/brain/shapesphysical_schaefer200.npz"
```

Fit and compare brain/LM ridge models:

```bash
thesis-neuro-structure run-analysis \
  --feature-run-dir "$THESIS_NEURO_DATA_ROOT/feature-runs/<run-name>" \
  --stimulus-id shapesphysical \
  --brain-targets-npz "$THESIS_NEURO_OUTPUT_ROOT/brain/shapesphysical_schaefer200.npz" \
  --output-dir "$THESIS_NEURO_OUTPUT_ROOT/structure/shapesphysical"
```

## 6. Audit Dashboard

```bash
thesis-neuro-audit \
  --analysis-dir "$THESIS_NEURO_OUTPUT_ROOT/<run-name>" \
  --transcript-dir "$THESIS_NEURO_OUTPUT_ROOT/<run-name>" \
  --dolma-dir "$THESIS_NEURO_OUTPUT_ROOT/<run-name>"
```

Open `http://127.0.0.1:8010`. The dashboard reads existing artifacts and writes targeted probe outputs under the configured output tree.

## 7. Optional Benchmark Track

The benchmark workflow is exploratory and does not ship benchmark results:

```bash
thesis-neuro-benchmark validate-items --items-path examples/benchmark/mock_boolq.jsonl
thesis-neuro-benchmark prepare-superglue \
  --task copa \
  --split validation \
  --output-path "$THESIS_NEURO_OUTPUT_ROOT/benchmark/copa/items.jsonl"
```

Preparing SuperGLUE requires network access unless the dataset is already cached. Feature extraction additionally requires model weights and external selected-feature artifacts.

## Determinism Notes

Run configuration records the random seed. Ridge splits are deterministic for fixed sample order and group labels. Exact model activations can still vary with library, hardware, and model-weight versions; retain run manifests outside Git for numerical audits.
