# Usage Guide

## 1. Install

Python 3.11 or 3.12 is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Use `python -m pip install -e ".[full]"` for model, fMRI, judge, notebook, and data-preparation dependencies. `configs/examples/smoke.yaml` is the same one-window run as the mock config but with real Gemma 2 2B weights, for checking a machine end to end.

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

See [Data Contracts](data-contracts.md) for the required fields in each artifact.

## 3. Offline Verification

These commands need no network, private data, or model cache:

```bash
make check
```

Or individually:

```bash
ruff check src scripts tests
python -m compileall -q src scripts tests
pytest
thesis-neuro --config configs/examples/mock.yaml mock-extract
thesis-neuro-benchmark validate-items --items-path examples/benchmark/mock_boolq.jsonl
```

## 4. Prepare Brain Data

The scripts in `scripts/data_prep/` build a task-specific BIDS subset, audit fMRIPrep inputs, and produce word-timing and TR-aligned transcript tables:

```bash
python scripts/data_prep/build_task_bids_subset.py --help
python scripts/data_prep/audit_fmriprep_task_inputs.py --help
python scripts/data_prep/prepare_tr_aligned_transcripts.py --help
```

## 5. Feature Discovery And Probing

With model dependencies and transcript assets available:

```bash
thesis-neuro --config configs/default.yaml discover-transcript-features --local-files-only
thesis-neuro --config configs/default.yaml collect-dolma-contexts --local-files-only
thesis-neuro --config configs/default.yaml analyze-features
thesis-neuro --config configs/default.yaml probe-feature --layer 4 --feature-id 464 --script-id shapessocial
```

Drop `--local-files-only` when downloads are intended. Set `OPENAI_API_KEY` only for judge and probe commands. `configs/examples/full-extraction.yaml` shows the settings used for a full extraction run, and `scripts/analysis/render_run_config.py` renders a config for another model by resolving SAE layers at matched relative depths.

## 6. TR And Brain Modeling

Build a TR feature bundle from an existing feature run:

```bash
thesis-neuro-structure build-tr-artifacts \
  --feature-run-dir "$THESIS_NEURO_DATA_ROOT/feature-runs/<run-name>" \
  --stimulus-id shapesphysical \
  --output-dir "$THESIS_NEURO_OUTPUT_ROOT/structure/gemma_2_2b"
```

Build cleaned brain targets:

```bash
thesis-neuro-structure build-clean-brain-targets \
  --stimulus-id shapesphysical \
  --output-path "$THESIS_NEURO_OUTPUT_ROOT/brain/shapesphysical_schaefer200.npz"
```

Fit and compare brain and LM ridge models:

```bash
thesis-neuro-structure run-analysis \
  --feature-run-dir "$THESIS_NEURO_DATA_ROOT/feature-runs/<run-name>" \
  --stimulus-id shapesphysical \
  --brain-targets-npz "$THESIS_NEURO_OUTPUT_ROOT/brain/shapesphysical_schaefer200.npz" \
  --output-dir "$THESIS_NEURO_OUTPUT_ROOT/structure/gemma_2_2b"
```

Analysis variants and figure builders live in `scripts/analysis/`. `notebooks/model_results_comparison.ipynb` discovers every `<output root>/structure/<model>/analysis_summary.json` that `run-analysis` writes and compares them side by side.

## 7. Audit Dashboard

```bash
thesis-neuro-audit \
  --analysis-dir "$THESIS_NEURO_OUTPUT_ROOT/<run-name>" \
  --transcript-dir "$THESIS_NEURO_OUTPUT_ROOT/<run-name>" \
  --dolma-dir "$THESIS_NEURO_OUTPUT_ROOT/<run-name>"
```

Open `http://127.0.0.1:8010`. The dashboard reads existing artifacts and writes targeted probe outputs under the output root.

## 8. Optional Benchmark Track

```bash
thesis-neuro-benchmark validate-items --items-path examples/benchmark/mock_boolq.jsonl
thesis-neuro-benchmark prepare-superglue \
  --task copa \
  --split validation \
  --output-path "$THESIS_NEURO_OUTPUT_ROOT/benchmark/copa/items.jsonl"
```

Preparing SuperGLUE needs network access unless the dataset is cached. Feature extraction additionally needs model weights and a completed feature run.

## Determinism

Run configuration records the random seed. Ridge splits are deterministic for a fixed sample order and group labels. Exact model activations can still vary with library, hardware, and model-weight versions, so keep run manifests with generated outputs for numerical audits.
