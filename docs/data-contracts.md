# Data Contracts

No research dataset or generated result is tracked in Git. Paths below are logical locations under `THESIS_NEURO_DATA_ROOT` and `THESIS_NEURO_OUTPUT_ROOT`; commands expose explicit path options where they apply.

## Transcript Assets

Each stimulus directory under `$THESIS_NEURO_DATA_ROOT/transcripts/<stimulus>/` contains:

| File | Required fields |
| --- | --- |
| `<stimulus>_transcript.txt` | Plain transcript text |
| `<stimulus>_words.tsv` | `word`, `start_s`, `end_s` |
| `<stimulus>_tr_aligned.tsv` | `tr_index`, `start_s`, `end_s`, `text` |
| `metadata.json` | `stimulus_onset_s` in seconds (defaults to 0.0 if absent), `tr_s`, plus free-form provenance |

Word midpoints are assigned to the matching TR interval after the configured stimulus onset. Alignment validation rejects non-contiguous token streams and word/TR mismatches.

## Feature Run

Transcript extraction produces JSONL artifacts such as:

- `transcript_paired_records.jsonl`: token-level hidden-state/SAE records with model, layer, token position, activation, and provenance fields.
- `transcript_feature_shortlist.jsonl` and `transcript_feature_stats.jsonl`: ranked transcript features and their aggregate statistics; the shortlist drives Dolma context collection.
- `feature_relevance.jsonl`: per-feature relevance ranking read by the analysis stage and the dashboard.
- `selected_features_for_alignment.jsonl`: ranked `layer` and `feature_id` keys.
- `feature_alignment.jsonl`: token/span counterfactual evidence.
- `feature_concepts.jsonl`: optional judge summaries.

Generated feature runs belong under the configured output root and are never committed.

## TR Feature Bundle

Feature matrices are NumPy `.npz` files with:

- `values`: `(n_tr, n_features)` predictor or target matrix.
- `sample_ids`: unique stimulus/TR identifiers.
- `feature_names` or `target_names`.
- `stimulus_id`.
- `layers`, `tr_indices`, `start_s`, and `end_s` where applicable.

`build-tr-artifacts` writes presence, activation-mass, average, peak, and active-count views; `run-analysis --predictor-view` fits mass, presence, or average.

## Brain Target Bundle

Required arrays:

- `values`: `(n_samples, n_parcels)` parcel responses.
- `sample_ids`, `subject_ids`, `run_ids`, and `tr_indices`.
- `target_names`: parcel labels aligned to the columns of `values`.

Cleaned bundles may also provide `censor_mask`, `framewise_displacement`, and `std_dvars`. Censored samples are removed after transcript/TR alignment and before grouped evaluation.

## Probe Run

A probe run stores `feature_probe_evidence.json`, `feature_probe_rounds.jsonl`, `feature_probe_tests.jsonl`, `feature_probe_steering.jsonl`, `feature_probe_report.json`, and `manifest.json` under `probing.output_dir` (default `$THESIS_NEURO_OUTPUT_ROOT/probe_runs`). Reports must include a final hypothesis, summary, uncertainty, and confidence in `[0, 1]`. Credentials and raw provider responses are not repository artifacts.

## Privacy And Storage

The repository intentionally excludes BIDS datasets, fMRIPrep derivatives, atlas binaries, model/SAE weights, run outputs, logs, and dashboard bundles. Users are responsible for obtaining data under the original dataset licenses and ethics constraints.
