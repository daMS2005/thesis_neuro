# Brain Data EDA

This area contains the curated exploratory data analysis and transcript-preparation surface for the OpenNeuro Narratives data used by the thesis.

Public materials are limited to:

- `notebooks/02_openneuro_brain_eda.ipynb`: BIDS metadata and image-header EDA.
- `notebooks/08_methodology_brain_figure.ipynb`: methodology-oriented preprocessing and alignment inspection.
- `scripts/audit_fmriprep_task_inputs.py`: input completeness audit.
- `scripts/audit_transcript_layout.py`: transcript contract audit.
- `scripts/build_task_bids_subset.py`: selective BIDS subset construction.
- `scripts/prepare_tr_aligned_transcripts.py`: word timing to TR tables.
- `scripts/run_gentle_alignment.py`: optional forced-alignment integration.

Datasets, fMRIPrep derivatives, atlas binaries, generated reports, and cluster job files are intentionally not tracked. Configure their parent with `THESIS_NEURO_DATA_ROOT`; see [`docs/data-contracts.md`](../docs/data-contracts.md) for the expected layout.
