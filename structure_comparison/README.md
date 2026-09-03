# Structure Comparison

This package builds transcript-aligned TR feature matrices, constructs parcel-level brain targets, fits grouped ridge encoders, and compares brain/LM model structure through learned weights, feature importance, and representational similarity.

Use the packaged interface rather than machine-specific wrappers:

```bash
thesis-neuro-structure --help
thesis-neuro-structure build-tr-artifacts \
  --feature-run-dir "$THESIS_NEURO_DATA_ROOT/feature-runs/<run-name>" \
  --stimulus-id shapesphysical
```

Focused modules expose the main contracts:

- `alignment.py`: token, word, and TR alignment.
- `artifacts.py`: feature and target bundles.
- `brain.py`: parcel extraction and confound cleaning.
- `modeling.py`: ridge validation and representational comparisons.
- `cli.py`: dependency-light command parsing.
- `workflow.py`: compatibility facade and established scientific implementation.

Inputs and generated outputs are external to Git. See [`docs/data-contracts.md`](../docs/data-contracts.md) and [`docs/reproducibility.md`](../docs/reproducibility.md).
