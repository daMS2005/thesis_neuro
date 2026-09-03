# Exploratory Benchmark Comparison

This optional track asks whether the transcript-selected SAE feature basis also predicts behavioral benchmark quantities and whether fitted benchmark importance resembles fitted brain-model importance.

It is an exploratory extension, not a reported benchmark result. The repository ships implementation and an offline item fixture only.

```bash
thesis-neuro-benchmark --help
thesis-neuro-benchmark validate-items \
  --items-path examples/benchmark/mock_boolq.jsonl
```

Supported normalization code covers BoolQ, CB, COPA, MultiRC, RTE, WiC, and WSC. Preparing SuperGLUE can require dataset access; extracting features requires separately obtained model weights and selected-feature artifacts. Generated benchmark data belongs under `THESIS_NEURO_OUTPUT_ROOT` and is ignored by Git.
