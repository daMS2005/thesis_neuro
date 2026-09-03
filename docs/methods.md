# Methods And Results

A compact description of what the pipeline measures and what it found. Numbers come from the thesis analyses; regenerating them needs the external inputs listed in [Data Contracts](data-contracts.md).

## Question

Do a language model and a human brain rely on the same interpretable features while processing the same story? The comparison is made between fitted encoding models, not between raw activation tensors: one transcript-derived feature basis is fit to both systems, and the fitted structures are compared.

## Data

| Source | Details |
| --- | --- |
| Brain | OpenNeuro Narratives (`ds002345`), stories `shapesphysical` (primary) and `shapessocial`; fMRIPrep derivatives in MNI space; Schaefer 2018 atlas with 200 cortical parcels |
| Models | Gemma 2 2B and Gemma 2 9B with Gemma Scope residual-stream SAEs; Llama 3.1 8B with Llama Scope |
| Layers | Six per model at 16, 30, 50, 65, 85, and 96 percent of depth, so early, middle, and late processing are sampled comparably across architectures |
| Corpus | Dolma, streamed, used only to explain features that the transcripts had already selected |

## Pipeline

1. **Transcript-first feature discovery.** Each transcript is windowed to the model's context length, passed through the model, and encoded by the SAE at the selected layers. Features are ranked by peak activation, total activation, persistence across windows, and sentence-pooled activation, then shortlisted per layer.
2. **External contexts and labels.** Dolma is streamed once for the shortlisted features, keeping the highest-activating token, span, sentence, and document snippets. A judge model turns the transcript and Dolma evidence into a concept label with supporting and opposing evidence and an uncertainty note.
3. **Localization and probing.** Counterfactual edits (token deletion, sentence masking, clause deletion) measure where each feature is grounded in the text. A multi-round probing agent can then test a hypothesis with synthetic probes, real transcript edits, and activation steering, writing evidence, tests, and a report with a confidence in [0, 1].
4. **TR-level predictors.** Tokens are regrouped into words with the model's own tokenizer, words are assigned to repetition-time (TR) bins by midpoint after the stimulus onset offset, and per-feature activations are averaged within each TR. The retained predictor bank is the top 128 transcript-ranked features.
5. **Brain targets.** Parcel time series are extracted from fMRIPrep outputs without smoothing or filtering, then cleaned with a 24-parameter motion model, six aCompCor components, detrending, and a 0.008 Hz high-pass filter. TRs with framewise displacement above 0.5 mm, standardized DVARS above 1.5, or non-steady-state flags are censored at fit time. Series are z-scored within run on uncensored TRs.
6. **Matched ridge models.** The brain model receives the predictors expanded with lags of 0 to 4 TRs and is evaluated with leave-one-run-out cross-validation. The language-model analog uses the same predictors without lags against held-out targets from the same model (SAE features not in the predictor set, or the final hidden state) with blocked five-fold cross-validation. Both use ridge regression with inner-loop alpha selection over five penalties.
7. **Comparison.** Held-out Pearson r and R² per target; cosine similarity between lag-collapsed brain weights and LM weights; Pearson correlation between the two systems' feature-importance vectors (L2 norm of each predictor's weights); and sample-level representational similarity analysis (RSA) between predicted brain and LM responses.

## Results

**Raw baseline.** The first end-to-end run used the `shapessocial` story with uncleaned parcels (59 runs, 17,995 samples). The pooled predictor family reached a brain correlation of 0.272, but brain R² was strongly negative and the strongest parcels sat in visual and dorsal-attention cortex, pointing to broad stimulus-locked variance rather than semantic structure.

**Cleaned analysis.** The main result uses the `shapesphysical` story with confound-cleaned parcels (48 of 59 runs, 14,116 retained samples) and the pooled all-layers predictor family for all three models.

| Model | Brain r | Brain R² | LM r | LM R² | Sample RSA | Feature-importance r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 2 2B | 0.1700 | 0.0337 | 0.4898 | 0.0209 | 0.0670 | 0.9584 |
| Gemma 2 9B | 0.1693 | 0.0339 | 0.5659 | 0.2022 | 0.0185 | 0.9610 |
| Llama 3.1 8B | 0.1699 | 0.0350 | 0.6217 | 0.3925 | 0.0500 | 0.9086 |

The Gemma 2 2B row uses a final-hidden-state target; the other two rows use the closest available hidden-state-style presentation.

**Aggregation matters.** For Gemma 2 2B, switching from summed activation mass to per-TR averages moved LM R² from -0.79 to a positive value without changing the brain fit, and switching the LM target to the final hidden state raised both RSA and feature-importance agreement.

**Anatomy.** All three cleaned models agree on their top parcels: bilateral somatomotor cortex, then default-mode temporal parcels and salience or ventral-attention parcels in opercular cortex. The visual and dorsal-attention pattern of the raw baseline disappears after cleaning.

## Interpretation

- Confound cleaning turned an uninterpretable fit into a calibrated one: positive brain R² and a stable anatomical pattern.
- The strongest cross-system regularity is which features matter, not the geometry of moment-by-moment states. Feature-importance agreement stays above 0.9 for every model; sample RSA stays below 0.1.
- Model scale improves the language-model side far more than the brain side. All three models tie on brain prediction.
- Averaging activations within a TR is more defensible than summing them for this comparison.

## Limitations

- Ridge prediction and RSA are associational. They do not establish that a feature causally controls a model or a brain response.
- Hemodynamic lag is handled with explicit lagged design matrices, not estimated per feature.
- Judge labels and probe reports can inherit language-model errors.
- The cleaned analysis covers 48 of 59 runs, so its numbers are interim.
- The strong somatomotor component means stimulus-locked, lower-level structure still contributes to the fit.
