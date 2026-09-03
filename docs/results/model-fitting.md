# Model Fitting Results

This page summarizes the completed structure-comparison analyses reported in the thesis. The numbers are copied from the [thesis results section](../thesis/thesis.md#results); the repository ships the code that produced them, not the data or the generated result bundles. Regenerating any table requires the external inputs described in [Data Contracts](../data-contracts.md) and the commands in [Reproducibility](../reproducibility.md).

## Scope

Two tiers of analysis were completed.

| Tier | Story | Brain targets | Runs | Samples | Models |
| --- | --- | --- | ---: | ---: | --- |
| Raw baseline | `shapessocial` | Raw Schaefer-200 parcels, no confound cleaning | 59 | 17,995 | Gemma 2 2B (layers 8, 13, 22) |
| Cleaned analysis | `shapesphysical` | fMRIPrep confound-cleaned Schaefer-200 parcels | 48 of 59 | 14,116 retained of 14,544 | Gemma 2 2B, Gemma 2 9B, Llama 3.1 8B |

The cleaned bundle covers preprocessing batches 00 through 05 with a mean censor fraction of about 0.029. Because 11 of the expected 59 runs were still unprocessed when the thesis was written, the cleaned results are strong interim results rather than the final endpoint of the study. All three cleaned model analyses share the same brain bundle, so they can be compared directly on the brain side.

Every family was evaluated with the same estimator and metrics: ridge regression with inner-loop alpha selection, leave-one-run-out cross-validation on the brain side, blocked five-fold cross-validation on the LM side, held-out Pearson correlation and R², sample-level representational similarity analysis (RSA), and Pearson correlation between brain-side and LM-side feature-importance vectors. See [Model Fitting Methodology](../methodology/model-fitting.md) for definitions.

## Raw `shapessocial` Baseline

The raw analysis was the first end-to-end run of the pipeline on real data.

**Table 1.** Family-level metrics for the raw `shapessocial` baseline.

| Family | Brain r | LM r | Sample RSA | Feature-importance r |
| --- | ---: | ---: | ---: | ---: |
| `layer8` | 0.173 | 0.233 | 0.0036 | 0.985 |
| `layer13` | 0.219 | 0.171 | 0.0117 | 0.418 |
| `layer22` | 0.223 | 0.367 | -0.0615 | 0.849 |
| `all_layers` | 0.272 | 0.360 | 0.0167 | 0.938 |

What the baseline established:

- The pooled `all_layers` family gave the highest brain correlation in the whole project (0.272), but brain R² was strongly negative in every family. The model recovered rank-order signal without explaining parcel variance in a calibrated way.
- Later-layer LM targets were easier to predict than earlier-layer targets.
- Feature-importance agreement was already high in the `layer8` and `all_layers` families, while sample RSA stayed near zero. One low-to-mid-layer feature, `layer8:feature13466`, was the top shared predictor in both families.
- The strongest parcels sat in right visual and dorsal-attention cortex (top parcel `7Networks_RH_DorsAttn_Post_2`, r = 0.558 in the pooled family), pointing to broad stimulus-locked structure rather than high-level semantics.

Two smaller `shapessocial` subsets (2 runs with 610 samples, and 9 runs with 2,745 samples) were run earlier purely to validate pipeline execution and output writing. Their metrics are not interpreted.

## Cleaned `shapesphysical` Results

### Cross-Model Comparison

**Table 2.** All-layers results for the cleaned `shapesphysical` analysis.

| Model | LM target framing | Brain r | Brain R² | LM r | LM R² | Sample RSA | Feature-importance r |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 2 2B | final hidden state | 0.1700 | 0.0337 | 0.4898 | 0.0209 | 0.0670 | 0.9584 |
| Gemma 2 9B | provisional hidden-state-style | 0.1693 | 0.0339 | 0.5659 | 0.2022 | 0.0185 | 0.9610 |
| Llama 3.1 8B | provisional hidden-state-style | 0.1699 | 0.0350 | 0.6217 | 0.3925 | 0.0500 | 0.9086 |

The final-hidden-state framing is exact for Gemma 2 2B, which was rerun with a true final-hidden-state target. For the other two models it is provisional: their hidden-state-style summaries closely match the available held-out-SAE outputs but had not been rerun at the time of writing.

Four points stand out:

1. Brain R² is positive for all three models, unlike the raw baseline. Confound cleaning made the brain-side fit interpretable.
2. The three models are nearly tied on the brain side, clustering at brain r of about 0.17 and brain R² of about 0.034.
3. The LM side separates the models clearly: Llama 3.1 8B, then Gemma 2 9B, then Gemma 2 2B.
4. Feature-importance agreement stays between 0.909 and 0.961, while sample RSA stays between 0.019 and 0.067.

### Gemma 2 2B Aggregation And Target Variants

Gemma 2 2B was rerun under three specifications with the same 128-feature predictor basis, the same 48-run brain bundle, and the same cross-validation scheme.

**Table 3.** Gemma 2 2B variants.

| Variant | Brain r | Brain R² | LM r | LM R² | Sample RSA | Feature-importance r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mass aggregation + held-out SAE targets | 0.1701 | 0.0340 | 0.4589 | -0.7917 | 0.0377 | 0.9607 |
| average aggregation + held-out SAE targets | 0.1700 | 0.0337 | 0.4345 | 0.0458 | 0.0446 | 0.9244 |
| average aggregation + final hidden state | 0.1700 | 0.0337 | 0.4898 | 0.0209 | 0.0670 | 0.9584 |

Moving from summed activation mass to per-TR averages fixed the negative LM R² without changing the brain fit. Switching the LM target to the final hidden state left the brain side untouched, kept LM R² positive, and raised both sample RSA and feature-importance agreement. The average-plus-final-hidden-state configuration is the reference Gemma 2 2B result.

### Single-Layer Findings

The pooled family was the strongest brain-side family for every model. Best single-layer values:

| Model | Best brain layer | Brain r | Brain R² | Best LM layer | LM r | LM R² | Highest sample RSA | Highest feature-importance r |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| Gemma 2 2B | 25 | 0.154 | 0.0305 | 17 | 0.389 | 0.0708 | layer 22 (0.0680) | layer 17 (0.986) |
| Gemma 2 9B | 36 | 0.152 | 0.0302 | 21 | 0.705 | 0.316 | layer 36 (0.0963) | layer 27 (0.998) |
| Llama 3.1 8B | 16 | 0.155 | 0.0298 | see note | | | layer 31 (0.0856) | |

Gemma 2 9B is the most internally coherent model: strong feature-importance agreement, the best non-anomalous LM recoverability, and the highest sample RSA in the project. Several Llama 3.1 8B LM-side layers are unusually strong relative to the brain side, so those values are reported cautiously pending an audit of the LM target construction rather than as evidence of model superiority.

### Anatomical Pattern

The cleaned all-layers runs agree closely on their top parcels across all three models:

1. `7Networks_LH_SomMot_2` (r about 0.695 to 0.697)
2. `7Networks_RH_SomMot_1` (r about 0.686 to 0.689)
3. `7Networks_RH_SomMot_2` (r about 0.648 to 0.651)
4. `7Networks_LH_SomMot_1` (r about 0.578 to 0.582)
5. left and right default-mode temporal parcels
6. salience and ventral-attention parcels in frontal and parietal opercular cortex

This somatomotor-temporal pattern replaces the visual and dorsal-attention pattern of the raw baseline. The strong somatomotor component means stimulus-locked, lower-level structure still contributes to the fit, so the result is not read as a purely semantic language network. The shift away from the raw pattern is still evidence that cleaning removed some of the broad confounding structure.

### Feature Importance Versus Geometry

Across raw and cleaned analyses and all three models, feature-importance agreement is far stronger than sample-level RSA. In the cleaned pooled families the former ranges from 0.909 to 0.961 and the latter from 0.019 to 0.067; even the strongest single-layer RSA (Gemma 2 9B layer 36, 0.0963) is modest. The brain and model fits therefore agree about which transcript-derived features matter much more than about the geometry of moment-by-moment representational states.

### Model Comparison

Gemma 2 9B improves on Gemma 2 2B on the LM side, and Llama 3.1 8B is stronger still in the current presentation, yet all three models remain nearly tied on brain prediction. Either the predictor basis already captures what parcel-level prediction can use, or the current brain targets and sample size are not sensitive enough to separate models. The data cannot yet distinguish these explanations, but they do show that better internal recoverability does not automatically buy stronger brain alignment.

## Conclusions

1. The structure-comparison framework recovers non-trivial signal: positive cleaned brain correlations, positive cleaned brain R², and high feature-importance agreement across models.
2. Preprocessing quality matters. The raw baseline had higher correlations but uninterpretable R² and a less convincing anatomy; the cleaned analysis is lower but trustworthy.
3. The strongest cross-system alignment lies in feature salience rather than sample geometry. This holds across raw and cleaned analyses and across all three models.
4. Model scaling improves the LM-side fit more clearly than the brain-side fit.
5. Average TR aggregation is more defensible than summed activation mass for this comparison.

Together these support a bounded claim: transcript-grounded interpretable features provide a partially shared explanatory basis across brains and language models, without evidence of a fully matched representational geometry. The remaining `shapesphysical` runs, once cleaned, are the natural test of whether these conclusions hold under the complete dataset.

## Output Artifacts

For each model family the analysis writes `brain_cv_summary.json`, `lm_cv_summary.json`, `brain_final_model.npz`, `lm_final_model.npz`, `brain_lm_weight_similarity.npz`, `sample_rsa.json`, `feature_importance_summary.json`, and `summary.json` under the configured output root. The notebooks in [`structure_comparison/notebooks/`](../../structure_comparison/notebooks/) read those files to produce the comparison tables above.
