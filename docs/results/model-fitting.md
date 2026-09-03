# Model Fitting Results

## Overview

This section summarizes the results currently available from the model-fitting pipeline. It distinguishes between:

1. completed raw-brain analyses on `shapessocial`
2. preprocessing-complete but still partial cleaned-brain analyses on `shapesphysical`
3. the current status of preliminary cleaned reruns using the updated feature outputs

The present document is intended as a thesis-ready draft results section based on the analyses that have already been executed. Where a result is still in progress, that is stated explicitly.

## Available Result Sets

At the time of writing, the following result categories are available:

1. **Raw `shapessocial`, partial run subset**
   - `structure_comparison/outputs/remote_experiment_l8_l13_l22_partial_real`
   - based on `2` resolved runs during early pipeline validation

2. **Raw `shapessocial`, current intermediate run subset**
   - `structure_comparison/outputs/remote_experiment_l8_l13_l22_current_real`
   - based on `9` resolved runs during intermediate validation

3. **Raw `shapessocial`, full run set**
   - `structure_comparison/outputs/remote_experiment_l8_l13_l22_full_real`
   - based on `59` runs and `17,995` subject-run-TR samples

4. **Raw `shapesphysical`, TR-artifact probe only**
   - `structure_comparison/outputs/remote_experiment_l8_l13_l22_shapesphysical_probe`
   - this probe successfully produced transcript-aligned TR feature artifacts, but it did not yield completed brain-model result bundles

5. **Cleaned `shapesphysical`, partial fMRIPrep bundle from batch 00**
   - `structure_comparison/brain_targets/shapesphysical_schaefer200_cleaned_batch00.npz`
   - `structure_comparison/brain_targets/shapesphysical_schaefer200_cleaned_batch00.summary.json`
   - based on `8` completed preprocessed runs

6. **Preliminary cleaned `shapesphysical` analysis**
   - currently launched against the updated `gemma_2_2b` feature run
   - output directory:
     `structure_comparison/outputs/final_output_retry_gemma_2_2b_shapesphysical_cleaned_batch00_prelim`
   - this run is in progress and therefore not interpreted below as a finished result

## Raw `shapessocial` Results

### Full-Runs Analysis

The most interpretable completed result set is the raw `shapessocial` full-runs analysis at:

`structure_comparison/outputs/remote_experiment_l8_l13_l22_full_real/analysis_summary.json`

This run used:

1. `59` runs
2. `17,995` retained subject-run-TR samples
3. transcript-derived predictor families from layers `8`, `13`, and `22`
4. an `all_layers` pooled family

The principal results were:

| Family | Brain mean test correlation | LM mean test correlation | Sample RSA | Feature-importance correlation |
|---|---:|---:|---:|---:|
| `layer8` | `0.173` | `0.233` | `0.0036` | `0.985` |
| `layer13` | `0.219` | `0.171` | `0.0117` | `0.418` |
| `layer22` | `0.223` | `0.367` | `-0.0615` | `0.849` |
| `all_layers` | `0.272` | `0.360` | `0.0167` | `0.938` |

### Main Observations From the Raw `shapessocial` Analysis

#### 1. The pooled predictor family performed best on the brain side.

The strongest brain-side held-out correlation came from the `all_layers` family (`0.272`). This exceeded the layer-specific families, suggesting that predictive signal was distributed across multiple LM depths rather than being captured by a single layer alone.

#### 2. Later-layer LM targets were easier to predict.

The strongest LM-side result came from `layer22` (`0.367`), with the pooled family close behind (`0.360`). This suggests that later-layer held-out SAE targets are more systematically recoverable from the transcript-derived feature basis than earlier-layer targets.

#### 3. Feature-importance agreement was much stronger than sample-geometry agreement.

The most encouraging cross-system result in the raw `shapessocial` analysis was the high agreement in feature importance, especially for:

1. `layer8` (`0.985`)
2. `layer22` (`0.849`)
3. `all_layers` (`0.938`)

By contrast, sample-level RSA remained near zero in all families. This indicates that the same predictors tended to matter in both systems, but the fitted sample-to-sample geometry was not yet strongly aligned.

#### 4. The strongest parcels were concentrated in visual and dorsal-attention territory.

For the raw full-runs analysis, top parcels frequently came from:

1. right visual cortex
2. right dorsal attention cortex
3. additional occipital and attentional parcels

For example:

1. `layer8` top parcel: `7Networks_RH_Vis_2` (`r = 0.348`)
2. `layer13` top parcel: `7Networks_RH_DorsAttn_Post_2` (`r = 0.464`)
3. `layer22` top parcel: `7Networks_RH_DorsAttn_Post_2` (`r = 0.468`)
4. `all_layers` top parcel: `7Networks_RH_DorsAttn_Post_2` (`r = 0.558`)

This pattern suggests that, in the raw analysis, a substantial fraction of the recoverable signal may still reflect broad stimulus-locked structure rather than exclusively high-level semantic abstraction.

#### 5. A small set of predictors dominated the shared explanatory structure.

The most prominent shared predictor in the pooled family was:

1. `layer8:feature13466`

This feature appeared as the top shared predictor in both the `layer8` family and the `all_layers` family, indicating that at least one low- to mid-layer transcript-derived feature was exerting strong influence on both brain and LM predictions.

### Interpretation of the Raw `shapessocial` Results

The raw `shapessocial` results show that the structure-comparison framework is capable of recovering nontrivial signal. In particular:

1. brain prediction is weak but not absent
2. LM prediction is moderate for several target sets
3. feature-importance structure aligns much more strongly than sample geometry

However, the raw analysis is also limited in important ways:

1. brain `R^2` values were massively negative across families
2. the data had not yet undergone full confound-cleaned preprocessing
3. the strongest parcels were concentrated in visual and attentional systems
4. representational geometry alignment remained near zero

Accordingly, these raw results should be treated as proof that the overall comparison framework can recover structure, but not yet as the strongest evidence for final thesis claims.

## Intermediate Validation Runs on `shapessocial`

Two earlier validation runs help contextualize the raw full analysis:

1. `partial_real`: `2` runs, `610` brain samples
2. `current_real`: `9` runs, `2,745` brain samples

These intermediate runs served primarily to validate:

1. pipeline execution
2. brain-target construction
3. family-level output writing
4. the stability of model-side results

As expected, the brain-side metrics varied substantially across these smaller subsets and should not be interpreted as substantive scientific findings. Their main role was engineering validation rather than final inference.

## Raw `shapesphysical` Probe Status

There is one earlier raw `shapesphysical` output directory:

`structure_comparison/outputs/remote_experiment_l8_l13_l22_shapesphysical_probe`

However, this probe should not be treated as a completed raw `shapesphysical` result set. It produced:

1. TR-level feature matrices
2. TR-level LM target matrices
3. transcript alignment summaries

It did not produce the full family-level model outputs needed for quantitative interpretation, such as:

1. `brain_cv_summary.json`
2. `lm_cv_summary.json`
3. `summary.json`

Accordingly, the raw `shapesphysical` probe is best understood as a stimulus-adaptation and engineering checkpoint rather than a finished empirical result.

## `shapesphysical` Results Status

### Cleaned Brain Data Availability

For `shapesphysical`, the current state is stronger in preprocessing quality but weaker in total sample coverage because preprocessing is still running in batches.

The currently available cleaned partial bundle is:

`structure_comparison/brain_targets/shapesphysical_schaefer200_cleaned_batch00.npz`

Its summary indicates:

1. `8` observed runs
2. `59` expected runs in the full dataset
3. `2,424` total subject-run-TR samples before censor exclusion at model-fit time
4. `2,368` retained uncensored samples
5. `200` parcels in the current cleaned bundle
6. mean censor fraction of approximately `0.023`

This means that a preliminary cleaned-brain `shapesphysical` analysis is already feasible, even though the full cleaned dataset is not yet complete.

### Ongoing Preprocessing

The remaining `shapesphysical` `fMRIPrep` preprocessing is running in sequential batches. At the time of writing:

1. `batch_00` completed successfully
2. `batch_01` is running
3. the queue watcher is active for later batches

This is methodologically important because the final cleaned `shapesphysical` result set will be substantially stronger than the current `8`-run preliminary bundle once all batches complete.

## Updated Model Runs for `shapesphysical`

The current intended comparison set for cleaned `shapesphysical` analyses is:

1. `Gemma-2-2B`
2. `Gemma-2-9B`
3. `Llama-3.1-8B`

At present:

1. `gemma_2_2b` is already local and ready
2. `gemma_2_9b` is still copying from the VM
3. `llama_3_1_8b` is still pending after the Gemma copy

The locally available updated run is:

`structure_comparison/remote_runs/final_output_retry/gemma_2_2b`

## Preliminary Cleaned `shapesphysical` Analysis

A preliminary cleaned `shapesphysical` analysis has been launched against:

1. the cleaned `batch_00` brain bundle
2. the updated `gemma_2_2b` transcript-first feature run

The output location is:

`structure_comparison/outputs/final_output_retry_gemma_2_2b_shapesphysical_cleaned_batch00_prelim`

Because this run is still in progress, it is not yet interpreted as a result in this draft section. Once it finishes, it will provide the first direct answer to the question of whether cleaned `shapesphysical` produces stronger and more trustworthy structure-comparison results than raw `shapessocial`.

## Provisional Conclusion

The current empirical picture is:

1. The raw `shapessocial` analysis demonstrates that the transcript-first structure-comparison pipeline can recover meaningful predictive signal and strong cross-system feature-importance agreement.
2. The raw `shapessocial` analysis also shows clear limitations, especially weak brain `R^2`, near-zero sample RSA, and likely contamination from broad stimulus-driven variance.
3. The cleaned `shapesphysical` pipeline is now operational and has already produced an initial `8`-run cleaned brain bundle suitable for preliminary model fitting.
4. The strongest final thesis claims should therefore depend on the cleaned `shapesphysical` results, not the earlier raw `shapessocial` analyses alone.

In other words, the raw `shapessocial` results are best understood as a successful proof of concept, while the cleaned `shapesphysical` analyses are the results that will determine the final strength of the thesis argument.
