# Model Fitting Methodology

## Scope and Aim

This section describes the final retained model-fitting methodology for the thesis: how transcript-derived language-model features are transformed into time-resolved predictors, how those predictors are fit to both human brain responses and language-model targets, and how the resulting fitted structures are compared across systems. The central methodological claim is not that raw brain recordings and raw model activations are directly identical. Instead, the analysis asks whether the same transcript-grounded feature basis predicts both systems and whether the fitted structures learned from that basis are organized similarly.

The comparison is therefore an encoding-style framework built around a shared stimulus space. A common predictor matrix is derived from transcript-aligned sparse autoencoder (SAE) features, and that matrix is used to fit:

1. Brain encoding models, where the targets are cleaned fMRI responses.
2. Language-model analog models, where the targets are held-out SAE feature time courses from the same model family.

The main text below describes the final analysis path rather than every intermediate experiment. Earlier alternatives, engineering constraints, and setup-specific details are summarized in an appendix-style section at the end.

## Models Included

The comparison uses transcript-first feature runs produced from three updated model outputs:

1. `Gemma-2-2B`
2. `Gemma-2-9B`
3. `Llama-3.1-8B`

Each model contributes a feature-discovery directory containing:

1. `transcript_paired_records.jsonl`
2. `selected_features_for_alignment.jsonl`
3. `feature_alignment.jsonl`
4. `feature_concepts.jsonl`
5. `feature_relevance.jsonl`
6. `transcript_feature_shortlist.jsonl`
7. `transcript_feature_stats.jsonl`
8. `manifest.json`

These artifacts define the predictor feature library, the judged concepts associated with each feature, and the transcript-window activations used to reconstruct time-resolved regressors.

## Stimulus Set and Shared Scaffold

The thesis is grounded in the Narratives dataset (`ds002345`). Within that dataset, `shapesphysical` serves as the primary cleaned-analysis story because it combines transcript regularity, protocol consistency, and a finished cleaned brain bundle. `shapessocial` was retained as the closest matched companion story, while `black`, `bronx`, `forgot`, and `piemanpni` expand the broader Narratives stimulus base beyond a single story.

The analysis depends on the transcript as a shared explanatory scaffold:

1. Human participants listened to the same naturalistic story stimuli.
2. SAE features were selected from transcript-linked model activity.
3. Brain responses and LM-side targets were both aligned back to the same TR grid.

This transcript-grounded design makes it possible to compare fitted structure across systems without claiming that the raw signals themselves are directly commensurate.

## Feature Library Construction

### Predictor Features

Predictor features are taken from `selected_features_for_alignment.jsonl`. These are treated as the main interpretable predictor bank because they are already filtered for transcript relevance and alignment quality.

Predictor keys are ordered by:

1. `transcript_relevance_rank`
2. `layer`
3. `feature_id`

The main analysis layers are:

1. layer `8`
2. layer `13`
3. layer `22`

Feature families are fit in four groupings:

1. `layer8`
2. `layer13`
3. `layer22`
4. `all_layers`

The predictor library can be truncated before family construction by retaining only the top `k` transcript-ranked predictors. The retained mainline setting is `k = 128`, while smaller predictor banks are treated as sensitivity analyses rather than the primary specification.

### Held-Out LM Targets

The LM-side targets are not the same features used as predictors. Instead, they are a disjoint target panel drawn from `feature_concepts.jsonl` under the following constraints:

1. the feature must belong to one of the analysis layers (`8`, `13`, `22`)
2. the feature must not already be in the predictor set
3. the feature must have `judge_status == "ok"`
4. features are ranked by descending judge confidence
5. ties are broken by `transcript_relevance_rank`

The default target panel size is:

1. `8` LM targets per layer
2. `24` LM targets total in the pooled `all_layers` family

This separation prevents the LM regression from collapsing into a trivial self-prediction problem.

## Transcript-to-TR Alignment

### Token Reconstruction

For each run directory, transcript activations are read from `transcript_paired_records.jsonl`. The workflow reconstructs a global token sequence for the stimulus using the minimum analysis layer as a canonical token stream. Global token indices are reconstructed from:

1. `window_start`
2. `token_position`

The implementation requires:

1. contiguous global token indices
2. token reconstruction beginning at index `0`
3. consistency of token identity at every global position

### Tokenizer-Specific Word Alignment

Token-to-word alignment is model specific. Rather than assuming a generic tokenizer or a simple token-prefix heuristic, the transcript text is re-tokenized using the cached tokenizer associated with each model family, and the resulting tokens are checked against the stored artifact tokens. This makes the alignment methodologically important rather than incidental, because Gemma and Llama tokenizers do not segment text in exactly the same way.

After model-specific tokenization, token groups are matched to timed words under two hard validation checks:

1. token-derived word count must equal timed word count
2. normalized token-derived words must match normalized timed words

The timed words must also reconstruct the normalized transcript text.

### TR Assignment

Each timed word is assigned to a TR bin by the midpoint of its start and end times, shifted by the stimulus onset time taken from the transcript metadata. This onset correction is especially important for the Shapes stories because both `shapesphysical` and `shapessocial` contain an introductory music segment before the main story content.

The result is a deterministic mapping:

1. token -> word
2. word -> TR

### TR-Level Feature Views

For each selected feature and each TR, the workflow stores several summary views, including:

1. `presence`
2. `mass`
3. `average`
4. `peak`
5. `count`

The retained analysis view is `average`, which divides TR-level activation mass by the number of active token events in that TR. This reduces the extent to which predictor magnitude simply scales with token density or total activation mass. Summed activation (`mass`) was retained only as an earlier alternative and sensitivity condition, because it made generalization harder to interpret when stories differed in token count or total activation magnitude.

The same transcript-to-TR alignment is used to construct the LM target matrix, and the retained LM target view is likewise the TR-level `average`.

## Brain Target Construction

### Cleaned Derivatives

Brain targets are built from `fMRIPrep` derivatives in `MNI152NLin6Asym` space rather than from raw BOLD files. The primary cleaned cortical bundle for the current structure-comparison analyses is the `shapesphysical` Schaefer-200 target set, which currently contains `48` cleaned runs out of `59` expected runs. This is the cortical bundle used by the main retained analyses.

The methods therefore describe the finalized analysis path using the cleaned bundles that were actually available, rather than implying that all `59` `shapesphysical` runs were complete at fit time.

### Main Cortical Target Space

The main target space uses the Schaefer 2018 atlas:

1. `200` cortical parcels
2. `7` large-scale functional networks
3. MNI `1 mm` atlas image

Parcels are extracted with `NiftiLabelsMasker` from preprocessed BOLD runs. At extraction time, the masker is configured with:

1. no smoothing
2. no detrending
3. no temporal filtering
4. no standardization

This keeps sampling separate from denoising and standardization so that the cleaning stage remains explicit and reproducible.

### Confound Regression and Temporal Cleaning

For each subject-run, nuisance regressors are built from the `fMRIPrep` confounds table using:

1. the `24`-parameter motion model
2. the first `6` aCompCor components

The cleaned parcel matrix is then produced with:

1. confound regression
2. detrending
3. high-pass filtering at `0.008 Hz`

No global signal regression is applied in the retained protocol.

### Censoring

A TR is marked censored if any of the following are true:

1. framewise displacement `> 0.5`
2. standardized DVARS `> 1.5`
3. any `non_steady_state_outlier` flag is present

Censored TRs are not physically removed from the saved bundle. Instead:

1. the full TR grid is preserved
2. censor flags are stored alongside the targets
3. censored samples are excluded only at model-fitting time

### Run-Wise Standardization and Target Matrix

After cleaning, parcel time series are z-scored within run using means and standard deviations estimated from uncensored TRs only. Because atlas resampling can cause parcel labels to vary slightly across runs, the final cortical bundle keeps only the intersection of labels present across the included runs.

The resulting target matrix stores one row per:

1. subject
2. run
3. TR

Core arrays include:

1. `values`
2. `sample_ids`
3. `subject_ids`
4. `run_ids`
5. `tr_indices`
6. `target_names`

Optional vectors include:

1. `censor_mask`
2. `framewise_displacement`
3. `std_dvars`

This representation preserves the repeated-measures structure needed for leave-one-run-out evaluation.

## Non-Cortical ROI Sensitivity Analysis

The main target space is cortical, but a secondary sensitivity track was added to test whether important signal was being missed outside Schaefer cortex. This secondary bundle uses bilateral spherical ROIs centered on:

1. hippocampus
2. amygdala
3. thalamus
4. striatum
5. cerebellum

The current non-cortical bundle contains `51` cleaned `shapesphysical` runs. It is treated as a sensitivity analysis rather than the primary target space, but it directly addresses the possibility that a cortex-only atlas could hide meaningful non-cortical structure.

## Predictor Design for Brain and LM Models

### Brain Models

The transcript-derived predictor matrix is expanded with a finite impulse response lag basis before fitting to the brain data. The retained lag set is:

1. `0`
2. `1`
3. `2`
4. `3`
5. `4`

For each original feature, the brain encoding model therefore receives five lagged copies. This allows the model to absorb the delayed and temporally extended nature of the BOLD response without imposing a single fixed canonical hemodynamic response function.

Brain design rows are retained only when:

1. the target TR exists in the transcript-aligned predictor matrix
2. the corresponding brain sample is not censored

Evaluation groups on the brain side are defined as `subject_id:run_id`, so each outer split leaves out an entire run from one subject.

### Analogous LM Models

The LM-side model uses the same transcript-derived predictor basis as the brain model, but without HRF lag expansion. Its target matrix consists of held-out SAE feature time courses aligned to the same TR grid.

This LM model is analogous to the brain encoding model in three ways:

1. it uses the same predictor family
2. it predicts a separate target system
3. it is fit with the same estimator family

The point is not to maximize the LM benchmark independently, but to keep the estimator class and predictor basis comparable across brain and LM fits.

## Estimator and Cross-Validation

### Ridge Regression

Both brain and LM models use ridge regression:

$$
\hat{W} = \arg\min_W \|Y - XW\|_2^2 + \alpha \|W\|_2^2
$$

Ridge regression was chosen because:

1. transcript-derived feature libraries are correlated
2. a stable linear estimator is easier to interpret than a more flexible nonlinear alternative
3. fitted weights can be compared directly across target systems

Within each fit:

1. `X_train` is z-scored column-wise
2. `Y_train` is z-scored column-wise
3. evaluation data are transformed using training means and scales only

The default ridge penalty grid is:

1. `0.1`
2. `1.0`
3. `10.0`
4. `100.0`
5. `1000.0`

An inner cross-validation loop selects the best alpha on the training data, and the reported family-level alpha is the consensus alpha most often selected across outer folds.

### Brain-Side Cross-Validation

The brain encoding model uses grouped cross-validation:

1. outer split: leave-one-run-out
2. inner split: leave-one-run-out within the training set

This keeps evaluation aligned with the repeated-measures structure of the fMRI data and prevents leakage across adjacent TRs from the same run.

### LM-Side Cross-Validation

The LM analog model uses blocked cross-validation over the transcript-aligned TR sequence:

1. `5` contiguous folds
2. folds defined as ordered TR blocks rather than random samples

This blocked design reduces optimistic leakage from temporal autocorrelation. LM-side leave-one-sample-out variants were exploratory and are not part of the retained mainline methodology.

## Model Families and Evaluation Metrics

### Model Families

Each feature run is fit as four families:

1. `layer8`
2. `layer13`
3. `layer22`
4. `all_layers`

For layer-specific families, both the predictor matrix and the LM target matrix are restricted to the corresponding layer. For `all_layers`, all selected predictors and all held-out LM targets are pooled.

### Prediction Metrics

For both brain and LM fits, the main held-out metrics are:

1. Pearson correlation
2. coefficient of determination (`R^2`)

Metrics are computed separately for each target and then averaged across targets. On the brain side, targets are parcels or ROIs; on the LM side, targets are held-out SAE features.

### Weight-Space Comparison

Brain and LM weight matrices are compared after collapsing lagged brain weights back to base features. The lag-collapsed brain weights are then compared to LM weights using cosine similarity, producing a parcel-by-LM-target similarity matrix.

### Feature-Importance Agreement

Feature importance is defined as the L2 norm of the fitted weights for each predictor feature. Brain-side and LM-side importance vectors are compared using Pearson correlation to test whether the same transcript-derived features matter in both systems.

### Sample Geometry

Predicted brain responses are averaged by TR, aligned to the LM prediction TR axis, and converted into sample-similarity matrices. The upper triangles of the resulting matrices are then correlated to produce a sample-level representational similarity analysis (RSA) score.

## Sensitivity Analyses

The main thesis methodology centers the retained analysis path, but several targeted sensitivities are methodologically important:

1. `average` versus `mass` TR aggregation
2. `k = 128` versus `k = 32` predictor truncation
3. cortical Schaefer targets versus non-cortical ROI targets

These sensitivity checks are used to test robustness, not to redefine the mainline method. In particular, the retained specification keeps `average` aggregation and `k = 128` because they preserve a stronger and more interpretable comparison basis across stories and models.

## Output Artifacts

For each model family, the workflow writes:

1. `brain_cv_summary.json`
2. `lm_cv_summary.json`
3. `brain_final_model.npz`
4. `lm_final_model.npz`
5. `brain_lm_weight_similarity.npz`
6. `sample_rsa.json`
7. `feature_importance_summary.json`
8. `summary.json`

These outputs support both quantitative reporting and later visualization.

## Recommended Thesis Wording

If this section is integrated directly into the thesis, the core methodological claim can be framed as follows:

1. transcript-derived interpretable LM features were used as a common explanatory basis
2. the same feature basis was fit to both cleaned brain targets and held-out LM targets
3. comparison was performed in prediction space, weight space, feature-importance space, and representational space

This makes clear that the contribution is not merely another encoding model, but a structured comparison pipeline for testing whether transcript-grounded feature organization is shared across human neural responses and language-model internal representations.

## Appendix A. Compute and Execution Logistics

The engineering details below are included for reproducibility, but they should not carry the interpretive weight of the main methods.

Large-scale transcript feature extraction and remote experiment storage were run on remote compute resources, while fMRI preprocessing, cleaned target construction, and many local comparison runs were executed from the project workspace. Early `shapesphysical` preprocessing was staged in conservative subject batches because `fMRIPrep` was being executed through a Linux AMD64 container on Apple Silicon hardware. Those batching decisions affected workflow stability and which cleaned bundles were available at a given time, but they did not change the conceptual comparison framework described above.
