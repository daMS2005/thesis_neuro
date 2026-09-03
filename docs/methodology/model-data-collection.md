# Model Data Collection Methodology

## Purpose of This Section

This section documents the model-side data collection procedure used to generate interpretability artifacts for the thesis. The goal of this stage was to identify, characterize, and interpret sparse autoencoder (SAE) features that were recruited by a small set of experimental stimulus transcripts, and then to contextualize those features using an external natural-language corpus. The resulting artifacts were intended to support later qualitative and quantitative analyses of feature content, feature selectivity, and transcript-specific activation structure.

The overall methodological design was informed most directly by the recent large-scale SAE literature, especially work on scalable SAE extraction and evaluation, large feature inventories, and automated feature interpretation (Cunningham et al., 2023; Gao et al., 2024; Anthropic, 2024; Paulo et al., 2024). The present workflow did not simply reproduce those pipelines. Instead, it adapted them into a transcript-first methodology in which experimental stimuli determined which features mattered, while external corpora and judge-style interpretation were used only after transcript relevance had already been established.

The pipeline was designed to answer four related questions:

1. Which latent features in the target language models are activated by the experimental transcripts?
2. Which of those features are sufficiently salient, persistent, or locally distinctive to merit further analysis?
3. What broader semantic or syntactic contexts are associated with those features in a large external corpus?
4. Can the resulting evidence be organized into a compact set of interpretable feature-level summaries suitable for later alignment, ablation, and manual inspection?

The methodology described here covers the complete model data collection stage for three model families:

- `google/gemma-2-2b`
- `google/gemma-2-9b`
- `meta-llama/Llama-3.1-8B`

For Gemma models, the corresponding SAE family was Gemma Scope. For Llama, the corresponding SAE family was Llama Scope.

## Experimental Inputs

### Transcript Stimuli

The transcript-first stage of the pipeline operated over a directory of experimental stimulus transcripts stored under:

`eda_brain_data/datasets/data/transcripts`

Files were selected with the glob pattern:

`*/*_transcript.txt`

The transcript run configuration allowed up to 10 files, but the final successful runs processed 6 transcript documents. These corresponded to the transcript set used throughout the broader project, including stimuli such as:

- `alice`
- `black`
- `bronx`
- `shapesphysical`
- `shapessocial`

The transcript stage was treated as the primary discovery corpus. In other words, the pipeline was not designed to discover arbitrary globally strong features first and only later ask whether they mattered for the transcripts. Instead, the methodology was intentionally transcript-centered: the transcripts were the entry point for feature discovery.

### External Interpretation Corpus

After transcript feature discovery, shortlisted features were contextualized using the Dolma corpus:

- dataset: `allenai/dolma`
- split: `train`
- streaming mode: enabled
- text field: `text`

Dolma was used as a large-scale interpretation corpus rather than a discovery corpus. This distinction is methodologically important. The transcript stimuli were used to determine which features mattered for the experiment, while Dolma was used to provide broader natural-language contexts for those already-selected features.

## Models and SAE Releases

### Base Models

Three base models were used:

1. `google/gemma-2-2b`
2. `google/gemma-2-9b`
3. `meta-llama/Llama-3.1-8B`

All three were used in causal language modeling mode via Hugging Face Transformers.

### Sparse Autoencoders

The following SAE releases were paired with the base models:

- Gemma 2 2B: `gemma-scope-2b-pt-res-canonical`
- Gemma 2 9B: `gemma-scope-9b-pt-res-canonical`
- Llama 3.1 8B: `llama_scope_lxr_8x`

The Gemma runs used residual-stream SAEs with width:

- `width_16k`

The Llama run used Llama Scope with width:

- `8x`

Each run used SAE encoding over selected residual layers rather than all possible layers. Layer subsets were not chosen arbitrarily; they were computed by relative depth so that early-, mid-, and late-model processing stages were all sampled in a comparable way across architectures.

## Layer Selection Strategy

Rather than exhaustively processing every layer, six target depths were selected per model using relative fractions of the model depth:

- 16%
- 30%
- 50%
- 65%
- 85%
- 96%

These fractions were converted to layer indices by rounding against the model’s `num_hidden_layers`, then validating that the requested layers existed in the corresponding SAE release.

This produced the following final layer sets:

- Gemma 2 2B, 26 layers total:
  - `4, 8, 13, 17, 22, 25`
- Gemma 2 9B, 42 layers total:
  - `7, 13, 21, 27, 36, 40`
- Llama 3.1 8B, 32 layers total:
  - `5, 10, 16, 21, 27, 31`

This strategy ensured that the final dataset sampled comparable functional depths across models while keeping the total run budget tractable.

## Hardware and Runtime Environment

### Final Successful Runtime

The final successful runs were executed on a Google Cloud VM with the following effective hardware characteristics:

- GPU: `NVIDIA L4`
- GPU memory: approximately `24 GB` VRAM
- System RAM: approximately `15 GiB`
- Swap: `32 GiB` swap file added during the run
- Root disk: `246 GB`

This hardware setup is important because it directly affected the engineering decisions made during collection.

### Why Hardware Details Matter

The Gemma 2 9B run repeatedly failed early in the project when the system attempted to load the full model through a CPU-heavy path on a machine with limited host RAM. The core issue was not that the L4 GPU was categorically incapable of running a 9B model. The problem was that the original loading path transiently required too much host memory while materializing model weights.

To address this, two runtime changes were introduced:

1. low-memory model loading
2. a 32 GiB swap file on the VM

The low-memory loading path used:

- `low_cpu_mem_usage=True`
- direct CUDA device placement during model load

This avoided fully loading the model into CPU memory before moving it to the GPU. After this change, Gemma 2 9B completed successfully on the same machine.

### Disk Utilization at Completion

At the end of the full retry run, the VM’s root disk was approximately:

- total: `246G`
- used: `186G`
- free: `51G`

Major storage contributors were:

- Hugging Face cache: `68G`
- `final_output_retry`: `22G`
- older `final_output`: `31G`
- earlier `outputs`: `16G`
- Python environment: `8.4G`

The three final retry output directories were individually about:

- Gemma 2 9B: `7.2G`
- Gemma 2 2B: `7.0G`
- Llama 3.1 8B: `7.4G`

The largest files were the transcript paired-record artifacts, each slightly above 5 GB.

## Software and Runtime Configuration

### Core Runtime Parameters

The final retry configuration used the following key settings:

- `seq_len: 1024`
- `add_special_tokens: false`
- token-level latent retention: `128`
- pooled latent retention: `128`
- top logits retained per token: `10`
- random seed: `42`

The final retry run wrote all outputs into:

`final_output_retry/`

with separate subdirectories for:

- `gemma_2_9b`
- `gemma_2_2b`
- `llama_3_1_8b`

### Feature Selection Parameters

For transcript feature selection, the pipeline retained and ranked candidate features using multiple summary criteria:

- top by peak activation: `96`
- top by total activation: `96`
- top by persistence: `96`
- top by sentence-pooled activation: `96`
- final shortlist per layer: `192`
- example rows retained per feature during transcript discovery: `8`

This produced per-model transcript shortlists that were later used for both Dolma collection and downstream analysis.

### Dolma Context Collection Parameters

For the final retry run, Dolma interpretation used:

- `max_windows: 10000`
- `top_contexts_per_feature: 25`
- token context radius: `4`
- top tokens per context: `5`
- top spans per context: `3`
- top sentences per context: `3`
- minimum activation threshold: `0.0`

The Dolma stage was therefore bounded but still large enough to gather a rich set of external contexts for each shortlisted feature.

### Analysis and Judge Parameters

For downstream analysis:

- top features for correlation: `96`
- top features for alignment: `128`
- top features for judge: `400`
- transcript examples per judged feature: `4`
- Dolma contexts per judged feature: `6`
- correlated features per judged feature: `6`

The LLM-as-judge stage used:

- model: `gpt-5-mini`
- timeout: `120 s`
- max retries: `6`
- max concurrency: `3`

### Alignment Parameters

Alignment was run as a later feature-focused stage rather than the main discovery stage. Parameters were:

- top features per window: `128`
- top windows per feature: `3`
- top token positions per feature-window: `16`
- top token alignments retained: `6`
- top span alignments retained: `6`

Alignment methods used in the final retry run:

- `deletion_retokenize`
- `pad_sentence_mask`
- `delete_clause_retokenize`

## Transcript Processing Procedure

### Transcript-First Discovery

For each transcript document:

1. The full text was tokenized with the model tokenizer.
2. A transcript window length was chosen as:
   - the minimum of:
     - the model’s maximum position capacity
     - the configured `seq_len`
3. If the transcript length exceeded that window size, the transcript was chunked into sequential windows.
4. Each window was decoded back to text and annotated with sentence and clause boundaries.

This transcript-windowing behavior was important for two reasons:

- it prevented windows from exceeding the model’s context size
- it preserved long transcripts while keeping the window definition explicit and reproducible

### Sentence and Clause Annotation

For transcript runs, sentence segmentation was performed with spaCy:

- model: `en_core_web_sm`

This transcript-specific choice was made because the transcript audit workflow required better sentence and clause structure than the lightweight heuristics used elsewhere.

The process was:

1. tokenize the text window
2. recover token-to-character offset mappings
3. segment the decoded text into sentences with spaCy
4. map sentence character spans back to token spans
5. derive clause spans within the sentence structure

This yielded:

- token-level sentence IDs
- sentence spans
- clause spans

These metadata were then stored alongside the latent activations.

### Per-Window Model Pass

For each transcript window and each selected layer:

1. the model produced hidden states and logits
2. the corresponding SAE encoded the residual stream for that layer
3. negative latent values were clamped away for positive-only feature summaries
4. token-level latent activations were retained up to the configured top-k
5. pooled sentence-level and window-level summaries were computed

The resulting transcript artifact preserved, for each token:

- token identity
- top latent activations
- sentence membership
- sentence and clause metadata
- local top logits

This structure was intentionally self-contained so that later stages did not need to rerun transcript extraction in order to inspect token-, sentence-, or window-level evidence.

## Transcript Feature Aggregation

After all transcript windows were processed, the pipeline aggregated feature-level statistics across the full transcript set.

For each `(layer, feature_id)` pair, the following kinds of information were accumulated:

- total positive activation
- peak activation
- active-token fraction
- sentence-pooled behavior
- persistence-like measures across windows
- representative high-activation transcript examples

The aggregated statistics were written to:

- `transcript_feature_stats.jsonl`

A shorter, ranked subset was then written to:

- `transcript_feature_shortlist.jsonl`

This shortlist served as the entry point to Dolma interpretation.

## Dolma Interpretation Procedure

### Rationale

The Dolma stage was intentionally downstream of transcript discovery. The logic was:

- the transcripts determine which features matter for the experiment
- Dolma helps explain what those features tend to mean more broadly in natural language

This avoids a methodology in which model features are first discovered generically and only later tested for transcript relevance.

### Procedure

For each shortlisted feature:

1. Dolma was streamed document-by-document.
2. The same base model and matching SAE were applied to the text windows.
3. Feature activations were monitored only for shortlisted features.
4. High-scoring windows were retained in a bounded set per feature.
5. For each retained context, multiple levels of evidence were preserved:
   - token-level snippets
   - span-level snippets
   - sentence-level snippets
   - document-level snippets

The Dolma outputs were written to:

- `dolma_feature_contexts.jsonl`

These contexts were later used for both human inspection and LLM-based conceptual labeling.

## Feature Analysis Procedure

After transcript and Dolma collection, the pipeline ran a feature analysis stage. This stage had several subgoals:

1. rank transcript-relevant features
2. compute correlations between highly ranked features
3. assemble judge inputs from transcript and Dolma evidence
4. obtain conceptual labels from the LLM judge
5. select a reduced set of features for alignment

### Streaming and Resume-Safe Analysis

This stage was refactored during the project to avoid loading all transcript and Dolma artifacts into memory at once. The final version processed the large JSONL artifacts incrementally and was resume-safe.

This was crucial because earlier versions of the analysis stage were terminated by the operating system (`exit code 137`) when attempting to process the full artifacts too eagerly on limited-RAM hardware.

### Outputs

The analysis stage produced:

- `feature_relevance.jsonl`
- `feature_correlations.jsonl`
- `feature_judge_input.jsonl`
- `feature_concepts.jsonl`
- `selected_features_for_alignment.jsonl`
- `selected_features_for_alignment_from_judge.jsonl`

## LLM-as-Judge Interpretation

The LLM judge was used as a structured conceptual interpretation step rather than as a discovery mechanism. It received compact evidence bundles assembled from:

- transcript relevance summaries
- transcript examples
- Dolma contexts
- correlated features
- alignment evidence when available

The judge model:

- `gpt-5-mini`

was asked to generate structured outputs describing likely feature meaning, evidence in favor of that meaning, evidence against it, uncertainty, and follow-up questions. The resulting rows were stored in:

- `feature_concepts.jsonl`

This stage was rate-limited and retry-aware. The final successful configuration used low concurrency and incremental writes so that partially completed judge runs could be resumed if needed.

## Alignment and Ablation Procedure

Alignment was the final stage in the collection pipeline. It was not used to discover all features. Instead, it was used to provide more localized evidence for a selected subset of already-ranked features.

The alignment stage used transcript windows and a narrowed feature set selected from the judged feature outputs. For each target feature, it retained the best windows and token positions, then evaluated localized perturbation methods such as token deletion and sentence masking.

The final alignment artifact was:

- `feature_alignment.jsonl`

This produced feature-centered evidence about where transcript features appeared to be grounded in the text.

## Final Successful Outputs

The final retry run completed successfully for all three models:

- `final_output_retry/gemma_2_9b`
- `final_output_retry/gemma_2_2b`
- `final_output_retry/llama_3_1_8b`

Each final model directory contained the complete artifact set:

- `transcript_paired_records.jsonl`
- `transcript_feature_stats.jsonl`
- `transcript_feature_shortlist.jsonl`
- `dolma_feature_contexts.jsonl`
- `feature_relevance.jsonl`
- `feature_judge_input.jsonl`
- `feature_concepts.jsonl`
- `feature_correlations.jsonl`
- `selected_features_for_alignment.jsonl`
- `selected_features_for_alignment_from_judge.jsonl`
- `feature_alignment.jsonl`
- `manifest.json`

This meant that the final data collection stage was completed end-to-end for:

- two Gemma models at different scales
- one Llama-family model
- matched SAE-based feature extraction
- transcript-first discovery
- Dolma-based interpretation
- judge-based conceptual summarization
- feature-focused alignment

## Practical Challenges and Methodological Adjustments

Several methodological decisions were shaped by engineering constraints encountered during the project:

### 1. Model loading for larger models

Gemma 2 9B initially failed during weight loading on the available VM. This was resolved by:

- switching to a low-memory load path
- loading directly to CUDA
- adding 32 GiB of swap

This is worth documenting because the final 9B methodology was not simply “run the original code on larger hardware.” The loading path was adapted to make the run feasible on a constrained system.

### 2. Analysis stage memory pressure

The initial analysis implementation attempted to hold too much transcript and Dolma evidence in memory simultaneously. This caused repeated `exit code 137` failures. The final methodology therefore used a streaming analysis path with explicit sub-stages and resumability.

### 3. Queueing and orchestration

Model runs were executed sequentially rather than in parallel. This minimized resource contention on the single-GPU VM and simplified recovery. When orchestration scripts misidentified stale processes, models were relaunched directly rather than rerunning already-finished earlier stages.

These adjustments did not change the conceptual goals of the pipeline, but they were part of the actual data collection methodology and should therefore be reported transparently.

## Summary

In summary, the model data collection stage used a transcript-first interpretability workflow across three language models and their corresponding SAE releases. Transcript stimuli were processed first to discover experiment-relevant latent features. Those features were then contextualized using Dolma, summarized using a structured LLM judge, and localized using feature-focused alignment procedures. The final successful run produced a complete, multi-stage interpretability dataset for Gemma 2 2B, Gemma 2 9B, and Llama 3.1 8B, all stored as model-specific JSONL artifact bundles suitable for downstream analysis and thesis reporting.
