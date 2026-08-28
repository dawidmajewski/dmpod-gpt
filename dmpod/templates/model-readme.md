$front_matter# $model_name

<!-- Review the intended use, limitations, license, and repository file names before publication. -->

This decoder-only causal language model was pretrained using a pinned nanoGPT
implementation. The generated card contains recorded run data; edit the prose
to add model-specific context before publishing.

$completion_notice

## Training Results

$results_table

## Training Curves And W&B Media

$wandb_section

The generated SVG curves use the same local scalar stream written to
`metrics.jsonl` and sent to W&B. Additional images below were copied from local
W&B media or directories supplied with `--wandb-media-dir`.

$training_media

## Benchmarks

$benchmark_note

$benchmark_table

### All Recorded Benchmark Metrics

$benchmark_metrics

$benchmark_links

## Model Architecture

$model_table

## Initialization

$source_table

## Training Setup

### Experiment

$experiment_table

### Optimizer And Schedule

$training_table

### Batch And Token Budget

$batch_table

### Runtime

$runtime_table

### Evaluation

$evaluation_table

### Logging

$logging_table

### W&B Configuration

$wandb_table

## Data And Tokenizer

$data_table

## Checkpoint Policy

$checkpoint_policy_table

## Published Checkpoints

$checkpoint_table

## Using The Native Checkpoint

$checkpoint_usage

## Reproduce Or Continue

```bash
$reproduce_command
```

The immutable configuration, source hashes, metric-file hash, benchmark output,
and publication metadata are also available in `report-context.json`.

## Provenance

$provenance_table

## Intended Uses

The default intended use is research and controlled experimentation with small
causal language models. Evaluate generation quality, safety, and task fitness
before using the model in another setting.

## Out-Of-Scope Uses

Do not rely on the model for factual, medical, legal, financial, safety-critical,
or other high-impact decisions. It was not trained or evaluated as a safety
classifier, retrieval system, or instruction-following assistant.

## Limitations, Bias, And Risks

Generated text can be incorrect, biased, repetitive, toxic, or unsafe. The model
inherits limitations and representational biases from its training corpus. The
benchmark table covers only the recorded tasks and does not establish broad
capability or deployment safety.

## License

$license_section

## Environmental Impact

The recorded hardware and wall time are listed above. No defensible energy or
carbon estimate was collected, so this card does not invent one.
