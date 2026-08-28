# DMPod GPT

Reproducible nanoGPT development and training image for RunPod. The image keeps
RunPod's standard `/start.sh`, SSH, and optional Jupyter services. It adds only a
pinned nanoGPT checkout, training dependencies, Codex CLI, Claude Code, and a
small set of commands for persistent work under `/workspace`.

## Image contents

- RunPod PyTorch 2.8.0 / CUDA 12.8.1 base pinned by digest.
- DMPod version from `DMPOD_VERSION`, embedded in the image metadata and environment.
- Official `karpathy/nanoGPT` cloned at the commit in `NANOGPT_REVISION`.
- The complete `/opt/nanogpt/.git` directory for local history and comparisons.
- Codex CLI `0.149.0` and Claude Code `2.1.240`, both checksum-verified.
- Machine-level DMPod guidance and a shared Hugging Face publication skill for
  Codex and Claude Code. Both agents are told that they are working on a RunPod
  GPU Pod and that unqualified training and benchmark questions refer to the
  local DMPod workflows.
- `transformers`, `datasets`, `tiktoken`, W&B, and related pinned dependencies.
- No datasets, model weights, checkpoints, credentials, or authentication state.

On first start the entrypoint copies `/opt/nanogpt`, including `.git`, to
`/workspace/nanogpt` and adds `AGENTS.md`, `CLAUDE.md`, and example configs. It
never overwrites an existing workspace. It then executes the base image's
`/start.sh`, so the Pod remains available after commands or training finish.
Interactive shells that start in `/workspace` or `/root` move to
`/workspace/nanogpt`, where both agents can discover the project instructions.

`/opt/nanogpt` stays as the clean image reference. User edits belong in
`/workspace/nanogpt`.

## Build

```bash
IMAGE_NAME=dawidmkrk/dmpod-gpt:v1.2.0 scripts/build.sh
```

The build targets `linux/amd64`. Change the image tag whenever code,
dependencies, agent versions, or the nanoGPT revision changes.

## First use

Attach a Pod volume or Network Volume at `/workspace`, then connect through SSH
or Jupyter and run:

```bash
dmpod-setup
```

`dmpod-setup` configures only storage paths, caches, and optional W&B and
Hugging Face access. It does not choose a model, dataset, source weights,
checkpoint, or training settings. Non-interactive examples:

```bash
dmpod-setup --wandb-from-env --hf-from-env --non-interactive
dmpod-setup --skip-wandb --skip-hf --non-interactive
```

`--hf-from-env` verifies `HF_TOKEN` without copying it to `/workspace`. Add
`--save-hf-token` only when the token should remain available on the attached
storage after the environment variable is gone.

If a W&B key is entered interactively, setup can save it at
`/workspace/.dmpod/secrets/wandb.key` with mode `0600`. Environment variable
`WANDB_API_KEY` always takes precedence. The key is never written to TOML or a
run manifest. Saving it on a Network Volume is explicit and optional.
`dmpod-setup` also registers a verified online key with the standard W&B client,
so `wandb` commands and agent tools use the same authenticated account.

Check the effective configuration from any new shell or agent process with:

```bash
dmpod-wandb-status
```

This resolves `WANDB_API_KEY` or the workspace/ephemeral key selected by
`dmpod-setup`, verifies the W&B account over the network, and never prints the
key. It is also the check used before an online training process starts. The
absence of `WANDB_API_KEY` from a later Codex or Claude process does not by
itself mean that W&B needs to be configured again.

Hugging Face login is optional and uses a hidden `HF_TOKEN` prompt. Tokens
entered interactively, or passed with explicit `--save-hf-token`, are registered
with the standard `huggingface_hub` client under
`HF_HOME=/workspace/cache/huggingface`, so `hf`, Transformers, datasets, and
agent tools reuse them across Pods attached to the same workspace. Environment
tokens are otherwise only verified and used from the environment. Tokens are
never written to DMPod configuration or run manifests.

For reusable Pod credentials, create a RunPod secret named `wandb_api_key` and
map it in the Pod template without putting the value in the template:

```text
WANDB_API_KEY={{ RUNPOD_SECRET_wandb_api_key }}
```

Add this mapping only to a private copy of the template. Each Pod created from
that template can then use `dmpod-setup --wandb-from-env --non-interactive`;
setup verifies the key but does not copy it to `/workspace`. The distributed
template intentionally omits the mapping so users can enter a key through the
hidden `dmpod-setup` prompt or choose offline mode.

Codex and Claude authentication is intentionally separate:

```bash
codex
claude
```

## Data

Built-in small presets:

```bash
dmpod-prepare-data shakespeare_char
dmpod-prepare-data shakespeare
dmpod-prepare-data tinystories_smoke
```

Register existing nanoGPT binaries from a mounted volume without copying:

```bash
dmpod-prepare-data existing my-data /workspace/datasets/my-data
```

Run a custom preparation script and validate its output:

```bash
dmpod-prepare-data custom \
  --script /workspace/datasets/my-data/prepare.py \
  --name my-data
```

A dataset must provide non-empty uint16 `train.bin` and `val.bin`; `meta.pkl` is
optional. The image includes Hugging Face `datasets`, but deliberately provides
no automated OpenWebText preset.

## Training definitions

Architecture and training settings can be supplied as separate nanoGPT Python
configs. A run compiles them into `config.json` and snapshots the
configs, tokenizer, dataset metadata, model source, and trainer under `sources/`.
All provenance and immutable file hashes are recorded in `manifest.json`:

```bash
dmpod-create-training demo \
  --model-config configs/models/tiny-gpt.py \
  --training-config configs/training/shakespeare-char.py \
  --source scratch \
  --experiment-revision r001 \
  --change baseline \
  --hypothesis "Establish a reproducible baseline for later comparisons."

dmpod-train demo
```

Resume is always explicit:

```bash
dmpod-train demo --resume
```

If a new scratch run fails before producing its first checkpoint, restart it
explicitly with `dmpod-train demo --restart`. Run configuration snapshots are
hash-checked before execution and cannot be edited after creation. Checkpoints
are replaced atomically. The included profiles refresh `ckpt-last.pt` at each
configured evaluation and at least once every 60 minutes, while retaining the
configured best, final, and data-pass checkpoints.

Request a planned interruption without stopping in the middle of an optimizer
update:

```bash
dmpod-stop demo
```

The trainer notices the request at a safe step boundary, refreshes
`ckpt-last.pt`, writes `state.json` with status `stopped`, and exits cleanly.
Wait for that status before stopping the Pod, then continue with
`dmpod-train demo --resume`. `SIGINT` and `SIGTERM` use the same checkpoint-first
path; an uncatchable process or Pod failure cannot guarantee a final save.

Remote Hugging Face initialization requires an immutable revision and supports
only GPT-2-compatible models whose architecture matches the model config:

```bash
dmpod-create-training hf-demo \
  --model-config configs/models/gpt2-124m.py \
  --training-config configs/training/shakespeare.py \
  --source hf \
  --model-id organization/model \
  --revision COMMIT_SHA \
  --experiment-revision r002 \
  --change hf-initialization \
  --hypothesis "Pinned pretrained initialization improves validation loss."
```

Initialize from a trusted DMPod checkpoint already on the volume:

```bash
dmpod-create-training continued \
  --model-config configs/models/gpt2-124m.py \
  --training-config configs/training/shakespeare.py \
  --source checkpoint \
  --checkpoint /workspace/models/previous/ckpt.pt \
  --experiment-revision r003 \
  --change continued-pretraining \
  --hypothesis "Continued pretraining improves validation loss without instability."
```

Changing LR, scheduling, batch size, or evaluation settings only requires a
training config. Changing `n_layer`, `n_head`, `n_embd`, `block_size`, `bias`,
or vocabulary is not supported during resume.

## Reproducible profiles

Profiles are another input for the same immutable run format, run-local trainer,
checkpoint schema, local JSONL metrics, and W&B reporting.

The canonical minimal-en profile uses 16 layers, 12 heads, width 768, context
2048, vocabulary 12,288, tied embeddings, and exactly 124,281,600 trainable
parameters. Create the first LR candidate with:

```bash
dmpod-create-training --profile minimal-en-125m --max-lr 6e-4 \
  --experiment-revision r001 \
  --change baseline \
  --hypothesis "Establish the canonical 125M baseline."
dmpod-train r001-baseline-s1337-lr0.0006 --tmux
```

Profiles require a complete `dataset.json` next to `train.bin`, `val.bin`,
and the tokenizer file. Start from `datasets/dataset.json.example`. Creation
scans the binaries once, verifies SHA-256, token counts, tokenizer provenance,
and that every token ID is below the configured vocabulary size.

W&B online mode is the default. A run fails before using the GPU if the
key is unavailable or W&B cannot be reached. Explicit offline mode remains available through
`dmpod-setup --wandb-mode offline`; local `metrics.jsonl`, `summary.json`, and
checkpoint records are always written. Local periodic checkpoints are not
uploaded to W&B. A completed online run uploads only the retained `best-val`
and `final` checkpoint artifacts.

Every new run uses the `nanogpt-training-v1` W&B contract. Its default group is
`<revision>-<change>`, its run name is
`<revision>-<change>-s<seed>-lr<max_lr>`, its primary x-axis is
`progress/tokens_seen`, and its comparison metric is `eval/val_loss` minimized.
Use `--experiment-revision` for the experiment sequence; `--revision` remains
the immutable Hugging Face model revision. Project, group, run name, job type,
and tags can be explicitly overridden with the corresponding `--wandb-*`
options. All resolved values are frozen in `config.json`.

Create the canonical W&B dashboard after setup:

```bash
dmpod-wandb-dashboard RUN_NAME
```

Generate a model card and pull-request body after training:

```bash
dmpod-export-run RUN_NAME --format all
```

The generated `README.md` is a Hugging Face-compatible, user-editable model
card. It includes Hub YAML metadata, model and training configuration, complete
summary statistics, benchmark tables, checkpoint hashes, native nanoGPT loading
instructions, provenance, W&B links, and SVG curves generated from the local
`metrics.jsonl` stream. The exporter does not invent a model-weights license:

```bash
dmpod-export-run RUN_NAME \
  --model-id SlayerLab/MODEL_NAME \
  --model-license apache-2.0
```

Profiles can provide default languages, datasets, pipeline tags, and tags.
Override or supplement them with `--language`, `--dataset-id`, and `--tag`.
If W&B screenshots or report images were downloaded separately, copy them into
the report assets and embed them with:

```bash
dmpod-export-run RUN_NAME \
  --wandb-media-dir /workspace/downloads/RUN_NAME-wandb
```

The output is stored in `runs/RUN_NAME/reports/README.md`, `PR_BODY.md`,
`report-context.json`, `metrics.jsonl`, and `assets/`. The JSON context records
the complete run configuration, state, summary, runtime, benchmarks, artifact
records, metric-file hash, and generated media manifest. W&B scalar charts are
generated locally because the charts displayed by the W&B UI are not ordinary
run media files.

Before public upload, review the intended-use and limitations prose, select the
model license, verify that benchmarks were not run with `--limit`, and confirm
that every linked W&B run and copied image is safe to publish. The
`dmpod-huggingface-publish` agent skill supplies this checklist. Agents must not
upload anything until the user explicitly approves the destination repository
and file set.

For a quick end-to-end validation using a pinned Hugging Face dataset:

```bash
dmpod-prepare-data tinystories_smoke
dmpod-create-training --profile smoke-tinystories \
  --experiment-revision r001 \
  --change trainer-validation \
  --hypothesis "The trainer completes and records the full W&B contract."
dmpod-train r001-trainer-validation-s1337-lr0.001 --tmux
```

## Benchmarks

Benchmark training throughput for an architecture on synthetic tokens:

```bash
run-benchmarks \
  --model-config configs/models/tiny-gpt.py \
  --batch-size 8 --steps 20
```

Results are printed and appended as JSONL to
`/workspace/benchmarks/results.jsonl` by default.

Evaluate any completed DMPod run for language quality:

```bash
dmpod-benchmark RUN_NAME
```

The interactive selector offers `All`, `English only`, `Polish only`, and
`Let me choose`, with separate English and Polish checklists for custom runs.
For unattended execution, pass benchmark IDs explicitly:

```bash
dmpod-benchmark RUN_NAME \
  --benchmarks blimp lambada hellaswag piqa sciq arc-easy arc-challenge \
  8tags polemo2-in polemo2-out
```

English evaluation uses the pinned lm-evaluation-harness task definitions.
The Polish classification tasks use zero-shot, length-normalized candidate
likelihood and report accuracy plus macro-F1; this is not supervised KLEJ
fine-tuning. Results are written to `runs/RUN_NAME/benchmarks/results.json`.
Run `dmpod-export-run RUN_NAME` afterward to add the table and benchmark links
to the generated README and PR body. `--limit N` is available only for quick
integration tests and should not be used for reported scores.

Codex and Claude interpret an unqualified request for model benchmarks as the
quality workflow through `dmpod-benchmark`. Use `run-benchmarks` explicitly for
synthetic architecture throughput or GPU performance measurements.

## Storage

- Container disk is disposable and may be cleared on stop/restart.
- Pod volume data under `/workspace` survives stop/restart, but not termination.
- Network Volume data survives Pod termination and can be attached to a new Pod
  in the same data center.

Before training starts, `dmpod-train` prints an estimate for all checkpoints the
run may retain and the current free space. It warns rather than blocks when the
volume may be too small, including room for the temporary file required by an
atomic checkpoint replacement. The default 50 GB volume can be sufficient for
small runs, but a warning should be resolved by increasing persistent storage
or removing unused runs before a long training job.

Critical checkpoints should also be backed up outside RunPod.

## GitHub Actions

`.github/workflows/docker.yml` builds a test stage, runs the CLI tests and an
entrypoint workspace smoke test inside the image, and only then builds and
pushes the final stage. A pushed Git tag must be exactly `v` followed by the
version in `DMPOD_VERSION`; manual runs may publish development tags such as
`edge` without changing the release version.
Configure these repository secrets before enabling publication:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN` (read/write Docker Hub access token)

No credentials belong in the repository, Dockerfile, image, or workflow inputs.
