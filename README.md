# DMPod GPT

Reproducible nanoGPT development and training image for RunPod. The image keeps
RunPod's standard `/start.sh`, SSH, and optional Jupyter services. It adds only a
pinned nanoGPT checkout, training dependencies, Codex CLI, Claude Code, and a
small set of commands for persistent work under `/workspace`.

## Image contents

- RunPod PyTorch 2.8.0 / CUDA 12.8.1 base pinned by digest.
- Official `karpathy/nanoGPT` cloned at the commit in `NANOGPT_REVISION`.
- The complete `/opt/nanogpt/.git` directory for local history and comparisons.
- Codex CLI `0.149.0` and Claude Code `2.1.240`, both checksum-verified.
- `transformers`, `datasets`, `tiktoken`, W&B, and related pinned dependencies.
- No datasets, model weights, checkpoints, credentials, or authentication state.

On first start the entrypoint copies `/opt/nanogpt`, including `.git`, to
`/workspace/nanogpt` and adds `AGENTS.md`, `CLAUDE.md`, and example configs. It
never overwrites an existing workspace. It then executes the base image's
`/start.sh`, so the Pod remains available after commands or training finish.

`/opt/nanogpt` stays as the clean image reference. User edits belong in
`/workspace/nanogpt`.

## Build

```bash
IMAGE_NAME=dawidmkrk/dmpod-gpt:v1.1.0 scripts/build.sh
```

The build targets `linux/amd64`. Change the image tag whenever code,
dependencies, agent versions, or the nanoGPT revision changes.

## First use

Attach a Pod volume or Network Volume at `/workspace`, then connect through SSH
or Jupyter and run:

```bash
dmpod-setup
```

`dmpod-setup` configures only storage paths, caches, and optional W&B access. It
does not choose a model, dataset, source weights, checkpoint, or training
settings. Non-interactive examples:

```bash
dmpod-setup --wandb-from-env --non-interactive
dmpod-setup --skip-wandb --non-interactive
```

If a W&B key is entered interactively, setup can save it at
`/workspace/.dmpod/secrets/wandb.key` with mode `0600`. Environment variable
`WANDB_API_KEY` always takes precedence. The key is never written to TOML or a
run manifest. Saving it on a Network Volume is explicit and optional.

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

Architecture and training settings are separate native nanoGPT Python configs.
A run snapshots both files, generates `runtime.py`, and records provenance in
`manifest.json`:

```bash
dmpod-create-training demo \
  --model-config configs/models/tiny-gpt.py \
  --training-config configs/training/shakespeare-char.py \
  --source scratch

dmpod-train demo
```

Resume is always explicit:

```bash
dmpod-train demo --resume
```

If a new scratch run fails before producing its first checkpoint, restart it
explicitly with `dmpod-train demo --restart`. Run configuration snapshots are
hash-checked before execution and cannot be edited after creation. Checkpoints
are replaced atomically, saved at each configured evaluation interval by the
included presets, and saved once more after a normal training-loop exit.

Remote Hugging Face initialization requires an immutable revision and supports
only GPT-2-compatible models whose architecture matches the model config:

```bash
dmpod-create-training hf-demo \
  --model-config configs/models/gpt2-124m.py \
  --training-config configs/training/shakespeare.py \
  --source hf \
  --model-id organization/model \
  --revision COMMIT_SHA
```

Initialize from a trusted native nanoGPT checkpoint already on the volume:

```bash
dmpod-create-training continued \
  --model-config configs/models/gpt2-124m.py \
  --training-config configs/training/shakespeare.py \
  --source checkpoint \
  --checkpoint /workspace/models/previous/ckpt.pt
```

Changing LR, scheduling, batch size, or evaluation settings only requires a
training config. Changing `n_layer`, `n_head`, `n_embd`, `block_size`, `bias`,
or vocabulary is not supported during resume.

## Reproducible profiles

Profile runs use a versioned, immutable run format with a run-local trainer and
model source. They always write the same metrics to local JSONL and W&B.

The canonical minimal-en profile uses 16 layers, 12 heads, width 768, context
2048, vocabulary 12,288, tied embeddings, and exactly 124,281,600 trainable
parameters. Create the first LR candidate with:

```bash
dmpod-create-training --profile minimal-en-125m --max-lr 6e-4
dmpod-train lr-0.0006_seed-1337_btok-262144 --tmux
```

Profile runs require a complete `dataset.json` next to `train.bin`, `val.bin`,
and the tokenizer file. Start from `datasets/dataset.json.example`. Creation
scans the binaries once, verifies SHA-256, token counts, tokenizer provenance,
and that every token ID is below the configured vocabulary size.

W&B online mode is the default. A profile run fails before using the GPU if the
key is unavailable. Explicit offline mode remains available through
`dmpod-setup --wandb-mode offline`; local `metrics.jsonl`, `summary.json`, and
checkpoint records are always written.

Create the canonical W&B dashboard after setup:

```bash
dmpod-wandb-dashboard --profile minimal-en-125m
```

Generate a model card and pull-request body after training:

```bash
dmpod-export-run RUN_NAME --format all
```

The output is stored in `runs/RUN_NAME/reports/README.md`, `PR_BODY.md`, and
`report-context.json`. The JSON context is the stable input for future HTML
converters.

For a quick end-to-end validation using a pinned Hugging Face dataset:

```bash
dmpod-prepare-data tinystories_smoke
dmpod-create-training --profile smoke-tinystories
dmpod-train lr-0.001_seed-1337_btok-8192 --tmux
```

## Benchmarks

```bash
run-benchmarks \
  --model-config configs/models/tiny-gpt.py \
  --batch-size 8 --steps 20
```

Results are printed and appended as JSONL to
`/workspace/benchmarks/results.jsonl` by default.

## Storage

- Container disk is disposable and may be cleared on stop/restart.
- Pod volume data under `/workspace` survives stop/restart, but not termination.
- Network Volume data survives Pod termination and can be attached to a new Pod
  in the same data center.

Critical checkpoints should also be backed up outside RunPod.

## GitHub Actions

`.github/workflows/docker.yml` builds and pushes version tags and manual runs.
Configure these repository secrets before enabling publication:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN` (read/write Docker Hub access token)

No credentials belong in the repository, Dockerfile, image, or workflow inputs.
