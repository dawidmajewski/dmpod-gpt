# DMPod GPT

A ready-to-use development and training environment for nanoGPT and compatible
forks on Runpod. DMPod GPT is based on
[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).

## Description

DMPod GPT provides a reproducible PyTorch 2.8 and CUDA 12.8 environment with
nanoGPT, training utilities, Codex CLI, and Claude Code preinstalled. Project
files, datasets, caches, and checkpoints are stored under `/workspace` so they
can persist between Pods.

DMPod tooling is installed outside the project under `/opt/dmpod`. To use an
existing compatible fork without injecting DMPod tooling into it, clone the
repository and select it during setup:

```bash
git clone https://github.com/OWNER/REPOSITORY.git /workspace/my-nanogpt
dmpod-setup --project-root /workspace/my-nanogpt
```

## Getting Started

### Dependencies

* A Runpod account and a compatible NVIDIA GPU.
* A persistent volume mounted at `/workspace` (50 GB or more recommended).
* SSH or Jupyter access to the Pod.
* Optional Hugging Face and Weights & Biases accounts.

### Using the template

Create a Pod from the template, connect through SSH or Jupyter, and initialize
the workspace:

```bash
dmpod-setup
dmpod-wandb-status
```

The status command checks the effective DMPod configuration, including a key
saved outside the current shell environment, without displaying credentials.

Prepare sample data and start a small training run:

```bash
dmpod-prepare-data shakespeare_char
dmpod-create-training demo \
  --model-config configs/models/tiny-gpt.py \
  --training-config configs/training/shakespeare-char.py \
  --source scratch \
  --experiment-revision r001 \
  --change baseline \
  --hypothesis "Establish a reproducible baseline."
dmpod-train demo --tmux
```

When online W&B access is configured, run the complete GPU and logging smoke
test with `dmpod-smoke-test`.

## Help

Use the built-in command help when a command or option is unclear:

```bash
dmpod-setup --help
dmpod-create-training --help
dmpod-train --help
```

If CUDA is unavailable, verify that the Pod uses an NVIDIA GPU on a host with
CUDA 12.8 or newer. If training runs out of GPU memory, reduce the configured
micro-batch size.

## Authors

Dawid Majewski
[@dawidmajewski](https://github.com/dawidmajewski)

## Version History

* 1.2.0
  * Adds persistent RunPod/DMPod context for Codex and Claude Code.
  * Adds authoritative W&B status verification for setup and training.
  * Expands reproducible training, benchmark, and model-publication workflows.
* 1.0.0
  * Initial release.
