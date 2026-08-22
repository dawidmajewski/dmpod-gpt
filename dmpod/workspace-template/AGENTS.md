# DMPod nanoGPT workspace

This directory is a persistent copy of the pinned upstream nanoGPT repository.
Its Git history is preserved. `/opt/nanogpt` is the clean image copy you can use
as a comparison point.

## Safety

- Keep datasets, model weights, checkpoints, caches, logs, and secrets under
  `/workspace`; never add them to Git or bake them into an image.
- Do not print or commit `WANDB_API_KEY`, `HF_TOKEN`, or agent credentials.
- Treat checkpoints loaded with `torch.load` as trusted executable inputs.
- Do not overwrite an existing run. Create a new named run instead.

## Commands

- `dmpod-setup`: configure storage, caches, and optional W&B access only. It
  never chooses a model, dataset, checkpoint, or training configuration.
- `dmpod-prepare-data shakespeare_char|shakespeare`: run an upstream preset.
- `dmpod-prepare-data existing NAME PATH`: register existing uint16 binaries
  from the attached volume without copying them.
- `dmpod-prepare-data custom --script PATH [--name NAME]`: execute and validate
  a custom dataset preparation script. The `datasets` and `tiktoken` packages
  are already installed.
- `dmpod-create-training`: snapshot model/training configs and select scratch,
  a GPT-2-compatible Hugging Face model, or a trusted native checkpoint.
- `dmpod-train NAME`: start a new run. Use `dmpod-train NAME --resume` to
  continue from a checkpoint, or `--restart` only when a scratch run failed
  before producing its first checkpoint.
- `run-benchmarks --model-config PATH`: benchmark an architecture on synthetic
  tokens and append machine-readable JSONL results under `/workspace`.

Model architecture and training settings live in separate Python config files.
Architecture includes `n_layer`, `n_head`, `n_embd`, `block_size`, and `bias`.
Training includes the dataset, batch/accumulation settings, optimizer, schedule,
evaluation, and checkpoint cadence. Architecture cannot change during resume.
Run snapshots are immutable; create a new run instead of editing one in place.

Codex CLI (`codex`) and Claude Code (`claude`) are installed in the image. Their
authentication is intentionally not handled by `dmpod-setup`.
