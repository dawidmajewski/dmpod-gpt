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
- Never run a profile with `python train.py`. Profile runs must go through
  `dmpod-create-training` and `dmpod-train` so provenance and metrics are saved.

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
- `dmpod-export-run NAME`: generate the trained model README, PR body, and
  machine-readable report context from a completed profile run.
- `dmpod-wandb-dashboard`: create or update the canonical LR-sweep workspace.
- `run-benchmarks --model-config PATH`: benchmark an architecture on synthetic
  tokens and append machine-readable JSONL results under `/workspace`.

Model architecture and training settings live in separate Python config files.
Architecture includes `n_layer`, `n_head`, `n_embd`, `block_size`, and `bias`.
Training includes the dataset, batch/accumulation settings, optimizer, schedule,
evaluation, and checkpoint cadence. Architecture cannot change during resume.
Run snapshots are immutable; create a new run instead of editing one in place.

## Agent process policy

- Before every training start, resume, or restart, verify that W&B is configured
  for online logging and reachable. If W&B is not connected or its status cannot
  be confirmed, stop and explicitly ask the user whether to proceed with
  local-only or offline logging. Do not run `dmpod-train` without an explicit
  affirmative answer.
- Run training and other long GPU or data jobs in a named tmux session. For
  training, always use `dmpod-train NAME --tmux`; do not hand-roll background
  shell processes.
- After starting training, report both `tmux attach -t dmpod-NAME` and
  `tail -f /workspace/runs/NAME/logs/training.log` to the user.
- Use a descriptive tmux name for long dataset or conversion jobs and report the
  exact attach command immediately after launch.
- Before declaring success, check `state.json`, `summary.json`, the final
  checkpoint, local `metrics.jsonl`, and `wandb.json`. W&B logging must not be
  silently disabled for profile runs; an explicit user-approved offline run is
  the only exception.
- For the canonical 125M LR sweep, change only `--max-lr` during the first sweep.
  Keep seeds, hashes, token budget, batch tokens, precision, and eval offsets
  identical.

Canonical profile example:

```bash
dmpod-create-training --profile minimal-en-125m --max-lr 6e-4
dmpod-train lr-0.0006_seed-1337_btok-262144 --tmux
```

Codex CLI (`codex`) and Claude Code (`claude`) are installed in the image. Their
authentication is intentionally not handled by `dmpod-setup`.
