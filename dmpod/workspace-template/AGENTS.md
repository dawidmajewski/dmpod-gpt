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
- Never run DMPod training with `python train.py`. All runs must go through
  `dmpod-create-training` and `dmpod-train` so provenance and metrics are saved.

## Commands

- `dmpod-setup`: configure storage, caches, and optional W&B and Hugging Face
  access only. It never chooses a model, dataset, checkpoint, or training
  configuration. `--hf-from-env` verifies `HF_TOKEN` without persisting it;
  combine it with `--save-hf-token` only when the user explicitly wants the
  token stored on the attached workspace.
- `dmpod-prepare-data shakespeare_char|shakespeare`: run an upstream preset.
- `dmpod-prepare-data existing NAME PATH`: register existing uint16 binaries
  from the attached volume without copying them.
- `dmpod-prepare-data custom --script PATH [--name NAME]`: execute and validate
  a custom dataset preparation script. The `datasets` and `tiktoken` packages
  are already installed.
- `dmpod-create-training`: snapshot model/training configs and select scratch,
  a GPT-2-compatible Hugging Face model, or a trusted DMPod checkpoint.
- `dmpod-train NAME`: start a new run. Use `dmpod-train NAME --resume` to
  continue from a checkpoint, or `--restart` only when a scratch run failed
  before producing its first checkpoint.
- `dmpod-stop NAME`: request a checkpoint and clean stop at the next safe
  training-step boundary. Resume the stopped run with `dmpod-train NAME --resume`.
- `dmpod-export-run NAME`: generate the trained model README, PR body, and
  machine-readable report context from a completed run.
- `dmpod-benchmark NAME`: interactively select English and Polish quality
  benchmarks for a trusted DMPod checkpoint. For unattended runs, pass
  `--benchmarks all|english|polish|ID...`. Results are saved as JSON under the
  run and included by the next `dmpod-export-run NAME`.
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
- For a planned interruption, use `dmpod-stop NAME` and wait until `state.json`
  reports `stopped` before stopping the Pod. Do not use `kill -9`; `SIGINT` and
  `SIGTERM` also request a checkpoint-first stop but cannot protect against an
  uncatchable process or Pod failure.
- Use a descriptive tmux name for long dataset or conversion jobs and report the
  exact attach command immediately after launch.
- Read the checkpoint storage estimate printed before training. If it warns,
  resolve the capacity risk before a long run by increasing persistent storage
  or removing unused runs; do not assume the default 50 GB is sufficient.
- Before declaring success, check `state.json`, `summary.json`, the final
  checkpoint, local `metrics.jsonl`, and `wandb.json`. W&B logging must not be
  silently disabled for any run; an explicit user-approved offline run is
  the only exception.
- Expect periodic and `latest` checkpoints to remain local. A completed online
  run uploads only the retained `best-val` and `final` checkpoints to W&B.
- After training succeeds, ask the user whether to run quality benchmarks. Do
  not silently start the full suite because it downloads evaluation datasets
  and can consume substantial GPU time. If accepted, run `dmpod-benchmark NAME`
  and refresh the reports with `dmpod-export-run NAME`.
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
