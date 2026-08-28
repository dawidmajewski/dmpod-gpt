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
- `dmpod-wandb-status`: verify the effective W&B mode, stored key, account, and
  network connection without printing the key. This command, not the presence
  of `WANDB_API_KEY` in the current shell, is the source of truth for W&B.
- `dmpod-prepare-data shakespeare_char|shakespeare`: run an upstream preset.
- `dmpod-prepare-data existing NAME PATH`: register existing uint16 binaries
  from the attached volume without copying them.
- `dmpod-prepare-data custom --script PATH [--name NAME]`: execute and validate
  a custom dataset preparation script. The `datasets` and `tiktoken` packages
  are already installed.
- `dmpod-create-training`: snapshot model/training configs and select scratch,
  a GPT-2-compatible Hugging Face model, or a trusted DMPod checkpoint. Every
  new run requires `--experiment-revision rNNN`, `--change KEBAB-SLUG`, and a
  concrete `--hypothesis`. The separate `--revision` option pins an HF model.
- `dmpod-train NAME`: start a new run. Use `dmpod-train NAME --resume` to
  continue from a checkpoint, or `--restart` only when a scratch run failed
  before producing its first checkpoint.
- `dmpod-stop NAME`: request a checkpoint and clean stop at the next safe
  training-step boundary. Resume the stopped run with `dmpod-train NAME --resume`.
- `dmpod-export-run NAME`: generate a Hugging Face-compatible model card, PR
  body, machine-readable report context, and SVG training curves from a
  completed run. Pass `--model-license ID` before public release. Use
  `--wandb-media-dir PATH` to include images downloaded from W&B.
- `dmpod-benchmark NAME`: interactively select English and Polish quality
  benchmarks for a trusted DMPod checkpoint. For unattended runs, pass
  `--benchmarks all|english|polish|ID...`. Results are saved as JSON under the
  run and included by the next `dmpod-export-run NAME`.
- `dmpod-wandb-dashboard NAME`: create or update the canonical workspace from
  the run's frozen W&B project and shared metric contract.
- `run-benchmarks --model-config PATH`: benchmark an architecture on synthetic
  tokens and append machine-readable JSONL results under `/workspace`.

Model architecture and training settings live in separate Python config files.
Architecture includes `n_layer`, `n_head`, `n_embd`, `block_size`, and `bias`.
Training includes the dataset, batch/accumulation settings, optimizer, schedule,
evaluation, and checkpoint cadence. Architecture cannot change during resume.
Run snapshots are immutable; create a new run instead of editing one in place.

## Agent process policy

- You are working inside a GPU Pod on RunPod. Unless the user explicitly names
  another project, "training" and "benchmarks" refer to the DMPod nanoGPT runs,
  profiles, and evaluation tools documented here. Inspect the existing local
  state before asking the user to explain those terms.
- For training information or status, inspect `/workspace/runs` and the relevant
  run's `state.json`, `config.json`, `summary.json`, `wandb.json`, and
  `benchmarks/results.json`. Do not infer run state from chat history.
- Before every training start, resume, or restart, run `dmpod-wandb-status`.
  Continue when it reports `W&B status: connected`; do not ask the user to
  configure W&B merely because `WANDB_API_KEY` is absent from the current
  process. If the command reports offline, disabled, unavailable, or cannot run,
  stop and explicitly ask whether to proceed with local-only or offline logging.
  Do not run `dmpod-train` without an explicit affirmative answer.
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
- Use `dmpod-benchmark NAME` for model-quality evaluation and `run-benchmarks`
  only for synthetic architecture throughput or GPU performance measurements.
- For Hugging Face repository URLs or IDs, use the installed `hf` CLI instead
  of enumerating files and downloading them with `curl` or `wget`. Use
  `--repo-type dataset` for dataset repositories and pin an immutable revision
  for reproducible inputs.
- Before preparing a public model repository, run `dmpod-export-run NAME`,
  review the generated `reports/README.md`, and resolve every publication
  warning. Do not infer a model-weights license from the dataset license.
- Never upload a model, checkpoint, tokenizer, report, or W&B media to
  Hugging Face without explicit user approval of the destination repository.
  Use the `dmpod-huggingface-publish` skill for the review and upload workflow.
- For the canonical 125M LR sweep, change only `--max-lr` during the first sweep.
  Keep seeds, hashes, token budget, batch tokens, precision, and eval offsets
  identical.
- Before creating a run, inspect existing run `config.json` files in
  `/workspace/runs`, select the next `rNNN` revision, name one concrete change,
  and state a falsifiable hypothesis. Do not use vague values such as `test`,
  `changes`, or `improvements`. Ask the user only when the intended comparison
  or next revision is genuinely ambiguous.
- Preserve the `nanogpt-training-v1` defaults unless the user explicitly requests
  a W&B override: group `<revision>-<change>`, run
  `<revision>-<change>-s<seed>-lr<max_lr>`, token x-axis, and minimized
  `eval/val_loss`.

Canonical profile example:

```bash
dmpod-create-training --profile minimal-en-125m --max-lr 6e-4 \
  --experiment-revision r001 \
  --change baseline \
  --hypothesis "Establish the canonical 125M baseline."
dmpod-train r001-baseline-s1337-lr0.0006 --tmux
```

Codex CLI (`codex`) and Claude Code (`claude`) are installed in the image. Their
authentication is intentionally not handled by `dmpod-setup`.
