# DMPod machine guidance

You are running inside a GPU Pod on RunPod. This image is dedicated to DMPod
nanoGPT model training and evaluation. The persistent workspace is `/workspace`,
the default project root is `/workspace/nanogpt`, and saved training runs are
under `/workspace/runs`. A user may select their own nanoGPT-compatible fork with
`dmpod-setup --project-root PATH`; work in the selected project and follow its
own `AGENTS.md` or `CLAUDE.md`. DMPod tooling is installed under `/opt/dmpod` and
must not be copied into or overwrite a user-provided repository.

- Prefer DMPod commands over direct nanoGPT training commands.
- Unless the user names another project, interpret questions about "training",
  "runs", or "training status" as questions about the DMPod model-training runs
  on this Pod. Inspect `/workspace/runs` and the relevant run's `state.json`,
  `config.json`, `summary.json`, `wandb.json`, and `benchmarks/results.json`
  before asking the user for context.
- Interpret model or quality benchmarks as `dmpod-benchmark NAME`. Interpret
  architecture throughput or GPU performance benchmarks as `run-benchmarks`.
  Use the existing DMPod profiles, runs, and results rather than assuming a
  generic benchmark suite.
- `dmpod-wandb-status` is the source of truth for W&B readiness. A missing
  `WANDB_API_KEY` in the agent process does not mean W&B is unconfigured because
  `dmpod-setup` may have stored the verified key outside the process environment.
  Do not tell the user to configure W&B when `dmpod-wandb-status` reports
  `W&B status: connected`.
- Use `hf` for Hugging Face model and dataset repositories, not hand-written
  parallel `curl` or `wget` downloads.
- Use `dmpod-export-run` to prepare model cards and publication metadata.
- Before creating a training run, inspect existing `/workspace/runs/*/config.json`
  files and choose the next `--experiment-revision rNNN`, one lowercase
  kebab-case `--change`, and a falsifiable `--hypothesis`. Keep `--revision` for
  an immutable Hugging Face model revision. Ask only when the intended
  experiment sequence is genuinely ambiguous.
- Keep the `nanogpt-training-v1` W&B identity and metric defaults unless the user
  explicitly requests an override. The frozen config, not `dmpod-setup`, is the
  source of truth for a run's project, group, name, tags, and comparison metric.
- Never print credentials or upload public artifacts without explicit user
  approval of the destination and model license.
