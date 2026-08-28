@AGENTS.md

Before starting, resuming, or restarting training, run `dmpod-wandb-status`.
Trust its effective DMPod configuration rather than checking only whether
`WANDB_API_KEY` exists in the current process. If it does not report
`W&B status: connected`, do not run `dmpod-train` until the user explicitly
confirms proceeding with local-only or offline logging.

After training succeeds, ask whether the user wants quality benchmarks. If the
answer is yes, use `dmpod-benchmark NAME` and then refresh the exported reports.
