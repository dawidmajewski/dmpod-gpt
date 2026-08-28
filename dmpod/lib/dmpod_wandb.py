from __future__ import annotations

import re
from typing import Any

SCHEMA = "nanogpt-training-v1"
PRIMARY_X_AXIS = "progress/tokens_seen"
PRIMARY_COMPARISON_METRIC = "eval/val_loss"
COMPARISON_GOAL = "minimize"

EXPERIMENT_REVISION_PATTERN = re.compile(r"^r[0-9]{3,}$")
SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

TRAIN_METRICS = frozenset(
    {
        "progress/update_step",
        PRIMARY_X_AXIS,
        "progress/data_pass_equivalent",
        "train/loss",
        "train/lr",
        "train/grad_norm",
        "train/grad_norm_max",
        "train/clip_fraction",
        "perf/tokens_per_sec",
        "perf/step_time_ms",
    }
)
EVAL_METRICS = frozenset(
    {
        "progress/update_step",
        PRIMARY_X_AXIS,
        "progress/data_pass_equivalent",
        "eval/train_loss",
        "eval/val_loss",
        "eval/val_perplexity",
        "eval/generalization_gap",
        "eval/reason",
    }
)
SUMMARY_KEYS = frozenset(
    {
        "final_val_loss",
        "min_val_loss",
        "tokens_at_min_val_loss",
        "final_train_eval_loss",
        "final_val_perplexity",
        "final_tokens_seen",
        "final_data_pass_equivalent",
        "completed_target_budget",
        "optimizer_updates_completed",
        "skipped_updates_total",
        "wall_time_hours",
        "mean_tokens_per_sec",
        "peak_gpu_memory_allocated_gb",
        "peak_gpu_memory_reserved_gb",
        "best_checkpoint_alias",
        "final_checkpoint_alias",
        "stop_reason",
        "exit_code",
    }
)

METRIC_DEFINITIONS = (
    {"name": PRIMARY_X_AXIS},
    {"name": "train/*", "step_metric": PRIMARY_X_AXIS},
    {"name": "eval/*", "step_metric": PRIMARY_X_AXIS},
    {"name": "perf/*", "step_metric": PRIMARY_X_AXIS},
    {"name": "amp/*", "step_metric": PRIMARY_X_AXIS},
    {
        "name": PRIMARY_COMPARISON_METRIC,
        "step_metric": PRIMARY_X_AXIS,
        "summary": "min",
    },
)

DASHBOARD_SECTIONS = (
    {
        "name": "Loss And Learning Rate",
        "panels": (
            ("Validation loss", ("eval/val_loss",)),
            ("Train loss", ("train/loss", "eval/train_loss")),
            ("Learning rate", ("train/lr",)),
        ),
    },
    {
        "name": "Optimization",
        "panels": (
            ("Gradient norms", ("train/grad_norm", "train/grad_norm_max")),
            ("Clip fraction", ("train/clip_fraction",)),
        ),
    },
    {
        "name": "Performance",
        "panels": (
            ("Tokens per second", ("perf/tokens_per_sec",)),
            ("Step time", ("perf/step_time_ms",)),
            ("Model FLOPs utilization", ("perf/mfu",)),
        ),
    },
)


def slug(value: str, label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not normalized:
        raise ValueError(f"{label} must contain a letter or number")
    return normalized


def dataset_project(dataset_name: str) -> str:
    return f"nanogpt-training-{slug(dataset_name, 'dataset name')}"


def validate_experiment(revision: str, change: str, hypothesis: str) -> None:
    if not EXPERIMENT_REVISION_PATTERN.fullmatch(revision):
        raise ValueError("--experiment-revision must use rNNN format, for example r001")
    if not SLUG_PATTERN.fullmatch(change):
        raise ValueError("--change must be a lowercase kebab-case slug")
    if not hypothesis.strip():
        raise ValueError("--hypothesis must not be empty")


def format_learning_rate(value: float) -> str:
    return format(float(value), ".8g")


def configure_run(
    config: dict[str, Any],
    *,
    revision: str,
    change: str,
    hypothesis: str,
    project: str | None = None,
    group: str | None = None,
    run_name: str | None = None,
    job_type: str | None = None,
    tags: list[str] | None = None,
) -> None:
    validate_experiment(revision, change, hypothesis)
    wandb = config["wandb"]
    profile = str(config["profile"])
    dataset = str(config["dataset"]["name"])
    default_project = (
        str(wandb["project"])
        if profile != "config" and wandb.get("project")
        else dataset_project(dataset)
    )
    default_group = f"{revision}-{change}"
    default_run_name = (
        f"{default_group}-s{config['runtime']['seed']}-"
        f"lr{format_learning_rate(config['optimizer']['max_lr'])}"
    )
    resolved_project = str(project or default_project).strip()
    resolved_group = str(group or default_group).strip()
    resolved_run_name = str(run_name or default_run_name).strip()
    resolved_job_type = str(job_type or wandb.get("job_type", "pretrain")).strip()
    for label, value in (
        ("W&B project", resolved_project),
        ("W&B group", resolved_group),
        ("W&B run name", resolved_run_name),
        ("W&B job type", resolved_job_type),
    ):
        if not value.strip():
            raise ValueError(f"{label} must not be empty")

    supplied_tags = [*wandb.get("tags", []), *(tags or [])]
    automatic_tags = [
        "nanogpt-training",
        f"schema:{SCHEMA}",
        f"profile:{slug(profile, 'profile')}",
        f"dataset:{slug(dataset, 'dataset name')}",
        f"revision:{revision}",
        f"change:{change}",
        f"job-type:{slug(resolved_job_type, 'W&B job type')}",
    ]
    resolved_tags = []
    for value in [*supplied_tags, *automatic_tags]:
        value = str(value).strip()
        if not value:
            raise ValueError("W&B tags must not be empty")
        if value not in resolved_tags:
            resolved_tags.append(value)

    config["experiment"] = {
        "schema": SCHEMA,
        "revision": revision,
        "change": change,
        "hypothesis": hypothesis.strip(),
    }
    config["wandb"] = {
        "schema": SCHEMA,
        "project": resolved_project,
        "group": resolved_group,
        "job_type": resolved_job_type,
        "run_name": resolved_run_name,
        "tags": resolved_tags,
        "primary_x_axis": PRIMARY_X_AXIS,
        "primary_comparison_metric": PRIMARY_COMPARISON_METRIC,
        "comparison_goal": COMPARISON_GOAL,
    }


def validate_metric_payload(values: dict[str, Any], kind: str) -> None:
    required = {"training": TRAIN_METRICS, "evaluation": EVAL_METRICS}.get(kind)
    if required is None:
        raise ValueError(f"Unknown metric payload kind: {kind!r}")
    missing = required - values.keys()
    if missing:
        raise ValueError(f"Missing {kind} metrics: {sorted(missing)}")


def validate_summary(values: dict[str, Any]) -> None:
    missing = SUMMARY_KEYS - values.keys()
    if missing:
        raise ValueError(f"Missing summary values: {sorted(missing)}")
