from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import inspect
import json
import math
import os
import shutil
import signal
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from dmpod_wandb import (
    METRIC_DEFINITIONS,
    validate_metric_payload,
    validate_summary,
)
from torch.nn.parallel import DistributedDataParallel as DDP


class GracefulStop(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_wandb_config(
    config: dict[str, Any], runtime: dict[str, Any]
) -> dict[str, Any]:
    model = config["model"]
    data = config["data"]
    optimizer = config["optimizer"]
    batch = config["batch"]
    execution = config["runtime"]
    dataset = config["dataset"]
    evaluation = config["evaluation"]
    logging = config["logging"]
    checkpoint = config["checkpoint"]
    experiment = config["experiment"]
    wandb = config["wandb"]
    values: dict[str, Any] = {
        "config_schema": config["schema"],
        "config_schema_version": config["schema_version"],
        "profile": config["profile"],
        "experiment_schema": experiment["schema"],
        "experiment_revision": experiment["revision"],
        "experiment_change": experiment["change"],
        "experiment_hypothesis": experiment["hypothesis"],
        "wandb_project": wandb["project"],
        "wandb_group": wandb["group"],
        "wandb_job_type": wandb["job_type"],
        "task_type": config["task"]["type"],
        "framework": config["task"]["framework"],
        "objective": config["task"]["objective"],
        "target_parameter_count": model["target_parameter_count"],
        "vocab_size": model["vocab_size"],
        "n_layer": model["n_layer"],
        "n_head": model["n_head"],
        "n_embd": model["n_embd"],
        "block_size": model["block_size"],
        "dropout": model["dropout"],
        "bias": model["bias"],
        "weight_tying": model["weight_tying"],
        "actual_parameters_total": model["actual_parameters_total"],
        "actual_parameters_trainable": model["actual_parameters_trainable"],
        "actual_parameters_non_embedding": model["actual_parameters_non_embedding"],
        "dataset_path": data["path"],
        "dataset_dtype": "uint16",
        "dataset_size_gib": dataset["dataset_size_gib"],
        "train_tokens_unique": data["train_tokens_unique"],
        "val_tokens": data["val_tokens"],
        "target_data_passes": data["target_data_passes"],
        "sampling_mode": data["sampling_mode"],
        "document_boundary_token_id": dataset["document_boundary_token_id"],
        "padding_token_id": dataset["padding_token_id"],
        "split_method": dataset["split_method"],
        "deduplication_method": dataset["deduplication_method"],
        "dataset_revision": dataset["revision"],
        "train_bin_sha256": dataset["files"]["train"]["sha256"],
        "val_bin_sha256": dataset["files"]["val"]["sha256"],
        "tokenizer_sha256": dataset["tokenizer"]["sha256"],
        "tokenizer_name": dataset["tokenizer"]["name"],
        "tokenizer_version": dataset["tokenizer"]["version"],
        "optimizer": optimizer["optimizer"],
        "max_lr": optimizer["max_lr"],
        "min_lr_ratio": optimizer["min_lr_ratio"],
        "min_lr": optimizer["min_lr"],
        "lr_schedule": optimizer["lr_schedule"],
        "warmup_ratio": optimizer["warmup_ratio"],
        "warmup_tokens": optimizer["warmup_tokens"],
        "adam_beta1": optimizer["adam_beta1"],
        "adam_beta2": optimizer["adam_beta2"],
        "adam_eps": optimizer["adam_eps"],
        "weight_decay": optimizer["weight_decay"],
        "grad_clip": optimizer["grad_clip"],
        "micro_batch_size_per_gpu": batch["micro_batch_size_per_gpu"],
        "global_gradient_accumulation_steps": batch[
            "global_gradient_accumulation_steps"
        ],
        "effective_batch_tokens": batch["effective_batch_tokens"],
        "target_update_steps": batch["target_update_steps"],
        "target_train_tokens": batch["target_train_tokens"],
        "actual_train_tokens": batch["actual_train_tokens"],
        "actual_data_passes": batch["actual_data_passes"],
        "seed": execution["seed"],
        "data_seed": execution["data_seed"],
        "eval_seed": execution["eval_seed"],
        "precision": execution["precision"],
        "allow_tf32": execution["allow_tf32"],
        "torch_compile": execution["torch_compile"],
        "gradient_checkpointing": execution["gradient_checkpointing"],
        "ddp_backend": execution["ddp_backend"],
        "grad_scaler_enabled": execution["precision"] == "float16",
        "eval_at_start": evaluation["eval_at_start"],
        "eval_at_warmup_end": evaluation["eval_at_warmup_end"],
        "eval_interval_tokens": evaluation["eval_interval_tokens"],
        "eval_at_data_passes": evaluation["eval_at_data_passes"],
        "final_eval_required": evaluation["final_eval_required"],
        "val_evaluation_mode": evaluation["val_evaluation_mode"],
        "fixed_eval_offsets": evaluation.get("val_eval_offsets", {}).get("count"),
        "fixed_eval_offsets_sha256": evaluation.get("val_eval_offsets", {}).get(
            "sha256"
        ),
        "train_eval_subset_enabled": evaluation["train_eval_subset_enabled"],
        "train_eval_subset_tokens": evaluation["train_eval_subset_tokens"],
        "train_eval_offsets_sha256": evaluation["train_eval_offsets"]["sha256"],
        "loss_reduction": evaluation["loss_reduction"],
        "train_log_interval_target_tokens": logging["train_log_interval_target_tokens"],
        "train_log_interval_steps": logging["train_log_interval_steps"],
        "aggregate_metrics_over_logging_window": logging[
            "aggregate_metrics_over_logging_window"
        ],
        "save_last_checkpoint": checkpoint["save_last_checkpoint"],
        "save_best_val_checkpoint": checkpoint["save_best_val_checkpoint"],
        "save_final_checkpoint": checkpoint["save_final_checkpoint"],
        "save_at_data_passes": checkpoint["save_at_data_passes"],
        "checkpoint_max_interval_minutes": checkpoint["max_interval_minutes"],
        "log_wandb_artifacts": checkpoint["log_wandb_artifacts"],
    }
    values.update(runtime)
    return values


class Reporter:
    def __init__(
        self,
        run_dir: Path,
        config: dict[str, Any],
        runtime: dict[str, Any],
        enabled: bool,
    ) -> None:
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.jsonl"
        self.artifacts_path = run_dir / "artifacts.json"
        self.run = None
        self.enabled = enabled
        if not enabled:
            return
        import wandb

        wandb_config = flatten_wandb_config(config, runtime)
        wandb_values = config["wandb"]
        self.run = wandb.init(
            id=wandb_values["run_id"],
            resume="allow",
            project=wandb_values["project"],
            group=wandb_values["group"],
            job_type=wandb_values["job_type"],
            name=wandb_values["run_name"],
            notes=config["experiment"]["hypothesis"],
            tags=wandb_values["tags"],
            config=wandb_config,
            dir=str(run_dir / "logs" / "wandb"),
        )
        atomic_json(
            run_dir / "wandb.json",
            {
                "entity": self.run.entity,
                "project": self.run.project,
                "run_id": self.run.id,
                "run_name": self.run.name,
                "url": self.run.url,
                "mode": os.environ.get("WANDB_MODE", "online"),
            },
        )
        for definition in METRIC_DEFINITIONS:
            values = dict(definition)
            name = values.pop("name")
            self.run.define_metric(name, **values)

    def log(self, values: dict[str, Any], kind: str) -> None:
        validate_metric_payload(values, kind)
        append_jsonl(
            self.metrics_path,
            {"timestamp": now(), "kind": kind, **values},
        )
        if self.run is not None:
            self.run.log(values)

    def summary(self, values: dict[str, Any]) -> None:
        validate_summary(values)
        atomic_json(self.run_dir / "summary.json", values)
        if self.run is not None:
            for key, value in values.items():
                self.run.summary[key] = value

    def record_checkpoint(
        self,
        checkpoint: Path,
        aliases: list[str],
        metadata: dict[str, Any],
    ) -> None:
        record = {
            "timestamp": now(),
            "path": str(checkpoint.relative_to(self.run_dir)),
            "sha256": file_sha256(checkpoint),
            "aliases": aliases,
            "metadata": metadata,
        }
        existing = (
            json.loads(self.artifacts_path.read_text(encoding="utf-8"))
            if self.artifacts_path.is_file()
            else {"version": 1, "checkpoints": []}
        )
        existing["checkpoints"] = [
            item
            for item in existing["checkpoints"]
            if item.get("path") != record["path"]
        ]
        existing["checkpoints"].append(record)
        atomic_json(self.artifacts_path, existing)

    def upload_recorded_checkpoint(self, checkpoint: Path, aliases: list[str]) -> None:
        if self.run is None or os.environ.get("WANDB_MODE", "online") != "online":
            return
        records = json.loads(self.artifacts_path.read_text(encoding="utf-8"))[
            "checkpoints"
        ]
        relative = str(checkpoint.relative_to(self.run_dir))
        record = next(
            (item for item in records if item.get("path") == relative),
            None,
        )
        if record is None or file_sha256(checkpoint) != record["sha256"]:
            raise RuntimeError(f"Checkpoint record is missing or stale: {checkpoint}")
        import wandb

        artifact = wandb.Artifact(
            f"{self.run.id}-checkpoint",
            type="model",
            metadata=record["metadata"],
        )
        artifact.add_file(str(checkpoint), name="checkpoint.pt")
        self.run.log_artifact(artifact, aliases=aliases)

    def finish(self, exit_code: int = 0) -> None:
        if self.run is not None:
            self.run.finish(exit_code=exit_code)


def reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def reduce_max(value: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value


def make_batch(
    data: np.memmap,
    offsets: Iterable[int],
    length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = list(offsets)
    x = torch.from_numpy(
        np.stack(
            [np.asarray(data[pos : pos + length], dtype=np.int64) for pos in positions]
        )
    )
    y = torch.from_numpy(
        np.stack(
            [
                np.asarray(data[pos + 1 : pos + 1 + length], dtype=np.int64)
                for pos in positions
            ]
        )
    )
    if device.type == "cuda":
        return (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate_offsets(
    model: torch.nn.Module,
    data: np.memmap,
    offsets: list[int],
    block_size: int,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    autocast: Any,
) -> float:
    local_offsets = offsets[rank::world_size]
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    token_count = torch.zeros((), dtype=torch.float64, device=device)
    model.eval()
    full = [offset for offset in local_offsets if offset + block_size + 1 <= len(data)]
    full_set = set(full)
    partial = [offset for offset in local_offsets if offset not in full_set]
    for start in range(0, len(full), batch_size):
        selected = full[start : start + batch_size]
        x, y = make_batch(data, selected, block_size, device)
        with autocast():
            _, loss = model(x, y)
        count = y.numel()
        loss_sum += loss.detach().double() * count
        token_count += count
    for offset in partial:
        length = len(data) - offset - 1
        if length < 1:
            continue
        x, y = make_batch(data, [offset], length, device)
        with autocast():
            _, loss = model(x, y)
        loss_sum += loss.detach().double() * length
        token_count += length
    reduce_sum(loss_sum)
    reduce_sum(token_count)
    model.train()
    if not token_count.item():
        raise RuntimeError("Evaluation selected no tokens")
    return float((loss_sum / token_count).item())


def optimizer_for(
    model: torch.nn.Module,
    config: dict[str, Any],
    device_type: str,
) -> torch.optim.Optimizer:
    parameters = {
        name: value for name, value in model.named_parameters() if value.requires_grad
    }
    decay = [value for value in parameters.values() if value.dim() >= 2]
    no_decay = [value for value in parameters.values() if value.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": config["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs: dict[str, Any] = {}
    if (
        "fused" in inspect.signature(torch.optim.AdamW).parameters
        and device_type == "cuda"
    ):
        kwargs["fused"] = True
    return torch.optim.AdamW(
        groups,
        lr=config["max_lr"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["adam_eps"],
        **kwargs,
    )


def learning_rate(
    config: dict[str, Any], tokens_seen: int, effective_tokens: int
) -> float:
    maximum = float(config["max_lr"])
    if config.get("lr_schedule", "cosine") == "constant":
        return maximum
    minimum = float(config["min_lr"])
    warmup = int(config["warmup_tokens"])
    decay_end = int(config["lr_decay_end_tokens"])
    position = min(tokens_seen + effective_tokens, decay_end)
    if warmup > 0 and position <= warmup:
        return maximum * position / warmup
    if position >= decay_end:
        return minimum
    ratio = (position - warmup) / (decay_end - warmup)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return minimum + coefficient * (maximum - minimum)


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def gather_rng_state(
    data_generator: torch.Generator,
    rank: int,
    world_size: int,
    device: torch.device,
) -> list[dict[str, Any]] | None:
    state = {
        "rank": rank,
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "data": data_generator.get_state(),
    }
    if world_size == 1:
        return [state]
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(state, gathered, dst=0)
    return gathered


def restore_rng_state(
    checkpoint: dict[str, Any],
    data_generator: torch.Generator,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    if checkpoint.get("world_size") != world_size:
        raise RuntimeError("Resume requires the same world_size as the checkpoint")
    states = checkpoint.get("rng_states")
    if not isinstance(states, list) or len(states) != world_size:
        raise RuntimeError("Checkpoint does not contain complete per-rank RNG state")
    state = states[rank]
    torch.set_rng_state(state["cpu"])
    data_generator.set_state(state["data"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def save_checkpoint(
    *,
    run_dir: Path,
    base_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    progress: dict[str, Any],
    best_val_loss: float,
    data_generator: torch.Generator,
    rank: int,
    world_size: int,
    device: torch.device,
    reporter: Reporter | None,
    aliases: list[str],
) -> None:
    rng_states = gather_rng_state(data_generator, rank, world_size, device)
    checkpoint_dir = run_dir / "checkpoints"
    last_path = checkpoint_dir / "ckpt-last.pt"
    if rank == 0:
        checkpoint = {
            "schema": "dmpod.checkpoint",
            "schema_version": 1,
            "model": base_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "model_args": config["model"],
            "scheduler": {
                "type": config["optimizer"]["lr_schedule"],
                "tokens_seen": progress["tokens_seen"],
            },
            "progress": progress,
            "best_val_loss": best_val_loss,
            "rng_states": rng_states,
            "world_size": world_size,
            "full_training_config": config,
            "tokenizer_reference": config["dataset"]["tokenizer"],
            "dataset_hashes": {
                "train": config["dataset"]["files"]["train"]["sha256"],
                "val": config["dataset"]["files"]["val"]["sha256"],
            },
            "git_commit": config["git"]["commit"],
        }
        temporary = last_path.with_name(f".{last_path.name}.tmp.{os.getpid()}")
        try:
            torch.save(checkpoint, temporary)
            os.replace(temporary, last_path)
        finally:
            temporary.unlink(missing_ok=True)
        saved_paths = [(last_path, ["latest"])]
        for alias in aliases:
            if alias == "latest":
                continue
            path = checkpoint_dir / f"ckpt-{alias}.pt"
            atomic_copy(last_path, path)
            saved_paths.append((path, [alias]))
        if reporter is not None:
            metadata = {
                "update_step": progress["update_step"],
                "tokens_seen": progress["tokens_seen"],
                "data_pass_equivalent": progress["data_pass_equivalent"],
                "best_val_loss": best_val_loss,
            }
            for path, path_aliases in saved_paths:
                reporter.record_checkpoint(path, path_aliases, metadata)
    if dist.is_initialized():
        dist.barrier()


def event_thresholds(config: dict[str, Any]) -> dict[int, set[str]]:
    thresholds: dict[int, set[str]] = {}

    def add(value: int, reason: str) -> None:
        if value > 0:
            thresholds.setdefault(value, set()).add(reason)

    actual = int(config["batch"]["actual_train_tokens"])
    interval = int(config["evaluation"]["eval_interval_tokens"])
    for value in range(interval, actual, interval):
        add(value, "interval")
    if config["evaluation"]["eval_at_warmup_end"]:
        add(int(config["optimizer"]["warmup_tokens"]), "warmup-end")
    unique = int(config["data"]["train_tokens_unique"])
    effective = int(config["batch"]["effective_batch_tokens"])
    for data_pass in config["evaluation"]["eval_at_data_passes"]:
        pass_tokens = max(
            effective,
            round(unique * float(data_pass) / effective) * effective,
        )
        add(pass_tokens, f"pass-{format(float(data_pass), 'g')}")
    add(actual, "final-budget")
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", action="store_true")
    initialization.add_argument("--init-from", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("schema") != "dmpod.config" or config.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported training configuration: {run_dir / 'config.json'}"
        )
    runtime_path = run_dir / "runtime.json"
    runtime = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.is_file()
        else {}
    )
    config["git"] = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))[
        "git"
    ]
    sys.path.insert(0, str(run_dir / "sources"))
    from model import GPT, GPTConfig

    ddp = int(os.environ.get("RANK", "-1")) >= 0
    if ddp:
        dist.init_process_group(backend=config["runtime"]["ddp_backend"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master = rank == 0
    global_accumulation = int(config["batch"]["global_gradient_accumulation_steps"])
    if global_accumulation % world_size:
        raise RuntimeError(
            "global_gradient_accumulation_steps must be divisible by world_size"
        )
    accumulation = global_accumulation // world_size
    micro_batch = int(config["batch"]["micro_batch_size_per_gpu"])
    block_size = int(config["model"]["block_size"])
    local_tokens_per_attempt = micro_batch * accumulation * block_size
    effective_tokens = local_tokens_per_attempt * world_size
    if effective_tokens != config["batch"]["effective_batch_tokens"]:
        raise RuntimeError("Runtime effective batch differs from config.json")

    torch.manual_seed(int(config["runtime"]["seed"]) + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed(int(config["runtime"]["seed"]) + rank)
    torch.backends.cuda.matmul.allow_tf32 = bool(config["runtime"]["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["runtime"]["allow_tf32"])
    data_generator = torch.Generator(device="cpu")
    data_generator.manual_seed(int(config["runtime"]["data_seed"]) + rank)
    precision = config["runtime"]["precision"]
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[precision]

    def autocast() -> Any:
        if device.type == "cpu" or dtype == torch.float32:
            return contextlib.nullcontext()
        return torch.amp.autocast("cuda", dtype=dtype)

    data_path = Path(config["data"]["path"])
    train_data = np.memmap(data_path / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_path / "val.bin", dtype=np.uint16, mode="r")
    model_config = config["model"]
    model = GPT(
        GPTConfig(
            block_size=model_config["block_size"],
            vocab_size=model_config["vocab_size"],
            n_layer=model_config["n_layer"],
            n_head=model_config["n_head"],
            n_embd=model_config["n_embd"],
            dropout=model_config["dropout"],
            bias=model_config["bias"],
        )
    ).to(device)
    actual_total = sum(parameter.numel() for parameter in model.parameters())
    actual_trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    actual_non_embedding = model.get_num_params(non_embedding=True)
    checks = {
        "actual_parameters_total": actual_total,
        "actual_parameters_trainable": actual_trainable,
        "actual_parameters_non_embedding": actual_non_embedding,
    }
    for key, value in checks.items():
        if value != model_config[key]:
            raise RuntimeError(f"Parameter count mismatch for {key}: {value}")
    optimizer = optimizer_for(model, config["optimizer"], device.type)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "float16")
    progress = {
        "train_step": 0,
        "update_step": 0,
        "tokens_seen": 0,
        "data_pass_equivalent": 0.0,
        "skipped_updates_total": 0,
        "tokens_at_min_val_loss": 0,
    }
    best_val_loss = math.inf
    checkpoint_path = run_dir / "checkpoints" / "ckpt-last.pt"
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if (
            checkpoint.get("schema") != "dmpod.checkpoint"
            or checkpoint.get("schema_version") != 1
        ):
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
        if checkpoint.get("full_training_config") != config:
            raise RuntimeError(
                "Checkpoint training configuration does not match the run"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler.is_enabled() and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        progress = dict(checkpoint["progress"])
        best_val_loss = float(checkpoint["best_val_loss"])
    elif args.init_from is not None:
        initialization_state = torch.load(
            args.init_from, map_location=device, weights_only=False
        )
        if (
            initialization_state.get("schema")
            not in {
                "dmpod.checkpoint",
                "dmpod.initial-weights",
            }
            or initialization_state.get("schema_version") != 1
        ):
            raise ValueError(f"Unsupported initialization format: {args.init_from}")
        state = {
            key.removeprefix("_orig_mod."): value
            for key, value in initialization_state["model"].items()
        }
        model.load_state_dict(state)

    base_model = model
    training_model: torch.nn.Module = model
    if config["runtime"]["torch_compile"]:
        training_model = torch.compile(training_model)
    if ddp:
        training_model = DDP(training_model, device_ids=[device.index])
    if checkpoint is not None:
        restore_rng_state(checkpoint, data_generator, rank, world_size, device)
    training_model.train()
    train_offsets_path = run_dir / config["evaluation"]["train_eval_offsets"]["path"]
    train_eval_offsets = np.load(train_offsets_path, allow_pickle=False).tolist()
    val_mode = config["evaluation"]["val_evaluation_mode"]
    if val_mode == "full_validation":
        val_offsets = list(range(0, len(val_data) - 1, block_size))
    elif val_mode == "fixed_subset":
        val_offsets_path = run_dir / config["evaluation"]["val_eval_offsets"]["path"]
        val_offsets = np.load(val_offsets_path, allow_pickle=False).tolist()
    else:
        raise ValueError(f"Unsupported val_evaluation_mode: {val_mode!r}")
    runtime.update(
        world_size=world_size,
        gradient_accumulation_steps_per_gpu=accumulation,
    )
    reporter = (
        Reporter(
            run_dir,
            config,
            runtime,
            enabled=master and os.environ.get("DMPOD_WANDB_ENABLED") == "1",
        )
        if master
        else None
    )

    checkpoint_config = config["checkpoint"]
    pass_checkpoint_aliases = {
        f"pass-{format(float(value), 'g')}"
        for value in checkpoint_config["save_at_data_passes"]
    }
    thresholds = event_thresholds(config)
    evaluated_thresholds: set[int] = set()
    min_val_loss = best_val_loss
    tokens_at_min = int(progress.get("tokens_at_min_val_loss", 0))
    final_train_loss = math.nan
    final_val_loss = math.nan
    total_training_seconds = 0.0
    started = time.monotonic()
    last_checkpoint_at = started
    last_checkpoint_update = -1
    last_evaluation_update = -1
    signal_action = 0
    stop_request_path = run_dir / "stop-request.json"
    window_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    window_token_count = torch.zeros((), dtype=torch.float64, device=device)
    window_grad_sum = torch.zeros((), dtype=torch.float64, device=device)
    window_grad_max = torch.zeros((), dtype=torch.float64, device=device)
    window_clip_count = torch.zeros((), dtype=torch.float64, device=device)
    window_steps = torch.zeros((), dtype=torch.float64, device=device)
    window_duration = torch.zeros((), dtype=torch.float64, device=device)

    def save_local_checkpoint(aliases: list[str]) -> None:
        nonlocal last_checkpoint_at, last_checkpoint_update
        save_checkpoint(
            run_dir=run_dir,
            base_model=base_model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            progress=progress,
            best_val_loss=best_val_loss,
            data_generator=data_generator,
            rank=rank,
            world_size=world_size,
            device=device,
            reporter=reporter,
            aliases=list(dict.fromkeys(aliases)),
        )
        last_checkpoint_at = time.monotonic()
        last_checkpoint_update = int(progress["update_step"])

    def evaluate(reasons: set[str]) -> tuple[float, float]:
        nonlocal best_val_loss, min_val_loss, tokens_at_min, last_evaluation_update
        train_loss = evaluate_offsets(
            training_model,
            train_data,
            train_eval_offsets,
            block_size,
            int(config["evaluation"]["eval_batch_size"]),
            rank,
            world_size,
            device,
            autocast,
        )
        val_loss = evaluate_offsets(
            training_model,
            val_data,
            val_offsets,
            block_size,
            int(config["evaluation"]["eval_batch_size"]),
            rank,
            world_size,
            device,
            autocast,
        )
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            tokens_at_min = int(progress["tokens_seen"])
            progress["tokens_at_min_val_loss"] = tokens_at_min
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        if master and reporter is not None:
            reporter.log(
                {
                    "progress/update_step": progress["update_step"],
                    "progress/tokens_seen": progress["tokens_seen"],
                    "progress/data_pass_equivalent": progress["data_pass_equivalent"],
                    "eval/train_loss": train_loss,
                    "eval/val_loss": val_loss,
                    "eval/val_perplexity": math.exp(min(val_loss, 80.0)),
                    "eval/generalization_gap": val_loss - train_loss,
                    "eval/reason": ",".join(sorted(reasons)),
                },
                "evaluation",
            )
        aliases = ["latest"]
        aliases.extend(
            sorted(reason for reason in reasons if reason in pass_checkpoint_aliases)
        )
        if (
            is_best
            and progress["update_step"] > 0
            and checkpoint_config["save_best_val_checkpoint"]
        ):
            aliases.append("best-val")
        if (
            reasons & {"final", "final-budget"}
            and checkpoint_config["save_final_checkpoint"]
        ):
            aliases.append("final")
        if progress["update_step"] > 0:
            save_local_checkpoint(aliases)
        last_evaluation_update = int(progress["update_step"])
        return train_loss, val_loss

    def write_partial_summary(stop_reason: str, exit_code: int) -> None:
        if not master or reporter is None:
            return
        wall_seconds = time.monotonic() - started
        values = {
            "final_val_loss": final_val_loss if math.isfinite(final_val_loss) else None,
            "min_val_loss": min_val_loss if math.isfinite(min_val_loss) else None,
            "tokens_at_min_val_loss": tokens_at_min,
            "final_train_eval_loss": final_train_loss
            if math.isfinite(final_train_loss)
            else None,
            "final_val_perplexity": math.exp(min(final_val_loss, 80.0))
            if math.isfinite(final_val_loss)
            else None,
            "final_tokens_seen": progress["tokens_seen"],
            "final_data_pass_equivalent": progress["data_pass_equivalent"],
            "completed_target_budget": progress["update_step"]
            >= config["batch"]["target_update_steps"],
            "optimizer_updates_completed": progress["update_step"],
            "skipped_updates_total": progress["skipped_updates_total"],
            "wall_time_hours": wall_seconds / 3600,
            "mean_tokens_per_sec": progress["tokens_seen"] / total_training_seconds
            if total_training_seconds
            else 0.0,
            "peak_gpu_memory_allocated_gb": torch.cuda.max_memory_allocated(device)
            / 1024**3
            if device.type == "cuda"
            else 0.0,
            "peak_gpu_memory_reserved_gb": torch.cuda.max_memory_reserved(device)
            / 1024**3
            if device.type == "cuda"
            else 0.0,
            "best_checkpoint_alias": "best-val"
            if (run_dir / "checkpoints" / "ckpt-best-val.pt").is_file()
            else None,
            "final_checkpoint_alias": "final"
            if stop_reason == "completed"
            and (run_dir / "checkpoints" / "ckpt-final.pt").is_file()
            else None,
            "stop_reason": stop_reason,
            "exit_code": exit_code,
        }
        reporter.summary(values)

    def signal_handler(signum: int, _frame: Any) -> None:
        nonlocal signal_action
        signal_action = max(signal_action, 3 if signum == signal.SIGTERM else 2)

    def requested_control_action() -> int:
        action = signal_action
        if master:
            if stop_request_path.is_file():
                try:
                    request = json.loads(stop_request_path.read_text(encoding="utf-8"))
                    state = json.loads(
                        (run_dir / "state.json").read_text(encoding="utf-8")
                    )
                    valid_request = (
                        request.get("schema") == "dmpod.stop-request"
                        and request.get("schema_version") == 1
                        and request.get("attempt") == state.get("attempts")
                    )
                except (OSError, TypeError, ValueError):
                    valid_request = False
                if valid_request:
                    action = max(action, 2)
                else:
                    stop_request_path.unlink(missing_ok=True)
            if (
                action == 0
                and time.monotonic() - last_checkpoint_at
                >= float(checkpoint_config["max_interval_minutes"]) * 60
            ):
                action = 1
        action_tensor = torch.tensor(action, dtype=torch.int64, device=device)
        if dist.is_initialized():
            dist.all_reduce(action_tensor, op=dist.ReduceOp.MAX)
        return int(action_tensor.item())

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    exit_code = 0
    try:
        if config["evaluation"]["eval_at_start"] and progress["tokens_seen"] == 0:
            final_train_loss, final_val_loss = evaluate({"start"})
        target_updates = int(config["batch"]["target_update_steps"])
        log_interval = int(config["logging"]["train_log_interval_steps"])
        control_interval = min(log_interval, 10)
        while progress["update_step"] < target_updates:
            lr = learning_rate(
                config["optimizer"], progress["tokens_seen"], effective_tokens
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                step_started = time.perf_counter()
            attempt_loss = torch.zeros((), dtype=torch.float64, device=device)
            attempt_tokens = 0
            optimizer.zero_grad(set_to_none=True)
            for micro_step in range(accumulation):
                if ddp:
                    training_model.require_backward_grad_sync = (
                        micro_step == accumulation - 1
                    )
                offsets = torch.randint(
                    len(train_data) - block_size,
                    (micro_batch,),
                    generator=data_generator,
                ).tolist()
                x, y = make_batch(train_data, offsets, block_size, device)
                with autocast():
                    _, raw_loss = training_model(x, y)
                    scaled_loss = raw_loss / accumulation
                attempt_loss += raw_loss.detach().double() * y.numel()
                attempt_tokens += y.numel()
                scaler.scale(scaled_loss).backward()
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_clip = float(config["optimizer"]["grad_clip"])
            grad_norm = torch.nn.utils.clip_grad_norm_(
                base_model.parameters(), grad_clip if grad_clip > 0 else math.inf
            )
            clipped = grad_clip > 0 and float(grad_norm) > grad_clip
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.is_enabled() and scaler.get_scale() < scale_before
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                end_event.record()
                end_event.synchronize()
                duration = start_event.elapsed_time(end_event) / 1000
            else:
                duration = time.perf_counter() - step_started
            progress["train_step"] += 1
            progress["tokens_seen"] += effective_tokens
            progress["data_pass_equivalent"] = (
                progress["tokens_seen"] / config["data"]["train_tokens_unique"]
            )
            if skipped:
                progress["skipped_updates_total"] += 1
            else:
                progress["update_step"] += 1
            total_training_seconds += duration
            window_loss_sum += attempt_loss
            window_token_count += attempt_tokens
            window_grad_sum += float(grad_norm)
            window_grad_max = torch.maximum(
                window_grad_max,
                torch.tensor(float(grad_norm), dtype=torch.float64, device=device),
            )
            window_clip_count += int(clipped)
            window_steps += 1
            window_duration += duration

            should_log = (
                progress["update_step"] % log_interval == 0
                or progress["update_step"] >= target_updates
            )
            if should_log:
                global_loss_sum = reduce_sum(window_loss_sum.clone())
                global_token_count = reduce_sum(window_token_count.clone())
                grad_sum = reduce_sum(window_grad_sum.clone())
                grad_max = reduce_max(window_grad_max.clone())
                clip_count = reduce_sum(window_clip_count.clone())
                step_count = reduce_sum(window_steps.clone()) / world_size
                duration_max = reduce_max(window_duration.clone())
                if master and reporter is not None:
                    payload = {
                        "progress/update_step": progress["update_step"],
                        "progress/tokens_seen": progress["tokens_seen"],
                        "progress/data_pass_equivalent": progress[
                            "data_pass_equivalent"
                        ],
                        "train/loss": float(global_loss_sum / global_token_count),
                        "train/lr": lr,
                        "train/grad_norm": float(grad_sum / (step_count * world_size)),
                        "train/grad_norm_max": float(grad_max),
                        "train/clip_fraction": float(
                            clip_count / (step_count * world_size)
                        ),
                        "perf/tokens_per_sec": float(global_token_count / duration_max),
                        "perf/step_time_ms": float(duration_max * 1000 / step_count),
                    }
                    if base_model.get_num_params() and duration_max.item() > 0:
                        payload["perf/mfu"] = base_model.estimate_mfu(
                            micro_batch * accumulation,
                            float(duration_max / step_count),
                        )
                    if scaler.is_enabled():
                        payload["amp/loss_scale"] = scaler.get_scale()
                        payload["amp/skipped_updates_total"] = progress[
                            "skipped_updates_total"
                        ]
                    reporter.log(payload, "training")
                window_loss_sum.zero_()
                window_token_count.zero_()
                window_grad_sum.zero_()
                window_grad_max.zero_()
                window_clip_count.zero_()
                window_steps.zero_()
                window_duration.zero_()

            crossed: set[str] = set()
            previous_tokens = progress["tokens_seen"] - effective_tokens
            for threshold, reasons in thresholds.items():
                if (
                    threshold not in evaluated_thresholds
                    and previous_tokens < threshold <= progress["tokens_seen"]
                ):
                    evaluated_thresholds.add(threshold)
                    crossed.update(reasons)
            if crossed:
                final_train_loss, final_val_loss = evaluate(crossed)

            if (
                progress["update_step"] < target_updates
                and progress["train_step"] % control_interval == 0
            ):
                action = requested_control_action()
                if action >= 2:
                    if last_checkpoint_update != progress["update_step"]:
                        save_local_checkpoint(["latest"])
                    if master:
                        stop_request_path.unlink(missing_ok=True)
                    reason = "preempted" if action == 3 else "manual"
                    raise GracefulStop(reason)
                if action == 1:
                    save_local_checkpoint(["latest"])

        if (
            config["evaluation"]["final_eval_required"]
            and last_evaluation_update != progress["update_step"]
        ):
            final_train_loss, final_val_loss = evaluate({"final"})
        if master:
            stop_request_path.unlink(missing_ok=True)
        if master and reporter is not None and checkpoint_config["log_wandb_artifacts"]:
            for alias in ("best-val", "final"):
                checkpoint_path = run_dir / "checkpoints" / f"ckpt-{alias}.pt"
                if checkpoint_path.is_file():
                    try:
                        reporter.upload_recorded_checkpoint(checkpoint_path, [alias])
                    except Exception as error:
                        print(
                            f"Warning: W&B {alias} checkpoint upload failed: "
                            f"{type(error).__name__}",
                            file=sys.stderr,
                            flush=True,
                        )
        write_partial_summary("completed", 0)
    except GracefulStop as stop:
        write_partial_summary(stop.reason, 0)
    except KeyboardInterrupt:
        exit_code = 130
        write_partial_summary("manual", exit_code)
        raise
    except torch.cuda.OutOfMemoryError:
        exit_code = 137
        write_partial_summary("oom", exit_code)
        raise
    except BaseException:
        exit_code = 1
        write_partial_summary("error", exit_code)
        raise
    finally:
        if reporter is not None:
            reporter.finish(exit_code=exit_code)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
