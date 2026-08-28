from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from dmpod_common import atomic_json, file_sha256
from dmpod_wandb import dataset_project, format_learning_rate

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def profile_root() -> Path:
    return Path(
        os.environ.get(
            "DMPOD_PROFILE_ROOT",
            str(Path(__file__).resolve().parents[1] / "profiles"),
        )
    ).resolve()


def load_profile(name: str) -> tuple[Path, dict[str, Any]]:
    path = profile_root() / f"{name}.json"
    if not path.is_file():
        available = ", ".join(
            item.stem for item in sorted(profile_root().glob("*.json"))
        )
        raise FileNotFoundError(f"Unknown profile {name!r}; available: {available}")
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1 or profile.get("name") != name:
        raise ValueError(f"Invalid profile header: {path}")
    return path, profile


def parameter_counts(model: dict[str, Any]) -> dict[str, int]:
    layers = int(model["n_layer"])
    embedding = int(model["n_embd"])
    vocab = int(model["vocab_size"])
    block = int(model["block_size"])
    bias = bool(model["bias"])
    tied = bool(model["weight_tying"])
    per_block = 12 * embedding * embedding + (13 if bias else 2) * embedding
    total = vocab * embedding + block * embedding + layers * per_block
    total += 2 * embedding if bias else embedding
    if not tied:
        total += vocab * embedding
    return {
        "actual_parameters_total": total,
        "actual_parameters_trainable": total,
        "actual_parameters_non_embedding": total - block * embedding,
    }


def validate_profile(profile: dict[str, Any]) -> None:
    model = profile["model"]
    for key in ("n_layer", "n_head", "n_embd", "block_size", "vocab_size"):
        if type(model.get(key)) is not int or model[key] < 1:
            raise ValueError(f"model.{key} must be a positive integer")
    if model["n_embd"] % model["n_head"]:
        raise ValueError("model.n_embd must be divisible by model.n_head")
    if model.get("weight_tying") is not True:
        raise ValueError("The pinned nanoGPT model requires weight_tying=true")
    if profile["runtime"].get("gradient_checkpointing") is not False:
        raise ValueError("gradient_checkpointing=true is not supported by this trainer")
    if profile["data"].get("sampling_mode") != "random_with_replacement":
        raise ValueError("Only random_with_replacement sampling is currently supported")
    val_mode = profile["evaluation"].get("val_evaluation_mode")
    if val_mode not in {"full_validation", "fixed_subset"}:
        raise ValueError(f"Unsupported val_evaluation_mode: {val_mode!r}")
    if val_mode == "fixed_subset" and (
        type(profile["evaluation"].get("val_eval_subset_tokens")) is not int
        or profile["evaluation"]["val_eval_subset_tokens"] < 1
    ):
        raise ValueError("val_eval_subset_tokens must be a positive integer")
    checkpoint_interval = profile["checkpoint"].get("max_interval_minutes")
    if (
        isinstance(checkpoint_interval, bool)
        or not isinstance(checkpoint_interval, (int, float))
        or checkpoint_interval <= 0
    ):
        raise ValueError("checkpoint.max_interval_minutes must be positive")
    counts = parameter_counts(model)
    expected = {
        "actual_parameters_total": model["expected_actual_parameters_total"],
        "actual_parameters_non_embedding": model[
            "expected_actual_parameters_non_embedding"
        ],
    }
    for key, value in expected.items():
        if counts[key] != value:
            raise ValueError(f"Profile {key}={value} but computed {counts[key]}")


def ensure_dataset_manifest(dataset_dir: Path, name: str) -> Path:
    dataset_dir = dataset_dir.resolve()
    manifest_path = dataset_dir / "dataset.json"
    if manifest_path.is_file():
        return manifest_path
    files: dict[str, dict[str, Any]] = {}
    revision = hashlib.sha256()
    maximum = 0
    for split in ("train", "val"):
        path = dataset_dir / f"{split}.bin"
        digest, tokens, split_maximum = _scan_uint16(path)
        revision.update(bytes.fromhex(digest))
        maximum = max(maximum, split_maximum)
        files[split] = {
            "path": path.name,
            "sha256": digest,
            "tokens": tokens,
        }

    meta_path = dataset_dir / "meta.pkl"
    tokenizer_path = dataset_dir / "tokenizer.json"
    if meta_path.is_file():
        with meta_path.open("rb") as source:
            meta = pickle.load(source)
        vocabulary = meta.get("stoi")
        if not isinstance(vocabulary, dict) or not vocabulary:
            raise ValueError(f"Dataset tokenizer metadata is missing stoi: {meta_path}")
        normalized: dict[str, int] = {}
        for token, token_id in vocabulary.items():
            if not isinstance(token, str) or type(token_id) is not int or token_id < 0:
                raise ValueError(f"Invalid character tokenizer entry in {meta_path}")
            normalized[token] = token_id
        atomic_json(
            tokenizer_path,
            {
                "implementation": "characters",
                "vocabulary": normalized,
            },
        )
        tokenizer_name = f"characters:{name}"
        tokenizer_version = "1"
        vocab_size = int(meta.get("vocab_size", max(normalized.values()) + 1))
        model_vocab_size = vocab_size
        document_boundary_token_id = 0
    else:
        try:
            tokenizer_version = importlib.metadata.version("tiktoken")
        except importlib.metadata.PackageNotFoundError:
            tokenizer_version = "0.14.0"
        atomic_json(
            tokenizer_path,
            {
                "implementation": "tiktoken",
                "encoding": "gpt2",
                "package_version": tokenizer_version,
                "vocab_size_with_padding": 50304,
            },
        )
        tokenizer_name = "tiktoken:gpt2"
        vocab_size = 50257
        model_vocab_size = 50304
        document_boundary_token_id = 50256
    if maximum >= vocab_size:
        raise ValueError(
            f"Dataset contains token id {maximum}, outside "
            f"tokenizer vocab_size={vocab_size}"
        )
    atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "name": name,
            "revision": f"sha256:{revision.hexdigest()}",
            "license": "unspecified",
            "dtype": "uint16",
            "files": files,
            "tokenizer": {
                "path": tokenizer_path.name,
                "sha256": file_sha256(tokenizer_path),
                "name": tokenizer_name,
                "version": tokenizer_version,
                "vocab_size": vocab_size,
                "model_vocab_size": model_vocab_size,
            },
            "document_boundary_token_id": document_boundary_token_id,
            "padding_token_id": None,
            "split_method": "existing train.bin and val.bin split",
            "deduplication_method": "unspecified",
        },
    )
    return manifest_path


def profile_from_configs(
    *,
    name: str,
    model_values: dict[str, Any],
    training_values: dict[str, Any],
    dataset_dir: Path,
    dataset_name: str,
    vocab_size: int,
) -> dict[str, Any]:
    model = {
        "n_layer": int(model_values["n_layer"]),
        "n_head": int(model_values["n_head"]),
        "n_embd": int(model_values["n_embd"]),
        "block_size": int(model_values["block_size"]),
        "vocab_size": int(vocab_size),
        "dropout": float(model_values.get("dropout", 0.0)),
        "bias": bool(model_values["bias"]),
        "weight_tying": True,
    }
    counts = parameter_counts(model)
    model.update(
        target_parameter_count=counts["actual_parameters_total"],
        expected_actual_parameters_total=counts["actual_parameters_total"],
        expected_actual_parameters_non_embedding=counts[
            "actual_parameters_non_embedding"
        ],
    )
    block_size = model["block_size"]
    micro_batch = int(training_values.get("batch_size", 12))
    accumulation = int(training_values.get("gradient_accumulation_steps", 40))
    max_iters = int(training_values.get("max_iters", 600000))
    for field, value in (
        ("batch_size", micro_batch),
        ("gradient_accumulation_steps", accumulation),
        ("max_iters", max_iters),
    ):
        if value < 1:
            raise ValueError(f"{field} must be a positive integer")
    effective_tokens = micro_batch * accumulation * block_size
    train_tokens = (dataset_dir / "train.bin").stat().st_size // 2
    actual_tokens = effective_tokens * max_iters
    max_lr = float(training_values.get("learning_rate", 6e-4))
    min_lr = float(training_values.get("min_lr", max_lr / 10))
    if max_lr <= 0 or min_lr < 0 or min_lr > max_lr:
        raise ValueError(
            "learning_rate and min_lr must satisfy 0 <= min_lr <= learning_rate"
        )
    decay_lr = bool(training_values.get("decay_lr", True))
    warmup_iters = int(training_values.get("warmup_iters", 2000)) if decay_lr else 0
    decay_iters = int(training_values.get("lr_decay_iters", max_iters))
    if warmup_iters < 0 or decay_iters < 1 or warmup_iters > decay_iters:
        raise ValueError(
            "LR schedule iterations must satisfy 0 <= warmup_iters <= lr_decay_iters"
        )
    eval_iters = int(training_values.get("eval_iters", 200))
    eval_interval = int(training_values.get("eval_interval", 2000))
    log_interval = int(training_values.get("log_interval", 1))
    for field, value in (
        ("eval_iters", eval_iters),
        ("eval_interval", eval_interval),
        ("log_interval", log_interval),
    ):
        if value < 1:
            raise ValueError(f"{field} must be a positive integer")
    if training_values.get("eval_only") is True:
        raise ValueError("eval_only is not a training run; use dmpod-benchmark")
    precision = str(training_values.get("dtype", "bfloat16"))
    if precision not in {"float32", "bfloat16", "float16"}:
        raise ValueError(f"Unsupported dtype: {precision!r}")
    return {
        "schema_version": 1,
        "name": name,
        "task": {
            "type": "causal_language_model_pretraining",
            "framework": "nanoGPT",
            "objective": "next_token_cross_entropy",
        },
        "model": model,
        "data": {
            "name": dataset_name,
            "path": str(dataset_dir),
            "sampling_mode": "random_with_replacement",
            "target_data_passes": actual_tokens / train_tokens,
        },
        "optimizer": {
            "optimizer": "AdamW",
            "max_lr": max_lr,
            "min_lr_ratio": min_lr / max_lr if decay_lr else 1.0,
            "lr_schedule": "cosine" if decay_lr else "constant",
            "warmup_ratio": warmup_iters / max_iters if max_iters else 0.0,
            "lr_decay_end_ratio": decay_iters / max_iters,
            "adam_beta1": float(training_values.get("beta1", 0.9)),
            "adam_beta2": float(training_values.get("beta2", 0.95)),
            "adam_eps": 1e-8,
            "weight_decay": float(training_values.get("weight_decay", 0.1)),
            "grad_clip": float(training_values.get("grad_clip", 1.0)),
        },
        "batch": {
            "micro_batch_size_per_gpu": micro_batch,
            "global_gradient_accumulation_steps": accumulation,
            "target_effective_batch_tokens": effective_tokens,
            "target_update_steps": max_iters,
        },
        "runtime": {
            "seed": int(training_values.get("seed", 1337)),
            "data_seed": int(training_values.get("data_seed", 1337)),
            "eval_seed": int(training_values.get("eval_seed", 4242)),
            "precision": precision,
            "allow_tf32": bool(training_values.get("allow_tf32", True)),
            "torch_compile": bool(training_values.get("compile", True)),
            "gradient_checkpointing": False,
            "ddp_backend": str(training_values.get("backend", "nccl")),
        },
        "evaluation": {
            "eval_at_start": True,
            "eval_at_warmup_end": warmup_iters > 0,
            "eval_interval_tokens": max(
                effective_tokens, eval_interval * effective_tokens
            ),
            "eval_at_data_passes": [],
            "final_eval_required": True,
            "val_evaluation_mode": "fixed_subset",
            "val_eval_subset_tokens": max(
                block_size, eval_iters * micro_batch * block_size
            ),
            "train_eval_subset_enabled": True,
            "train_eval_subset_tokens": max(
                block_size, eval_iters * micro_batch * block_size
            ),
            "eval_batch_size": micro_batch,
            "loss_reduction": "token_weighted_mean",
        },
        "logging": {
            "train_log_interval_target_tokens": max(
                effective_tokens, log_interval * effective_tokens
            ),
            "aggregate_metrics_over_logging_window": True,
            "log_from_rank_zero_only": True,
            "synchronize_distributed_metrics_before_logging": True,
            "exclude_evaluation_from_step_timing": True,
            "exclude_checkpointing_from_step_timing": True,
            "exclude_wandb_upload_from_step_timing": True,
        },
        "checkpoint": {
            "max_interval_minutes": float(
                training_values.get("checkpoint_interval_minutes", 60.0)
            ),
            "save_last_checkpoint": True,
            "save_best_val_checkpoint": True,
            "save_final_checkpoint": True,
            "save_at_data_passes": [],
            "log_wandb_artifacts": True,
        },
        "wandb": {
            "project": dataset_project(dataset_name),
            "job_type": "pretrain",
            "tags": ["nanogpt", "causal-lm"],
        },
    }


def _reject_placeholders(value: Any, path: str = "dataset") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_placeholders(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_placeholders(child, f"{path}[{index}]")
    elif isinstance(value, str) and (
        not value.strip()
        or "REPLACE_WITH" in value
        or (value.startswith("<") and value.endswith(">"))
    ):
        raise ValueError(f"Unresolved placeholder at {path}")


def _scan_uint16(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    maximum = 0
    tokens = 0
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
            if len(chunk) % 2:
                raise ValueError(f"Odd-sized uint16 dataset file: {path}")
            values = np.frombuffer(chunk, dtype=np.uint16)
            if values.size:
                maximum = max(maximum, int(values.max()))
                tokens += int(values.size)
    return digest.hexdigest(), tokens, maximum


def validate_dataset_manifest(
    manifest_path: Path,
    profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _reject_placeholders(manifest)
    if manifest.get("schema_version") != 1:
        raise ValueError("dataset manifest schema_version must be 1")
    if manifest.get("name") != profile["data"]["name"]:
        raise ValueError("dataset manifest name does not match the profile")
    if manifest.get("dtype") != "uint16":
        raise ValueError("dataset dtype must be uint16")
    dataset_dir = manifest_path.parent
    validation: dict[str, Any] = {
        "name": manifest["name"],
        "revision": manifest["revision"],
        "license": manifest["license"],
        "dtype": manifest["dtype"],
        "document_boundary_token_id": manifest.get("document_boundary_token_id"),
        "padding_token_id": manifest.get("padding_token_id"),
        "split_method": manifest["split_method"],
        "deduplication_method": manifest["deduplication_method"],
        "files": {},
    }
    for split in ("train", "val"):
        entry = manifest.get("files", {}).get(split, {})
        expected_hash = entry.get("sha256")
        if not isinstance(expected_hash, str) or not SHA256_PATTERN.fullmatch(
            expected_hash
        ):
            raise ValueError(f"Invalid SHA-256 for dataset {split}")
        binary = (dataset_dir / entry["path"]).resolve()
        if binary.parent != dataset_dir:
            raise ValueError(f"Dataset {split} path must remain inside {dataset_dir}")
        digest, tokens, maximum = _scan_uint16(binary)
        if digest != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {binary}")
        if tokens != entry.get("tokens"):
            raise ValueError(f"Token count mismatch for {binary}: {tokens}")
        if tokens <= profile["model"]["block_size"]:
            raise ValueError(
                f"{binary} has {tokens} tokens, not enough for block_size="
                f"{profile['model']['block_size']}"
            )
        if maximum >= profile["model"]["vocab_size"]:
            raise ValueError(
                f"{binary} contains token id {maximum}, outside vocab_size="
                f"{profile['model']['vocab_size']}"
            )
        validation["files"][split] = {
            "path": str(binary),
            "sha256": digest,
            "tokens": tokens,
            "bytes": binary.stat().st_size,
            "max_token_id": maximum,
        }
    tokenizer = manifest.get("tokenizer", {})
    tokenizer_hash = tokenizer.get("sha256")
    if not isinstance(tokenizer_hash, str) or not SHA256_PATTERN.fullmatch(
        tokenizer_hash
    ):
        raise ValueError("Invalid tokenizer SHA-256")
    tokenizer_path = (dataset_dir / tokenizer["path"]).resolve()
    if tokenizer_path.parent != dataset_dir or not tokenizer_path.is_file():
        raise ValueError("Tokenizer path must be a file inside the dataset directory")
    if file_sha256(tokenizer_path) != tokenizer_hash:
        raise ValueError(f"SHA-256 mismatch for {tokenizer_path}")
    tokenizer_vocab_size = tokenizer.get("vocab_size")
    if type(tokenizer_vocab_size) is not int or tokenizer_vocab_size < 1:
        raise ValueError("Tokenizer vocab_size must be a positive integer")
    model_vocab_size = tokenizer.get("model_vocab_size", tokenizer_vocab_size)
    if type(model_vocab_size) is not int or model_vocab_size < tokenizer_vocab_size:
        raise ValueError("Tokenizer model_vocab_size must cover its vocabulary")
    if tokenizer_vocab_size > profile["model"]["vocab_size"]:
        raise ValueError("Tokenizer vocab_size exceeds the model vocabulary")
    if any(
        validation["files"][split]["max_token_id"] >= tokenizer_vocab_size
        for split in ("train", "val")
    ):
        raise ValueError("Dataset contains a token outside tokenizer vocab_size")
    for key in ("document_boundary_token_id", "padding_token_id"):
        token_id = manifest.get(key)
        if token_id is not None and (
            type(token_id) is not int
            or token_id < 0
            or token_id >= profile["model"]["vocab_size"]
        ):
            raise ValueError(f"Invalid {key}: {token_id!r}")
    validation["tokenizer"] = {**tokenizer, "path": str(tokenizer_path)}
    expected_train = profile["data"].get("expected_train_tokens_unique")
    expected_val = profile["data"].get("expected_val_tokens")
    if (
        expected_train is not None
        and validation["files"]["train"]["tokens"] != expected_train
    ):
        raise ValueError("Profile train token count does not match the dataset")
    if (
        expected_val is not None
        and validation["files"]["val"]["tokens"] != expected_val
    ):
        raise ValueError("Profile val token count does not match the dataset")
    validation["manifest_path"] = str(manifest_path)
    validation["manifest_sha256"] = file_sha256(manifest_path)
    validation["dataset_size_gib"] = (
        sum(validation["files"][split]["bytes"] for split in ("train", "val")) / 1024**3
    )
    return manifest, validation


def resolve_profile(
    profile: dict[str, Any],
    *,
    max_lr: float | None,
    seed: int | None,
    dataset_validation: dict[str, Any],
) -> dict[str, Any]:
    resolved = copy.deepcopy(profile)
    validate_profile(resolved)
    resolved["schema"] = "dmpod.config"
    if max_lr is not None:
        if max_lr <= 0:
            raise ValueError("--max-lr must be positive")
        resolved["optimizer"]["max_lr"] = max_lr
    if resolved["optimizer"].get("max_lr") is None:
        raise ValueError(f"Profile {resolved['name']} requires --max-lr")
    if seed is not None:
        if seed < 0:
            raise ValueError("--seed cannot be negative")
        resolved["runtime"]["seed"] = seed
    model = resolved["model"]
    counts = parameter_counts(model)
    resolved["model"].update(counts)
    train_tokens = dataset_validation["files"]["train"]["tokens"]
    val_tokens = dataset_validation["files"]["val"]["tokens"]
    target_tokens = round(train_tokens * float(resolved["data"]["target_data_passes"]))
    batch = resolved["batch"]
    effective_tokens = (
        int(batch["micro_batch_size_per_gpu"])
        * int(batch["global_gradient_accumulation_steps"])
        * int(model["block_size"])
    )
    if effective_tokens != batch["target_effective_batch_tokens"]:
        raise ValueError("Profile effective batch token check failed")
    configured_steps = batch.get("target_update_steps")
    if configured_steps is not None and (
        type(configured_steps) is not int or configured_steps < 1
    ):
        raise ValueError("batch.target_update_steps must be a positive integer")
    target_steps = (
        int(configured_steps)
        if configured_steps is not None
        else max(1, round(target_tokens / effective_tokens))
    )
    actual_tokens = target_steps * effective_tokens
    optimizer = resolved["optimizer"]
    optimizer["min_lr"] = optimizer["max_lr"] * optimizer["min_lr_ratio"]
    optimizer["warmup_tokens"] = round(actual_tokens * optimizer["warmup_ratio"])
    optimizer["lr_decay_end_tokens"] = round(
        actual_tokens * float(optimizer.get("lr_decay_end_ratio", 1.0))
    )
    batch.update(
        effective_batch_tokens=effective_tokens,
        target_update_steps=target_steps,
        target_train_tokens=target_tokens,
        actual_train_tokens=actual_tokens,
        actual_data_passes=actual_tokens / train_tokens,
    )
    resolved["logging"]["train_log_interval_steps"] = max(
        1,
        round(
            resolved["logging"]["train_log_interval_target_tokens"] / effective_tokens
        ),
    )
    resolved["data"]["train_tokens_unique"] = train_tokens
    resolved["data"]["val_tokens"] = val_tokens
    resolved["dataset"] = copy.deepcopy(dataset_validation)
    return resolved


def canonical_run_name(resolved: dict[str, Any]) -> str:
    experiment = resolved.get("experiment")
    if experiment:
        return (
            f"{experiment['revision']}-{experiment['change']}-"
            f"s{resolved['runtime']['seed']}-"
            f"lr{format_learning_rate(resolved['optimizer']['max_lr'])}"
        )
    values = {
        "max_lr": format(float(resolved["optimizer"]["max_lr"]), ".8g"),
        "seed": resolved["runtime"]["seed"],
        "effective_batch_tokens": resolved["batch"]["effective_batch_tokens"],
    }
    return resolved["wandb"]["run_name_template"].format(**values)


def write_train_eval_offsets(run_dir: Path, resolved: dict[str, Any]) -> dict[str, Any]:
    evaluation = resolved["evaluation"]
    block_size = int(resolved["model"]["block_size"])
    train_tokens = int(resolved["data"]["train_tokens_unique"])
    requested_tokens = min(
        int(evaluation["train_eval_subset_tokens"]), train_tokens - 1
    )
    count = max(1, math.ceil(requested_tokens / block_size))
    rng = np.random.default_rng(int(resolved["runtime"]["eval_seed"]))
    offsets = rng.integers(0, train_tokens - block_size, size=count, dtype=np.int64)
    path = run_dir / "eval" / "train_offsets.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as output:
        np.save(output, offsets, allow_pickle=False)
    os.replace(temporary, path)
    return {
        "path": str(path.relative_to(run_dir)),
        "sha256": file_sha256(path),
        "count": count,
        "target_tokens": requested_tokens,
    }


def write_val_eval_offsets(run_dir: Path, resolved: dict[str, Any]) -> dict[str, Any]:
    evaluation = resolved["evaluation"]
    block_size = int(resolved["model"]["block_size"])
    val_tokens = int(resolved["data"]["val_tokens"])
    requested_tokens = min(int(evaluation["val_eval_subset_tokens"]), val_tokens - 1)
    count = max(1, math.ceil(requested_tokens / block_size))
    rng = np.random.default_rng(int(resolved["runtime"]["eval_seed"]) + 1)
    offsets = rng.integers(0, val_tokens - block_size, size=count, dtype=np.int64)
    path = run_dir / "eval" / "val_offsets.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as output:
        np.save(output, offsets, allow_pickle=False)
    os.replace(temporary, path)
    return {
        "path": str(path.relative_to(run_dir)),
        "sha256": file_sha256(path),
        "count": count,
        "target_tokens": requested_tokens,
    }


def write_config(path: Path, config: dict[str, Any]) -> None:
    atomic_json(path, config)
