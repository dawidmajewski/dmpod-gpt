from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from dmpod_common import atomic_json, file_sha256

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
    if tokenizer.get("vocab_size") != profile["model"]["vocab_size"]:
        raise ValueError("Tokenizer vocab_size does not match the model")
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
    validation["dataset_size_gib"] = sum(
        validation["files"][split]["bytes"] for split in ("train", "val")
    ) / 1024**3
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
    target_steps = max(1, round(target_tokens / effective_tokens))
    actual_tokens = target_steps * effective_tokens
    optimizer = resolved["optimizer"]
    optimizer["min_lr"] = optimizer["max_lr"] * optimizer["min_lr_ratio"]
    optimizer["warmup_tokens"] = round(actual_tokens * optimizer["warmup_ratio"])
    optimizer["lr_decay_end_tokens"] = actual_tokens
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
    resolved["dataset"] = dataset_validation
    return resolved


def canonical_run_name(resolved: dict[str, Any]) -> str:
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


def write_resolved_config(path: Path, resolved: dict[str, Any]) -> None:
    atomic_json(path, resolved)
