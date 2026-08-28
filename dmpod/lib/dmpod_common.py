from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import tomllib
from pathlib import Path
from typing import Any


NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def workspace_root() -> Path:
    return Path(os.environ.get("DMPOD_WORKSPACE", "/workspace")).resolve()


def project_root() -> Path:
    configured = os.environ.get("DMPOD_PROJECT_ROOT") or os.environ.get(
        "DMPOD_NANOGPT_ROOT"
    )
    if not configured:
        path = config_path()
        if path.is_file():
            with path.open("rb") as source:
                environment = tomllib.load(source)
            configured = environment.get("project_root") or environment.get(
                "nanogpt_root"
            )
    return Path(configured or workspace_root() / "nanogpt").expanduser().resolve()


def state_root() -> Path:
    return workspace_root() / ".dmpod"


def config_path() -> Path:
    return state_root() / "config.toml"


def require_name(value: str, label: str = "name") -> str:
    if not NAME_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must start with a letter or number and contain only "
            "letters, numbers, dots, underscores, or hyphens"
        )
    return value


def load_environment_config(required: bool = True) -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Run dmpod-setup first; missing {path}")
        return {}
    with path.open("rb") as source:
        return tomllib.load(source)


def toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def write_environment_config(config: dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    wandb = config["wandb"]
    lines = [
        f'version = {int(config["version"])}',
        f'workspace_root = {toml_string(config["workspace_root"])}',
        f'project_root = {toml_string(config["project_root"])}',
        f'datasets_root = {toml_string(config["datasets_root"])}',
        f'models_root = {toml_string(config["models_root"])}',
        f'runs_root = {toml_string(config["runs_root"])}',
        f'hf_cache = {toml_string(config["hf_cache"])}',
        "",
        "[wandb]",
        f'enabled = {str(bool(wandb["enabled"])).lower()}',
        f'mode = {toml_string(wandb["mode"])}',
        f'project = {toml_string(wandb["project"])}',
        f'entity = {toml_string(wandb["entity"])}',
        f'key_source = {toml_string(wandb["key_source"])}',
        "",
    ]
    atomic_write(path, "\n".join(lines), mode=0o600)


def atomic_write(path: Path, content: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            descriptor = -1
            output.write(content)
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any, mode: int = 0o644) -> None:
    atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        mode=mode,
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def execute_configs(paths: list[Path]) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    for path in paths:
        code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
        exec(code, namespace)
    return {
        key: value
        for key, value in namespace.items()
        if not key.startswith("__")
        and isinstance(value, (bool, float, int, str))
    }


def validate_model_config(values: dict[str, Any]) -> None:
    required = ("n_layer", "n_head", "n_embd", "block_size", "bias")
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"Model config is missing: {', '.join(missing)}")
    for name in ("n_layer", "n_head", "n_embd", "block_size"):
        if type(values[name]) is not int or values[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(values["bias"]) is not bool:
        raise ValueError("bias must be a boolean")
    if values["n_embd"] % values["n_head"]:
        raise ValueError("n_embd must be divisible by n_head")


def validate_dataset(path: Path, block_size: int | None = None) -> dict[str, int]:
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")
    result: dict[str, int] = {}
    for name in ("train.bin", "val.bin"):
        binary = path / name
        if not binary.is_file():
            raise FileNotFoundError(f"Dataset is missing {binary}")
        size = binary.stat().st_size
        if size < 2 or size % 2:
            raise ValueError(f"{binary} is not a non-empty uint16 binary file")
        tokens = size // 2
        if block_size is not None and tokens <= block_size:
            raise ValueError(
                f"{binary} has {tokens} tokens, but nanoGPT needs more than "
                f"block_size={block_size}"
            )
        result[name] = tokens
    return result


def export_wandb_key(configured_source: str) -> str:
    if os.environ.get("WANDB_API_KEY"):
        return "environment"
    candidates = {
        "workspace": state_root() / "secrets" / "wandb.key",
        "ephemeral": Path.home() / ".config" / "dmpod" / "wandb.key",
    }
    path = candidates.get(configured_source)
    if path is not None and path.is_file():
        key = path.read_text(encoding="utf-8").strip()
        if key:
            os.environ["WANDB_API_KEY"] = key
            return configured_source
    return "none"


def verify_wandb_key(key: str, timeout: int = 30) -> str:
    if not key:
        raise RuntimeError("W&B online mode requires a non-empty API key")
    import wandb

    try:
        wandb.login(key=key, verify=True)
        api = wandb.Api(api_key=key, timeout=timeout)
        _ = api.viewer
        entity = api.default_entity
    except Exception as error:
        raise RuntimeError("W&B online connection verification failed") from error
    if not entity:
        raise RuntimeError("W&B did not return a default entity for this account")
    return str(entity)


def verify_configured_wandb(
    wandb_config: dict[str, Any], timeout: int = 30
) -> tuple[str, str]:
    if not wandb_config.get("enabled") or wandb_config.get("mode") == "disabled":
        raise RuntimeError("W&B is disabled in DMPod configuration")
    if wandb_config.get("mode") != "online":
        raise RuntimeError("W&B is configured for offline logging")
    source = export_wandb_key(str(wandb_config.get("key_source", "none")))
    if source == "none":
        raise RuntimeError(
            "W&B online mode requires WANDB_API_KEY or a configured key file"
        )
    entity = verify_wandb_key(os.environ["WANDB_API_KEY"], timeout=timeout)
    return source, entity
