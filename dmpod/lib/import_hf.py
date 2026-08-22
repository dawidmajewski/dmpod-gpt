from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

TRANSPOSED_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def import_hf_checkpoint(
    *,
    nanogpt_root: Path,
    model_id: str,
    revision: str | None,
    output: Path,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    dataset: str,
    cache_dir: Path,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    sys.path.insert(0, str(nanogpt_root))
    from model import GPT, GPTConfig

    source_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        trust_remote_code=False,
    ).cpu().eval()
    source_config = source_model.config
    if source_config.model_type != "gpt2":
        raise ValueError(
            f"Only GPT-2-compatible Hugging Face models are supported, got "
            f"{source_config.model_type!r}"
        )

    model_args = {
        "n_layer": int(model_config["n_layer"]),
        "n_head": int(model_config["n_head"]),
        "n_embd": int(model_config["n_embd"]),
        "block_size": int(model_config["block_size"]),
        "bias": bool(model_config["bias"]),
        "vocab_size": int(source_config.vocab_size),
        "dropout": float(model_config.get("dropout", 0.0)),
    }
    native = GPT(GPTConfig(**model_args))
    source_state = source_model.state_dict()
    converted: dict[str, torch.Tensor] = {}
    for key, target in native.state_dict().items():
        source_key = "transformer.wte.weight" if key == "lm_head.weight" else key
        if source_key not in source_state:
            raise KeyError(f"Missing Hugging Face tensor: {source_key}")
        value = source_state[source_key]
        if source_key.endswith(TRANSPOSED_SUFFIXES):
            value = value.t()
        if value.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {key}: {tuple(value.shape)} != {tuple(target.shape)}"
            )
        converted[key] = value.contiguous()
    native.load_state_dict(converted)

    optimizer = native.configure_optimizers(
        float(training_config.get("weight_decay", 0.1)),
        float(training_config.get("learning_rate", 6e-4)),
        (
            float(training_config.get("beta1", 0.9)),
            float(training_config.get("beta2", 0.95)),
        ),
        "cpu",
    )
    checkpoint = {
        "model": native.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": 0,
        "best_val_loss": 1e9,
        "config": {
            "dataset": dataset,
            "source_hf_model": model_id,
            "source_hf_revision": revision,
            "weights_only_restart": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
