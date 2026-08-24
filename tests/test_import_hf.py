from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "dmpod" / "lib"))

from import_hf import import_hf_weights


class ImportHfTests(unittest.TestCase):
    def test_import_writes_initial_weights_artifact(self) -> None:
        source_model = types.SimpleNamespace(
            config=types.SimpleNamespace(model_type="gpt2", vocab_size=4),
            state_dict=lambda: {"transformer.wte.weight": torch.ones(4, 2)},
        )
        source_model.cpu = lambda: source_model
        source_model.eval = lambda: source_model

        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                return source_model

        transformers = types.ModuleType("transformers")
        transformers.AutoModelForCausalLM = AutoModelForCausalLM
        previous_transformers = sys.modules.get("transformers")
        previous_model = sys.modules.pop("model", None)
        sys.modules["transformers"] = transformers
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "model.py").write_text(
                    """\
import torch


class GPTConfig:
    def __init__(self, **values):
        self.__dict__.update(values)


class GPT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.wte = torch.nn.Embedding(
            config.vocab_size, config.n_embd
        )
        self.lm_head = torch.nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )
        self.lm_head.weight = self.transformer.wte.weight
""",
                    encoding="utf-8",
                )
                output = root / "initial-weights.pt"
                import_hf_weights(
                    nanogpt_root=root,
                    model_id="test/model",
                    revision="abc123",
                    output=output,
                    model_config={
                        "n_layer": 1,
                        "n_head": 1,
                        "n_embd": 2,
                        "block_size": 4,
                        "bias": False,
                        "dropout": 0.0,
                    },
                    cache_dir=root / "cache",
                )
                artifact = torch.load(output, map_location="cpu", weights_only=False)
                self.assertEqual(artifact["schema"], "dmpod.initial-weights")
                self.assertEqual(artifact["schema_version"], 1)
                self.assertEqual(artifact["source"]["revision"], "abc123")
                self.assertIn("transformer.wte.weight", artifact["model"])
                self.assertIn("lm_head.weight", artifact["model"])
                self.assertNotIn("optimizer", artifact)
        finally:
            sys.modules.pop("model", None)
            if previous_model is not None:
                sys.modules["model"] = previous_model
            if previous_transformers is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = previous_transformers


if __name__ == "__main__":
    unittest.main()
