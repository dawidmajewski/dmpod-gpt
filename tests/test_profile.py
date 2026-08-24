from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "dmpod" / "lib"))

from dmpod_profile import load_profile, parameter_counts, validate_profile
from trainer import Reporter, event_thresholds, learning_rate


class ProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = __import__("os").environ.get("DMPOD_PROFILE_ROOT")
        __import__("os").environ["DMPOD_PROFILE_ROOT"] = str(
            PROJECT / "dmpod" / "profiles"
        )

    def tearDown(self) -> None:
        if self.previous is None:
            __import__("os").environ.pop("DMPOD_PROFILE_ROOT", None)
        else:
            __import__("os").environ["DMPOD_PROFILE_ROOT"] = self.previous

    def test_minimal_en_parameter_count_is_exact(self) -> None:
        profile = load_profile("minimal-en-125m")[1]
        validate_profile(profile)
        self.assertEqual(
            parameter_counts(profile["model"]),
            {
                "actual_parameters_total": 124281600,
                "actual_parameters_trainable": 124281600,
                "actual_parameters_non_embedding": 122708736,
            },
        )

    def test_token_scheduler_reaches_minimum(self) -> None:
        optimizer = {
            "max_lr": 1e-3,
            "min_lr": 1e-4,
            "warmup_tokens": 100,
            "lr_decay_end_tokens": 1000,
        }
        self.assertAlmostEqual(learning_rate(optimizer, 0, 10), 1e-4)
        self.assertAlmostEqual(learning_rate(optimizer, 990, 10), 1e-4)

    def test_evaluation_thresholds_include_passes_and_final(self) -> None:
        config = {
            "batch": {"actual_train_tokens": 400, "effective_batch_tokens": 10},
            "optimizer": {"warmup_tokens": 20},
            "data": {"train_tokens_unique": 100},
            "evaluation": {
                "eval_interval_tokens": 75,
                "eval_at_warmup_end": True,
                "eval_at_data_passes": [1.0, 2.0, 4.0],
            },
        }
        thresholds = event_thresholds(config)
        self.assertIn("warmup-end", thresholds[20])
        self.assertIn("pass-1", thresholds[100])
        self.assertIn("pass-4", thresholds[400])
        self.assertIn("final-budget", thresholds[400])

    def test_reporter_uses_tokens_axis_and_writes_local_metrics(self) -> None:
        profile = load_profile("smoke-tinystories")[1]
        profile["schema"] = "dmpod.config"
        profile["profile"] = profile["name"]
        profile["name"] = "smoke-run"
        profile["model"].update(parameter_counts(profile["model"]))
        profile["data"].update(
            path="/tmp/data", train_tokens_unique=1000, val_tokens=100
        )
        profile["optimizer"].update(
            min_lr=1e-4,
            warmup_tokens=100,
            lr_decay_end_tokens=1000,
        )
        profile["batch"].update(
            effective_batch_tokens=8192,
            target_update_steps=1,
            target_train_tokens=1000,
            actual_train_tokens=8192,
            actual_data_passes=8.192,
        )
        profile["evaluation"]["train_eval_offsets"] = {
            "path": "eval/train_offsets.npy",
            "sha256": "e" * 64,
            "count": 1,
            "target_tokens": 100,
        }
        profile["logging"]["train_log_interval_steps"] = 1
        profile["dataset"] = {
            "name": "test",
            "revision": "revision",
            "dataset_size_gib": 0.1,
            "manifest_sha256": "a" * 64,
            "document_boundary_token_id": 0,
            "padding_token_id": None,
            "split_method": "test",
            "deduplication_method": "none",
            "files": {
                "train": {"sha256": "b" * 64},
                "val": {"sha256": "c" * 64},
            },
            "tokenizer": {
                "sha256": "d" * 64,
                "name": "tokenizer",
                "version": "1",
            },
        }
        profile["wandb"]["run_id"] = "test1234"

        class FakeRun:
            entity = "entity"
            project = "project"
            id = "test1234"
            name = "smoke-run"
            url = "https://wandb.invalid/run"

            def __init__(self) -> None:
                self.summary = {}
                self.defined = []
                self.logged = []
                self.artifacts = []
                self.exit_code = None

            def define_metric(self, *args, **kwargs):
                self.defined.append((args, kwargs))

            def log(self, values):
                self.logged.append(values)

            def log_artifact(self, artifact, aliases):
                self.artifacts.append((artifact, aliases))

            def finish(self, exit_code=0):
                self.exit_code = exit_code

        class FakeArtifact:
            def __init__(self, name, type, metadata):
                self.name = name
                self.type = type
                self.metadata = metadata
                self.files = []

            def add_file(self, path, name):
                self.files.append((path, name))

        fake_run = FakeRun()
        fake_wandb = types.ModuleType("wandb")
        fake_wandb.init = lambda **_kwargs: fake_run
        fake_wandb.Artifact = FakeArtifact
        previous = sys.modules.get("wandb")
        sys.modules["wandb"] = fake_wandb
        try:
            with tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary)
                (run_dir / "logs" / "wandb").mkdir(parents=True)
                reporter = Reporter(
                    run_dir,
                    profile,
                    {
                        "gpu_model": ["test"],
                        "gpu_count": 1,
                        "cuda_version": "test",
                        "pytorch_version": "test",
                        "wandb_version": "test",
                        "git_commit": "commit",
                        "git_dirty": False,
                    },
                    enabled=True,
                )
                reporter.log(
                    {
                        "progress/tokens_seen": 8192,
                        "train/loss": 4.0,
                    },
                    "training",
                )
                checkpoint = run_dir / "checkpoint.pt"
                checkpoint.write_bytes(b"checkpoint")
                reporter.artifact(
                    checkpoint,
                    ["latest", "final"],
                    {"tokens_seen": 8192},
                    upload=True,
                )
                reporter.summary({"final_val_loss": 3.9})
                reporter.finish()
                self.assertTrue((run_dir / "metrics.jsonl").is_file())
                self.assertTrue((run_dir / "wandb.json").is_file())
                self.assertIn(
                    (("train/*",), {"step_metric": "progress/tokens_seen"}),
                    fake_run.defined,
                )
                self.assertEqual(fake_run.logged[0]["train/loss"], 4.0)
                self.assertEqual(fake_run.artifacts[0][1], ["latest", "final"])
                self.assertEqual(fake_run.summary["final_val_loss"], 3.9)
        finally:
            if previous is None:
                sys.modules.pop("wandb", None)
            else:
                sys.modules["wandb"] = previous


if __name__ == "__main__":
    unittest.main()
