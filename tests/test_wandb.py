from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "dmpod" / "lib"))

from dmpod_profile import canonical_run_name
from dmpod_wandb import (
    PRIMARY_COMPARISON_METRIC,
    SCHEMA,
    configure_run,
    validate_metric_payload,
)


def config(profile: str = "minimal-en-125m") -> dict:
    return {
        "profile": profile,
        "dataset": {"name": "minimal-en-corpus-2.5b"},
        "runtime": {"seed": 1337},
        "optimizer": {"max_lr": 0.0006},
        "wandb": {
            "project": "profile-project",
            "job_type": "pretrain",
            "tags": ["nanogpt"],
        },
    }


class WandbContractTests(unittest.TestCase):
    def test_configure_run_builds_canonical_identity(self) -> None:
        value = config()
        configure_run(
            value,
            revision="r006",
            change="longer-warmup",
            hypothesis="A longer warmup reduces early gradient spikes.",
        )

        self.assertEqual(value["experiment"]["schema"], SCHEMA)
        self.assertEqual(value["wandb"]["project"], "profile-project")
        self.assertEqual(value["wandb"]["group"], "r006-longer-warmup")
        self.assertEqual(
            value["wandb"]["run_name"],
            "r006-longer-warmup-s1337-lr0.0006",
        )
        self.assertEqual(
            value["wandb"]["primary_comparison_metric"],
            PRIMARY_COMPARISON_METRIC,
        )
        self.assertIn("schema:nanogpt-training-v1", value["wandb"]["tags"])
        self.assertIn("nanogpt-training", value["wandb"]["tags"])
        self.assertNotIn("dmpod", value["wandb"]["tags"])
        self.assertIn("revision:r006", value["wandb"]["tags"])

    def test_config_run_project_is_derived_from_dataset(self) -> None:
        value = config(profile="config")
        configure_run(
            value,
            revision="r001",
            change="baseline",
            hypothesis="Establish the baseline.",
        )
        self.assertEqual(
            value["wandb"]["project"],
            "nanogpt-training-minimal-en-corpus-2-5b",
        )

    def test_explicit_wandb_values_override_convention(self) -> None:
        value = config()
        configure_run(
            value,
            revision="r002",
            change="override-check",
            hypothesis="Overrides remain available when explicitly requested.",
            project="custom-project",
            group="custom-group",
            run_name="custom W&B run",
            job_type="smoke-test",
            tags=["requested"],
        )
        self.assertEqual(value["wandb"]["project"], "custom-project")
        self.assertEqual(value["wandb"]["group"], "custom-group")
        self.assertEqual(value["wandb"]["run_name"], "custom W&B run")
        self.assertEqual(
            canonical_run_name(value), "r002-override-check-s1337-lr0.0006"
        )
        self.assertEqual(value["wandb"]["job_type"], "smoke-test")
        self.assertIn("requested", value["wandb"]["tags"])

    def test_experiment_and_metric_validation_rejects_ambiguity(self) -> None:
        with self.assertRaisesRegex(ValueError, "rNNN"):
            configure_run(
                config(),
                revision="6",
                change="baseline",
                hypothesis="Establish the baseline.",
            )
        with self.assertRaisesRegex(ValueError, "kebab-case"):
            configure_run(
                config(),
                revision="r006",
                change="Longer Warmup",
                hypothesis="Test a longer warmup.",
            )
        with self.assertRaisesRegex(ValueError, "Missing training metrics"):
            validate_metric_payload({"progress/tokens_seen": 1}, "training")


if __name__ == "__main__":
    unittest.main()
