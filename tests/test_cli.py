from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
BIN = PROJECT / "dmpod" / "bin"
ENTRYPOINT = PROJECT / "scripts" / "container-entrypoint.sh"
TEMPLATE = PROJECT / "dmpod" / "workspace-template"


class DMPodCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.image_nanogpt = self.root / "image-nanogpt"
        self.image_nanogpt.mkdir()
        (self.image_nanogpt / "train.py").write_text("# test\n", encoding="utf-8")
        (self.image_nanogpt / "model.py").write_text("# test\n", encoding="utf-8")
        (self.image_nanogpt / ".git").mkdir()
        (self.image_nanogpt / ".git" / "HEAD").write_text(
            "ref: refs/heads/test\n", encoding="utf-8"
        )
        self.env = os.environ.copy()
        self.env.update(
            DMPOD_WORKSPACE=str(self.workspace),
            DMPOD_NANOGPT_ROOT=str(self.workspace / "nanogpt"),
            DMPOD_IMAGE_NANOGPT=str(self.image_nanogpt),
            DMPOD_TEMPLATE_ROOT=str(TEMPLATE),
            DMPOD_NANOGPT_PATCH="",
            DMPOD_GPU_COUNT="1",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_command(self, *command: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            command,
            env=self.env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"command failed: {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        return result

    def run_failure(self, *command: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            command,
            env=self.env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0, "command unexpectedly succeeded")
        return result

    def initialize_workspace(self) -> None:
        self.run_command(str(ENTRYPOINT), "/usr/bin/true")

    def setup_environment(self) -> None:
        self.run_command(
            str(BIN / "dmpod-setup"),
            "--wandb-mode",
            "offline",
            "--non-interactive",
        )

    def test_entrypoint_copies_git_and_does_not_overwrite_workspace(self) -> None:
        self.initialize_workspace()
        nanogpt = self.workspace / "nanogpt"
        self.assertTrue((nanogpt / ".git" / "HEAD").is_file())
        self.assertTrue((nanogpt / "AGENTS.md").is_file())
        marker = nanogpt / "user-change.txt"
        marker.write_text("keep\n", encoding="utf-8")
        self.initialize_workspace()
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep\n")

    def test_setup_writes_environment_only_config_with_private_mode(self) -> None:
        self.initialize_workspace()
        self.setup_environment()
        config = self.workspace / ".dmpod" / "config.toml"
        self.assertTrue(config.is_file())
        self.assertEqual(stat.S_IMODE(config.stat().st_mode), 0o600)
        content = config.read_text(encoding="utf-8")
        self.assertIn('mode = "offline"', content)
        self.assertNotIn("model_id", content)
        self.assertNotIn("learning_rate", content)

    def test_registers_existing_dataset_without_copying(self) -> None:
        self.initialize_workspace()
        self.setup_environment()
        source = self.workspace / "datasets" / "existing"
        source.mkdir(parents=True)
        (source / "train.bin").write_bytes(b"\x00\x00\x01\x00")
        (source / "val.bin").write_bytes(b"\x00\x00")
        self.run_command(
            str(BIN / "dmpod-prepare-data"),
            "existing",
            "existing",
            str(source),
        )
        target = self.workspace / "nanogpt" / "data" / "existing"
        self.assertTrue(target.is_symlink())
        self.assertEqual(target.resolve(), source.resolve())

    def test_creates_snapshot_for_scratch_training(self) -> None:
        self.initialize_workspace()
        self.setup_environment()
        nanogpt = self.workspace / "nanogpt"
        subprocess.run(["git", "init", "-q", str(nanogpt)], check=True)
        subprocess.run(
            ["git", "-C", str(nanogpt), "-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-qm", "test"],
            check=True,
        )
        dataset = nanogpt / "data" / "tiny"
        dataset.mkdir(parents=True)
        (dataset / "train.bin").write_bytes(bytes(128))
        (dataset / "val.bin").write_bytes(bytes(128))
        model = self.root / "model.py"
        model.write_text(
            "n_layer=2\nn_head=2\nn_embd=64\nblock_size=32\nbias=False\n",
            encoding="utf-8",
        )
        training = self.root / "training.py"
        training.write_text(
            'dataset="tiny"\nbatch_size=2\ngradient_accumulation_steps=1\n',
            encoding="utf-8",
        )
        self.run_command(
            str(BIN / "dmpod-create-training"),
            "test-run",
            "--model-config",
            str(model),
            "--training-config",
            str(training),
            "--source",
            "scratch",
        )
        run = self.workspace / "runs" / "test-run"
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source"]["type"], "scratch")
        self.assertEqual(manifest["architecture"]["n_head"], 2)
        self.assertTrue((run / "model.py").is_file())
        self.assertTrue((run / "training.py").is_file())
        self.assertTrue((run / "runtime.py").is_file())
        self.assertIn("Training command:", self.run_command(
            str(BIN / "dmpod-train"), "test-run", "--dry-run"
        ).stdout)

        (run / "state.json").write_text(
            json.dumps({"version": 1, "status": "failed", "attempts": 1}),
            encoding="utf-8",
        )
        restarted = self.run_command(
            str(BIN / "dmpod-train"), "test-run", "--restart", "--dry-run"
        )
        self.assertIn("--init_from=scratch", restarted.stdout)

        with (run / "training.py").open("a", encoding="utf-8") as config:
            config.write("learning_rate=1e-4\n")
        changed = self.run_failure(
            str(BIN / "dmpod-train"), "test-run", "--restart", "--dry-run"
        )
        self.assertIn("Run config changed after creation", changed.stderr)


if __name__ == "__main__":
    unittest.main()
