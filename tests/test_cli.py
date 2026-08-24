from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
BIN = PROJECT / "dmpod" / "bin"
BANNER = PROJECT / "dmpod" / "banner.txt"
SHELL_BANNER = PROJECT / "dmpod" / "shell-banner.sh"
ENTRYPOINT = PROJECT / "scripts" / "container-entrypoint.sh"
TEMPLATE = PROJECT / "dmpod" / "workspace-template"


class DMPodCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.home = self.root / "home"
        self.home.mkdir()
        self.image_nanogpt = self.root / "image-nanogpt"
        self.image_nanogpt.mkdir()
        (self.image_nanogpt / "train.py").write_text("# test\n", encoding="utf-8")
        (self.image_nanogpt / "model.py").write_text("# test\n", encoding="utf-8")
        (self.image_nanogpt / ".git").mkdir()
        (self.image_nanogpt / ".git" / "HEAD").write_text(
            "ref: refs/heads/test\n", encoding="utf-8"
        )
        self.env = os.environ.copy()
        self.env.pop("WANDB_API_KEY", None)
        self.env.pop("HF_TOKEN", None)
        self.env.pop("HUGGING_FACE_HUB_TOKEN", None)
        self.env.update(
            DMPOD_WORKSPACE=str(self.workspace),
            DMPOD_NANOGPT_ROOT=str(self.workspace / "nanogpt"),
            DMPOD_IMAGE_NANOGPT=str(self.image_nanogpt),
            DMPOD_TEMPLATE_ROOT=str(TEMPLATE),
            DMPOD_NANOGPT_PATCH="",
            DMPOD_GPU_COUNT="1",
            DMPOD_TEST_GPU_NAME="Test GPU",
            DMPOD_TEST_WANDB_VERSION="test",
            HF_HOME=str(self.workspace / "cache" / "huggingface"),
            HOME=str(self.home),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_command(
        self, *command: str, input_text: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            command,
            env=self.env,
            check=False,
            capture_output=True,
            text=True,
            input=input_text,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"command failed: {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        return result

    def run_failure(
        self, *command: str, input_text: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            command,
            env=self.env,
            check=False,
            capture_output=True,
            text=True,
            input=input_text,
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
            "--skip-hf",
            "--non-interactive",
        )

    def install_fake_wandb(self, expected_credential: str) -> None:
        modules = self.root / "test-modules"
        modules.mkdir()
        (modules / "wandb.py").write_text(
            """\
import os
from pathlib import Path


def login(key, verify):
    if not verify or key != os.environ["DMPOD_TEST_WANDB_CREDENTIAL"]:
        raise RuntimeError("rejected test credential")
    netrc = Path.home() / ".netrc"
    netrc.write_text("test W&B credentials configured\\n", encoding="utf-8")
    netrc.chmod(0o600)
    return True


class Api:
    def __init__(self, api_key, timeout):
        if api_key != os.environ["DMPOD_TEST_WANDB_CREDENTIAL"]:
            raise RuntimeError("rejected test credential")
        self.viewer = {"username": "test-user"}
        self.default_entity = "test-entity"
""",
            encoding="utf-8",
        )
        self.env["DMPOD_TEST_WANDB_CREDENTIAL"] = expected_credential
        self.env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(modules), self.env.get("PYTHONPATH")))
        )

    def install_fake_huggingface(self, expected_credential: str) -> None:
        modules = self.root / "test-modules"
        modules.mkdir(exist_ok=True)
        package = modules / "huggingface_hub"
        package.mkdir()
        (package / "__init__.py").write_text(
            """\
import os
from pathlib import Path


def _token_path():
    return Path(os.environ["HF_HOME"]) / "token"


def get_token():
    if token := os.environ.get("HF_TOKEN"):
        return token
    if _token_path().is_file():
        return os.environ["DMPOD_TEST_HF_CREDENTIAL"]
    return None


def login(token, add_to_git_credential, skip_if_logged_in):
    if (
        token != os.environ["DMPOD_TEST_HF_CREDENTIAL"]
        or add_to_git_credential
        or skip_if_logged_in
    ):
        raise ValueError("rejected test credential")
    path = _token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test Hugging Face credentials configured\\n", encoding="utf-8")
    path.chmod(0o600)


class HfApi:
    def __init__(self, token):
        self.token = token

    def whoami(self):
        if self.token != os.environ["DMPOD_TEST_HF_CREDENTIAL"]:
            raise ValueError("rejected test credential")
        return {"name": "test-hf-user"}
""",
            encoding="utf-8",
        )
        self.env["DMPOD_TEST_HF_CREDENTIAL"] = expected_credential
        self.env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(modules), self.env.get("PYTHONPATH")))
        )

    def install_fake_tmux(self) -> Path:
        binaries = self.root / "test-bin"
        binaries.mkdir()
        capture = self.root / "tmux-arguments.txt"
        tmux = binaries / "tmux"
        tmux.write_text(
            """\
#!/bin/sh
if [ "$1" = "has-session" ]; then
    exit 1
fi
printf '%s\\n' "$@" > "$DMPOD_TEST_TMUX_ARGUMENTS"
""",
            encoding="utf-8",
        )
        tmux.chmod(0o755)
        self.env["DMPOD_TEST_TMUX_ARGUMENTS"] = str(capture)
        self.env["PATH"] = os.pathsep.join((str(binaries), self.env["PATH"]))
        return capture

    def test_entrypoint_copies_git_and_does_not_overwrite_workspace(self) -> None:
        self.initialize_workspace()
        nanogpt = self.workspace / "nanogpt"
        self.assertTrue((nanogpt / ".git" / "HEAD").is_file())
        self.assertTrue((nanogpt / "AGENTS.md").is_file())
        marker = nanogpt / "user-change.txt"
        marker.write_text("keep\n", encoding="utf-8")
        self.initialize_workspace()
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep\n")

    def test_banner_identifies_slayerlab_and_links_instructions(self) -> None:
        banner = BANNER.read_text(encoding="utf-8")
        self.assertIn("Academy", banner)
        self.assertIn("https://huggingface.co/SlayerLab", banner)
        self.assertIn("https://github.com/dawidmajewski/dmpod-gpt", banner)

        before_setup = self.run_command("/bin/bash", str(SHELL_BANNER))
        self.assertIn('Run "dmpod-setup"', before_setup.stdout)

        config = self.workspace / ".dmpod" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text("version = 1\n", encoding="utf-8")
        after_setup = self.run_command("/bin/bash", str(SHELL_BANNER))
        self.assertNotIn('Run "dmpod-setup"', after_setup.stdout)

    def test_smoke_test_help_does_not_require_a_configured_workspace(self) -> None:
        result = self.run_command(str(BIN / "dmpod-smoke-test"), "--help")
        self.assertIn("TinyStories GPU and W&B smoke test", result.stdout)

    def test_quality_benchmark_catalog_does_not_require_a_workspace(self) -> None:
        result = self.run_command(str(BIN / "dmpod-benchmark"), "--list")
        self.assertIn("blimp", result.stdout)
        self.assertIn("arc-challenge", result.stdout)
        self.assertIn("8tags", result.stdout)
        self.assertIn("polemo2-out", result.stdout)

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

    def test_setup_accepts_hidden_credential_and_reuses_workspace_copy(self) -> None:
        self.initialize_workspace()
        credential = secrets.token_hex(24)
        self.install_fake_wandb(credential)

        result = self.run_command(
            str(BIN / "dmpod-setup"),
            "--skip-hf",
            input_text=f"\n{credential}\ny\n",
        )

        key_path = self.workspace / ".dmpod" / "secrets" / "wandb.key"
        config_path = self.workspace / ".dmpod" / "config.toml"
        self.assertTrue(key_path.is_file())
        self.assertEqual(stat.S_IMODE(key_path.stat().st_mode), 0o600)
        config = config_path.read_text(encoding="utf-8")
        self.assertIn('entity = "test-entity"', config)
        self.assertIn('key_source = "workspace"', config)
        self.assertNotIn(credential, config)
        self.assertNotIn(credential, result.stdout)
        self.assertNotIn(credential, result.stderr)
        netrc = self.home / ".netrc"
        self.assertTrue(netrc.is_file())
        self.assertEqual(stat.S_IMODE(netrc.stat().st_mode), 0o600)

        reused = self.run_command(
            str(BIN / "dmpod-setup"),
            "--skip-hf",
            "--non-interactive",
        )
        self.assertIn("key source: workspace", reused.stdout)

    def test_setup_does_not_save_rejected_interactive_credential(self) -> None:
        self.initialize_workspace()
        self.install_fake_wandb(secrets.token_hex(24))
        rejected_credential = secrets.token_hex(24)

        result = self.run_failure(
            str(BIN / "dmpod-setup"),
            "--skip-hf",
            input_text=f"\n{rejected_credential}\ny\n",
        )

        self.assertIn("W&B rejected the configured API key", result.stderr)
        self.assertNotIn(rejected_credential, result.stdout)
        self.assertNotIn(rejected_credential, result.stderr)
        self.assertFalse(
            (self.workspace / ".dmpod" / "secrets" / "wandb.key").exists()
        )
        self.assertFalse((self.workspace / ".dmpod" / "config.toml").exists())
        self.assertFalse((self.home / ".netrc").exists())

    def test_setup_keeps_interactive_credential_ephemeral_by_default(self) -> None:
        self.initialize_workspace()
        credential = secrets.token_hex(24)
        self.install_fake_wandb(credential)

        result = self.run_command(
            str(BIN / "dmpod-setup"),
            "--skip-hf",
            input_text=f"\n{credential}\n\n",
        )

        key_path = self.home / ".config" / "dmpod" / "wandb.key"
        self.assertTrue(key_path.is_file())
        self.assertEqual(stat.S_IMODE(key_path.stat().st_mode), 0o600)
        self.assertFalse(
            (self.workspace / ".dmpod" / "secrets" / "wandb.key").exists()
        )
        config = (self.workspace / ".dmpod" / "config.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn('key_source = "ephemeral"', config)
        self.assertNotIn(credential, config)
        self.assertNotIn(credential, result.stdout)
        self.assertNotIn(credential, result.stderr)

    def test_setup_logs_into_huggingface_and_reuses_cached_token(self) -> None:
        self.initialize_workspace()
        credential = secrets.token_hex(24)
        self.install_fake_huggingface(credential)

        result = self.run_command(
            str(BIN / "dmpod-setup"),
            "--skip-wandb",
            input_text=f"\n{credential}\n",
        )

        token_path = Path(self.env["HF_HOME"]) / "token"
        self.assertTrue(token_path.is_file())
        self.assertEqual(stat.S_IMODE(token_path.stat().st_mode), 0o600)
        self.assertIn("Hugging Face: connected as test-hf-user", result.stdout)
        self.assertNotIn(credential, result.stdout)
        self.assertNotIn(credential, result.stderr)

        reused = self.run_command(
            str(BIN / "dmpod-setup"),
            "--skip-wandb",
            "--non-interactive",
        )
        self.assertIn("token source: cached", reused.stdout)

    def test_setup_does_not_save_rejected_huggingface_token(self) -> None:
        self.initialize_workspace()
        self.install_fake_huggingface(secrets.token_hex(24))
        rejected_credential = secrets.token_hex(24)

        result = self.run_failure(
            str(BIN / "dmpod-setup"),
            "--skip-wandb",
            input_text=f"\n{rejected_credential}\n",
        )

        self.assertIn("Hugging Face rejected the configured token", result.stderr)
        self.assertNotIn(rejected_credential, result.stdout)
        self.assertNotIn(rejected_credential, result.stderr)
        self.assertFalse((Path(self.env["HF_HOME"]) / "token").exists())
        self.assertFalse((self.workspace / ".dmpod" / "config.toml").exists())

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
            [
                "git",
                "-C",
                str(nanogpt),
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "--allow-empty",
                "-qm",
                "test",
            ],
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
        self.assertIn(
            "Training command:",
            self.run_command(str(BIN / "dmpod-train"), "test-run", "--dry-run").stdout,
        )

        tmux_arguments = self.install_fake_tmux()
        launched = self.run_command(
            str(BIN / "dmpod-train"), "test-run", "--tmux"
        )
        shell_command = tmux_arguments.read_text(encoding="utf-8").splitlines()[-1]
        self.assertIn("set -o pipefail", shell_command)
        self.assertIn("PYTHONUNBUFFERED=1", shell_command)
        self.assertIn("2>&1 | tee -a", shell_command)
        self.assertIn("logs/training.log", shell_command)
        self.assertIn("tmux attach -t dmpod-test-run", launched.stdout)

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

    def create_profile_dataset(self) -> Path:
        dataset = self.workspace / "nanogpt" / "data" / "tinystories-smoke"
        dataset.mkdir(parents=True)
        train = dataset / "train.bin"
        val = dataset / "val.bin"
        tokenizer = dataset / "tokenizer.json"
        train.write_bytes(bytes(2048))
        val.write_bytes(bytes(1024))
        tokenizer.write_text('{"encoding":"test"}\n', encoding="utf-8")
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            "schema_version": 1,
            "name": "tinystories-smoke",
            "revision": "test-revision",
            "license": "test-only",
            "dtype": "uint16",
            "files": {
                "train": {
                    "path": "train.bin",
                    "sha256": digest(train),
                    "tokens": 1024,
                },
                "val": {
                    "path": "val.bin",
                    "sha256": digest(val),
                    "tokens": 512,
                },
            },
            "tokenizer": {
                "path": "tokenizer.json",
                "sha256": digest(tokenizer),
                "name": "test-tokenizer",
                "version": "1",
                "vocab_size": 50304,
            },
            "document_boundary_token_id": 0,
            "padding_token_id": None,
            "split_method": "fixed test split",
            "deduplication_method": "none",
        }
        (dataset / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
        return dataset

    def test_profile_run_snapshots_resolved_config_and_dataset_hashes(self) -> None:
        self.initialize_workspace()
        self.setup_environment()
        nanogpt = self.workspace / "nanogpt"
        subprocess.run(["git", "init", "-q", str(nanogpt)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(nanogpt),
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "--allow-empty",
                "-qm",
                "test",
            ],
            check=True,
        )
        dataset = self.create_profile_dataset()
        result = self.run_command(
            str(BIN / "dmpod-create-training"),
            "--profile",
            "smoke-tinystories",
        )
        name = "lr-0.001_seed-1337_btok-8192"
        self.assertIn(name, result.stdout)
        run = self.workspace / "runs" / name
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        resolved = json.loads(
            (run / "resolved-config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["version"], 2)
        self.assertEqual(resolved["model"]["actual_parameters_total"], 16091392)
        self.assertEqual(resolved["dataset"]["files"]["train"]["tokens"], 1024)
        self.assertTrue((run / "sources" / "trainer.py").is_file())
        self.assertTrue((run / "eval" / "train_offsets.npy").is_file())
        dry_run = self.run_command(str(BIN / "dmpod-train"), name, "--dry-run")
        self.assertIn("trainer.py", dry_run.stdout)

        (run / "summary.json").write_text(
            json.dumps(
                {
                    "final_val_loss": 4.0,
                    "min_val_loss": 3.9,
                    "tokens_at_min_val_loss": 8192,
                    "final_train_eval_loss": 3.8,
                    "final_val_perplexity": 54.6,
                    "final_tokens_seen": 8192,
                    "final_data_pass_equivalent": 8.0,
                    "completed_target_budget": True,
                    "optimizer_updates_completed": 1,
                    "skipped_updates_total": 0,
                    "wall_time_hours": 0.01,
                    "mean_tokens_per_sec": 1000.0,
                    "peak_gpu_memory_allocated_gb": 1.0,
                    "peak_gpu_memory_reserved_gb": 1.2,
                    "best_checkpoint_alias": "best-val",
                    "final_checkpoint_alias": "final",
                    "stop_reason": "completed",
                    "exit_code": 0,
                }
            ),
            encoding="utf-8",
        )
        (run / "artifacts.json").write_text(
            json.dumps({"version": 1, "checkpoints": []}), encoding="utf-8"
        )
        (run / "benchmarks").mkdir()
        (run / "benchmarks" / "results.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "selected": ["blimp", "8tags"],
                    "results": {
                        "blimp": {
                            "name": "BLiMP",
                            "language": "English",
                            "primary_metric": "acc",
                            "primary_value": 0.51,
                            "samples": 67000,
                            "links": [
                                {
                                    "label": "BLiMP",
                                    "url": "https://github.com/alexwarstadt/blimp",
                                }
                            ],
                        },
                        "8tags": {
                            "name": "8Tags",
                            "language": "Polish",
                            "primary_metric": "accuracy",
                            "primary_value": 0.25,
                            "samples": 4372,
                            "links": [
                                {
                                    "label": "8Tags",
                                    "url": "https://huggingface.co/datasets/sdadas/8tags",
                                }
                            ],
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        self.run_command(str(BIN / "dmpod-export-run"), name)
        readme = (run / "reports" / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "Final validation loss",
            readme,
        )
        self.assertIn("| BLiMP | English | `acc` | 0.510000 | 67000 |", readme)
        self.assertGreater(readme.index("[BLiMP]"), readme.index("| BLiMP |"))
        self.assertIn(
            "Target token budget completed",
            (run / "reports" / "PR_BODY.md").read_text(encoding="utf-8"),
        )
        report_context = json.loads(
            (run / "reports" / "report-context.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report_context["benchmarks"]["selected"], ["blimp", "8tags"])

        (dataset / "train.bin").write_bytes(b"\x01\x00" + bytes(2046))
        changed = self.run_failure(str(BIN / "dmpod-train"), name, "--dry-run")
        self.assertIn("SHA-256 mismatch", changed.stderr)

    def test_profile_rejects_unresolved_dataset_manifest(self) -> None:
        self.initialize_workspace()
        self.setup_environment()
        nanogpt = self.workspace / "nanogpt"
        subprocess.run(["git", "init", "-q", str(nanogpt)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(nanogpt),
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "--allow-empty",
                "-qm",
                "test",
            ],
            check=True,
        )
        dataset = self.create_profile_dataset()
        manifest = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
        manifest["revision"] = "REPLACE_WITH_REVISION"
        (dataset / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
        failed = self.run_failure(
            str(BIN / "dmpod-create-training"),
            "--profile",
            "smoke-tinystories",
        )
        self.assertIn("Unresolved placeholder", failed.stderr)


if __name__ == "__main__":
    unittest.main()
