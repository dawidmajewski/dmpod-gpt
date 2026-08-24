from __future__ import annotations

import contextlib
import datetime as dt
import importlib.metadata
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from dmpod_common import file_sha256


@dataclass(frozen=True)
class Benchmark:
    benchmark_id: str
    name: str
    language: str
    url: str
    harness_task: str | None = None
    dataset: str | None = None
    revision: str | None = None
    split: str | None = None
    primary_metric: str = "accuracy"


BENCHMARKS = (
    Benchmark(
        "blimp",
        "BLiMP",
        "English",
        "https://github.com/alexwarstadt/blimp",
        harness_task="blimp",
        primary_metric="acc",
    ),
    Benchmark(
        "lambada",
        "LAMBADA",
        "English",
        "https://huggingface.co/datasets/EleutherAI/lambada_openai",
        harness_task="lambada_openai",
        primary_metric="acc",
    ),
    Benchmark(
        "hellaswag",
        "HellaSwag",
        "English",
        "https://rowanzellers.com/hellaswag/",
        harness_task="hellaswag",
        primary_metric="acc_norm",
    ),
    Benchmark(
        "piqa",
        "PIQA",
        "English",
        "https://yonatanbisk.com/piqa/",
        harness_task="piqa",
        primary_metric="acc_norm",
    ),
    Benchmark(
        "sciq",
        "SciQ",
        "English",
        "https://allenai.org/data/sciq",
        harness_task="sciq",
        primary_metric="acc_norm",
    ),
    Benchmark(
        "arc-easy",
        "ARC-Easy",
        "English",
        "https://allenai.org/data/arc",
        harness_task="arc_easy",
        primary_metric="acc_norm",
    ),
    Benchmark(
        "arc-challenge",
        "ARC-Challenge",
        "English",
        "https://allenai.org/data/arc",
        harness_task="arc_challenge",
        primary_metric="acc_norm",
    ),
    Benchmark(
        "8tags",
        "8Tags",
        "Polish",
        "https://huggingface.co/datasets/sdadas/8tags",
        dataset="sdadas/8tags",
        revision="82ddf9c0d06ffc982aeccf2473b7ce31f2167adf",
        split="test",
    ),
    Benchmark(
        "polemo2-in",
        "PolEmo2-IN",
        "Polish",
        "https://huggingface.co/datasets/allegro/klej-polemo2-in",
        dataset="allegro/klej-polemo2-in",
        revision="9843af31facd613eb091deaa4141df4331d4f918",
        split="test",
    ),
    Benchmark(
        "polemo2-out",
        "PolEmo2-OUT",
        "Polish",
        "https://huggingface.co/datasets/allegro/klej-polemo2-out",
        dataset="allegro/klej-polemo2-out",
        revision="b1a18b27a37cfb366969e1aa5721f8bf6ec5e179",
        split="test",
    ),
)
BENCHMARK_BY_ID = {benchmark.benchmark_id: benchmark for benchmark in BENCHMARKS}


def benchmark_ids(language: str | None = None) -> list[str]:
    return [
        benchmark.benchmark_id
        for benchmark in BENCHMARKS
        if language is None or benchmark.language == language
    ]


def parse_benchmarks(values: Iterable[str]) -> list[str]:
    aliases = {
        "all": set(benchmark_ids()),
        "english": set(benchmark_ids("English")),
        "english-only": set(benchmark_ids("English")),
        "polish": set(benchmark_ids("Polish")),
        "polish-only": set(benchmark_ids("Polish")),
    }
    selected: set[str] = set()
    unknown: list[str] = []
    for value in values:
        for item in value.split(","):
            normalized = item.strip().lower()
            if not normalized:
                continue
            if normalized in aliases:
                selected.update(aliases[normalized])
            elif normalized in BENCHMARK_BY_ID:
                selected.add(normalized)
            else:
                unknown.append(item.strip())
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {', '.join(unknown)}")
    if not selected:
        raise ValueError("Select at least one benchmark")
    return [item for item in benchmark_ids() if item in selected]


def interactive_benchmarks() -> list[str]:
    import questionary
    from questionary import Choice

    mode = questionary.select(
        "Which benchmarks should run?",
        choices=["All", "English only", "Polish only", "Let me choose"],
    ).ask()
    if mode is None:
        raise KeyboardInterrupt
    if mode == "All":
        return benchmark_ids()
    if mode == "English only":
        return benchmark_ids("English")
    if mode == "Polish only":
        return benchmark_ids("Polish")

    english = questionary.checkbox(
        "English benchmarks",
        choices=[
            Choice(BENCHMARK_BY_ID[item].name, value=item)
            for item in benchmark_ids("English")
        ],
    ).ask()
    if english is None:
        raise KeyboardInterrupt
    polish = questionary.checkbox(
        "Polish benchmarks",
        choices=[
            Choice(BENCHMARK_BY_ID[item].name, value=item)
            for item in benchmark_ids("Polish")
        ],
    ).ask()
    if polish is None:
        raise KeyboardInterrupt
    return parse_benchmarks([*english, *polish])


class TextTokenizer:
    def __init__(self, reference: dict[str, Any]) -> None:
        path = Path(reference["path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Tokenizer file does not exist: {path}")
        expected_hash = reference.get("sha256")
        if expected_hash and file_sha256(path) != expected_hash:
            raise RuntimeError(f"Tokenizer SHA-256 mismatch: {path}")
        self.name = str(reference.get("name", path.name))
        definition = json.loads(path.read_text(encoding="utf-8"))
        if definition.get("implementation") == "tiktoken":
            import tiktoken

            encoding = tiktoken.get_encoding(definition["encoding"])
            self._encode = encoding.encode_ordinary
        else:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(path))
            self._encode = lambda text: tokenizer.encode(
                text, add_special_tokens=False
            ).ids

    def encode(self, text: str) -> list[int]:
        return list(self._encode(text))


class NanoGPTBackend:
    def __init__(
        self,
        run_dir: Path,
        checkpoint_path: Path,
        device: torch.device,
        dtype: str,
        batch_size: int,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.checkpoint_path = checkpoint_path.resolve()
        self.device = device
        self.batch_size = batch_size
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        if checkpoint.get("version") != 2:
            raise ValueError("Quality benchmarks require a native version 2 checkpoint")
        model_args = checkpoint["model_args"]
        self.block_size = int(model_args["block_size"])
        self.vocab_size = int(model_args["vocab_size"])
        training_config = checkpoint["full_training_config"]
        boundary_id = training_config["dataset"].get("document_boundary_token_id")
        if type(boundary_id) is not int:
            raise ValueError(
                "The checkpoint dataset must define document_boundary_token_id "
                "for empty-context benchmark prompts"
            )
        self.eot_token_id = boundary_id
        self.tokenizer = TextTokenizer(checkpoint["tokenizer_reference"])
        self.tokenizer_reference = dict(checkpoint["tokenizer_reference"])
        self.precision = self._resolve_precision(dtype, training_config)
        self.truncated_requests = 0

        source = self.run_dir / "sources" / "model.py"
        if not source.is_file():
            raise FileNotFoundError(f"Run is missing the snapshotted model source: {source}")
        module_name = f"dmpod_benchmark_model_{id(self)}"
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load model source: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        config = module.GPTConfig(
            block_size=self.block_size,
            vocab_size=self.vocab_size,
            n_layer=int(model_args["n_layer"]),
            n_head=int(model_args["n_head"]),
            n_embd=int(model_args["n_embd"]),
            dropout=float(model_args.get("dropout", 0.0)),
            bias=bool(model_args["bias"]),
        )
        self.model = module.GPT(config)
        state = {
            key.removeprefix("_orig_mod."): value
            for key, value in checkpoint["model"].items()
        }
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        del checkpoint, state

    def _resolve_precision(self, dtype: str, config: dict[str, Any]) -> torch.dtype:
        selected = config["runtime"]["precision"] if dtype == "auto" else dtype
        if self.device.type == "cpu":
            return torch.float32
        precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[selected]
        if precision == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support bfloat16")
        return precision

    def _autocast(self):
        if self.device.type != "cuda" or self.precision == torch.float32:
            return contextlib.nullcontext()
        return torch.amp.autocast("cuda", dtype=self.precision)

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], int]:
        if not continuation:
            raise ValueError("Benchmark continuation cannot be empty")
        spaces = len(context) - len(context.rstrip())
        if spaces:
            continuation = context[-spaces:] + continuation
            context = context[:-spaces]
        if not context:
            continuation_ids = self.tokenizer.encode(continuation)
            if continuation_ids and continuation_ids[0] == self.eot_token_id:
                combined = continuation_ids
                continuation_ids = continuation_ids[1:]
            else:
                combined = [self.eot_token_id, *continuation_ids]
        else:
            context_ids = self.tokenizer.encode(context)
            combined = self.tokenizer.encode(context + continuation)
            continuation_ids = combined[len(context_ids) :]
        if not continuation_ids:
            raise ValueError("Benchmark continuation produced no tokens")
        if len(continuation_ids) > self.block_size:
            raise ValueError(
                "Benchmark continuation exceeds the model context length: "
                f"{len(continuation_ids)} > {self.block_size}"
            )
        if not combined or min(combined) < 0 or max(combined) >= self.vocab_size:
            raise ValueError("Tokenizer produced an ID outside the model vocabulary")
        return combined, len(continuation_ids)

    def _full_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, length = input_ids.shape
        positions = torch.arange(length, dtype=torch.long, device=self.device)
        transformer = self.model.transformer
        hidden = transformer.drop(
            transformer.wte(input_ids) + transformer.wpe(positions)
        )
        for block in transformer.h:
            hidden = block(hidden)
        return self.model.lm_head(transformer.ln_f(hidden))

    @torch.inference_mode()
    def score_pairs(
        self, pairs: list[tuple[str, str]], description: str
    ) -> list[tuple[float, bool, int]]:
        from tqdm import tqdm

        encoded = [self._encode_pair(*pair) for pair in pairs]
        results: list[tuple[float, bool, int]] = []
        batches = range(0, len(encoded), self.batch_size)
        for start in tqdm(batches, desc=description):
            batch = encoded[start : start + self.batch_size]
            prepared: list[tuple[list[int], int]] = []
            for combined, continuation_length in batch:
                if len(combined) > self.block_size + 1:
                    self.truncated_requests += 1
                prepared.append(
                    (combined[-(self.block_size + 1) :], continuation_length)
                )
            maximum = max(len(tokens) - 1 for tokens, _ in prepared)
            inputs = torch.full(
                (len(prepared), maximum),
                self.eot_token_id,
                dtype=torch.long,
                device=self.device,
            )
            for index, (tokens, _) in enumerate(prepared):
                values = torch.tensor(tokens[:-1], dtype=torch.long, device=self.device)
                inputs[index, : len(values)] = values
            with self._autocast():
                logits = self._full_logits(inputs)
            for index, (tokens, continuation_length) in enumerate(prepared):
                input_length = len(tokens) - 1
                first = input_length - continuation_length
                selected = logits[index, first:input_length].float()
                targets = torch.tensor(
                    tokens[-continuation_length:],
                    dtype=torch.long,
                    device=self.device,
                )
                log_probs = F.log_softmax(selected, dim=-1)
                token_scores = log_probs.gather(1, targets[:, None]).squeeze(1)
                greedy = bool(torch.equal(selected.argmax(dim=-1), targets))
                results.append(
                    (float(token_scores.sum().item()), greedy, continuation_length)
                )
        return results


def make_harness_model(backend: NanoGPTBackend):
    from lm_eval.api.model import LM

    class NanoGPTHarnessLM(LM):
        def __init__(self) -> None:
            super().__init__()
            self._device = backend.device

        @property
        def tokenizer_name(self) -> str:
            return backend.tokenizer.name

        @property
        def eot_token_id(self) -> int:
            return backend.eot_token_id

        @property
        def max_length(self) -> int:
            return backend.block_size

        def get_model_info(self) -> dict[str, Any]:
            return {
                "checkpoint": str(backend.checkpoint_path),
                "max_length": backend.block_size,
                "tokenizer": backend.tokenizer.name,
            }

        def loglikelihood(self, requests) -> list[tuple[float, bool]]:
            pairs = [request.args for request in requests]
            scored = backend.score_pairs(pairs, "English likelihoods")
            results = [(score, greedy) for score, greedy, _ in scored]
            for pair, result in zip(pairs, results, strict=True):
                self.cache_hook.add_partial("loglikelihood", pair, result)
            return results

        def loglikelihood_rolling(self, requests) -> list[float]:
            raise NotImplementedError("The selected benchmark requires rolling likelihood")

        def generate_until(self, requests) -> list[str]:
            raise NotImplementedError("The selected benchmark requires generation")

    return NanoGPTHarnessLM()


def _metric_value(record: dict[str, Any], metric: str) -> float:
    for key, value in record.items():
        if key.split(",", 1)[0] == metric and isinstance(value, (int, float)):
            return float(value)
    raise KeyError(f"Metric {metric!r} is absent from {sorted(record)}")


def _numeric_metrics(record: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in record.items()
        if isinstance(value, (int, float)) and not key.endswith("_stderr")
    }


def _sample_count(raw: dict[str, Any], benchmark: Benchmark) -> int:
    samples = raw.get("n-samples", {})
    if benchmark.benchmark_id == "blimp":
        return sum(
            int(value["effective"])
            for key, value in samples.items()
            if key.startswith("blimp_")
        )
    value = samples.get(benchmark.harness_task or "", {})
    return int(value.get("effective", 0))


def evaluate_english(
    backend: NanoGPTBackend, selected: list[str], limit: int | None
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from lm_eval import evaluator

    benchmarks = [BENCHMARK_BY_ID[item] for item in selected]
    raw = evaluator.simple_evaluate(
        model=make_harness_model(backend),
        tasks=[benchmark.harness_task for benchmark in benchmarks],
        num_fewshot=0,
        batch_size=backend.batch_size,
        device=str(backend.device),
        limit=limit,
        bootstrap_iters=0,
        log_samples=False,
    )
    if raw is None:
        raise RuntimeError("lm-evaluation-harness returned no results")
    results: dict[str, dict[str, Any]] = {}
    for benchmark in benchmarks:
        source = "groups" if benchmark.benchmark_id == "blimp" else "results"
        key = "blimp" if benchmark.benchmark_id == "blimp" else benchmark.harness_task
        record = raw[source][key]
        primary_value = _metric_value(record, benchmark.primary_metric)
        results[benchmark.benchmark_id] = {
            "name": benchmark.name,
            "language": benchmark.language,
            "protocol": {
                "implementation": "lm-evaluation-harness",
                "version": importlib.metadata.version("lm_eval"),
                "task": benchmark.harness_task,
                "num_fewshot": 0,
            },
            "primary_metric": benchmark.primary_metric,
            "primary_value": primary_value,
            "metrics": _numeric_metrics(record),
            "samples": _sample_count(raw, benchmark),
            "links": [{"label": benchmark.name, "url": benchmark.url}],
        }
    return results, json_ready(raw)


TOPIC_LABELS = (
    "film",
    "historia",
    "jedzenie",
    "medycyna",
    "motoryzacja",
    "praca",
    "sport",
    "technologia",
)
SENTIMENT_LABELS = (
    "negatywny",
    "neutralny",
    "pozytywny",
    "niejednoznaczny",
)
SENTIMENT_TARGETS = {
    "__label__meta_minus_m": 0,
    "__label__meta_zero": 1,
    "__label__meta_plus_m": 2,
    "__label__meta_amb": 3,
}


def polish_example(
    benchmark_id: str, row: dict[str, Any]
) -> tuple[str, tuple[str, ...], int]:
    if benchmark_id == "8tags":
        prompt = (
            "Przypisz tekst do jednej kategorii: film, historia, jedzenie, "
            "medycyna, motoryzacja, praca, sport, technologia.\n"
            f"Tekst: {row['sentence']}\nKategoria:"
        )
        target = int(row["label"])
        return prompt, TOPIC_LABELS, target
    prompt = (
        "Określ wydźwięk recenzji. Możliwe odpowiedzi: negatywny, neutralny, "
        "pozytywny, niejednoznaczny.\n"
        f"Recenzja: {row['sentence']}\nWydźwięk:"
    )
    target_value = row["target"]
    if target_value not in SENTIMENT_TARGETS:
        raise ValueError(f"Unknown PolEmo2 target: {target_value!r}")
    return prompt, SENTIMENT_LABELS, SENTIMENT_TARGETS[target_value]


def macro_f1(predictions: list[int], references: list[int], classes: int) -> float:
    scores: list[float] = []
    for label in range(classes):
        true_positive = sum(
            prediction == label and reference == label
            for prediction, reference in zip(predictions, references, strict=True)
        )
        false_positive = sum(
            prediction == label and reference != label
            for prediction, reference in zip(predictions, references, strict=True)
        )
        false_negative = sum(
            prediction != label and reference == label
            for prediction, reference in zip(predictions, references, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def evaluate_polish(
    backend: NanoGPTBackend, benchmark_id: str, limit: int | None
) -> dict[str, Any]:
    from datasets import load_dataset

    benchmark = BENCHMARK_BY_ID[benchmark_id]
    dataset = load_dataset(
        benchmark.dataset,
        revision=benchmark.revision,
        split=benchmark.split,
    )
    count = min(len(dataset), limit) if limit is not None else len(dataset)
    pairs: list[tuple[str, str]] = []
    references: list[int] = []
    choice_counts: list[int] = []
    for index in range(count):
        prompt, choices, target = polish_example(benchmark_id, dataset[index])
        pairs.extend((prompt, f" {choice}") for choice in choices)
        references.append(target)
        choice_counts.append(len(choices))
    scored = backend.score_pairs(pairs, f"{benchmark.name} likelihoods")
    predictions: list[int] = []
    offset = 0
    for choices in choice_counts:
        candidates = scored[offset : offset + choices]
        predictions.append(
            max(
                range(choices),
                key=lambda index: candidates[index][0] / candidates[index][2],
            )
        )
        offset += choices
    accuracy = sum(
        prediction == reference
        for prediction, reference in zip(predictions, references, strict=True)
    ) / len(references)
    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1(predictions, references, choice_counts[0]),
    }
    return {
        "name": benchmark.name,
        "language": benchmark.language,
        "protocol": {
            "implementation": "DMPod zero-shot classification",
            "dataset": benchmark.dataset,
            "revision": benchmark.revision,
            "split": benchmark.split,
            "num_fewshot": 0,
            "choice_score": "mean_loglikelihood_per_token",
            "note": "This generative protocol is not supervised KLEJ fine-tuning.",
        },
        "primary_metric": benchmark.primary_metric,
        "primary_value": metrics[benchmark.primary_metric],
        "metrics": metrics,
        "samples": count,
        "links": [{"label": benchmark.name, "url": benchmark.url}],
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def result_document(
    *,
    run_name: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    backend: NanoGPTBackend,
    selected: list[str],
    results: dict[str, dict[str, Any]],
    harness: dict[str, Any] | None,
    limit: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": run_name,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
            "trusted_input_required": True,
        },
        "execution": {
            "device": str(backend.device),
            "precision": str(backend.precision).removeprefix("torch."),
            "batch_size": backend.batch_size,
            "limit_per_benchmark": limit,
            "truncated_requests": backend.truncated_requests,
        },
        "tokenizer": {
            "name": backend.tokenizer.name,
            "reference": backend.tokenizer_reference,
        },
        "selected": selected,
        "results": results,
        "lm_evaluation_harness": harness,
    }


def format_score(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    return f"{value:.6f}"
