from __future__ import annotations

import html
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

from dmpod_common import atomic_write, file_sha256


MEDIA_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".webp"}
COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed")


def render_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return format(value, ".8g")
    if isinstance(value, (list, tuple)):
        return ", ".join(render_value(item) for item in value)
    return str(value)


def table(rows: list[tuple[str, Any]]) -> str:
    lines = ["| Field | Value |", "| --- | --- |"]
    for key, value in rows:
        rendered = render_value(value).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {key} | {rendered} |")
    return "\n".join(lines)


def checkpoint_table(artifacts: dict[str, Any] | None) -> str:
    if not artifacts or not artifacts.get("checkpoints"):
        return "No checkpoint records are available."
    lines = [
        "| Aliases | File | Tokens seen | Data passes | SHA-256 |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for record in artifacts["checkpoints"]:
        metadata = record.get("metadata", {})
        lines.append(
            f"| {', '.join(record['aliases'])} | `{record['path']}` | "
            f"{render_value(metadata.get('tokens_seen'))} | "
            f"{render_value(metadata.get('data_pass_equivalent'))} | "
            f"`{record['sha256']}` |"
        )
    return "\n".join(lines)


def benchmark_sections(
    benchmarks: dict[str, Any] | None,
) -> tuple[str, str, str, str]:
    if not benchmarks or not benchmarks.get("results"):
        return (
            "No quality benchmark results are available. Run `dmpod-benchmark NAME` "
            "before publishing the model.",
            "",
            "",
            "",
        )
    lines = [
        "| Benchmark | Language | Protocol | Primary metric | Score | Samples |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    metric_lines = [
        "| Benchmark | Metric | Value |",
        "| --- | --- | ---: |",
    ]
    links: list[str] = []
    seen_links: set[tuple[str, str]] = set()
    order = benchmarks.get("selected") or list(benchmarks["results"])
    for benchmark_id in order:
        result = benchmarks["results"][benchmark_id]
        protocol = result.get("protocol", {})
        implementation = protocol.get("implementation", "unspecified")
        num_fewshot = protocol.get("num_fewshot")
        if num_fewshot is not None:
            implementation = f"{implementation} ({num_fewshot}-shot)"
        lines.append(
            f"| {result['name']} | {result['language']} | {implementation} | "
            f"`{result['primary_metric']}` | "
            f"{render_value(result['primary_value'])} | {result['samples']} |"
        )
        metrics = result.get("metrics") or {
            result["primary_metric"]: result["primary_value"]
        }
        for metric, value in sorted(metrics.items()):
            metric_lines.append(
                f"| {result['name']} | `{metric}` | {render_value(value)} |"
            )
        for link in result.get("links", []):
            item = (str(link["label"]), str(link["url"]))
            if item not in seen_links:
                seen_links.add(item)
                links.append(f"- [{item[0]}]({item[1]})")
    references = "Benchmark references:\n\n" + "\n".join(links) if links else ""
    limit = benchmarks.get("execution", {}).get("limit_per_benchmark")
    note = ""
    if limit is not None:
        note = (
            f"> **Not publication-ready:** these benchmarks were limited to {limit} "
            "samples per benchmark and are intended only for integration testing."
        )
    return "\n".join(lines), "\n".join(metric_lines), references, note


def load_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Invalid metrics record at {path}:{line_number}")
            records.append(value)
    return records


def _compact_number(value: float) -> str:
    absolute = abs(value)
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if absolute >= scale:
            return f"{value / scale:.3g}{suffix}"
    return f"{value:.4g}"


def _write_svg_chart(
    path: Path,
    title: str,
    y_label: str,
    series: list[tuple[str, list[tuple[float, float]]]],
) -> bool:
    series = [(label, points) for label, points in series if points]
    if not series:
        return False
    width, height = 960, 480
    left, right, top, bottom = 92, 30, 56, 68
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [point for _, points in series for point in points]
    x_min = min(point[0] for point in values)
    x_max = max(point[0] for point in values)
    y_min = min(point[1] for point in values)
    y_max = max(point[1] for point in values)
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        padding = abs(y_min) * 0.05 or 1.0
        y_min -= padding
        y_max += padding

    def x_position(value: float) -> float:
        return left + (value - x_min) * plot_width / (x_max - x_min)

    def y_position(value: float) -> float:
        return top + (y_max - value) * plot_height / (y_max - y_min)

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="480" viewBox="0 0 960 480">',
        '<rect width="960" height="480" fill="#ffffff"/>',
        f'<text x="{left}" y="32" font-family="sans-serif" font-size="22" font-weight="600">{html.escape(title)}</text>',
    ]
    for index in range(6):
        ratio = index / 5
        y = top + ratio * plot_height
        value = y_max - ratio * (y_max - y_min)
        lines.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#4b5563">{html.escape(_compact_number(value))}</text>',
            ]
        )
    for index in range(6):
        ratio = index / 5
        x = left + ratio * plot_width
        value = x_min + ratio * (x_max - x_min)
        lines.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#f3f4f6"/>',
                f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#4b5563">{html.escape(_compact_number(value))}</text>',
            ]
        )
    lines.extend(
        [
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>',
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 14}" text-anchor="middle" font-family="sans-serif" font-size="13">Tokens seen</text>',
            f'<text x="20" y="{top + plot_height / 2:.2f}" text-anchor="middle" transform="rotate(-90 20 {top + plot_height / 2:.2f})" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>',
        ]
    )
    legend_x = left
    for index, (label, points) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        coordinates = " ".join(
            f"{x_position(x):.2f},{y_position(y):.2f}" for x, y in points
        )
        if len(points) > 1:
            lines.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        for x, y in points:
            lines.append(
                f'<circle cx="{x_position(x):.2f}" cy="{y_position(y):.2f}" r="3" fill="{color}"/>'
            )
        legend_y = height - 42
        lines.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 28}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{html.escape(label)}</text>',
            ]
        )
        legend_x += 42 + len(label) * 7
    lines.append("</svg>")
    atomic_write(path, "\n".join(lines) + "\n")
    return True


def generate_training_charts(
    metrics: list[dict[str, Any]], output_dir: Path
) -> list[dict[str, str]]:
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    def points(key: str) -> list[tuple[float, float]]:
        result: list[tuple[float, float]] = []
        for record in metrics:
            x = record.get("progress/tokens_seen")
            y = record.get(key)
            if (
                isinstance(x, (int, float))
                and not isinstance(x, bool)
                and isinstance(y, (int, float))
                and not isinstance(y, bool)
                and math.isfinite(float(x))
                and math.isfinite(float(y))
            ):
                result.append((float(x), float(y)))
        return result

    definitions = (
        (
            "training-loss.svg",
            "Training and evaluation loss",
            "Cross-entropy loss",
            [
                ("Training", points("train/loss")),
                ("Train evaluation", points("eval/train_loss")),
                ("Validation", points("eval/val_loss")),
            ],
        ),
        (
            "learning-rate.svg",
            "Learning-rate schedule",
            "Learning rate",
            [("Learning rate", points("train/lr"))],
        ),
        (
            "throughput.svg",
            "Training throughput",
            "Tokens per second",
            [("Throughput", points("perf/tokens_per_sec"))],
        ),
    )
    generated: list[dict[str, str]] = []
    for filename, title, y_label, series in definitions:
        destination = assets_dir / filename
        if _write_svg_chart(destination, title, y_label, series):
            generated.append(
                {
                    "label": title,
                    "path": destination.relative_to(output_dir).as_posix(),
                    "source": "metrics.jsonl",
                    "sha256": file_sha256(destination),
                }
            )
    return generated


def collect_wandb_media(
    run_dir: Path, output_dir: Path, supplied_directories: list[Path]
) -> list[dict[str, str]]:
    candidates: list[tuple[Path, str]] = []
    local_wandb = run_dir / "logs" / "wandb"
    if local_wandb.is_dir():
        candidates.extend(
            (path, "W&B local media")
            for path in local_wandb.glob("**/files/media/images/**/*")
            if path.is_file()
        )
    for directory in supplied_directories:
        directory = directory.expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        candidates.extend(
            (path, "W&B downloaded media")
            for path in directory.rglob("*")
            if path.is_file()
        )
    assets_dir = output_dir / "assets" / "wandb"
    copied: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    for source, label in sorted(candidates, key=lambda item: str(item[0])):
        suffix = source.suffix.lower()
        if suffix not in MEDIA_SUFFIXES:
            continue
        digest = file_sha256(source)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", source.stem).strip("-._")
        stem = stem or "wandb-image"
        destination = assets_dir / f"{stem}-{digest[:8]}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        copied.append(
            {
                "label": f"{label}: {source.stem}",
                "path": destination.relative_to(output_dir).as_posix(),
                "source": label,
                "sha256": digest,
            }
        )
    return copied


def media_markdown(items: list[dict[str, str]]) -> str:
    if not items:
        return "No training-curve or W&B media files were available."
    return "\n\n".join(
        f"![{item['label'].replace(']', '&#93;')}]({item['path']})" for item in items
    )


def yaml_string(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=True)


def public_report_value(value: Any, run_dir: Path) -> Any:
    if isinstance(value, dict):
        return {
            str(key): public_report_value(item, run_dir) for key, item in value.items()
        }
    if isinstance(value, list):
        return [public_report_value(item, run_dir) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        path = Path(value)
        try:
            return path.relative_to(run_dir).as_posix()
        except ValueError:
            return f"<local>/{path.name}"
    return value


def model_card_front_matter(
    *,
    model_name: str,
    config: dict[str, Any],
    manifest: dict[str, Any],
    summary: dict[str, Any],
    benchmarks: dict[str, Any] | None,
    languages: list[str],
    dataset_ids: list[str],
    model_license: str | None,
    extra_tags: list[str],
) -> str:
    publication = config.get("publication", {})
    training_tags = {
        tag
        for tag in config.get("wandb", {}).get("tags", [])
        if tag != "dmpod" and not tag.startswith("schema:dmpod-training-")
    }
    tags = sorted(
        {
            "causal-lm",
            "nanogpt",
            "nanogpt-training",
            *training_tags,
            *publication.get("tags", []),
            *extra_tags,
        }
    )
    metrics = {"loss", "perplexity"}
    publishable_benchmarks = bool(
        benchmarks
        and benchmarks.get("execution", {}).get("limit_per_benchmark") is None
    )
    if publishable_benchmarks and benchmarks:
        metrics.update(
            result["primary_metric"]
            for result in benchmarks.get("results", {}).values()
        )
    lines = ["---"]
    if languages:
        lines.append("language:")
        lines.extend(f"- {yaml_string(value)}" for value in languages)
    if model_license:
        lines.append(f"license: {yaml_string(model_license)}")
    lines.append(
        f"pipeline_tag: {yaml_string(publication.get('pipeline_tag', 'text-generation'))}"
    )
    lines.append("tags:")
    lines.extend(f"- {yaml_string(value)}" for value in tags)
    if dataset_ids:
        lines.append("datasets:")
        lines.extend(f"- {yaml_string(value)}" for value in dataset_ids)
    lines.append("metrics:")
    lines.extend(f"- {yaml_string(value)}" for value in sorted(metrics))
    source = manifest.get("source", {})
    if source.get("type") == "hf" and not source.get("local"):
        lines.append(f"base_model: {yaml_string(source['model_id'])}")

    results: list[dict[str, Any]] = []
    validation_dataset = dataset_ids[0] if dataset_ids else config["dataset"]["name"]
    for metric, key in (
        ("loss", "final_val_loss"),
        ("perplexity", "final_val_perplexity"),
    ):
        value = summary.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            results.append(
                {
                    "dataset_name": config["dataset"]["name"],
                    "dataset_type": validation_dataset,
                    "metric": metric,
                    "value": value,
                }
            )
    if publishable_benchmarks and benchmarks:
        for benchmark_id in benchmarks.get("selected", []):
            result = benchmarks["results"][benchmark_id]
            protocol = result.get("protocol", {})
            results.append(
                {
                    "dataset_name": result["name"],
                    "dataset_type": protocol.get("dataset", benchmark_id),
                    "metric": result["primary_metric"],
                    "value": result["primary_value"],
                }
            )
    if results:
        lines.extend(
            [
                "model-index:",
                f"- name: {yaml_string(model_name)}",
                "  results:",
            ]
        )
        for result in results:
            lines.extend(
                [
                    "  - task:",
                    '      type: "text-generation"',
                    '      name: "Text Generation"',
                    "    dataset:",
                    f"      name: {yaml_string(result['dataset_name'])}",
                    f"      type: {yaml_string(result['dataset_type'])}",
                    "    metrics:",
                    f"    - type: {yaml_string(result['metric'])}",
                    f"      value: {render_value(result['value'])}",
                ]
            )
    lines.extend(["---", ""])
    return "\n".join(lines)


def native_checkpoint_usage() -> str:
    return """This repository uses a native trusted nanoGPT training checkpoint; it is
not a Transformers `AutoModel` package unless a separate conversion is provided. The
checkpoint contains Python-pickled state and must be treated as trusted executable input.

```python
import sys
import torch

sys.path.insert(0, "sources")
from model import GPT, GPTConfig

checkpoint = torch.load("checkpoint.pt", map_location="cpu", weights_only=False)
args = checkpoint["model_args"]
model = GPT(GPTConfig(
    block_size=args["block_size"],
    vocab_size=args["vocab_size"],
    n_layer=args["n_layer"],
    n_head=args["n_head"],
    n_embd=args["n_embd"],
    dropout=args["dropout"],
    bias=args["bias"],
))
model.load_state_dict(checkpoint["model"])
model.eval()
```"""
