"""Create source-stratified and paired summaries for the clean ResNet screen."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / "data/split_manifest_v1.csv"
PREDICTIONS = PROJECT / "results/vision_clean_resnet_v1/paired_predictions.csv"
SUMMARY = PROJECT / "results/vision_clean_resnet_v1/summary.json"
OUTPUT_JSON = PROJECT / "results/vision_clean_resnet_v1/analysis.json"
OUTPUT_MD = PROJECT / "results/vision_clean_resnet_v1/analysis.md"
CHECKPOINTS = PROJECT / "checkpoints/clean_resnet_v1"
SEED = 42
CLASS_NAMES = json.loads((PROJECT / "data/class_names.json").read_text())


def file_identity(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "relative_path": str(path.relative_to(PROJECT)),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def bootstrap_difference(rows: list[dict], left: str, right: str) -> list[float]:
    groups = sorted({row["group_id"] for row in rows})
    group_index = {name: index for index, name in enumerate(groups)}
    sizes = np.zeros(len(groups), dtype=int)
    differences = np.zeros(len(groups), dtype=int)
    for row in rows:
        index = group_index[row["group_id"]]
        sizes[index] += 1
        differences[index] += int(row[left] == row["truth"]) - int(row[right] == row["truth"])
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(2000):
        selected = rng.integers(0, len(groups), len(groups))
        samples.append(100.0 * differences[selected].sum() / sizes[selected].sum())
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def accuracy(rows: list[dict], model: str) -> float:
    return sum(row[model] == row["truth"] for row in rows) / len(rows)


def main() -> None:
    source_by_id = {}
    with MANIFEST.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "test":
                source_by_id[row["id"]] = row["source"]

    rows = []
    by_source = defaultdict(list)
    with PREDICTIONS.open(newline="") as stream:
        for row in csv.DictReader(stream):
            for field in ("truth", "base", "ce", "f2"):
                row[field] = int(row[field])
            row["source"] = source_by_id[row["id"]]
            rows.append(row)
            by_source[row["source"]].append(row)

    stored = json.loads(SUMMARY.read_text())
    comparisons = {}
    for left, right in (("ce", "base"), ("f2", "base"), ("f2", "ce")):
        differences = [
            int(row[left] == row["truth"]) - int(row[right] == row["truth"])
            for row in rows
        ]
        comparisons[f"{left}_minus_{right}"] = {
            "difference_pp": 100.0 * sum(differences) / len(rows),
            "group_bootstrap_95pct_ci_pp": bootstrap_difference(rows, left, right),
            "left_wrong_right_right": sum(value == -1 for value in differences),
            "left_right_right_wrong": sum(value == 1 for value in differences),
        }

    source_results = {}
    for source, source_rows in sorted(by_source.items()):
        source_results[source] = {
            "n": len(source_rows),
            "base_top1": accuracy(source_rows, "base"),
            "ce_top1": accuracy(source_rows, "ce"),
            "f2_top1": accuracy(source_rows, "f2"),
            "f2_minus_ce_pp": 100.0 * (
                accuracy(source_rows, "f2") - accuracy(source_rows, "ce")
            ),
        }

    error_categories = {}
    for model in ("base", "ce", "f2"):
        counts = Counter()
        for row in rows:
            truth = CLASS_NAMES[row["truth"]]
            predicted = CLASS_NAMES[row[model]]
            if predicted == truth:
                counts["correct"] += 1
            elif predicted.split("__", 1)[0] == truth.split("__", 1)[0]:
                counts["wrong_condition_same_crop"] += 1
            else:
                counts["wrong_crop"] += 1
        error_categories[model] = dict(counts)

    result = {
        "manifest_sha256": stored["manifest_sha256"],
        "seed": SEED,
        "n_test": len(rows),
        "selected_epochs_zero_based": stored["selected_epochs_zero_based"],
        "selected_checkpoint_files": {
            stage: file_identity(CHECKPOINTS / f"{stage}_best.pth")
            for stage in ("base", "ce", "f2")
        },
        "comparisons": comparisons,
        "source_results": source_results,
        "error_categories": error_categories,
        "interpretation": (
            "One-seed internal screen. Group bootstrap reflects test-group sampling only; "
            "source, seed, and external-domain uncertainty are not represented."
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# Clean ResNet50 screen analysis",
        "",
        f"Frozen manifest: `{result['manifest_sha256']}`; seed {SEED}; test images {len(rows):,}.",
        "",
        "## Selected checkpoints",
        "",
        "| Stage | Zero-based epoch | Bytes | SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for stage, identity in result["selected_checkpoint_files"].items():
        lines.append(
            f"| {stage.upper()} | {result['selected_epochs_zero_based'][stage]} | "
            f"{identity['bytes']:,} | `{identity['sha256']}` |"
        )
    lines.extend([
        "",
        "## Paired comparisons",
        "",
        "| Comparison | Difference (pp) | Group-bootstrap 95% CI (pp) | Left wins | Right wins |",
        "|---|---:|---:|---:|---:|",
    ])
    for key, value in comparisons.items():
        left, _, right = key.partition("_minus_")
        low, high = value["group_bootstrap_95pct_ci_pp"]
        lines.append(
            f"| {left.upper()} minus {right.upper()} | {value['difference_pp']:.3f} | "
            f"[{low:.3f}, {high:.3f}] | {value['left_right_right_wrong']} | "
            f"{value['left_wrong_right_right']} |"
        )
    lines.extend([
        "",
        "| Source | Test n | Base (%) | CE (%) | F2 (%) | F2-CE (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for source, value in source_results.items():
        lines.append(
            f"| {source} | {value['n']:,} | {100*value['base_top1']:.2f} | "
            f"{100*value['ce_top1']:.2f} | {100*value['f2_top1']:.2f} | "
            f"{value['f2_minus_ce_pp']:.2f} |"
        )
    lines.extend([
        "",
        "## Post hoc error categories",
        "",
        "| Checkpoint | Correct | Wrong condition, same crop | Wrong crop |",
        "|---|---:|---:|---:|",
    ])
    for model, counts in error_categories.items():
        lines.append(
            f"| {model.upper()} | {counts['correct']:,} | "
            f"{counts['wrong_condition_same_crop']:,} | {counts['wrong_crop']:,} |"
        )
    lines.extend([
        "",
        "The primary CE-versus-F2 result is negative for F2. The interval represents "
        "resampling of test groups only and does not justify a multi-seed or external-domain claim. "
        "The error-category analysis was performed after the test read and is descriptive.",
        "",
    ])
    OUTPUT_MD.write_text("\n".join(lines))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
