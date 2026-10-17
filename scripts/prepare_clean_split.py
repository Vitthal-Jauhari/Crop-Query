"""Create a prospective CropQuery image split from the nine pinned sources.

Exact file hashes, 64-bit image hashes, and source-local filename families
stay in the same partition. This is a lightweight leakage sanity check, not
proof that every related photograph has been found.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError


PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path("/workspace/datasets/cropquery_sources")
OUTPUT = PROJECT / "data" / "split_manifest_v1.csv"
AUDIT = PROJECT / "results" / "split_audit_v1.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def notebook_label_maps() -> dict:
    notebook = json.loads((PROJECT / "notebooks/training/ensemble-training.ipynb").read_text())
    source = "".join(notebook["cells"][6]["source"])
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "dataset_label_maps"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("could not find notebook label mapping")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(image.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return f"{bits:016x}"


def label_for(path: Path, source: str, maps: dict, mexwell: dict) -> str | None:
    if source == "mexwell":
        return mexwell.get(path.name)
    if source == "tushar5harma":
        return maps[source].get((path.parent.parent.parent.name, path.parent.name.strip()))
    label = path.parent.name
    if source == "shubham2703" and label == "Healthy":
        candidate = f"{path.parent.parent.name}__Healthy"
        if candidate in CLASS_INDEX:
            return candidate
    return maps[source].get(label)


def filename_family(path: Path, source: str) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"(?:_flip(?:lr|ud)?|_rot(?:\d+)?|_aug\d+)$", "", stem)
    parent_parts = path.relative_to(ROOT).parts[3:-1]
    parent_parts = [part.lower() for part in parent_parts
                    if part.lower() not in {"train", "test", "val", "validation", "train_set", "test_set"}]
    return f"{source}/{'/'.join(parent_parts)}/{stem}"


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def main() -> None:
    maps = notebook_label_maps()
    expected = sorted({v for mapping in maps.values() for v in mapping.values() if v})
    if expected != CLASS_NAMES:
        raise RuntimeError("notebook and project class maps differ")
    manifests = sorted(ROOT.glob("*/*/source_manifest.json"))
    if len(manifests) != 9:
        raise RuntimeError(f"expected nine completed source downloads; found {len(manifests)}")

    mexwell_labels = {}
    mexwell_root = ROOT / "mexwell/crop-diseases-classification/images"
    maps_json = list(mexwell_root.rglob("label_num_to_disease_map.json"))
    train_csv = list(mexwell_root.rglob("train.csv"))
    if len(maps_json) != 1 or len(train_csv) != 1:
        raise RuntimeError("Mexwell label files are missing or ambiguous")
    number_to_name = json.loads(maps_json[0].read_text())
    with train_csv[0].open(newline="") as stream:
        for row in csv.DictReader(stream):
            canonical = maps["mexwell"].get(number_to_name[str(row["label"])])
            if canonical:
                mexwell_labels[row["image_id"]] = canonical

    rows = []
    skipped = Counter()
    by_source = Counter()
    for manifest in manifests:
        metadata = json.loads(manifest.read_text())
        source = metadata["slug"].split("/")[0]
        for path in sorted((manifest.parent / "images").rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
                continue
            label = label_for(path, source, maps, mexwell_labels)
            if not label:
                skipped[source] += 1
                continue
            try:
                file_hash, image_hash = digest_file(path), dhash(path)
            except (OSError, UnidentifiedImageError):
                skipped[source] += 1
                continue
            relative = path.relative_to(ROOT).as_posix()
            rows.append({
                "id": hashlib.sha256(relative.encode()).hexdigest()[:20],
                "relative_path": relative,
                "class": label,
                "class_index": CLASS_INDEX[label],
                "source": metadata["slug"],
                "source_version": metadata["version"],
                "sha256": file_hash,
                "dhash": image_hash,
                "filename_family": filename_family(path, source),
            })
            by_source[source] += 1
        print(f"SCANNED {source}: {by_source[source]} mapped, {skipped[source]} skipped", flush=True)

    groups = DisjointSet(len(rows))
    seen = {}
    for index, row in enumerate(rows):
        for key in (("sha", row["sha256"]), ("dhash", row["dhash"]),
                    ("family", row["filename_family"])):
            if key in seen:
                groups.union(index, seen[key])
            else:
                seen[key] = index
    members = defaultdict(list)
    for index in range(len(rows)):
        members[groups.find(index)].append(index)

    grouped_by_class = defaultdict(list)
    excluded_conflicts = 0
    for indices in members.values():
        labels = {rows[i]["class"] for i in indices}
        if len(labels) != 1:
            excluded_conflicts += len(indices)
            for index in indices:
                rows[index]["split"] = "exclude_conflicting_labels"
            continue
        group_id = hashlib.sha256("|".join(sorted(rows[i]["id"] for i in indices)).encode()).hexdigest()[:20]
        for index in indices:
            rows[index]["group_id"] = group_id
        grouped_by_class[next(iter(labels))].append(indices)

    rng = random.Random(SEED)
    for label, class_groups in sorted(grouped_by_class.items()):
        rng.shuffle(class_groups)
        count = len(class_groups)
        if count < 3:
            test_n, val_n = 0, 0
        else:
            test_n = max(1, round(count * 0.15))
            val_n = max(1, round(count * 0.05))
            if test_n + val_n >= count:
                val_n = max(0, count - test_n - 1)
        for position, indices in enumerate(class_groups):
            split = "test" if position < test_n else "val" if position < test_n + val_n else "train"
            for index in indices:
                rows[index]["split"] = split

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "relative_path", "class", "class_index", "source", "source_version",
              "sha256", "dhash", "filename_family", "group_id", "split"]
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["split"] for row in rows)
    per_class = defaultdict(Counter)
    for row in rows:
        per_class[row["class"]][row["split"]] += 1
    audit = {
        "seed": SEED,
        "source_root": str(ROOT),
        "mapped_images": len(rows),
        "unique_groups": len(members),
        "class_count": len(grouped_by_class),
        "split_counts": dict(counts),
        "per_class_split_counts": {label: dict(split_counts) for label, split_counts in sorted(per_class.items())},
        "largest_group_images": max((len(indices) for indices in members.values()), default=0),
        "source_counts": dict(by_source),
        "skipped_unmapped_or_invalid": dict(skipped),
        "excluded_conflicting_labels": excluded_conflicts,
        "class_groups_below_three": [label for label, g in grouped_by_class.items() if len(g) < 3],
        "manifest_sha256": digest_file(OUTPUT),
        "limitation": "Exact and 64-bit image hashes plus filename families do not find all near duplicates or plant-level repeats.",
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"split_counts": audit["split_counts"],
                      "excluded_conflicting_labels": excluded_conflicts,
                      "manifest_sha256": audit["manifest_sha256"]}), flush=True)


CLASS_NAMES = json.loads((PROJECT / "data/class_names.json").read_text())
CLASS_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

if __name__ == "__main__":
    main()
