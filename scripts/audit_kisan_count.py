"""Count Q&A records retained by src/knowledge_base.py at a pinned HF revision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[1]
FILE = Path("/workspace/datasets/cropquery_kisan/data/train-00000-of-00001.parquet")
REVISION = "57535f94e2feebe7bb1b320edbc4b2d5cf3406bb"
OUTPUT = PROJECT / "results/kisan_count_audit.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value(text) -> str:
    return str(text).strip() if text is not None and str(text).strip() else ""


def main() -> None:
    table = pq.read_table(FILE, columns=["questions", "answers", "text"])
    retained = 0
    for batch in table.to_batches():
        questions = batch.column("questions").to_pylist()
        answers = batch.column("answers").to_pylist()
        fallback = batch.column("text").to_pylist()
        retained += sum(bool(value(q) or value(t)) and bool(value(a))
                        for q, a, t in zip(questions, answers, fallback))
    audit = {
        "dataset": "Franklin01/Dataset_prep01",
        "revision": REVISION,
        "parquet_sha256": file_sha256(FILE),
        "raw_rows": table.num_rows,
        "loader_retained_qa": retained,
        "loader_skipped": table.num_rows - retained,
        "embedded_v2_crop_entries": 39,
        "expected_two_source_documents": retained + 39,
        "scope": "Count reconstruction from current pinned inputs; not a verification of every historical Chroma record or agronomic quality.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
