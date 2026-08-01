"""
CropQuery — Knowledge Base Builder
"""
import os, re, json
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, HF_KISAN_DATASET, HF_AGROBENCH_DATASET

_KISAN_QUESTION_COLS = ["question","questions","Question","Questions","query","Query","farmer_query","input","text"]
_KISAN_ANSWER_COLS   = ["answer","answers","Answer","Answers","expert_answer","response","Response","output","label"]

def _get_col(row: dict, candidates: list) -> str:
    for key in candidates:
        val = row.get(key)
        if val and str(val).strip():
            return str(val).strip()
    return ""


def _normalize_label(s) -> str:
    """
    Must exactly match _normalize_label() in retriever.py — this is what lets
    an $eq metadata filter succeed regardless of the casing/spacing/underscores
    used in the source dataset vs. the predicted class label. If you change
    the normalization here, change it in retriever.py too and rebuild the KB
    (force_rebuild=True) so old un-normalized entries are replaced.
    """
    if not s:
        return ""
    return re.sub(r"[\s_\-]+", " ", str(s)).strip().lower()

def _parse_kisan_dataset() -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("[KB] datasets not installed — skipping Kisan source")
        return []
    print("[KB] Loading Kisan Call Centre dataset...")
    try:
        ds = load_dataset(HF_KISAN_DATASET, split="train")
    except Exception as e:
        print(f"[KB] Could not load Kisan dataset: {e}")
        return []
    if len(ds) == 0:
        return []
    print(f"[KB] Kisan columns: {list(ds[0].keys())}")
    chunks, skipped = [], 0
    for i, row in enumerate(ds):
        question = _get_col(row, _KISAN_QUESTION_COLS)
        answer   = _get_col(row, _KISAN_ANSWER_COLS)
        if not question or not answer:
            skipped += 1
            continue
        text = f"Farmer Query: {question}\nExpert Answer: {answer}"
        # FIXED: truncate at sentence boundary, not mid-word
        if len(text) > 1500:
            text = text[:1500].rsplit(". ", 1)[0] + "."
        chunks.append({
            "id":       f"kisan_{i}",
            "text":     text,
            "metadata": {
                "source":         "kisan_call_centre",
                "crop":           "general",
                "condition":      "general",
                "crop_norm":      "general",
                "condition_norm": "general",
                "chunk_type":     "qa_pair",
                "language":       "en",
            },
        })
    if skipped:
        print(f"[KB] Kisan: skipped {skipped} rows with missing question/answer")
    print(f"[KB] Kisan: {len(chunks)} Q&A pairs loaded")
    return chunks

def _build_agrobench_chunks() -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
        import os
        token = os.environ.get("HF_TOKEN")
        ds = load_dataset(
            HF_AGROBENCH_DATASET,
            split="train",
            token=token,
        )
        chunks = []
        for i, row in enumerate(ds):
            chunks.append({
                "id":       f"agrobench_{i}",
                "text":     row["text"],
                "metadata": {
                    "source":         "agrobench",
                    "crop":           row["crop"],
                    "condition":      row["condition"],
                    "crop_norm":      _normalize_label(row["crop"]),
                    "condition_norm": _normalize_label(row["condition"]),
                    "chunk_type":     "disease_entry",
                    "language":       "en",
                },
            })
        print(f"[KB] AgroBench: {len(chunks)} entries loaded from HuggingFace")
        return chunks
    except Exception as e:
        print(f"[KB] AgroBench HF load failed: {e}")
        return []

class KnowledgeBaseBuilder:
    def __init__(self, chroma_dir: str = CHROMA_DIR, embed_model: str = EMBED_MODEL):
        os.makedirs(chroma_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.ef = SentenceTransformerEmbeddingFunction(model_name=embed_model)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def build(self, force_rebuild: bool = False, batch_size: int = 64) -> int:
        if force_rebuild:
            print("[KB] Rebuilding from scratch...")
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.ef,
                metadata={"hnsw:space": "cosine"},
            )
        elif self.collection.count() > 0:
            print(f"[KB] Collection already has {self.collection.count()} docs. "
                  "Pass force_rebuild=True to overwrite.")
            return self.collection.count()

        all_chunks = _build_agrobench_chunks() + _parse_kisan_dataset()
        if not all_chunks:
            print("[KB] No chunks to ingest!")
            return 0

        seen, unique = set(), []
        for c in all_chunks:
            if c["id"] not in seen:
                seen.add(c["id"])
                unique.append(c)

        total = len(unique)
        print(f"[KB] Upserting {total} chunks into ChromaDB...")
        for start in range(0, total, batch_size):
            batch = unique[start:start + batch_size]
            self.collection.upsert(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )
        print(f"[KB] Done. Collection size: {self.collection.count()} documents")
        return self.collection.count()

    def stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        sources: Dict[str, int] = {}
        sample = self.collection.get(limit=min(count, 1000), include=["metadatas"])
        for meta in sample["metadatas"]:
            src = meta.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        return {"total_documents": count, "source_breakdown": sources}
