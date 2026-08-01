"""
CropQuery — RAG Retriever
"""
import re
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .config import (
    CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL,
    TOP_CHUNKS, TOP_K_AGROBENCH, TOP_K_KISAN,
)

_RAG_JARGON_MAP = {
    "new flush":          "new leaves",
    "flush emergence":    "new leaves appearing",
    "flush":              "new leaves",
    "lesions":            "spots",
    "necrosis":           "dead patches",
    "necrotic":           "dead",
    "inoculum":           "disease source",
    "sporulation":        "disease spreading",
    "oviposition":        "egg-laying",
    "larval mining":      "pest damage inside leaves",
    "foliar application": "leaf spray",
    "phytotoxicity":      "plant damage from chemicals",
    "chlorosis":          "yellowing",
    "defoliation":        "leaf fall",
    "stippling":          "tiny spots",
    "instar":             "pest stage",
}

def _clean_rag_chunk(text: str) -> str:
    for term, replacement in _RAG_JARGON_MAP.items():
        text = re.sub(re.escape(term), replacement, text, flags=re.IGNORECASE)
    return text


def _normalize_label(s: Optional[str]) -> str:
    """
    Collapse whitespace/underscores/hyphens and lowercase, so 'Downy_Mildew',
    'downy mildew', and 'Downy  Mildew' all compare equal. Must match the
    normalization applied to crop_norm/condition_norm at ingest time in
    knowledge_base.py — if you change one, change both and rebuild the KB.
    """
    if not s:
        return ""
    return re.sub(r"[\s_\-]+", " ", str(s)).strip().lower()


class CropRetriever:

    def __init__(
        self,
        chroma_dir: str = CHROMA_DIR,
        embed_model: str = EMBED_MODEL,
        top_k: int = TOP_CHUNKS,
    ):
        self.top_k = top_k
        client = chromadb.PersistentClient(path=chroma_dir)
        ef = SentenceTransformerEmbeddingFunction(model_name=embed_model)
        self.collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )
        print(f"[RAG] Connected to ChromaDB. {self.collection.count()} documents.")

    def retrieve(
        self,
        predicted_class: Optional[str],
        farmer_question: str = "What should I do?",
    ) -> List[Dict[str, Any]]:
        query     = self._build_query(predicted_class, farmer_question)
        crop      = self._extract_crop(predicted_class)
        condition = self._extract_condition(predicted_class)

        crop_chunks  = []
        kisan_chunks = []

        # 1. Crop + condition specific agrobench chunks
        if crop:
            try:
                where_filter = {"$and": [
                    {"source":    {"$eq": "agrobench"}},
                    {"crop_norm": {"$eq": crop}},
                ]}
                if condition:
                    where_filter["$and"].append({"condition_norm": {"$eq": condition}})
                r = self.collection.query(
                    query_texts=[query],
                    n_results=TOP_K_AGROBENCH,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"],
                )
                crop_chunks = self._parse_results(r)

                # If condition filter returned nothing, retry crop-only
                if not crop_chunks and condition:
                    r = self.collection.query(
                        query_texts=[query],
                        n_results=TOP_K_AGROBENCH,
                        where={"$and": [
                            {"source":    {"$eq": "agrobench"}},
                            {"crop_norm": {"$eq": crop}},
                        ]},
                        include=["documents", "metadatas", "distances"],
                    )
                    crop_chunks = self._parse_results(r)
            except Exception as e:
                print(f"[RAG] Crop query failed: {e}")

        # 2. Kisan Q&A chunks
        try:
            r = self.collection.query(
                query_texts=[query],
                n_results=TOP_K_KISAN,
                where={"source": {"$eq": "kisan_call_centre"}},
                include=["documents", "metadatas", "distances"],
            )
            kisan_chunks = self._parse_results(r)
        except Exception as e:
            print(f"[RAG] Kisan query failed: {e}")

        # 3. Fallback — unfiltered if still nothing crop-specific
        if not crop_chunks:
            try:
                r = self.collection.query(
                    query_texts=[query],
                    n_results=TOP_K_AGROBENCH,
                    include=["documents", "metadatas", "distances"],
                )
                crop_chunks = self._parse_results(r)
                print("[RAG] Warning: using unfiltered fallback — verify chunks are relevant")
            except Exception as e:
                print(f"[RAG] Unfiltered fallback failed: {e}")

        # 4. Post-filter by condition to reduce mixed chunks
        if condition and crop_chunks:
            filtered = [
                c for c in crop_chunks
                if c["metadata"].get("source") != "agrobench"
                or c["metadata"].get("condition_norm", "") == condition
                or c["metadata"].get("condition_norm", "") in ("general", "unknown", "")
            ]
            if filtered:
                crop_chunks = filtered

        return (crop_chunks + kisan_chunks)[:(TOP_K_AGROBENCH + TOP_K_KISAN)]

    def build_rag_context(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "No specific knowledge retrieved. Use general agronomy principles."
        MAX_CHUNK_CHARS = 400
        lines = ["=== Retrieved Agricultural Knowledge ==="]
        for i, chunk in enumerate(chunks, 1):
            src  = chunk["metadata"].get("source", "unknown")
            crop = chunk["metadata"].get("crop", "")
            cond = chunk["metadata"].get("condition", "")
            label = f"[{i}] Source: {src}"
            if crop and crop not in ("Unknown", "general"):
                label += f" | Crop: {crop}"
            if cond and cond != "general":
                label += f" | Disease: {cond}"
            lines.append(label)
            text = _clean_rag_chunk(chunk["text"])
            if len(text) > MAX_CHUNK_CHARS:
                text = text[:MAX_CHUNK_CHARS].rsplit(". ", 1)[0] + "."
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _extract_crop(predicted_class: Optional[str]) -> Optional[str]:
        if not predicted_class:
            return None
        return _normalize_label(predicted_class.split("__")[0]) or None

    @staticmethod
    def _extract_condition(predicted_class: Optional[str]) -> Optional[str]:
        if not predicted_class or "__" not in predicted_class:
            return None
        return _normalize_label(predicted_class.split("__", 1)[1]) or None

    @staticmethod
    def _build_query(predicted_class: Optional[str], farmer_question: str) -> str:
        parts = []
        if predicted_class:
            crop, _, cond = predicted_class.partition("__")
            parts.append(f"{crop.replace('_', ' ')} {cond.replace('_', ' ')} disease symptoms treatment")
        parts.append(farmer_question)
        return " ".join(parts)

    @staticmethod
    def _parse_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        if not results or not results.get("documents"):
            return chunks
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({"text": doc, "metadata": meta, "distance": dist})
        return chunks
