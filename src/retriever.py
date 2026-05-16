"""
CropQuery — RAG Retriever
Retrieves top-k relevant knowledge chunks for a given crop/condition query.

Logic
-----
1. Primary:  filter by predicted crop, semantic search over farmer question
2. Fallback: if crop returns < MIN_CROP_CHUNKS results, run unfiltered search
"""

from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .config import (
    CHROMA_DIR, COLLECTION_NAME,
    EMBED_MODEL, TOP_CHUNKS, MIN_CROP_CHUNKS,
)


class CropRetriever:
    """
    Wraps ChromaDB for RAG retrieval.

    Usage
    -----
    retriever = CropRetriever()
    chunks = retriever.retrieve("Tomato__Late_Blight", "What fungicide should I spray?")
    prompt  = retriever.build_rag_context(chunks)
    """

    def __init__(
        self,
        chroma_dir: str = CHROMA_DIR,
        embed_model: str = EMBED_MODEL,
        top_k: int = TOP_CHUNKS,
        min_crop_chunks: int = MIN_CROP_CHUNKS,
    ):
        self.top_k = top_k
        self.min_crop_chunks = min_crop_chunks

        client = chromadb.PersistentClient(path=chroma_dir)
        ef = SentenceTransformerEmbeddingFunction(model_name=embed_model)
        self.collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )
        print(f"[RAG] Connected to ChromaDB. {self.collection.count()} documents in collection.")

    def retrieve(
        self,
        predicted_class: Optional[str],
        farmer_question: str = "What should I do?",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks.

        Parameters
        ----------
        predicted_class : str, optional
            ResNet50 prediction in 'Crop__Condition' format, e.g. 'Tomato__Late_Blight'.
            If None, runs unfiltered retrieval.
        farmer_question : str
            Free-form question from the farmer.

        Returns
        -------
        List of dicts: {text, metadata, distance}
        """
        query = self._build_query(predicted_class, farmer_question)
        crop  = self._extract_crop(predicted_class)

        results = None

        # ── Crop-filtered retrieval ──────────────────────────────────────────
        if crop:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=self.top_k,
                    where={"crop": {"$eq": crop}},
                    include=["documents", "metadatas", "distances"],
                )
                n_found = len(results["documents"][0]) if results["documents"] else 0
                if n_found < self.min_crop_chunks:
                    print(f"[RAG] Only {n_found} crop-specific chunks — falling back to unfiltered")
                    results = None   # trigger fallback
            except Exception as e:
                print(f"[RAG] Crop-filtered query failed ({e}) — falling back to unfiltered")
                results = None

        # ── Unfiltered fallback ──────────────────────────────────────────────
        if results is None:
            results = self.collection.query(
                query_texts=[query],
                n_results=self.top_k,
                include=["documents", "metadatas", "distances"],
            )

        return self._parse_results(results)

    # ── Prompt builder ───────────────────────────────────────────────────────

    def build_rag_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into the RAG context block injected into the prompt.
        """
        if not chunks:
            return "No specific knowledge retrieved for this disease. Use general agronomy principles."

        lines = ["=== Retrieved Agricultural Knowledge ==="]
        for i, chunk in enumerate(chunks, 1):
            src  = chunk["metadata"].get("source", "unknown")
            crop = chunk["metadata"].get("crop", "")
            cond = chunk["metadata"].get("condition", "")
            label = f"[{i}] Source: {src}"
            if crop and crop != "Unknown":
                label += f" | Crop: {crop}"
            if cond and cond != "general":
                label += f" | Disease: {cond}"
            lines.append(label)
            lines.append(chunk["text"])
            lines.append("")

        return "\n".join(lines)

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_crop(predicted_class: Optional[str]) -> Optional[str]:
        if not predicted_class:
            return None
        return predicted_class.split("__")[0].replace("_", " ")

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
        docs  = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        for doc, meta, dist in zip(docs, metas, dists):
            chunks.append({"text": doc, "metadata": meta, "distance": dist})
        return chunks
