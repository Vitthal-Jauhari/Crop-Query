"""
CropQuery — Pipeline Orchestrator
Wires all components together into a single callable.

Flow
----
Image
  → ResNet50       → top-3 classes + confidence
  → Confidence gate → hint if conf ≥ 85%, else None
  → ChromaDB RAG   → top-5 relevant chunks
  → LLaVA-1.5-7B  → structured English advice
  → IndicTrans2    → regional language output
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from PIL import Image

from .config import CONFIDENCE_GATE


@dataclass
class CropQueryResult:
    """Full output of one CropQuery inference."""

    # Vision
    predictions: List[Tuple[str, float]] = field(default_factory=list)
    resnet_hint: Optional[str]           = None

    # Retrieved knowledge
    rag_chunks: list                     = field(default_factory=list)

    # Language output
    english_advice: str                  = ""
    translated_advice: str               = ""
    target_language: str                 = "English"

    # Convenience
    @property
    def top_class(self) -> Optional[str]:
        return self.predictions[0][0] if self.predictions else None

    @property
    def top_confidence(self) -> Optional[float]:
        return self.predictions[0][1] if self.predictions else None

    @property
    def is_high_confidence(self) -> bool:
        return bool(self.top_confidence and self.top_confidence >= CONFIDENCE_GATE)

    @property
    def advice(self) -> str:
        """Return translated advice if available, else English."""
        return self.translated_advice or self.english_advice

    def summary(self) -> str:
        lines = ["── CropQuery Result ──────────────────────────"]
        for cls, conf in self.predictions:
            lines.append(f"  {'★' if conf >= CONFIDENCE_GATE else '·'} {cls}: {conf:.1f}%")
        lines.append(f"  Language: {self.target_language}")
        lines.append("─────────────────────────────────────────────")
        return "\n".join(lines)


class CropQueryPipeline:
    """
    Full CropQuery inference pipeline.

    Parameters
    ----------
    model_path : str
        Path to phaseC_best.pth (ResNet50 weights).
    class_names_path : str
        Path to class_names.json (64-class list).
    skip_llava : bool
        If True, loads only ResNet50 + RAG (useful for testing without VRAM).

    Usage
    -----
    pipeline = CropQueryPipeline(model_path="model.pth", class_names_path="class_names.json")
    result   = pipeline.run(image=pil_img, farmer_question="...", target_language="Hindi")
    print(result.advice)
    """

    def __init__(
        self,
        model_path: str      = "model.pth",
        class_names_path: str = "class_names.json",
        skip_llava: bool     = False,
    ):
        from .vision     import CropVisionModel
        from .retriever  import CropRetriever
        from .translator import IndicTranslator

        print("[Pipeline] Initialising CropQuery...")
        self.vision     = CropVisionModel(model_path=model_path,
                                          class_names_path=class_names_path)
        self.retriever  = CropRetriever()
        self.translator = IndicTranslator()

        self.advisor = None
        if not skip_llava:
            from .advisor import LLaVAAdvisor
            self.advisor = LLaVAAdvisor()

        print("[Pipeline] Ready.")

    def run(
        self,
        image: Image.Image,
        farmer_question: str = "What should I do?",
        target_language: str = "Hindi",
    ) -> CropQueryResult:
        """
        Run full inference on a PIL image.

        Parameters
        ----------
        image : PIL.Image.Image
        farmer_question : str
            Free-form question from the farmer.
        target_language : str
            One of IndicTranslator.available_languages().

        Returns
        -------
        CropQueryResult
        """
        result = CropQueryResult(target_language=target_language)

        # ── Step 1: ResNet50 ─────────────────────────────────────────────────
        result.predictions = self.vision.predict(image)
        result.resnet_hint = self.vision.gate(result.predictions)

        if result.is_high_confidence:
            print(f"[Pipeline] High confidence: {result.top_class} ({result.top_confidence:.1f}%)")
        else:
            top = result.top_class or "Unknown"
            print(f"[Pipeline] Low confidence ({result.top_confidence:.1f}%) — LLaVA reasons from image alone")

        # ── Step 2: RAG retrieval ─────────────────────────────────────────────
        result.rag_chunks = self.retriever.retrieve(
            predicted_class=result.top_class,
            farmer_question=farmer_question,
        )
        rag_context = self.retriever.build_rag_context(result.rag_chunks)
        print(f"[Pipeline] Retrieved {len(result.rag_chunks)} RAG chunks")

        # ── Step 3: LLaVA ────────────────────────────────────────────────────
        if self.advisor:
            result.english_advice = self.advisor.generate(
                image           = image,
                rag_context     = rag_context,
                farmer_question = farmer_question,
                resnet_hint     = result.resnet_hint,
                confidence      = result.top_confidence,
            )
        else:
            # Skip-LLaVA mode — return a RAG-only text summary
            result.english_advice = (
                f"[RAG-only mode — LLaVA not loaded]\n\n"
                f"Top prediction: {result.top_class} ({result.top_confidence:.1f}%)\n\n"
                + rag_context
            )

        # ── Step 4: Translation ───────────────────────────────────────────────
        if target_language != "English":
            result.translated_advice = self.translator.translate(
                result.english_advice, target_language
            )

        print(f"[Pipeline] Done. Advice generated in {target_language}.")
        return result
