from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from PIL import Image

from .config import CONFIDENCE_GATE, BACKBONE_DEVICE


@dataclass
class CropQueryResult:
    """Full output of one CropQuery inference."""

    # Vision
    predictions: List[Tuple[str, float]] = field(default_factory=list)
    resnet_hint: Optional[str]           = None

    # Retrieved knowledge
    rag_chunks: list                     = field(default_factory=list)

    # Context (B1, B2, D1)
    weather_ctx: Optional[dict]          = None
    location_ctx: Optional[dict]         = None
    irrigation_advice: str               = ""

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
        lines = [" CropQuery Result "]
        for cls, conf in self.predictions:
            lines.append(f"  {'★' if conf >= CONFIDENCE_GATE else '·'} {cls}: {conf:.1f}%")
        lines.append(f"  Language: {self.target_language}")
        lines.append("─")
        return "\n".join(lines)


class CropQueryPipeline:
    """
    Full CropQuery inference pipeline.

    Parameters
    ----------
    model_path : str
        Path to ResNet50 weights.
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
        skip_llava: bool = False,
    ):
        from .vision          import CropVisionModel
        from .retriever       import CropRetriever
        from .translator      import IndicTranslator
        from .context_fetcher import ContextFetcher

        print("[Pipeline] Initialising CropQuery...")
        self.vision = CropVisionModel(device=BACKBONE_DEVICE)

        self.retriever       = CropRetriever()
        self.translator      = IndicTranslator()
        self.context_fetcher = ContextFetcher()   # B1, B2

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

        #  Step 1: ResNet50 ─
        result.predictions = self.vision.predict(image)
        result.resnet_hint = self.vision.gate(result.predictions, self.vision._last_tensor)

        if result.is_high_confidence:
            print(f"[Pipeline] High confidence: {result.top_class} ({result.top_confidence:.1f}%)")
        else:
            top = result.top_class or "Unknown"
            print(f"[Pipeline] Low confidence ({result.top_confidence:.1f}%) — LLaVA reasons from image alone")

        #  Step 1b: Context fetch (B1, B2) ─
        try:
            result.weather_ctx  = self.context_fetcher.get_weather()
            result.location_ctx = self.context_fetcher.get_location()
            print(f"[Pipeline] Context: {result.location_ctx.get('district', 'unknown')} | "
                  f"Temp: {result.weather_ctx.get('temp_c', 'N/A')}°C")
        except Exception as e:
            print(f"[Pipeline] Context fetch failed (non-fatal): {e}")

        #  Step 1c: Irrigation advisory (D1) ─
        try:
            result.irrigation_advice = self.context_fetcher.get_irrigation_advice(
                crop=result.top_class.split("__")[0].replace("_", " ") if result.top_class else None,
                weather_ctx=result.weather_ctx,
            )
        except Exception as e:
            print(f"[Pipeline] Irrigation advisory failed (non-fatal): {e}")

        #  Step 2: RAG retrieval ─
        result.rag_chunks = self.retriever.retrieve(
            predicted_class=result.top_class,
            farmer_question=farmer_question,
        )
        rag_context = self.retriever.build_rag_context(result.rag_chunks)
        print(f"[Pipeline] Retrieved {len(result.rag_chunks)} RAG chunks")

        #  Step 3: LLaVA 
        if self.advisor:
            result.english_advice = self.advisor.generate(
                image           = image,
                rag_context     = rag_context,
                farmer_question = farmer_question,
                resnet_hint     = result.resnet_hint,
                confidence      = result.top_confidence,
                weather_ctx     = result.weather_ctx,    # A2/B1
                location_ctx    = result.location_ctx,  # B2
            )
        else:
            # Skip-LLaVA mode — return a RAG-only text summary
            result.english_advice = (
                f"[RAG-only mode — LLaVA not loaded]\n\n"
                f"Top prediction: {result.top_class} ({result.top_confidence:.1f}%)\n\n"
                + rag_context
            )

        # Step 4: Translation
        if target_language != "English":
            from .advisor import _clean_jargon   # reuse the same map
            raw_translation = self.translator.translate(
                result.english_advice, target_language
            )
            # Clean any jargon that leaked through translation
            result.translated_advice = _clean_jargon(raw_translation)

        print(f"[Pipeline] Done. Advice generated in {target_language}.")
        return result
