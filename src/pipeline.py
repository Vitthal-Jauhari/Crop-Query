from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from PIL import Image
import re

from .config import CONFIDENCE_GATE, BACKBONE_DEVICE

_CHEMICAL_PRESERVE = [
    "Carbendazim 50 WP","Carbendazim","Mancozeb 75 WP","Mancozeb",
    "Copper Oxychloride","Copper Hydroxide","Metalaxyl","Propiconazole",
    "Tebuconazole","Chlorothalonil","Captan","Myclobutanil","Azoxystrobin",
    "Trifloxystrobin","Hexaconazole","Abamectin","Dimethoate","Quinalphos",
    "Imidacloprid","Thiamethoxam","Bordeaux mixture","Tricyclazole",
    "Isoprothiolane","Fosetyl-Al","Cymoxanil","Spiromesifen",
]
_CHEMICAL_PRESERVE.sort(key=len, reverse=True)


def _protect_chemicals(text: str):
    placeholders = {}
    for i, name in enumerate(_CHEMICAL_PRESERVE):
        key = f"CHEM_{i}"
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        if pattern.search(text):
            placeholders[key] = pattern.search(text).group(0)
            text = pattern.sub(key, text)
    return text, placeholders


def _restore_chemicals(text: str, placeholders: dict) -> str:
    for key, original in placeholders.items():
        text = text.replace(key, original)
    return text


@dataclass
class CropQueryResult:
    predictions:       List[Tuple[str, float]] = field(default_factory=list)
    resnet_hint:       Optional[str]           = None
    used_faiss:        bool                    = False
    faiss_similarity:  Optional[float]         = None
    rag_chunks:        list                    = field(default_factory=list)
    weather_ctx:       Optional[dict]          = None
    location_ctx:      Optional[dict]          = None
    irrigation_advice: str                     = ""
    english_advice:    str                     = ""
    translated_advice: str                     = ""
    target_language:   str                     = "English"

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
        return self.translated_advice or self.english_advice


class CropQueryPipeline:

    def __init__(self, skip_llava: bool = False):
        from .vision          import CropVisionModel
        from .retriever       import CropRetriever
        from .translator      import IndicTranslator
        from .context_fetcher import ContextFetcher

        print("[Pipeline] Initialising CropQuery...")
        self.vision          = CropVisionModel(device=BACKBONE_DEVICE)
        self.retriever       = CropRetriever()
        self.translator      = IndicTranslator()
        self.context_fetcher = ContextFetcher()
        self.advisor         = None

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
        result = CropQueryResult(target_language=target_language)

        result.predictions, img_tensor = self.vision.predict(image)
        result.resnet_hint, result.used_faiss, result.faiss_similarity = \
            self.vision.gate(result.predictions, img_tensor)

        try:
            result.weather_ctx  = self.context_fetcher.get_weather()
            result.location_ctx = self.context_fetcher.get_location()
        except Exception as e:
            print(f"[Pipeline] Context fetch failed (non-fatal): {e}")

        try:
            result.irrigation_advice = self.context_fetcher.get_irrigation_advice(
                crop=(result.top_class.split("__")[0].replace("_", " ")
                      if result.top_class else None),
                weather_ctx=result.weather_ctx,
            )
        except Exception as e:
            print(f"[Pipeline] Irrigation advisory failed (non-fatal): {e}")

        english_question = self.translator.to_english(farmer_question) if farmer_question else farmer_question

        result.rag_chunks = self.retriever.retrieve(
            predicted_class=result.top_class,
            farmer_question=english_question or farmer_question,
        )
        rag_context = self.retriever.build_rag_context(result.rag_chunks)

        if self.advisor:
            result.english_advice = self.advisor.generate(
                image=image, rag_context=rag_context,
                farmer_question=farmer_question,
                resnet_hint=result.resnet_hint,
                confidence=result.top_confidence,
                weather_ctx=result.weather_ctx,
                location_ctx=result.location_ctx,
            )
        else:
            result.english_advice = (
                f"[RAG-only mode]\n\nTop prediction: "
                f"{result.top_class} ({result.top_confidence:.1f}%)\n\n"
                + rag_context
            )

        from .advisor import _clean_jargon, _validate_advice

        result.english_advice = _validate_advice(
            result.english_advice, result.top_class
        )

        if target_language != "English":
            protected, placeholders = _protect_chemicals(result.english_advice)
            translated = self.translator.translate(protected, target_language)
            result.translated_advice = _clean_jargon(
                _restore_chemicals(translated, placeholders)
            )

        return result
