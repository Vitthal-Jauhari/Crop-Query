"""
CropQuery — AI-Powered Crop Disease Detection & Farmer Advisory System

Architecture
------------
- Vision:       3-backbone ensemble (ResNet50 + EfficientNet-B3 + DenseNet121)
                + FAISS k-NN fallback for low-confidence cases
- Advisory:     LLaVA-1.6-Mistral-7B (4-bit quantised, cuda:1)
- Knowledge:    ChromaDB RAG — AgroBench entries + Kisan Call Centre Q&A
- Translation:  deep-translator (GoogleTranslator) — 11 Indian languages
- Context:      Open-Meteo weather + IP geolocation + irrigation advisory

Quick start
-----------
    from cropquery import CropQueryPipeline
    from PIL import Image

    pipeline = CropQueryPipeline()
    result   = pipeline.run(
        image           = Image.open("leaf.jpg"),
        farmer_question = "What should I spray?",
        target_language = "Hindi",
    )
    print(result.advice)
"""

from .pipeline        import CropQueryPipeline, CropQueryResult
from .context_fetcher import ContextFetcher
from .vision          import CropVisionModel
from .retriever       import CropRetriever
from .translator      import IndicTranslator
from .knowledge_base  import KnowledgeBaseBuilder
from .advisor         import LLaVAAdvisor
from . import config

__all__ = [
    "CropQueryPipeline",
    "CropQueryResult",
    "CropVisionModel",
    "CropRetriever",
    "IndicTranslator",
    "KnowledgeBaseBuilder",
    "LLaVAAdvisor",
    "config",
]
