"""
CropQuery — AI-Powered Crop Disease Detection & Farmer Advisory System

Quick start
-----------
    from cropquery import CropQueryPipeline
    from PIL import Image

    pipeline = CropQueryPipeline(
        model_path       = "model.pth",
        class_names_path = "class_names.json",
    )
    result = pipeline.run(
        image            = Image.open("leaf.jpg"),
        farmer_question  = "What should I spray?",
        target_language  = "Hindi",
    )
    print(result.advice)

Submodules
----------
- cropquery.vision         — ResNet50 classifier
- cropquery.knowledge_base — ChromaDB builder (run once)
- cropquery.retriever      — RAG retrieval
- cropquery.advisor        — LLaVA-1.5-7B advice generation
- cropquery.translator     — IndicTrans2 multilingual output
- cropquery.pipeline       — full orchestrator
"""

from .pipeline        import CropQueryPipeline, CropQueryResult
from .context_fetcher import ContextFetcher
from .vision         import CropVisionModel
from .retriever      import CropRetriever
from .translator     import IndicTranslator
from .knowledge_base import KnowledgeBaseBuilder
from .advisor        import LLaVAAdvisor
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
