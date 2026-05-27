# Crop-Query

A multilingual vision-language system for crop disease detection and farmer advisory.

## Overview
Crop-Query combines an ensemble vision classifier with RAG and LLaVA to produce structured advice, then translates it into regional languages. The Gradio UI lives in apps/app.py and the core package is in src/.

## Project Structure
- [apps/app.py](apps/app.py): Gradio demo entrypoint.
- [src/](src/): Core pipeline, models, RAG, translation, and context fetcher.
  - [src/pipeline.py](src/pipeline.py): Orchestrates vision -> RAG -> LLaVA -> translation.
  - [src/vision.py](src/vision.py): Ensemble vision model + FAISS fallback.
  - [src/retriever.py](src/retriever.py): ChromaDB RAG retrieval.
  - [src/advisor.py](src/advisor.py): LLaVA prompt + generation.
  - [src/translator.py](src/translator.py): Multilingual translation.
  - [src/context_fetcher.py](src/context_fetcher.py): Weather/location context + irrigation advice.
- [data/](data/): Metadata like class_names.json.
- [models/](models/): Model weights.
- [notebooks/](notebooks/): Experiments and demos.
- [notebooks/training/](notebooks/training/): Training notebooks.

## Quickstart
```bash
python apps/app.py
```

## Notes
- Default paths in [src/config.py](src/config.py) are Kaggle-oriented. Update paths for local use.
- If you only want the package, import from [src/](src/) instead of running the app.
