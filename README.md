# CropQuery

**A multilingual vision-language system for crop disease detection and farmer advisory.**

CropQuery diagnoses crop diseases from leaf photos and gives farmers plain-language, actionable treatment advice in 11 languages (10 Indian languages + English). Built as part of the PRISM AI Mentorship Program, targeting an IEEE conference paper with a longer-horizon journal extension.

📱 **Mobile app:** [CropQuery-Mobile](https://github.com/Vitthal-Jauhari/CropQuery-Mobile) — Expo/React Native frontend that talks to this backend.

---

## What it does

1. A farmer photographs a diseased crop leaf and asks a question (in any supported language)
2. A 3-backbone CNN ensemble classifies the disease (64 classes, 18 crops)
3. A RAG pipeline retrieves grounded agricultural knowledge (curated expert entries + real farmer call-center transcripts)
4. LLaVA-Next generates a plain-language diagnosis and treatment plan, safety-checked and translated back into the farmer's language

---

## Architecture

| Layer | Details |
|---|---|
| **Vision** | 3-backbone ensemble — ResNet50 + EfficientNet-B3 + DenseNet121 (`cuda:0`), softmax-averaged, 64 disease classes across 9 combined datasets |
| **Advisory LLM** | LLaVA-Next 7B, 4-bit NF4 quantized (`cuda:1`), two-stage inference: Stage 1 produces a visual description of the leaf, Stage 2 generates the advisory using that description + retrieved context |
| **Retrieval** | FAISS (4608-dim, `IndexFlatIP`, k=7) for visual similarity + ChromaDB RAG over 39 curated AgroBench entries and 178,815 real Kisan Call Centre farmer-advisor transcripts |
| **Translation** | Google Translate (`deep-translator`), with chemical/drug names protected from mistranslation before/after the translation call |
| **Serving** | Gradio UI for interactive use, FastAPI `/diagnose` + `/health` endpoints for the mobile app, exposed via ngrok during development |

---

## Results

**Vision (held-out test set, 19,984 samples):**

| Model | Top-1 | Top-5 | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| ResNet50 | 96.69% | — | — | — |
| EfficientNet-B3 | 95.66% | — | — | — |
| DenseNet121 | 96.66% | — | — | — |
| **Ensemble (F2, final)** | **97.24%** | 99.97% | **0.9493** | **0.9722** |
| Ensemble + FAISS (Phase D, paper headline) | **97.83%** | — | — | — |

> F2 consistency-loss training (`CE + λ·KL(pred_augA, pred_augB)`, λ=0.10) is the project's primary novel research contribution — see the paper for full details. FAISS override is confirmed net-negative after F2 training across a full threshold sweep and is disabled for F2 checkpoints; retained only for showing farmers visually similar reference images.

**Advisory (AgroBench evaluation, ROUGE-1):**

| System | ROUGE-1 |
|---|---|
| CropQuery (LLaVA-Next 4-bit + RAG) | 53.98% |
| GPT-4o baseline | 73.45% |

The gap is explained by model scale and by CropQuery deliberately favoring simpler, less jargon-heavy language for low-literacy farmers over verbose completeness.

---

## Repository structure

```
Crop-Query/
├── data/           # dataset references and preprocessing
├── models/         # trained checkpoints (see note below)
├── notebooks/       # training and evaluation notebooks (Kaggle-run)
├── src/            # pipeline, advisor, retriever, vision, translation code
├── web/            # Gradio UI
├── cropquery_system_architecture.pdf
└── README.md
```

Model checkpoints and FAISS feature banks are hosted as a Kaggle dataset (`vitthaljauhari2/cropquerydataset`) rather than in this repo, since training and inference both run inside Kaggle notebooks (T4×2 GPUs).

---

## Running it

Training and inference are designed to run inside Kaggle notebooks with dual T4 GPUs:

1. Open `notebooks/cropquery-main.ipynb` on Kaggle, attach the `vitthaljauhari2/cropquerydataset` dataset
2. Run cells sequentially — the notebook builds the vision ensemble, RAG index, and advisory pipeline
3. The FastAPI + ngrok cells at the end expose a public `/diagnose` and `/health` endpoint for the mobile app to hit
4. Point [CropQuery-Mobile](https://github.com/Vitthal-Jauhari/CropQuery-Mobile)'s `API_URL` at the printed ngrok URL

For evaluation only, use `notebooks/cropquery-eval.ipynb`.

---

## Status

- ✅ Vision ensemble, RAG pipeline, LLaVA advisory, translation layer, Gradio UI — feature-complete
- ✅ Mobile app (CropQuery-Mobile) — working end-to-end against the FastAPI backend
- ✅ IEEE paper — feature-complete, pending resolution of a few dataset-count/split reconciliation items before submission
- 🟡 A few safety-critical translation edge cases still being hardened (chemical-name protection in two remaining code paths)
- 🟡 Eval notebook reproducibility — being restored (currently 0/7 items independently re-run and confirmed)

Full task-level tracking lives in the project's Notion workspace.

---

## Acknowledgments

Built under the PRISM AI Mentorship Program. Uses AgroBench for expert-curated evaluation data and the Kisan Call Centre dataset for real-world farmer advisory grounding.
