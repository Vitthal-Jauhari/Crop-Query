"""
CropQuery — Central Configuration
All paths, model IDs, thresholds, and language codes live here.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH        = os.path.join(BASE_DIR, "model.pth")          # phaseC_best.pth
CLASS_NAMES_PATH  = os.path.join(BASE_DIR, "class_names.json")   # list of 64 Crop__Condition labels
CHROMA_DIR        = os.path.join(BASE_DIR, "chroma_db")          # persisted vector store

# ── Vision model ─────────────────────────────────────────────────────────────
NUM_CLASSES       = 64
DROPOUT           = 0.4
IMG_SIZE          = 224
CONFIDENCE_GATE   = 85.0   # ResNet hint passed to LLaVA only when conf ≥ this
TOP_K             = 3      # top-k predictions returned by ResNet50

# ── Embedding / RAG ──────────────────────────────────────────────────────────
EMBED_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME   = "cropquery_knowledge"
TOP_CHUNKS        = 5
MIN_CROP_CHUNKS   = 3      # fallback to unfiltered retrieval below this

# ── LLaVA ────────────────────────────────────────────────────────────────────
LLAVA_MODEL_ID    = "llava-hf/llava-1.5-7b-hf"
LLAVA_MAX_TOKENS  = 512
LOAD_IN_4BIT      = True   # BitsAndBytesConfig — keeps it within T4 VRAM

# ── IndicTrans2 ───────────────────────────────────────────────────────────────
INDICTRANS_MODEL  = "ai4bharat/indictrans2-en-indic-1B"
DEFAULT_LANGUAGE  = "hin_Deva"   # Hindi

# Language display name → IndicTrans2 language code
LANGUAGE_CODES = {
    "Hindi":     "hin_Deva",
    "Marathi":   "mar_Deva",
    "Bengali":   "ben_Beng",
    "Telugu":    "tel_Telu",
    "Tamil":     "tam_Taml",
    "Kannada":   "kan_Knda",
    "Gujarati":  "guj_Gujr",
    "Punjabi":   "pan_Guru",
    "Odia":      "ory_Orya",
    "Malayalam": "mal_Mlym",
    "English":   None,           # no translation needed
}

# ── Data sources (for knowledge base builder) ─────────────────────────────────
KAGGLE_WHEAT_RUST_SLUG  = "ivanxavier/wheat-rust-vqa"           # WheatRustVQA
HF_KISAN_DATASET        = "Franklin01/Dataset_prep01"           # Kisan Call Centre
