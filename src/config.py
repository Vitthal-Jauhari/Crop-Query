import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = "/kaggle/working/chroma_db"

# ── Device assignment ──────────────────────────────────────────────────────
BACKBONE_DEVICE = "cuda:0"
LLAVA_DEVICE    = "cuda:1"

# ── Ensemble checkpoint paths ──────────────────────────────────────────────
RUNS_DIR          = "/kaggle/input/datasets/vitthaljauhari2/cropquerydataset"
RESNET_CKPT       = os.path.join(RUNS_DIR, "resnet50_best.pth")
EFFICIENTNET_CKPT = os.path.join(RUNS_DIR, "efficientnet_best.pth")
DENSENET_CKPT     = os.path.join(RUNS_DIR, "densenet_best.pth")

# ── FAISS feature bank ─────────────────────────────────────────────────────
BANK_PREFIX   = "/kaggle/input/datasets/vitthaljauhari2/cropquerydataset/feature_bank"
BANK_FEATURES = f"{BANK_PREFIX}_features.npy"
BANK_LABELS   = f"{BANK_PREFIX}_labels.npy"

# ── Class map ──────────────────────────────────────────────────────────────
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
NUM_CLASSES      = 64

# ── Vision / combiner ──────────────────────────────────────────────────────
IMG_SIZE        = 224
CONFIDENCE_GATE = 80.0
KNN_K           = 7
TOP_K           = 3

# ── Embedding / RAG ────────────────────────────────────────────────────────
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME  = "cropquery_knowledge"
TOP_CHUNKS       = 5
MIN_CROP_CHUNKS  = 2
TOP_K_AGROBENCH  = 3
TOP_K_KISAN      = 2

# ── LLaVA ─────────────────────────────────────────────────────────────────
LLAVA_MODEL_ID   = "llava-hf/llava-v1.6-mistral-7b-hf"
LLAVA_MAX_TOKENS = 768
LOAD_IN_4BIT     = True

# ── IndicTrans2 ────────────────────────────────────────────────────────────
INDICTRANS_MODEL = "ai4bharat/indictrans2-en-indic-1B"
DEFAULT_LANGUAGE = "hin_Deva"
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
    "English":   None,
}

# ── Weather / Location ─────────────────────────────────────────────────────
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_API_URL = "https://nominatim.openstreetmap.org/reverse"

# ── HuggingFace dataset ────────────────────────────────────────────────────
HF_KISAN_DATASET = "Franklin01/Dataset_prep01"
