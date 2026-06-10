import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = "/kaggle/working/chroma_db"

# ── Device assignment ──────────────────────────────────────────────
BACKBONE_DEVICE = "cuda:0"
LLAVA_DEVICE    = "cuda:1"

# ── Checkpoint paths ── single source of truth ─────────────────────
RUNS_DIR          = "/kaggle/input/datasets/vitthaljauhari2/cropquerydataset"
RESNET_CKPT       = os.path.join(RUNS_DIR, "resnet50_best.pth")
EFFICIENTNET_CKPT = os.path.join(RUNS_DIR, "efficientnet_best.pth")
DENSENET_CKPT     = os.path.join(RUNS_DIR, "densenet_best.pth")

# ── FAISS feature bank ─────────────────────────────────────────────
BANK_PREFIX   = os.path.join(RUNS_DIR, "feature_bank")
BANK_FEATURES = f"{BANK_PREFIX}_features.npy"
BANK_LABELS   = f"{BANK_PREFIX}_labels.npy"

# ── Class map ──────────────────────────────────────────────────────
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
NUM_CLASSES      = 64

# ── Vision / combiner ──────────────────────────────────────────────
IMG_SIZE             = 224
CONFIDENCE_GATE      = 80.0
FAISS_MIN_SIMILARITY = 0.65
KNN_K                = 7
TOP_K                = 3

# ── Embedding / RAG ────────────────────────────────────────────────
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "cropquery_knowledge"
TOP_CHUNKS      = 5
TOP_K_AGROBENCH = 3
TOP_K_KISAN     = 2

# ── LLaVA ─────────────────────────────────────────────────────────
LLAVA_CACHE_DIR  = "/kaggle/working/llava_cache" 
LLAVA_MODEL_ID   = LLAVA_CACHE_DIR       
LLAVA_MAX_TOKENS = 768
LOAD_IN_4BIT     = True

# ── Translation ────────────────────────────────────────────────────
DEFAULT_LANGUAGE = "Hindi"
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

# ── Weather / Location ─────────────────────────────────────────────
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_API_URL = "https://nominatim.openstreetmap.org/reverse"

# ── HuggingFace dataset ────────────────────────────────────────────
HF_AGROBENCH_DATASET = "vitthaljauhari/agrobench"  # single source of truth — matches Cell 5
HF_KISAN_DATASET     = "Franklin01/Dataset_prep01"
