"""
CropQuery — Kaggle Notebook Bootstrap
Copy each cell into a Kaggle notebook.

BEFORE RUNNING:
  1. Upload model.pth and class_names.json as a Kaggle dataset
  2. Add datasets: wheat-rust-vqa (if available), Franklin01/Dataset_prep01 (HF)
  3. Enable internet (for HuggingFace downloads)
  4. Accelerator: T4 x2
"""

# ╔══════════════════════════════════════════════════════╗
# ║  CELL 1 — Install dependencies                       ║
# ╚══════════════════════════════════════════════════════╝
INSTALL = """
pip install -q \
    transformers>=4.40 \
    accelerate \
    bitsandbytes \
    chromadb \
    sentence-transformers \
    datasets \
    gradio \
    torch torchvision \
    Pillow
"""


# ╔══════════════════════════════════════════════════════╗
# ║  CELL 2 — Set paths and verify                       ║
# ╚══════════════════════════════════════════════════════╝
CELL2 = """
import os, json

# ── Adjust these to match your Kaggle dataset slugs ──
MODEL_PATH       = "/kaggle/input/YOUR-MODEL-DATASET/model.pth"
CLASS_NAMES_PATH = "/kaggle/input/YOUR-MODEL-DATASET/class_names.json"

# Verify
assert os.path.exists(MODEL_PATH),       f"model.pth not found at {MODEL_PATH}"
assert os.path.exists(CLASS_NAMES_PATH), f"class_names.json not found"

with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)
print(f"✓ {len(class_names)} classes loaded")
print("Sample:", class_names[:5])
"""


# ╔══════════════════════════════════════════════════════╗
# ║  CELL 3 — Build knowledge base (run once)            ║
# ╚══════════════════════════════════════════════════════╝
CELL3 = """
import sys
sys.path.insert(0, "/kaggle/working")   # add CropQuery package to path

from cropquery.knowledge_base import KnowledgeBaseBuilder

builder = KnowledgeBaseBuilder(
    chroma_dir  = "/kaggle/working/chroma_db",
    embed_model = "sentence-transformers/all-MiniLM-L6-v2",
)
n = builder.build(
    force_rebuild = False,    # set True to rebuild from scratch
    vqa_dir       = "/kaggle/input/wheat-rust-vqa",   # optional
)
print(f"\\nKnowledge base: {n} documents")
print(builder.stats())
"""


# ╔══════════════════════════════════════════════════════╗
# ║  CELL 4 — Test ResNet50 inference                    ║
# ╚══════════════════════════════════════════════════════╝
CELL4 = """
from cropquery.vision import CropVisionModel
from PIL import Image

vm = CropVisionModel(
    model_path        = MODEL_PATH,
    class_names_path  = CLASS_NAMES_PATH,
)

# Test on a sample image
test_img = Image.open("/kaggle/input/SOME-TEST-IMAGE/test.jpg")
preds    = vm.predict(test_img)
hint     = vm.gate(preds)

print("Predictions:")
for cls, conf in preds:
    print(f"  {cls}: {conf:.1f}%")
print(f"Confidence gate: {'PASS →' + hint if hint else 'FAIL → LLaVA reasons alone'}")
"""


# ╔══════════════════════════════════════════════════════╗
# ║  CELL 5 — Test RAG retrieval                         ║
# ╚══════════════════════════════════════════════════════╝
CELL5 = """
from cropquery.retriever import CropRetriever

retriever = CropRetriever(chroma_dir="/kaggle/working/chroma_db")
chunks    = retriever.retrieve(
    predicted_class = "Tomato__Late_Blight",
    farmer_question = "What fungicide should I spray?",
)
print(f"Retrieved {len(chunks)} chunks\\n")
for i, c in enumerate(chunks, 1):
    print(f"[{i}] {c['metadata']['source']} | {c['metadata']['crop']}")
    print(c['text'][:200])
    print()
"""


# ╔══════════════════════════════════════════════════════╗
# ║  CELL 6 — Test full pipeline (no Gradio)             ║
# ╚══════════════════════════════════════════════════════╝
CELL6 = """
from cropquery import CropQueryPipeline
from PIL import Image

pipeline = CropQueryPipeline(
    model_path       = MODEL_PATH,
    class_names_path = CLASS_NAMES_PATH,
    skip_llava       = False,   # set True to test without VRAM usage
)

img    = Image.open("/kaggle/input/SOME-TEST-IMAGE/test.jpg")
result = pipeline.run(
    image           = img,
    farmer_question = "What spray should I use?",
    target_language = "Hindi",
)

print(result.summary())
print("\\n=== Advice (Hindi) ===")
print(result.advice)
"""


# ╔══════════════════════════════════════════════════════╗
# ║  CELL 7 — Launch Gradio demo                         ║
# ╚══════════════════════════════════════════════════════╝
CELL7 = """
# IMPORTANT: Update config.py paths before running
import cropquery.config as cfg
cfg.MODEL_PATH        = MODEL_PATH
cfg.CLASS_NAMES_PATH  = CLASS_NAMES_PATH
cfg.CHROMA_DIR        = "/kaggle/working/chroma_db"

# Launch
import subprocess
proc = subprocess.Popen(["python", "/kaggle/working/cropquery/app.py"])

# The public URL will appear in the output below
# Share it with your PRISM AI mentor for the demo
"""

print("Kaggle notebook cells ready.")
print("Copy each CELL_N string into a separate notebook cell.")
print()
print("CHECKLIST BEFORE FIRST RUN:")
print("  [ ] Add model.pth + class_names.json as a Kaggle dataset")
print("  [ ] Enable internet in notebook settings")
print("  [ ] Accelerator: T4 x2")
print("  [ ] Run CELL 3 (build KB) once, then skip it on subsequent runs")
