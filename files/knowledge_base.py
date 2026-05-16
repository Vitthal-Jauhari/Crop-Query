"""
CropQuery — Knowledge Base Builder
Ingests three data sources into ChromaDB with per-chunk metadata.

Run once to build the vector store; subsequent runs skip if collection
already has documents (use force_rebuild=True to overwrite).

Sources
-------
1. WheatRustVQA (Kaggle)             — image-text Q&A pairs (text only)
2. Kisan Call Centre (HuggingFace)   — real farmer–expert transcripts
3. AgroBench                         — encyclopaedic disease knowledge (built-in)

Usage (Kaggle notebook cell)
-----------------------------
    from cropquery.knowledge_base import KnowledgeBaseBuilder
    builder = KnowledgeBaseBuilder()
    builder.build()
"""

import os
import re
import json
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .config import (
    CHROMA_DIR, COLLECTION_NAME,
    EMBED_MODEL, HF_KISAN_DATASET,
)


# ── AgroBench knowledge (curated subset, always available) ───────────────────
# 30 high-value disease entries covering the 64 model classes.
# Expand this list with the full AgroBench dataset when available.
AGROBENCH_ENTRIES: List[Dict[str, str]] = [
    {
        "crop": "Tomato", "condition": "Late Blight",
        "text": (
            "Tomato Late Blight (Phytophthora infestans). "
            "Symptoms: Water-soaked lesions on leaves that turn brown with a pale green border; "
            "white sporulation on undersides; fruit shows greasy dark patches. "
            "Severity: High — can destroy an entire crop in 7–10 days under humid conditions. "
            "Organic control: Copper-based fungicide (Bordeaux mixture) spray every 7 days. "
            "Chemical control: Mancozeb 75 WP (2g/L) or Metalaxyl + Mancozeb. "
            "Prevention: Avoid overhead irrigation; maintain plant spacing for air circulation; "
            "destroy infected debris; use certified disease-free seeds."
        ),
    },
    {
        "crop": "Tomato", "condition": "Early Blight",
        "text": (
            "Tomato Early Blight (Alternaria solani). "
            "Symptoms: Dark brown spots with concentric rings (target-board pattern) on older leaves; "
            "yellowing around lesions; stem cankers at soil level. "
            "Severity: Moderate — progresses from bottom leaves upward. "
            "Chemical control: Chlorothalonil 75 WP (2g/L) or Mancozeb spray at 10-day intervals. "
            "Organic control: Neem oil (5 mL/L) spray. "
            "Prevention: Crop rotation (avoid Solanaceae for 2–3 years); remove infected leaves promptly; "
            "balanced nitrogen fertilisation (excess N promotes disease)."
        ),
    },
    {
        "crop": "Tomato", "condition": "Bacterial Spot",
        "text": (
            "Tomato Bacterial Spot (Xanthomonas campestris). "
            "Symptoms: Small, water-soaked spots on leaves, stems, and fruit; spots turn dark and "
            "scabby on fruit; defoliation under severe infection. "
            "Chemical control: Copper hydroxide (2g/L) applied preventively; do not apply during rain. "
            "Prevention: Use resistant varieties; avoid working in field when wet; drip irrigation preferred. "
            "Note: Antibiotics (streptomycin) are banned for vegetable use in India."
        ),
    },
    {
        "crop": "Tomato", "condition": "Leaf Mold",
        "text": (
            "Tomato Leaf Mold (Passalora fulva). "
            "Symptoms: Pale yellowish spots on upper leaf surface; olive-green to brown velvety mold "
            "on lower surface. Common in greenhouses and humid regions. "
            "Control: Improve ventilation; reduce humidity below 85%; "
            "Chlorothalonil or Mancozeb spray every 10 days. "
            "Resistant varieties available (Ty-1 gene lines)."
        ),
    },
    {
        "crop": "Potato", "condition": "Late Blight",
        "text": (
            "Potato Late Blight (Phytophthora infestans). "
            "Symptoms: Dark, water-soaked lesions on leaves; white fungal growth on leaf undersides; "
            "tubers show dry rot with reddish-brown discoloration. "
            "High risk when temperature is 10–25°C and humidity >90% for 2+ days. "
            "Chemical control: Metalaxyl + Mancozeb (Ridomil Gold) 2.5g/L; Cymoxanil + Mancozeb. "
            "Spray schedule: Every 7–10 days during risk periods. "
            "Harvest guidance: Stop irrigation 2 weeks before harvest; harvest on dry days."
        ),
    },
    {
        "crop": "Potato", "condition": "Early Blight",
        "text": (
            "Potato Early Blight (Alternaria solani). "
            "Symptoms: Circular dark spots with concentric rings on older leaves; premature defoliation. "
            "Most severe on stressed or nutrient-deficient plants. "
            "Control: Ensure adequate potassium; Mancozeb 75 WP 2g/L spray; Azoxystrobin. "
            "Prevention: Proper plant spacing; hilling to prevent soil splash."
        ),
    },
    {
        "crop": "Corn", "condition": "Common Rust",
        "text": (
            "Corn Common Rust (Puccinia sorghi). "
            "Symptoms: Small, circular to elongated cinnamon-brown pustules on both leaf surfaces; "
            "pustules rupture releasing rusty orange spores. "
            "Severity: Moderate in most seasons; severe in cool and humid conditions. "
            "Control: Propiconazole or Tebuconazole foliar spray at early infection. "
            "Planting: Use resistant hybrids; early planting avoids peak rust season."
        ),
    },
    {
        "crop": "Corn", "condition": "Gray Leaf Spot",
        "text": (
            "Corn Gray Leaf Spot (Cercospora zeae-maydis). "
            "Symptoms: Rectangular, tan-gray lesions with parallel edges running along leaf veins; "
            "lesions coalesce causing large dead areas. Severe in warm humid regions. "
            "Control: Strobilurin fungicides (Azoxystrobin); Triazoles. "
            "Agronomy: Crop rotation with non-host crops; tillage to bury residue reduces inoculum."
        ),
    },
    {
        "crop": "Corn", "condition": "Northern Leaf Blight",
        "text": (
            "Corn Northern Leaf Blight (Exserohilum turcicum). "
            "Symptoms: Long (5–15 cm) cigar-shaped lesions, tan with dark borders; "
            "lesions begin on lower leaves and progress upward; dark spores visible in lesions. "
            "Control: Mancozeb + Carbendazim spray; Propiconazole. "
            "Resistant varieties are the primary management strategy."
        ),
    },
    {
        "crop": "Wheat", "condition": "Leaf Rust",
        "text": (
            "Wheat Leaf Rust (Puccinia triticina). "
            "Symptoms: Small, round to oval orange-red pustules on upper leaf surface; "
            "spores orange-yellow. Spreads rapidly in warm (15–25°C), moist conditions. "
            "Chemical control: Propiconazole 25 EC (1 mL/L) at flag leaf stage; repeat after 15 days. "
            "Mancozeb as preventive. "
            "Importance: One of the most economically damaging wheat diseases in India. "
            "Genetic resistance: Use recommended varieties from ICAR for your agro-climatic zone."
        ),
    },
    {
        "crop": "Wheat", "condition": "Yellow Rust",
        "text": (
            "Wheat Yellow Rust / Stripe Rust (Puccinia striiformis). "
            "Symptoms: Yellow-orange pustules arranged in stripes along leaf veins; "
            "severe on flag leaf. Favoured by cool (10–15°C), moist conditions. "
            "Control: Propiconazole or Tebuconazole spray; apply when 5% plants show symptoms. "
            "India risk zones: North-west plains, hills. Monitor weather advisories from ICAR-IIWBR."
        ),
    },
    {
        "crop": "Rice", "condition": "Leaf Blast",
        "text": (
            "Rice Leaf Blast (Magnaporthe oryzae). "
            "Symptoms: Diamond-shaped lesions with grey centre and brown border on leaves; "
            "neck blast causes panicle collapse (most damaging). "
            "Favoured by excessive nitrogen, humidity >90%, temperature 24–28°C. "
            "Chemical control: Tricyclazole 75 WP (0.6g/L) or Isoprothiolane; apply at early tillering "
            "and again at panicle initiation. "
            "Organic: Pseudomonas fluorescens bioagent spray (10g/L). "
            "Avoid excess urea; use split nitrogen application."
        ),
    },
    {
        "crop": "Rice", "condition": "Brown Spot",
        "text": (
            "Rice Brown Spot (Cochliobolus miyabeanus). "
            "Symptoms: Oval to circular brown spots with yellow halo; severe on potassium-deficient soils. "
            "Often indicates nutritional stress. "
            "Management: Apply potassium (MOP) before planting; Mancozeb 75 WP 2g/L spray. "
            "Seed treatment with Thiram or Carbendazim (2g/kg seed)."
        ),
    },
    {
        "crop": "Grape", "condition": "Black Rot",
        "text": (
            "Grape Black Rot (Guignardia bidwellii). "
            "Symptoms: Tan-brown circular lesions on leaves with dark border; "
            "infected berries turn black and shrivel into mummified 'raisins'. "
            "Critical spray window: Bloom to 3 weeks post-bloom. "
            "Chemical control: Myclobutanil, Mancozeb, or Captan at 10-day intervals. "
            "Sanitation: Remove mummified berries and infected canes during dormant pruning — "
            "primary inoculum source."
        ),
    },
    {
        "crop": "Grape", "condition": "Black Measles",
        "text": (
            "Grape Black Measles / Esca (fungal complex). "
            "Symptoms: Interveinal chlorosis and necrosis in a 'tiger stripe' pattern; "
            "apoplexy (sudden vine collapse) in severe cases; "
            "dark streaking in wood when canes are cut. "
            "Management: No curative chemical treatment available. "
            "Preventive: Protect pruning wounds with fungicide or wound sealant immediately after cutting; "
            "avoid large pruning cuts. Infected vines may persist for years — mark and monitor."
        ),
    },
    {
        "crop": "Apple", "condition": "Apple Scab",
        "text": (
            "Apple Scab (Venturia inaequalis). "
            "Symptoms: Olive-green to brown scabby lesions on leaves and fruit; "
            "defoliation in severe cases; fruit lesions reduce marketability. "
            "Critical infection periods: Spring when leaves are expanding during wet weather. "
            "Control: Captan, Mancozeb, or Myclobutanil at 7–10 day intervals from green tip to post-petal fall. "
            "Scab models (e.g. RIMpro) help time sprays. "
            "Resistant varieties: Use scab-resistant cultivars where available."
        ),
    },
    {
        "crop": "Cashew", "condition": "Anthracnose",
        "text": (
            "Cashew Anthracnose (Colletotrichum gloeosporioides). "
            "Symptoms: Dark brown to black lesions on leaves, young shoots, and nuts; "
            "tip dieback; severe in humid coastal conditions. "
            "Chemical control: Carbendazim 50 WP (1g/L) or Copper oxychloride (3g/L); "
            "3 sprays at 15-day intervals from flushing stage. "
            "Organic: Pseudomonas fluorescens or Trichoderma viride soil drench. "
            "Pruning: Remove and burn affected branches."
        ),
    },
    {
        "crop": "Coffee", "condition": "Leaf Rust",
        "text": (
            "Coffee Leaf Rust (Hemileia vastatrix). "
            "Symptoms: Pale yellow spots on upper leaf surface; orange powdery spore masses "
            "on lower surface; severe defoliation reduces yield next season. "
            "Critical management: First spray at onset of monsoon; "
            "Copper oxychloride 50 WP (3g/L) or Hexaconazole 5 EC (1 mL/L). "
            "4 sprays at 45-day intervals covers the entire monsoon season. "
            "Shade management: Proper shade regulation improves airflow and reduces infection."
        ),
    },
    {
        "crop": "Sugarcane", "condition": "Red Rot",
        "text": (
            "Sugarcane Red Rot (Colletotrichum falcatum). "
            "Symptoms: Reddening of internal stalk tissues with white patches; "
            "leaves yellow from top; sour smell from infected stalks. "
            "Management: Plant disease-free setts from certified nurseries; "
            "treat setts with Carbendazim (1g/L) before planting. "
            "No effective foliar spray once infection is established — rogue out infected plants. "
            "Resistant varieties recommended by ICAR for your state."
        ),
    },
    {
        "crop": "Pea", "condition": "Powdery Mildew",
        "text": (
            "Pea Powdery Mildew (Erysiphe pisi). "
            "Symptoms: White powdery coating on leaves, stems, and pods; "
            "severely affected leaves turn yellow and dry. Late-season disease. "
            "Chemical control: Karathane (Dinocap) 1 mL/L or Wettable Sulphur 3g/L spray; "
            "repeat at 15-day intervals. "
            "Avoid spraying sulphur when temperature >35°C — phytotoxic. "
            "Resistant varieties: Bonneville, Meteor."
        ),
    },
    {
        "crop": "Pumpkin", "condition": "Downy Mildew",
        "text": (
            "Cucurbit Downy Mildew (Pseudoperonospora cubensis). "
            "Symptoms: Angular yellow spots on upper leaf surface limited by veins; "
            "grey-purple sporulation on lower surface; rapid defoliation. "
            "High risk: Humid nights + warm days. "
            "Chemical control: Cymoxanil + Mancozeb (2.5g/L) or Fosetyl-Al at 7-day intervals. "
            "Drip irrigation and mulching reduce humidity at canopy; "
            "avoid overhead irrigation especially in evenings."
        ),
    },
    {
        "crop": "Melon", "condition": "Nitrogen Deficiency",
        "text": (
            "Melon Nitrogen Deficiency. "
            "Symptoms: Pale yellow-green leaves starting from oldest (lower) leaves; "
            "yellowing uniform across entire leaf; stunted growth. "
            "Correction: Top-dress with urea (2%) foliar spray for quick response; "
            "soil application of Ammonium Sulphate or Urea at recommended dose. "
            "Diagnosis tip: Confirm with soil test before heavy N application — "
            "excess N promotes vegetative growth at expense of fruit set and increases disease susceptibility."
        ),
    },
    {
        "crop": "Tomato", "condition": "Spider Mites Two-spotted",
        "text": (
            "Tomato Two-spotted Spider Mite (Tetranychus urticae). "
            "Symptoms: Stippled, bronzed leaves; fine webbing on underside; "
            "severe: defoliation and plant death. Thrives in hot, dry weather. "
            "Miticide control: Abamectin (0.5 mL/L), Spiromesifen, or Fenpyroximate; "
            "rotate modes of action to prevent resistance. "
            "Organic: Neem oil (5 mL/L) + wetting agent; spray on undersides. "
            "Avoid broad-spectrum insecticides that kill natural predators (Phytoseiid mites)."
        ),
    },
    {
        "crop": "Corn", "condition": "Cercospora Leaf Spot",
        "text": (
            "Corn Cercospora Leaf Spot / Gray Leaf Spot (Cercospora zeae-maydis). "
            "Symptoms: Tan to grey rectangular lesions bounded by veins; "
            "lesions expand in humid weather and coalesce into large blighted areas. "
            "Control: Strobilurin + Triazole tank mix (Azoxystrobin + Propiconazole); "
            "apply at VT/R1 stage for maximum economic benefit. "
            "Hybrid selection: Use GLS-tolerant hybrids in endemic areas."
        ),
    },
    {
        "crop": "Potato", "condition": "Virus Y",
        "text": (
            "Potato Virus Y (PVY). "
            "Symptoms: Mosaic or mottling; necrotic spots or streaks on leaves; "
            "stunting; tuber necrosis in severe strains. "
            "Spread: Non-persistently transmitted by aphids (Myzus persicae). "
            "Management: No cure — plant certified seed potato; control aphid vectors with Imidacloprid "
            "or Thiamethoxam soil drench at planting; mineral oil sprays reduce aphid probing. "
            "Rogue out symptomatic plants early to reduce spread."
        ),
    },
    {
        "crop": "Strawberry", "condition": "Leaf Scorch",
        "text": (
            "Strawberry Leaf Scorch (Diplocarpon earlianum). "
            "Symptoms: Small, irregular dark purple spots on upper leaf surface; "
            "spots do not have a defined border (distinguishes from leaf spot); "
            "severe infection causes leaf margins to appear scorched and brown. "
            "Chemical control: Captan or Myclobutanil spray every 10 days after first signs. "
            "Cultural: Remove old leaves at renovation; improve air circulation; "
            "drip irrigation keeps foliage dry."
        ),
    },
    {
        "crop": "Apple", "condition": "Cedar Apple Rust",
        "text": (
            "Apple Cedar-Apple Rust (Gymnosporangium juniperi-virginianae). "
            "Symptoms: Bright orange-yellow spots on upper leaf surface; "
            "orange tube-like structures on lower surface. Requires alternate host (junipers/cedars). "
            "Chemical control: Myclobutanil or Mancozeb from pink bud through first cover sprays (May). "
            "Cultural: Remove nearby juniper hosts within 300m where feasible. "
            "Resistant apple varieties (Enterprise, Redfree) provide best long-term management."
        ),
    },
    {
        "crop": "Pepper", "condition": "Bacterial Spot",
        "text": (
            "Pepper Bacterial Spot (Xanthomonas euvesicatoria). "
            "Symptoms: Small water-soaked spots on leaves becoming brown with yellow halo; "
            "raised, scabby spots on green fruit; severe defoliation exposes fruit to sunscald. "
            "Chemical control: Copper hydroxide (2g/L) preventive sprays from transplanting; "
            "do not apply in hot weather (phytotoxic). "
            "Resistance: Copper resistance is common — rotate with Acibenzolar-S-methyl (SAR inducer). "
            "Use certified disease-free transplants."
        ),
    },
    {
        "crop": "Soybean", "condition": "Frogeye Leaf Spot",
        "text": (
            "Soybean Frogeye Leaf Spot (Cercospora sojina). "
            "Symptoms: Circular spots with brown border and grey-white centre (frog-eye appearance); "
            "seed infection causes bleached, shrivelled pods and reduced germination. "
            "Chemical control: Trifloxystrobin + Tebuconazole at R3 (beginning pod) stage. "
            "Resistance: Some QoI-resistant strains present — use alternate fungicide classes. "
            "Seed treatment: Thiram or Iprodione reduces seed-to-seedling transmission."
        ),
    },
    {
        "crop": "Tomato", "condition": "Mosaic Virus",
        "text": (
            "Tomato Mosaic Virus (ToMV). "
            "Symptoms: Light and dark green mosaic pattern on leaves; leaves may be distorted or fern-like; "
            "stunting; internal browning of fruit (chocolate fruit). "
            "Spread: Seed-borne and mechanically transmitted (tools, hands). NOT aphid-transmitted. "
            "Management: No cure — use resistant varieties (Tm-2 gene); "
            "disinfect tools with 10% bleach or milk solution between plants; "
            "wash hands after handling tobacco (ToMV source); do not transplant symptomatic seedlings."
        ),
    },
]


def _parse_kisan_dataset() -> List[Dict[str, Any]]:
    """
    Download and parse the Kisan Call Centre dataset from HuggingFace.
    Returns list of chunk dicts ready for ChromaDB.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[KB] 'datasets' library not installed — skipping Kisan Call Centre source")
        return []

    print("[KB] Loading Kisan Call Centre dataset from HuggingFace...")
    try:
        ds = load_dataset(HF_KISAN_DATASET, split="train", trust_remote_code=True)
    except Exception as e:
        print(f"[KB] Could not load Kisan dataset: {e}")
        return []

    chunks = []
    for i, row in enumerate(ds):
        # Dataset has varying column names — probe for common ones
        question = (
            row.get("question") or row.get("query") or
            row.get("Question") or row.get("input") or ""
        )
        answer = (
            row.get("answer") or row.get("response") or
            row.get("Answer") or row.get("output") or ""
        )
        crop = row.get("crop") or row.get("Crop") or "Unknown"

        if not question or not answer:
            continue

        text = f"Farmer Query: {question}\nExpert Answer: {answer}"
        chunks.append({
            "id":       f"kisan_{i}",
            "text":     text[:1500],   # cap at 1500 chars
            "metadata": {
                "source":     "kisan_call_centre",
                "crop":       str(crop),
                "condition":  "general",
                "chunk_type": "qa_pair",
                "language":   "en",
            },
        })
    print(f"[KB] Kisan Call Centre: {len(chunks)} Q&A pairs loaded")
    return chunks


def _parse_wheat_rust_vqa(vqa_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Parse WheatRustVQA text Q&A pairs.
    vqa_dir: path to downloaded dataset. If None, attempts Kaggle input path.
    """
    # Try common Kaggle paths
    search_dirs = [
        vqa_dir,
        "/kaggle/input/wheat-rust-vqa",
        "/kaggle/input/datasets/wheat-rust-vqa",
    ]
    qa_path = None
    for d in search_dirs:
        if d and os.path.isdir(d):
            for fname in ["qa_pairs.json", "annotations.json", "data.json", "train.json"]:
                candidate = os.path.join(d, fname)
                if os.path.exists(candidate):
                    qa_path = candidate
                    break
        if qa_path:
            break

    if not qa_path:
        print("[KB] WheatRustVQA not found — skipping. Download from Kaggle and pass vqa_dir.")
        return []

    with open(qa_path) as f:
        data = json.load(f)

    chunks = []
    items = data if isinstance(data, list) else data.get("annotations", data.get("data", []))
    for i, item in enumerate(items):
        q = item.get("question", "")
        a = item.get("answer", "")
        if not q or not a:
            continue
        text = f"Q: {q}\nA: {a}"
        chunks.append({
            "id":       f"wheat_vqa_{i}",
            "text":     text[:1000],
            "metadata": {
                "source":     "wheat_rust_vqa",
                "crop":       "Wheat",
                "condition":  "rust",
                "chunk_type": "qa_pair",
                "language":   "en",
            },
        })
    print(f"[KB] WheatRustVQA: {len(chunks)} Q&A pairs loaded")
    return chunks


def _build_agrobench_chunks() -> List[Dict[str, Any]]:
    """Convert built-in AgroBench entries to ChromaDB chunk format."""
    chunks = []
    for i, entry in enumerate(AGROBENCH_ENTRIES):
        chunks.append({
            "id":       f"agrobench_{i}",
            "text":     entry["text"],
            "metadata": {
                "source":     "agrobench",
                "crop":       entry["crop"],
                "condition":  entry["condition"],
                "chunk_type": "disease_entry",
                "language":   "en",
            },
        })
    print(f"[KB] AgroBench: {len(chunks)} disease entries loaded")
    return chunks


class KnowledgeBaseBuilder:
    """
    Builds and manages the ChromaDB vector store for CropQuery.

    Parameters
    ----------
    chroma_dir : str
        Directory to persist the ChromaDB collection.
    embed_model : str
        SentenceTransformer model ID.
    """

    def __init__(
        self,
        chroma_dir: str = CHROMA_DIR,
        embed_model: str = EMBED_MODEL,
    ):
        os.makedirs(chroma_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.ef = SentenceTransformerEmbeddingFunction(model_name=embed_model)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def build(
        self,
        force_rebuild: bool = False,
        vqa_dir: Optional[str] = None,
        batch_size: int = 64,
    ) -> int:
        """
        Build the knowledge base from all three sources.

        Parameters
        ----------
        force_rebuild : bool
            If True, delete the collection and rebuild from scratch.
        vqa_dir : str, optional
            Path to WheatRustVQA dataset directory.
        batch_size : int
            Upsert batch size for ChromaDB.

        Returns
        -------
        Total number of documents in collection after build.
        """
        if force_rebuild:
            print("[KB] Rebuilding from scratch...")
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.ef,
                metadata={"hnsw:space": "cosine"},
            )
        elif self.collection.count() > 0:
            print(f"[KB] Collection already has {self.collection.count()} docs. "
                  f"Pass force_rebuild=True to overwrite.")
            return self.collection.count()

        # Gather all chunks
        all_chunks: List[Dict[str, Any]] = []
        all_chunks.extend(_build_agrobench_chunks())
        all_chunks.extend(_parse_kisan_dataset())
        all_chunks.extend(_parse_wheat_rust_vqa(vqa_dir=vqa_dir))

        if not all_chunks:
            print("[KB] No chunks to ingest!")
            return 0

        # Deduplicate IDs
        seen_ids = set()
        unique_chunks = []
        for c in all_chunks:
            if c["id"] not in seen_ids:
                seen_ids.add(c["id"])
                unique_chunks.append(c)

        # Upsert in batches
        total = len(unique_chunks)
        print(f"[KB] Upserting {total} chunks into ChromaDB...")
        for start in range(0, total, batch_size):
            batch = unique_chunks[start : start + batch_size]
            self.collection.upsert(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )
        print(f"[KB] Done. Collection size: {self.collection.count()} documents")
        return self.collection.count()

    def stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        count = self.collection.count()
        sources: Dict[str, int] = {}
        # Sample up to 1000 to compute source breakdown
        sample = self.collection.get(limit=min(count, 1000), include=["metadatas"])
        for meta in sample["metadatas"]:
            src = meta.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        return {"total_documents": count, "source_breakdown": sources}
