"""
CropQuery — LLaVA Advisor

Dual-stream inference:
  Stream A  →  3-backbone ensemble (cuda:0)  →  class label + confidence
  Stream B  →  LLaVA vision encoder (cuda:1) →  visual symptom description

Stream B runs in two stages:
  Stage 1: short visual description prompt  (~80 tokens) — fast
  Stage 2: full advisory generation prompt  (~768 tokens)

Stage 1 output feeds into Stage 2 prompt as `visual_symptoms`,
replacing the old text-only "Predicted class: ..." placeholder.
This is what justifies using LLaVA instead of plain LLaMA.
"""
import re
from typing import Optional, Generator, List, Tuple
from PIL import Image
from transformers import TextIteratorStreamer
from threading import Thread

from .config import (
    LLAVA_MODEL_ID, LLAVA_MAX_TOKENS, LOAD_IN_4BIT,
    CONFIDENCE_GATE, LLAVA_DEVICE,
)


# ── Spray timing helper ────────────────────────────────────────────────────────

def _spray_timing_warning(weather_ctx: dict) -> str:
    if not weather_ctx:
        return ""
    from .context_fetcher import spray_timing_advisory
    return spray_timing_advisory(weather_ctx)


# ── Jargon simplifier ─────────────────────────────────────────────────────────

_JARGON_MAP = {
    "new flush":           "new leaves",
    "flush emergence":     "new leaves appearing",
    "flush":               "new leaves",
    "inoculum":            "disease source",
    "sporulation":         "disease spreading",
    "lesions":             "spots",
    "necrosis":            "dead patches",
    "necrotic":            "dead",
    "vector":              "carrier insect",
    "infestation":         "attack",
    "larval mining":       "pest damage inside leaves",
    "oviposition":         "egg-laying",
    "instar":              "stage of pest",
    "systemic":            "absorbed by the plant",
    "contact insecticide": "spray that kills on contact",
    "foliar application":  "leaf spray",
    "phytotoxicity":       "plant damage from chemicals",
    "stippling":           "tiny spots",
    "chlorosis":           "yellowing",
    "defoliation":         "leaf fall",
}


def _clean_jargon(text: str) -> str:
    for jargon, simple in _JARGON_MAP.items():
        text = re.sub(re.escape(jargon), simple, text, flags=re.IGNORECASE)
    return text


# ── Safety validator ───────────────────────────────────────────────────────────

_FUNGICIDE_KEYWORDS = {
    "carbendazim", "mancozeb", "copper oxychloride", "chlorothalonil",
    "metalaxyl", "propiconazole", "tebuconazole", "captan", "myclobutanil",
    "trifloxystrobin", "azoxystrobin", "hexaconazole", "bordeaux",
    "copper hydroxide", "fosetyl", "cymoxanil",
}

_PEST_KEYWORDS = {
    "leaf miner", "spider mite", "aphid", "whitefly", "mealybug",
    "thrips", "scale", "weevil", "borer", "caterpillar", "locust",
    "fruit fly", "jassid", "leafhopper",
}


def _validate_advice(advice: str, predicted_class: Optional[str]) -> str:
    if not predicted_class:
        return advice
    condition = predicted_class.split("__", 1)[-1].replace("_", " ").lower()
    is_pest = any(kw in condition for kw in _PEST_KEYWORDS)
    if not is_pest:
        return advice
    advice_lower = advice.lower()
    flagged = [kw for kw in _FUNGICIDE_KEYWORDS if kw in advice_lower]
    if flagged:
        warning = (
            "\n\n⚠️ [Safety Check] This appears to be a PEST problem, not a fungal disease. "
            f"The following chemicals listed above are fungicides and will NOT work against this pest: "
            f"{', '.join(flagged).title()}. "
            "Please consult your local agriculture officer for the correct insecticide."
        )
        print(f"[Validator] PEST/FUNGICIDE MISMATCH detected: {flagged}")
        return advice + warning
    return advice


# ── Closing note guarantee ─────────────────────────────────────────────────────

_CLOSING_NOTE = (
    "\n\nKeep watching your crop over the next 7-10 days. "
    "If the problem does not improve after treatment, contact your local "
    "agricultural extension officer."
)


def _ensure_closing_note(advice: str) -> str:
    """
    Guarantee a consistent, translator-safe closing line regardless of whether the
    model remembered to write one. This is deliberately injected in code rather
    than left to the LLM: a fixed short sentence translates reliably and
    identically across languages, whereas trusting each independent generation to
    both write it AND phrase it the same way is exactly what produced the
    cross-language inconsistency seen in review (English/Hindi/Bengali outputs
    ending differently, some omitting a conclusion entirely).
    """
    advice_lower = advice.lower()
    if "extension" in advice_lower or "keep watching" in advice_lower:
        return advice
    return advice.rstrip() + _CLOSING_NOTE


# ── Intent classifier ─────────────────────────────────────────────────────────

# Maps farmer question keywords → intent label
# Intent label drives which sections the model must include vs. skip.
_INTENT_KEYWORDS = {
    "curable":      ["can this be cured", "is it curable", "will it recover", "ठीक हो सकता", "ठीक होगा", "ठीक हो जाएगा"],
    "treatment":    ["spray", "chemical", "medicine", "treat", "fungicide", "pesticide",
                     "दवाई", "दवा", "स्प्रे", "इलाज"],
    "fertilizer":   ["fertilizer", "fertiliser", "urea", "npk", "nutrient", "खाद", "उर्वरक"],
    "irrigation":   ["water", "irrigat", "drip", "flood", "सिंचाई", "पानी"],
    "severity":     ["how bad", "serious", "damage", "loss", "कितना नुकसान", "कितना खराब", "गंभीर"],
    "prevention":   ["prevent", "avoid", "protect", "stop from", "रोकें", "बचाव"],
    # NOTE: "diagnosis" intent bucket removed (was: "what is", "what happened",
    # "identify", "disease", "problem"). These phrases are too generic — they are
    # the MOST common way a farmer actually asks a question ("What happened to
    # my crop?"), and the old "diagnosis" intent's required-sections list did
    # NOT include "treatment". That meant the single most common farmer question
    # silently produced a diagnosis-only answer with no treatment advice.
    # These questions now fall through to the "general" intent bucket below,
    # whose required sections are ["visual_reasoning", "diagnosis", "treatment"]
    # — a strict superset of what "diagnosis" used to require.
    "spray_timing": ["when to spray", "best time to spray", "कब स्प्रे", "सही समय"],
    "yield_loss":   ["yield", "production", "harvest", "crop loss", "उत्पादन", "पैदावार"],
    "cost":         ["cost", "price", "cheap", "expensive", "कितने का", "सस्ता", "महंगा"],
}

# Which sections are REQUIRED for each intent vs. always optional
_INTENT_SECTION_RULES = {
    "curable": {
        "lead": "Answer YES or NO immediately. Give a 1-line reason. Then state what treatment is needed.",
        "required": ["curable_answer", "treatment"],
        "skip": ["severity_detail", "next_7_days", "weather"],
    },
    "treatment": {
        "lead": "Lead with the specific chemical or organic treatment. State dosage and frequency.",
        "required": ["treatment"],
        "skip": ["weather"],
    },
    "fertilizer": {
        "lead": "Explain clearly that fertilizer does NOT cure this problem. Then state what actually will.",
        "required": ["fertilizer_note", "treatment"],
        "skip": ["severity_detail", "next_7_days", "weather"],
    },
    "irrigation": {
        "lead": "Answer the irrigation question directly. Give a water amount or schedule.",
        "required": ["irrigation"],
        "skip": ["treatment", "next_7_days"],
    },
    "severity": {
        "lead": "State severity using the visible leaf damage % estimate (not confidence). Explain what damage you can see in the image.",
        "required": ["severity_detail", "visual_reasoning"],
        "skip": ["treatment_detail", "weather"],
    },
    "prevention": {
        "lead": "Lead with 2-3 concrete prevention steps. Reference what the image shows.",
        "required": ["prevention"],
        "skip": ["severity_detail"],
    },
    "spray_timing": {
        "lead": "Give the best spray timing first. Mention weather only if it affects timing.",
        "required": ["spray_timing", "weather"],
        "skip": ["severity_detail", "next_7_days"],
    },
    "yield_loss": {
        "lead": "State expected yield impact clearly. Tie it to the severity visible in the image.",
        "required": ["severity_detail", "yield_impact"],
        "skip": ["weather"],
    },
    "cost": {
        "lead": "Give approximate cost of recommended treatment. Mention cheaper alternatives if any.",
        "required": ["treatment", "cost_note"],
        "skip": ["severity_detail", "next_7_days", "weather"],
    },
    "general": {
        "lead": "Describe what you see, name the problem, briefly explain what causes it, and give the most useful advice for this situation.",
        "required": ["visual_reasoning", "diagnosis", "cause_explanation", "treatment"],
        "skip": [],
    },
}


def _classify_intent(question: str) -> str:
    """
    Classify farmer's question into one of the intent labels.
    Returns 'general' if no specific intent is detected (this is also the
    fallback for generic/ambiguous questions like "What happened to my crop?"
    or an empty question — see note above _INTENT_KEYWORDS).
    """
    if not question or not question.strip():
        return "general"
    q_lower = question.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return intent
    return "general"


# ── Non-leaf detection ────────────────────────────────────────────────────────

_NO_LEAF_PHRASES = [
    "no plant", "no leaf", "not a leaf", "no crop", "no vegetation",
    "cannot identify", "not a plant", "no disease visible",
    "no agricultural", "unclear image", "no leaves",
]

_NO_LEAF_RESPONSE = """
* No plant or leaf was detected in this image.
* Please upload a clear photo of a leaf or crop.

🌿 Problem
* Image does not show a plant leaf.
* Cannot diagnose without a clear leaf photo.

✅ What to do now
* Take a close-up photo of the affected leaf in natural daylight.
* Make sure the leaf fills most of the frame.
* Avoid blurry or very dark photos.

⚠️ Safety Notice
* Always confirm chemical use with your local agriculture officer before applying."""


def _is_no_leaf(visual_description: str) -> bool:
    desc_lower = visual_description.lower()
    return any(phrase in desc_lower for phrase in _NO_LEAF_PHRASES)


# ── Healthy-leaf detection (Stage 1 second layer) ─────────────────────────────
# The primary healthy short-circuit lives in pipeline.py, keyed off the ensemble's
# high-confidence "Healthy" class prediction. This is a second, independent layer:
# even if the ensemble misses it, LLaVA's own Stage 1 read can still catch a
# healthy leaf and skip the full advisory generation — same shape, same reasoning
# as _is_no_leaf above.

_HEALTHY_PHRASES = ["appears healthy", "no visible symptoms", "looks healthy"]

_HEALTHY_RESPONSE = """✅ Good news — your crop looks healthy.

🌿 What we found
* No signs of disease or pest damage detected in the image.
* Leaf colour, pattern, and texture look normal.

📋 What to do
* Continue your regular care routine (watering, fertilizing).
* Recheck if you notice any yellowing, spots, or wilting in the coming days.

⚠️ Safety Notice
* No treatment is needed right now. Avoid spraying chemicals on a healthy plant."""


def _is_healthy_leaf(visual_description: str) -> bool:
    desc_lower = visual_description.lower()
    return any(phrase in desc_lower for phrase in _HEALTHY_PHRASES)


# ── Visual description prompt (Stage 1) ───────────────────────────────────────

_VISUAL_DESCRIPTION_PROMPT = """Look at this image carefully.

If there is NO plant or leaf visible in this image, reply with exactly:
"No plant or leaf visible in this image."

If the leaf looks healthy — no discoloration, spots, holes, or damage —
reply with exactly:
"Leaf appears healthy, no visible symptoms."

Otherwise, describe ONLY what you actually see on the leaf in 3-4 short bullets:
- Colour of affected areas, if any
- Location (edges, centre, whole leaf, underside, veins)
- Spread (small patches, large areas, most of leaf)
- Pattern: vein-limited/angular or diffuse/irregular
- Any visible pests, webbing, or eggs — only mention if actually visible

Then, on its own final line:
AFFECTED AREA: <range>%
(or "AFFECTED AREA: unclear" if you cannot tell, or "AFFECTED AREA: 0%" if healthy)

Keep your description under 90 words. Do not diagnose. Do not recommend treatment."""


_AFFECTED_AREA_RE = re.compile(r"AFFECTED AREA:\s*([^\n]+)", re.IGNORECASE)


def _extract_visual_severity(visual_description: str) -> str:
    """
    Pull the Stage 1 'AFFECTED AREA: ...' line out of the visual description.
    This is deliberately a SEPARATE signal from ensemble prediction confidence:
    confidence is how sure the vision model is about WHICH disease this is;
    this is how much of the leaf visibly looks damaged. The two must never be
    conflated in the advisory text — see STRICT RULE 12 in _SYSTEM_PROMPT.
    """
    m = _AFFECTED_AREA_RE.search(visual_description)
    if not m:
        return "Not estimated"
    val = m.group(1).strip().rstrip(". ")
    return val if val else "Not estimated"


# ── Main advisory prompt (Stage 2) ────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a senior agricultural extension officer with decades of field experience across India.
You speak like someone who has physically inspected this crop — calm, honest, and direct.
You do not sound like a form or a template. You sound like a person who knows farming.
{output_language_instruction}

========================
YOUR INPUTS
========================
What you can see in the image: {visual_symptoms}
Visible leaf damage — % of leaf area affected (Stage 1 estimate): {visual_severity}
Knowledge from disease database: {retrieved_context}
Farmer's Question: {user_query}
Diagnosis: {diagnosis_line}
Image Confidence: {confidence_display}
Other possibilities considered (ranked lower): {runner_up_classes}

========================
INTENT: {intent_label}
========================
{intent_instruction}

========================
GROUNDING — YOUR SINGLE MOST IMPORTANT RULE
========================
Use ONLY the structured inputs above. Never invent, infer, or assume anything not
explicitly stated in them:
- If a CONFIRMED DIAGNOSIS is given, your first sentence must state it plainly
  ("The vision model detected {diagnosis_line} with {confidence_display} confidence.")
  — not "I see..." or "it appears...".
- Describe only what "What you can see in the image" actually says. If it doesn't
  mention the underside, pests, or eggs, don't claim you checked or confirmed them —
  just omit them.
- This is one still photo. Never say the disease is "spreading" or "getting worse" —
  describe only the extent visible right now; frame urgency as forward-looking risk.
- "Image Confidence" and "Visible leaf damage" are different things — confidence is
  which disease; visible damage is how much of the leaf looks affected. State the
  damage percentage if it's a real estimate; if it's "Not estimated"/"unclear", say so
  plainly rather than inventing a number.
- If "Other possibilities" lists alternatives, add one short sentence on why you
  favoured the main diagnosis over them.
- Name the causal organism/biology ONLY if it appears in retrieved_context. Otherwise
  describe the cause generically but usefully (e.g. "a fungal disease that spreads
  through wind and splashing water in humid weather"), never a bare word like
  "pathogen" with nothing actionable attached.

========================
CONFIDENCE LANGUAGE
========================
Match your phrasing to the confidence level — don't hedge on strong evidence, don't
overstate weak evidence:
- At/above {confidence_gate}%: "This is clearly..." / "has been identified as..."
- 65–{confidence_gate}%: "This most likely is..." / "The symptoms suggest..."
- Below 65%: "The image is not clear enough for a confident diagnosis..." and mention
  alternative possibilities; ask for a better image before giving strong advice.

========================
TREATMENT RULES
========================
- Recommend ONLY chemicals/treatments found in retrieved_context. Copy chemical names
  and dosages character-for-character — never invent, round, or restate them
  differently anywhere in your answer.
- Before or alongside the chemical recommendation, mention 1-2 relevant cultural
  practices when applicable (removing infected leaves, improving airflow, avoiding
  overhead irrigation) — these aren't restricted to RAG-only.
- If retrieved_context gives a repeat-spray schedule, add one short sentence on why
  repeat treatment matters, grounded only in retrieved_context or visible symptoms.
- If weather context is provided and relevant, fold it into your causal reasoning as
  one connected statement (e.g. "the heavy rain has kept humidity high, which is
  exactly what this disease needs to spread") — don't restate weather as a separate
  afterthought.
- If listing 4+ treatment/action items, split under two short labels ("Immediate
  treatment" / "Ongoing care"); for 3 or fewer, a flat list is clearer.
- Never say "definitely", "100% cure", "certainly cured". Always end with the safety
  notice if chemicals are mentioned.
- If your answer covers chemical treatment, end with one short sentence pointing to
  the separate irrigation panel below — don't invent irrigation numbers yourself.
- Use simple, everyday words — avoid technical jargon.

========================
RESPONSE STRATEGY
========================
1. Answer what the farmer is specifically asking, first.
2. Order: what you can see → what it means → what to do about it. Don't blend these.
3. Use short paragraphs by default; switch to bullets only for 3+ actionable items.
4. Include only sections relevant to this question — don't pad.
5. Give each major section a short emoji + label (e.g. "🔍 What I see", "💊
   Treatment", "📅 Follow-up") so a farmer can scan quickly on a phone. One emoji
   per header only — never inside sentences or bullets.
6. Unless the question is narrow enough that it would feel like padding, close with
   one short sentence on continued monitoring.
7. Never write a disclaimer like "based on the information available in the
   agricultural knowledge base" — cite naturally or don't cite at all.
"""


def _build_confidence_display(confidence: Optional[float]) -> str:
    if confidence is None:
        return "Unknown"
    if confidence >= CONFIDENCE_GATE:
        return f"{confidence:.1f}% — High (strong symptom match)"
    elif confidence >= 65:
        return f"{confidence:.1f}% — Medium (likely but verify)"
    else:
        return f"{confidence:.1f}% — Low (image may be unclear or ambiguous)"


def _build_diagnosis_line(resnet_hint: Optional[str], confidence: Optional[float]) -> str:
    if not resnet_hint:
        return "Not yet determined"
    crop, condition = (
        resnet_hint.split("__", 1) if "__" in resnet_hint
        else (resnet_hint, "Unknown")
    )
    crop      = crop.replace("_", " ").title()
    condition = condition.replace("_", " ").title()
    conf_str  = f"{confidence:.1f}%" if confidence else "?"
    return f"{condition} in {crop} ({conf_str} confidence)"


def _build_runner_up_block(
    predictions: Optional[List[Tuple[str, float]]],
    resnet_hint: Optional[str],
) -> str:
    """
    Format the 2nd/3rd-place ensemble predictions so the model can reason about
    why it favoured the top diagnosis over close alternatives (differential
    reasoning). Returns a short human-readable string, never raw class labels.
    """
    if not predictions or len(predictions) < 2:
        return "None — no close alternatives above the reporting threshold."
    lines = []
    for cls, conf in predictions[1:3]:
        if cls == resnet_hint:
            continue
        _, condition = (
            cls.split("__", 1) if "__" in cls else (cls, "Unknown")
        )
        condition = condition.replace("_", " ").title()
        lines.append(f"{condition} ({conf:.1f}%)")
    return ", ".join(lines) if lines else "None — no close alternatives above the reporting threshold."


def _build_prompt(
    rag_context: str,
    farmer_question: str,
    resnet_hint: Optional[str],
    confidence: Optional[float],
    visual_description: str,
    weather_ctx: Optional[dict] = None,
    location_ctx: Optional[dict] = None,
    predictions: Optional[List[Tuple[str, float]]] = None,
) -> str:
    # ── Classify intent ───────────────────────────────────────────────
    q = farmer_question.strip() if farmer_question else "What should I do about this?"
    intent = _classify_intent(q)
    intent_rules = _INTENT_SECTION_RULES.get(intent, _INTENT_SECTION_RULES["general"])

    # ── Build context block ───────────────────────────────────────────
    parts = []

    # Enforce diagnosis context only when confidence is high enough
    if resnet_hint and confidence and confidence >= CONFIDENCE_GATE:
        crop, condition = (
            resnet_hint.split("__", 1) if "__" in resnet_hint
            else (resnet_hint, "Unknown")
        )
        crop      = crop.replace("_", " ").title()
        condition = condition.replace("_", " ").title()
        parts.append(
            f"CONFIRMED DIAGNOSIS: Vision model identified this as "
            f"'{condition}' in '{crop}' with {confidence:.1f}% confidence.\n"
            f"Your ENTIRE response must be about '{condition}' in '{crop}'. "
            f"Do NOT discuss any other disease or pest."
        )
        parts.append("")

    if location_ctx:
        loc_parts = []
        for key, label in [
            ("state", "State"), ("district", "District"),
            ("agro_climatic_zone", "Zone"), ("current_season", "Season"),
        ]:
            val = location_ctx.get(key)
            if val:
                loc_parts.append(f"{label}: {val}")
        if loc_parts:
            parts.append("=== Farmer Location ===")
            parts.append(" | ".join(loc_parts))
            parts.append("")

    # Weather: only inject if relevant to intent
    weather_intents = {"spray_timing", "irrigation", "general", "treatment", "prevention"}
    if weather_ctx and intent in weather_intents:
        w = weather_ctx
        weather_lines = ["=== Current Weather ==="]
        if w.get("temp_c") is not None:
            weather_lines.append(
                f"Temp: {w['temp_c']}°C  |  Humidity: {w.get('humidity_pct', 'N/A')}%"
            )
        if w.get("wind_kmh") is not None:
            weather_lines.append(
                f"Wind: {w['wind_kmh']} km/h  |  "
                f"Rain last 14 days: {w.get('rainfall_14d_mm', 'N/A')} mm"
            )
        if w.get("forecast_7d"):
            forecast_str = ", ".join(
                f"{d['date']}: {d.get('rain_mm', 0):.0f}mm"
                for d in w["forecast_7d"][:3]
            )
            weather_lines.append(f"3-day rain forecast: {forecast_str}")
        parts.append("\n".join(weather_lines))
        parts.append("")
        spray_warn = _spray_timing_warning(weather_ctx)
        if spray_warn:
            parts.append("=== Spray Timing Warning ===")
            parts.append(spray_warn)
            parts.append("")

    parts.append(f"=== Farmer's Question ===\n{q}")
    parts.append("")
    parts.append(
        f"INTENT DETECTED: {intent}\n"
        f"LEAD WITH: {intent_rules['lead']}\n"
        f"SKIP sections: {', '.join(intent_rules['skip']) if intent_rules['skip'] else 'none'}"
    )

    context_block = "\n".join(parts)

    # NOTE ON THIS BLOCK: added after review found the model silently skipping the
    # CONFIRMED DIAGNOSIS echo, the visible-damage-percentage statement, the
    # still-photo constraint, and the natural-sourcing (no disclaimer sentence)
    # instruction, despite them being present in GROUNDING / RESPONSE STRATEGY
    # above — in a long, ~8k-char prompt for a 7B model, instructions stated once
    # in the middle are the most likely to get dropped. This restates the failing
    # items in compressed imperative form, as the LAST thing before generation
    # starts (recency), on top of — not instead of — their home in GROUNDING /
    # RESPONSE STRATEGY above.
    final_checklist = (
        "\n\n========================\n"
        "BEFORE YOU ANSWER — CHECK THESE\n"
        "========================\n"
        "- If a CONFIRMED DIAGNOSIS was given above, your FIRST sentence must state it "
        "plainly (\"The vision model detected... with ...% confidence\") — not \"I see...\" "
        "or \"it appears...\".\n"
        "- Do NOT write \"based on the information available in the agricultural knowledge "
        "base\" or any similar disclaimer sentence.\n"
        "- If \"Visible leaf damage\" above is a real percentage range, state it. If it's "
        "\"Not estimated\"/\"unclear\", say so plainly — don't just omit it.\n"
        "- Do NOT say the disease is \"spreading\" or \"getting worse\" as something you "
        "observed — this is one still photo, not a video.\n"
        + (
            "- This question requires a cause_explanation section: include one short "
            "sentence on what causes this and how it spreads (see GROUNDING section) — "
            "do not skip it for brevity.\n"
            if "cause_explanation" in intent_rules.get("required", []) else ""
        )
        + "Now write your answer."
    )

    return _SYSTEM_PROMPT.format(
        visual_symptoms=visual_description,
        visual_severity=_extract_visual_severity(visual_description),
        retrieved_context=rag_context,
        user_query=q,
        diagnosis_line=_build_diagnosis_line(resnet_hint, confidence),
        confidence_display=_build_confidence_display(confidence),
        confidence_gate=f"{CONFIDENCE_GATE:.0f}",
        intent_label=intent.upper(),
        intent_instruction=intent_rules["lead"],
        runner_up_classes=_build_runner_up_block(predictions, resnet_hint),
        output_language_instruction=(
            "Respond entirely in English. "
            "Write in a natural, conversational tone — like a person speaking, not a form. "
            "Use bullet points for better understanding"
            "adapt to sound natural in context."
        ),
    ) + "\n\n" + context_block + final_checklist


# ── LLaVAAdvisor ──────────────────────────────────────────────────────────────

class LLaVAAdvisor:
    """
    Wraps LLaVA-Next-Mistral-7B for dual-stream image-grounded agricultural advice.

    Stage 1: _describe_image() — short visual description (~80 tokens).
             Checks for non-leaf images and short-circuits if detected.
    Stage 2: generate() / generate_streaming() — full advisory generation.
             visual_description from Stage 1 is injected as {visual_symptoms}
             in the system prompt, so LLaVA grounds its advice in what it sees.

    NOTE ON DETERMINISM: Stage 2 uses greedy decoding (do_sample=False), not
    temperature sampling. With sampling on, re-running the same image produced
    materially different advisories from run to run — different structure,
    different invented details (e.g. one run claiming to see the leaf
    underside, another inventing a fungicide dosage in a different unit than
    retrieved_context specified). Since translator.py is a literal translator
    with no rewriting, that variance was actually coming from independent LLaVA
    samples, not from translation — greedy decoding removes that source of
    cross-run and cross-language inconsistency. This trades a little fluency
    variety for much more consistent, checkable output, which matters more for
    a farmer-facing advisory tool than stylistic variation would.
    """

    def __init__(
        self,
        model_id: str = LLAVA_MODEL_ID,
        load_in_4bit: bool = LOAD_IN_4BIT,
        device_map: str = LLAVA_DEVICE,
    ):
        import torch, gc
        from transformers import (
            LlavaNextProcessor,
            LlavaNextForConditionalGeneration,
            BitsAndBytesConfig,
        )

        # ── Flush GPU 1 before loading ─────────────────────────────────
        gc.collect()
        torch.cuda.empty_cache()
        free_gb = (torch.cuda.get_device_properties(1).total_memory
                   - torch.cuda.memory_allocated(1)) / 1e9
        print(f"[LLaVA] GPU 1 free before load: {free_gb:.2f} GiB")
        if free_gb < 9.0:
            raise RuntimeError(
                f"[LLaVA] GPU 1 only has {free_gb:.1f} GiB free — need ~9 GiB for 4-bit LLaVA. "
                "Clear GPU 1 memory before loading."
            )

        print(f"[LLaVA] Loading {model_id} ({'4-bit' if load_in_4bit else 'full'})...")
        quant_cfg = None
        if load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quant_cfg,
            device_map={"": 1},          # ← explicit: entire model on GPU 1
            torch_dtype=torch.float16 if not load_in_4bit else None,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        print("[LLaVA] Model loaded.")

    def _prepare_inputs(self, image: Image.Image, prompt: str):
        if image.mode != "RGB":
            image = image.convert("RGB")
        conversation = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        return self.processor(
            images=image, text=prompt_text, return_tensors="pt"
        ).to(self.model.device)

    def _describe_image(self, image: Image.Image) -> str:
        """
        Stage 1: Run a short visual description pass.
        Returns a 3-4 bullet description of what the model sees on the leaf plus
        an "AFFECTED AREA: <range>%" line, or a no-leaf sentinel string if no
        plant is visible. ~100-120 tokens — fast.
        """
        import torch
        inputs = self._prepare_inputs(image, _VISUAL_DESCRIPTION_PROMPT)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=130,
                do_sample=False,           # greedy — deterministic description
            )
        input_len = inputs["input_ids"].shape[1]
        description = self.processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()
        print(f"[LLaVA] Visual description: {description[:120]}...")
        return description

    def generate(
        self,
        image: Image.Image,
        rag_context: str,
        farmer_question: str = "What should I do?",
        resnet_hint: Optional[str] = None,
        confidence: Optional[float] = None,
        max_new_tokens: int = LLAVA_MAX_TOKENS,
        weather_ctx: Optional[dict] = None,
        location_ctx: Optional[dict] = None,
        predictions: Optional[List[Tuple[str, float]]] = None,
    ) -> str:
        import torch

        # Stage 1: get visual description (or detect non-leaf / healthy leaf)
        visual_description = self._describe_image(image)
        if _is_no_leaf(visual_description):
            print("[LLaVA] Non-leaf image detected — skipping advisory generation.")
            return _NO_LEAF_RESPONSE
        if _is_healthy_leaf(visual_description):
            print("[LLaVA] Healthy leaf detected via Stage 1 read — skipping advisory generation.")
            return _HEALTHY_RESPONSE

        prompt = _build_prompt(
            rag_context, farmer_question, resnet_hint,
            confidence, visual_description, weather_ctx, location_ctx,
            predictions=predictions,
        )
        inputs = self._prepare_inputs(image, prompt)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # FIXED: greedy, not sampled — see note below
                repetition_penalty=1.15,
            )
        input_len = inputs["input_ids"].shape[1]
        raw = self.processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()
        raw = _ensure_closing_note(_clean_jargon(raw))
        return _validate_advice(raw, resnet_hint)

    def generate_streaming(
        self,
        image: Image.Image,
        rag_context: str,
        farmer_question: str = "What should I do?",
        resnet_hint: Optional[str] = None,
        confidence: Optional[float] = None,
        weather_ctx: Optional[dict] = None,
        location_ctx: Optional[dict] = None,
        predictions: Optional[List[Tuple[str, float]]] = None,
    ) -> TextIteratorStreamer:
        # Stage 1: visual description (non-streaming — it's short)
        visual_description = self._describe_image(image)
        if _is_no_leaf(visual_description):
            print("[LLaVA] Non-leaf image detected — returning static response.")
            class _StaticStreamer:
                def __iter__(self):
                    yield _NO_LEAF_RESPONSE
            return _StaticStreamer()
        if _is_healthy_leaf(visual_description):
            print("[LLaVA] Healthy leaf detected via Stage 1 read — returning static response.")
            class _StaticStreamer:
                def __iter__(self):
                    yield _HEALTHY_RESPONSE
            return _StaticStreamer()

        prompt = _build_prompt(
            rag_context, farmer_question, resnet_hint,
            confidence, visual_description, weather_ctx, location_ctx,
            predictions=predictions,
        )
        inputs = self._prepare_inputs(image, prompt)
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        Thread(
            target=self.model.generate,
            kwargs=dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=LLAVA_MAX_TOKENS,
                do_sample=False,          # FIXED: greedy, matches generate() — see
                                          # determinism note on the class docstring
                repetition_penalty=1.15,
            ),
        ).start()
        return streamer