"""
CropQuery — LLaVA Advisor
"""

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
from typing import Optional, Generator
from PIL import Image
from transformers import TextIteratorStreamer
from threading import Thread

from .config import (
    LLAVA_MODEL_ID, LLAVA_MAX_TOKENS, LOAD_IN_4BIT,
    CONFIDENCE_GATE, LLAVA_DEVICE,
)


def _spray_timing_warning(weather_ctx: dict) -> str:
    if not weather_ctx:
        return ""
    from .context_fetcher import spray_timing_advisory
    return spray_timing_advisory(weather_ctx)


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


_INTENT_KEYWORDS = {
    "curable":      ["can this be cured", "is it curable", "will it recover", "ठीक हो सकता", "ठीक होगा", "ठीक हो जाएगा"],
    "treatment":    ["spray", "chemical", "medicine", "treat", "fungicide", "pesticide",
                     "दवाई", "दवा", "स्प्रे", "इलाज"],
    "fertilizer":   ["fertilizer", "fertiliser", "urea", "npk", "nutrient", "खाद", "उर्वरक"],
    "irrigation":   ["water", "irrigat", "drip", "flood", "सिंचाई", "पानी"],
    "severity":     ["how bad", "serious", "damage", "loss", "कितना नुकसान", "कितना खराब", "गंभीर"],
    "prevention":   ["prevent", "avoid", "protect", "stop from", "रोकें", "बचाव"],
    "diagnosis":    ["what is", "what happened", "identify", "disease", "problem",
                     "क्या हुआ", "क्या है", "पहचान", "बीमारी"],
    "spray_timing": ["when to spray", "best time to spray", "कब स्प्रे", "सही समय"],
    "yield_loss":   ["yield", "production", "harvest", "crop loss", "उत्पादन", "पैदावार"],
    "cost":         ["cost", "price", "cheap", "expensive", "कितने का", "सस्ता", "महंगा"],
}

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
        "lead": "State severity level immediately. Explain what damage you can see in the image.",
        "required": ["severity_detail", "visual_reasoning"],
        "skip": ["treatment_detail", "weather"],
    },
    "prevention": {
        "lead": "Lead with 2-3 concrete prevention steps. Reference what the image shows.",
        "required": ["prevention"],
        "skip": ["severity_detail"],
    },
    "diagnosis": {
        "lead": "Describe what you can see in the image first. Then name the diagnosis and explain why.",
        "required": ["visual_reasoning", "diagnosis", "uncertainty"],
        "skip": [],
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
        "lead": "Describe what you see, name the problem, and give the most useful advice for this situation.",
        "required": ["visual_reasoning", "diagnosis", "treatment"],
        "skip": [],
    },
}


def _classify_intent(question: str) -> str:
    """
    Classify farmer's question into one of the intent labels.
    Returns 'general' if no specific intent is detected.
    """
    if not question or not question.strip():
        return "diagnosis"
    q_lower = question.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return intent
    return "general"


_NO_LEAF_PHRASES = [
    "no plant", "no leaf", "not a leaf", "no crop", "no vegetation",
    "cannot identify", "not a plant", "no disease visible",
    "no agricultural", "unclear image", "no leaves",
]

_NO_LEAF_RESPONSE = """❓ आपका जवाब (Your Answer)
* No plant or leaf was detected in this image.
* Please upload a clear photo of a leaf or crop to get disease advice.

🌿 समस्या (Problem)
* Image does not show a plant leaf.
* Cannot diagnose without a clear leaf photo.

✅ क्या करें (What to do now)
* Take a close-up photo of the affected leaf in natural daylight.
* Make sure the leaf fills most of the frame.
* Avoid blurry or very dark photos.

⚠️ सुरक्षा (Safety Notice)
* Always confirm chemical use with your local agriculture officer before applying."""


def _is_no_leaf(visual_description: str) -> bool:
    desc_lower = visual_description.lower()
    return any(phrase in desc_lower for phrase in _NO_LEAF_PHRASES)


_VISUAL_DESCRIPTION_PROMPT = """Look at this image carefully.

If you can see a plant leaf or crop:
Describe ONLY what you see on the leaf in 2-3 short bullet points:
- Colour of affected areas (yellow, brown, black, white spots?)
- Location on leaf (edges, centre, whole leaf, underside?)
- Spread (small patches, large areas, covering most of leaf?)
- Any visible pests, webbing, or powdery coating?

If there is NO plant or leaf visible in this image, reply with exactly:
"No plant or leaf visible in this image."

Keep your answer under 60 words. Do not diagnose. Do not recommend treatment. Just describe."""


_SYSTEM_PROMPT = """You are a senior agricultural extension officer with decades of field experience across India.
You speak like someone who has physically inspected this crop — calm, honest, and direct.
You do not sound like a form or a template. You sound like a person who knows farming.

========================
YOUR INPUTS
========================
What you can see in the image: {visual_symptoms}
Knowledge from disease database: {retrieved_context}
Farmer's Question: {user_query}
Diagnosis: {diagnosis_line}
Image Confidence: {confidence_display}

========================
RESPONSE STRATEGY
========================
1. Understand what the farmer is specifically asking.
2. Answer that question directly in your first sentence.
3. Use what you can see in the image to justify your answer.
4. Include only the sections that are relevant to this question.
5. Do not force in sections that add no value here.

========================
INTENT: {intent_label}
========================
{intent_instruction}

========================
STRICT RULES — FOLLOW EXACTLY
========================
1. RAG ONLY: Recommend ONLY chemicals and treatments found in the knowledge database.
   Never invent pesticide names, dosages, or schedules.
2. NO JARGON: Use only simple words. Replace automatically:
   "lesions"→"spots", "necrosis"→"dead patches", "inoculum"→"disease source",
   "sporulation"→"disease spreading", "vector"→"carrier insect",
   "foliar application"→"leaf spray", "systemic"→"absorbed into the plant".
3. VISUAL REASONING FIRST: Before giving advice, briefly describe what you see
   and why it points to this diagnosis. Do not skip this.
4. UNCERTAINTY: If confidence is below 90%, mention alternative possibilities.
   If confidence is below 70%, ask for a better image before giving strong advice.
5. CONFIDENCE LANGUAGE:
   - Above 90%: "This is clearly..." / "I can see..."
   - 70–90%: "This most likely is..." / "The symptoms suggest..."
   - Below 70%: "The image is not clear enough for a confident diagnosis..."
6. SHORT LINES: Maximum 10 words per bullet. No paragraphs.
7. NO GUARANTEES: Never say "definitely", "100% cure", "certainly cured".
8. SAFETY: Always end with the safety notice if chemicals are mentioned.

========================
CONVERSATIONAL STYLE
========================
Write like a person, not a form. Examples:

Instead of:
  Problem: Downy Mildew
  Severity: Severe

Write like:
  Based on what I can see, the leaf shows widespread yellowing and dark patches
  underneath — this strongly points to Downy Mildew.

Instead of:
  Action: Apply fungicide

Write like:
  At this stage I would spray Metalaxyl (2g/litre) on both sides of the leaf,
  ideally in the early morning when the wind is low.

========================
OUTPUT LANGUAGE
========================
{output_language_instruction}

========================
SAFETY NOTICE (always include if chemicals mentioned)
========================
⚠️ सुरक्षा (Safety)
* Confirm chemical use with your local agriculture officer before applying.
* Keep children and animals away from sprayed areas for 48 hours.
* Wear gloves and mask when mixing chemicals.
"""


def _build_confidence_display(confidence: Optional[float]) -> str:
    if confidence is None:
        return "Unknown"
    if confidence >= 90:
        return f"{confidence:.1f}% — High (strong symptom match)"
    elif confidence >= 70:
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


def _build_prompt(
    rag_context: str,
    farmer_question: str,
    resnet_hint: Optional[str],
    confidence: Optional[float],
    visual_description: str,
    weather_ctx: Optional[dict] = None,
    location_ctx: Optional[dict] = None,
) -> str:
    q = farmer_question.strip() if farmer_question else "What should I do about this?"
    intent = _classify_intent(q)
    intent_rules = _INTENT_SECTION_RULES.get(intent, _INTENT_SECTION_RULES["general"])

    parts = []

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

    return _SYSTEM_PROMPT.format(
        visual_symptoms=visual_description,
        retrieved_context=rag_context,
        user_query=q,
        diagnosis_line=_build_diagnosis_line(resnet_hint, confidence),
        confidence_display=_build_confidence_display(confidence),
        intent_label=intent.upper(),
        intent_instruction=intent_rules["lead"],
        output_language_instruction=(
            "Respond entirely in English. "
            "Write in a natural, conversational tone — like a person speaking, not a form. "
            "Use bullet points only where they genuinely help. "
            "Do not reproduce the section headers from the system prompt literally — "
            "adapt them to sound natural in context."
        ),
    ) + "\n\n" + context_block


class LLaVAAdvisor:
    """
    Wraps LLaVA-Next-Mistral-7B for dual-stream image-grounded agricultural advice.

    Stage 1: _describe_image() — short visual description (~80 tokens).
             Checks for non-leaf images and short-circuits if detected.
    Stage 2: generate() / generate_streaming() — full advisory generation.
             visual_description from Stage 1 is injected as {visual_symptoms}
             in the system prompt, so LLaVA grounds its advice in what it sees.
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
            device_map={"": 1},
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
        import torch
        inputs = self._prepare_inputs(image, _VISUAL_DESCRIPTION_PROMPT)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
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
    ) -> str:
        import torch

        visual_description = self._describe_image(image)
        if _is_no_leaf(visual_description):
            print("[LLaVA] Non-leaf image detected — skipping advisory generation.")
            return _NO_LEAF_RESPONSE

        prompt = _build_prompt(
            rag_context, farmer_question, resnet_hint,
            confidence, visual_description, weather_ctx, location_ctx,
        )
        inputs = self._prepare_inputs(image, prompt)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                repetition_penalty=1.15,
            )
        input_len = inputs["input_ids"].shape[1]
        raw = self.processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()
        return _validate_advice(_clean_jargon(raw), resnet_hint)

    def generate_streaming(
        self,
        image: Image.Image,
        rag_context: str,
        farmer_question: str = "What should I do?",
        resnet_hint: Optional[str] = None,
        confidence: Optional[float] = None,
        weather_ctx: Optional[dict] = None,
        location_ctx: Optional[dict] = None,
    ) -> TextIteratorStreamer:
        visual_description = self._describe_image(image)
        if _is_no_leaf(visual_description):
            print("[LLaVA] Non-leaf image detected — returning static response.")
            class _StaticStreamer:
                def __iter__(self):
                    yield _NO_LEAF_RESPONSE
            return _StaticStreamer()

        prompt = _build_prompt(
            rag_context, farmer_question, resnet_hint,
            confidence, visual_description, weather_ctx, location_ctx,
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
                do_sample=True,
                temperature=0.3,
                repetition_penalty=1.15,
            ),
        ).start()
        return streamer

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
            device_map=LLAVA_DEVICE,
            torch_dtype=torch.float16 if not load_in_4bit else None,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        print("[LLaVA] Model loaded.")

    def _prepare_inputs(self, image: Image.Image, prompt: str):
        if image.mode != "RGB":
            image = image.convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        return self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt",
        ).to(self.model.device)

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
    ) -> str:
        import torch

        prompt = _build_prompt(
            rag_context, farmer_question, resnet_hint,
            confidence, weather_ctx, location_ctx,
        )
        inputs = self._prepare_inputs(image, prompt)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                repetition_penalty=1.15,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        raw = self.processor.decode(generated, skip_special_tokens=True).strip()
        return _clean_jargon(raw)

    def generate_streaming(
        self,
        image: Image.Image,
        rag_context: str,
        farmer_question: str = "What should I do?",
        resnet_hint: Optional[str] = None,
        confidence: Optional[float] = None,
        weather_ctx: Optional[dict] = None,
        location_ctx: Optional[dict] = None,
    ) -> TextIteratorStreamer:
        """Token-by-token streaming variant."""
        prompt = _build_prompt(
            rag_context, farmer_question, resnet_hint,
            confidence, weather_ctx, location_ctx,
        )
        inputs = self._prepare_inputs(image, prompt)

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        thread = Thread(
            target=self.model.generate,
            kwargs=dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=LLAVA_MAX_TOKENS,
                do_sample=True,
                temperature=0.3,
                repetition_penalty=1.15,
            ),
        )
        thread.start()
        return streamer
