"""
CropQuery — LLaVA Advisor
"""

from typing import Optional
from PIL import Image
from transformers import TextIteratorStreamer
from threading import Thread

from .config import (
    LLAVA_MODEL_ID,
    LLAVA_MAX_TOKENS,
    LOAD_IN_4BIT,
    CONFIDENCE_GATE,
    LLAVA_DEVICE,
)

# ---------------------------------------------------------------------------
# Spray timing rule (weather-based)
# ---------------------------------------------------------------------------

def _spray_timing_warning(weather_ctx: dict) -> str:
    if not weather_ctx:
        return ""
    from .context_fetcher import spray_timing_advisory
    return spray_timing_advisory(weather_ctx)


# ---------------------------------------------------------------------------
# Jargon replacement — run on final output before returning
# ---------------------------------------------------------------------------

_JARGON_MAP = {
    "new flush":        "new leaves",
    "flush emergence":  "new leaves appearing",
    "flush":            "new leaves",
    "inoculum":         "disease source",
    "sporulation":      "disease spreading",
    "lesions":          "spots",
    "necrosis":         "dead patches",
    "necrotic":         "dead",
    "vector":           "carrier insect",
    "infestation":      "attack",
    "larval mining":    "pest damage inside leaves",
    "oviposition":      "egg-laying",
    "instar":           "stage of pest",
    "systemic":         "absorbed by the plant",
    "contact insecticide": "spray that kills on contact",
    "foliar application":  "leaf spray",
    "phytotoxicity":    "plant damage from chemicals",
    "stippling":        "tiny spots",
    "chlorosis":        "yellowing",
    "defoliation":      "leaf fall",
}

def _clean_jargon(text: str) -> str:
    """Replace technical terms with farmer-friendly language."""
    import re
    for jargon, simple in _JARGON_MAP.items():
        # case-insensitive whole-word replacement
        text = re.sub(re.escape(jargon), simple, text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are CropQuery — a trusted, practical agricultural advisor for Indian farmers.
You speak like a helpful village extension worker: calm, honest, simple, and caring.

========================
YOUR INPUTS
========================
Visual Symptoms: {visual_symptoms}
Knowledge from database: {retrieved_context}
Farmer's Question: {user_query}

========================
STRICT RULES — FOLLOW EXACTLY
========================
1. RAG ONLY: Recommend ONLY chemicals and treatments found in the knowledge database.
   Never invent pesticide names, dosages, or schedules.
2. NO JARGON: Use only simple words a farmer with no schooling can understand.
   BANNED WORDS — never use these, use the replacement instead:
   - "flush" or "new flush" → say "new leaves"
   - "flush emergence" → say "new leaves appearing"  
   - "lesions" → say "spots"
   - "necrosis" / "necrotic" → say "dead patches"
   - "inoculum" → say "disease source"
   - "sporulation" → say "disease spreading"
   - "vector" → say "carrier insect"
   - "instar" → say "pest stage"
   - "oviposition" → say "egg-laying"
   - "foliar application" → say "leaf spray"
   - "systemic" → say "absorbed into the plant"
3. ANSWER THE QUESTION: The farmer asked a specific question. You MUST answer it 
   directly, clearly, and first — before giving general advice.
4. FERTILIZER RULE: If asked about fertilizer for a pest or disease problem, 
   explain clearly: "Fertilizer does not cure this problem. It is caused by 
   [pest/fungus], not by lack of nutrients. Too much nitrogen can actually 
   make the problem worse." Then give the correct treatment.
5. CURABLE RULE: If asked whether the disease/pest is curable, answer YES or NO 
   clearly in the first line. Leaf miners, most fungal diseases — YES, manageable 
   with treatment. Be honest if something has no cure (e.g. viral wilt).
6. NO GUARANTEES: Never say "definitely", "100% cure", "certainly cured".
7. SHORT LINES: Maximum 10 words per bullet point. No paragraphs.
8. UNCERTAINTY: If symptoms and database do not match well, say so honestly.

========================
OUTPUT FORMAT — COPY EXACTLY, fill in advice only
========================

❓ आपका सवाल (Your Answer)
* [Direct answer to the farmer's question in 1-2 simple sentences. YES/NO first if applicable.]
* [If fertilizer was asked about: explain it won't help and why, in simple words.]

🌿 समस्या (Problem)
* [Disease or pest name — simple name only]
* [Confidence: High / Medium / Low]
* [One sentence: what you see on the leaf that tells you this]

⚠️ स्थिति (Severity)
* [Mild / Moderate / Severe]
* [One sentence: why this severity, what damage it is doing]

✅ क्या करें (What to do now)
* Organic: [one simple organic action, everyday words]
* Chemical: [chemical name + dosage from database only, or "Ask your local agricultural shop for the correct dose"]
* Prevent: [one simple preventive action]

📅 अगले कुछ दिन (Next 7 days)
* Check: [what to look for, in simple words — no jargon]
* Act if: [specific sign that means the problem is getting worse]

🌦️ मौसम की सलाह (Weather Advice)
* [One simple rule about when NOT to spray]

📞 कब सलाह लें (When to seek help)
* [One clear condition: when should the farmer call an officer]


========================
OUTPUT LANGUAGE
========================
{output_language_instruction}
"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    rag_context: str,
    farmer_question: str,
    resnet_hint: Optional[str],
    confidence: Optional[float],
    weather_ctx: Optional[dict] = None,
    location_ctx: Optional[dict] = None,
) -> str:
    parts = []

    # ── Confirmed diagnosis block ──────────────────────────────────────────
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

    # ── Location block ─────────────────────────────────────────────────────
    if location_ctx:
        loc_parts = []
        for key, label in [
            ("state", "State"),
            ("district", "District"),
            ("agro_climatic_zone", "Zone"),
            ("current_season", "Season"),
        ]:
            if location_ctx.get(key):
                loc_parts.append(f"{label}: {location_ctx[key]}")
        if loc_parts:
            parts.append("=== Farmer Location ===")
            parts.append(" | ".join(loc_parts))
            parts.append("")

    # ── Weather block ──────────────────────────────────────────────────────
    if weather_ctx:
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

    # ── Farmer question ────────────────────────────────────────────────────
    q = farmer_question.strip() if farmer_question else "What should I do about this?"
    parts.append(f"=== Farmer's Question ===\n{q}")
    parts.append("")
    parts.append(
        "IMPORTANT REMINDER: Answer the farmer's question FIRST and DIRECTLY. "
        "If they asked about fertilizer, explain it won't cure this. "
        "If they asked if it is curable, say YES or NO clearly."
    )

    # ── Build visual_symptoms string ───────────────────────────────────────
    if resnet_hint and confidence is not None:
        visual_symptoms = (
            f"Predicted class: {resnet_hint.replace('__', ' — ').replace('_', ' ')} "
            f"(confidence: {confidence:.1f}%)"
        )
    else:
        visual_symptoms = "Unknown — low confidence, reasoning from image directly"

    # ── Assemble final prompt via .format() ────────────────────────────────
    context_block = "\n".join(parts)

    return _SYSTEM_PROMPT.format(
        visual_symptoms=visual_symptoms,
        retrieved_context=rag_context,
        user_query=q,
        output_language_instruction=(
            "Respond entirely in English. "
            "Use the exact emoji section headers shown above. "
            "Do not write paragraphs — use short bullet points only."
        ),
    ) + "\n\n" + context_block


# ---------------------------------------------------------------------------
# LLaVA Advisor
# ---------------------------------------------------------------------------

class LLaVAAdvisor:
    """Wraps LLaVA-Next-Mistral-7B for image-grounded agricultural advice."""

    def __init__(
        self,
        model_id: str = LLAVA_MODEL_ID,
        load_in_4bit: bool = LOAD_IN_4BIT,
        device_map: str = "auto",
    ):
        import torch
        from transformers import (
            LlavaNextProcessor,
            LlavaNextForConditionalGeneration,
            BitsAndBytesConfig,
        )

        print(f"[LLaVA] Loading {model_id} "
              f"({'4-bit' if load_in_4bit else 'full precision'})...")

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
