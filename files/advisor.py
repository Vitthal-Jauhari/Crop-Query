"""
CropQuery — LLaVA Advisor
Runs LLaVA-1.5-7B in 4-bit quantization.
Image + RAG chunks + optional ResNet50 hint → structured English advice.

VRAM budget on Kaggle T4 (16 GB):
    LLaVA-1.5-7B  4-bit  ≈  8 GB
    ResNet50             ≈  0.1 GB
    Sentence Transformer ≈  0.1 GB
    Headroom             ≈  7.8 GB  ✓
"""

from typing import List, Dict, Any, Optional
from PIL import Image

from .config import (
    LLAVA_MODEL_ID,
    LLAVA_MAX_TOKENS,
    LOAD_IN_4BIT,
    CONFIDENCE_GATE,
)

# ── System prompt template ────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are CropQuery, an expert agricultural advisor for Indian farmers.
Use ONLY the retrieved agricultural knowledge below to ground your response.
Do NOT fabricate chemical names, dosages, or treatments not mentioned in the context.
Be practical, concise, and compassionate — the farmer may be in distress.

Structure your response in exactly these four sections:
1. **What this is** — disease name, causative agent (fungal/bacterial/viral/nutrient)
2. **How serious** — assess severity from what you see in the image (mild / moderate / severe)
3. **Immediate action** — what to do TODAY (within 24 hours)
4. **7-day plan** — step-by-step actions over the next week

Keep each section to 2–3 sentences. Use simple language."""


def _build_prompt(
    rag_context: str,
    farmer_question: str,
    resnet_hint: Optional[str],
    confidence: Optional[float],
) -> str:
    parts = [_SYSTEM_PROMPT, "", rag_context, ""]

    if resnet_hint and confidence and confidence >= CONFIDENCE_GATE:
        parts.append(
            f"ResNet50 Classification: {resnet_hint} "
            f"({confidence:.1f}% confidence — use as a hint, verify visually)"
        )
        parts.append("")

    parts.append(f"Farmer's Question: {farmer_question or 'What should I do about this?'}")
    return "\n".join(parts)


class LLaVAAdvisor:
    """
    Wraps LLaVA-1.5-7B-HF for image-grounded agricultural advice.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID.
    load_in_4bit : bool
        Whether to quantize with BitsAndBytes (required for T4).
    device_map : str
        Device map — 'auto' works for single or multi-GPU.

    Usage
    -----
    advisor = LLaVAAdvisor()
    advice  = advisor.generate(
        image=pil_image,
        rag_context="...",
        farmer_question="What spray should I use?",
        resnet_hint="Tomato__Late_Blight",
        confidence=91.2,
    )
    """

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

        print(f"[LLaVA] Loading {model_id} ({'4-bit' if load_in_4bit else 'full precision'})...")

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
            device_map=device_map,
            torch_dtype=torch.float16 if not load_in_4bit else None,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        print("[LLaVA] Model loaded.")

    def generate(
        self,
        image: Image.Image,
        rag_context: str,
        farmer_question: str = "What should I do?",
        resnet_hint: Optional[str] = None,
        confidence: Optional[float] = None,
        max_new_tokens: int = LLAVA_MAX_TOKENS,
    ) -> str:
        """
        Generate agricultural advice conditioned on the image and RAG context.

        Returns
        -------
        str — structured English advice (4 sections)
        """
        import torch

        if image.mode != "RGB":
            image = image.convert("RGB")

        text_prompt = _build_prompt(rag_context, farmer_question, resnet_hint, confidence)

        # LLaVA-1.5 conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
            )

        # Decode only the generated portion (skip input tokens)
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        advice = self.processor.decode(generated, skip_special_tokens=True).strip()
        return advice
