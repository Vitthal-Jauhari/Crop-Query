"""
CropQuery — Multilingual Translation
Translates English advice to Indian regional languages using IndicTrans2.

Model: ai4bharat/indictrans2-en-indic-1B
Runs on CPU — no VRAM impact, no API cost.
Handles agricultural vocabulary better than general VLM multilingual output.
"""

from typing import Optional

from .config import INDICTRANS_MODEL, LANGUAGE_CODES, DEFAULT_LANGUAGE


class IndicTranslator:
    """
    Wraps IndicTrans2 for English → Indian language translation.

    The model is loaded lazily on first call to avoid startup overhead
    when running in English-only mode.

    Usage
    -----
    translator = IndicTranslator()
    hindi_text = translator.translate(english_advice, "Hindi")
    """

    def __init__(self, model_id: str = INDICTRANS_MODEL):
        self.model_id = model_id
        self._pipeline = None   # lazy load

    def _load(self):
        """Load IndicTrans2 pipeline on first use."""
        if self._pipeline is not None:
            return
        print(f"[Trans] Loading IndicTrans2 from {self.model_id}...")
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self._model.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            self._torch = torch
            print(f"[Trans] IndicTrans2 loaded on {self._device}.")
        except Exception as e:
            print(f"[Trans] WARNING: Could not load IndicTrans2: {e}")
            print("[Trans] Translation will return the original English text.")
            self._pipeline = False   # sentinel: load failed

    def translate(
        self,
        text: str,
        target_language: str = "Hindi",
        src_lang: str = "eng_Latn",
    ) -> str:
        """
        Translate English text to the target Indian language.

        Parameters
        ----------
        text : str
            English text to translate (typically the 4-section advice).
        target_language : str
            Display name matching LANGUAGE_CODES keys, e.g. 'Hindi', 'Telugu'.
        src_lang : str
            IndicTrans2 source language code (always English here).

        Returns
        -------
        str — translated text, or original text if language is English
               or if the model failed to load.
        """
        tgt_code = LANGUAGE_CODES.get(target_language)

        # English — no translation needed
        if tgt_code is None:
            return text

        self._load()

        # Model load failed — return English
        if self._pipeline is False:
            return text

        try:
            return self._run_translation(text, src_lang, tgt_code)
        except Exception as e:
            print(f"[Trans] Translation failed: {e} — returning English")
            return text

    def _run_translation(self, text: str, src: str, tgt: str) -> str:
        """Run the actual IndicTrans2 inference."""
        # IndicTrans2 uses a sentence-level tokenizer with language tags
        inputs = self._tokenizer(
            text,
            src_lang=src,
            tgt_lang=tgt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self._device)

        with self._torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                num_beams=5,
                max_length=512,
                early_stopping=True,
            )

        translated = self._tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return translated[0].strip() if translated else text

    @staticmethod
    def available_languages() -> list:
        """Return list of supported language display names."""
        return list(LANGUAGE_CODES.keys())

    @staticmethod
    def language_code(display_name: str) -> Optional[str]:
        """Return the IndicTrans2 code for a display name."""
        return LANGUAGE_CODES.get(display_name)
