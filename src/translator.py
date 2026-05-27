"""
CropQuery — Multilingual Translation (deep-translator backend)
"""

from typing import Optional
from .config import LANGUAGE_CODES, DEFAULT_LANGUAGE

# deep-translator language codes differ from IndicTrans2 codes.
# Map display names → Google Translate BCP-47 codes.
_GOOGLE_LANG_CODES = {
    "Hindi":     "hi",
    "Tamil":     "ta",
    "Telugu":    "te",
    "Kannada":   "kn",
    "Malayalam": "ml",
    "Bengali":   "bn",
    "Marathi":   "mr",
    "Gujarati":  "gu",
    "Punjabi":   "pa",
    "Odia":      "or",
    "Urdu":      "ur",
}


class IndicTranslator:
    """
    Wraps deep-translator (GoogleTranslator) for English → Indian language
    translation. Same public interface as the original IndicTrans2 class.
    """

    def __init__(self, model_id: str = None):
        # model_id kept for API compatibility, unused here
        self._loaded: Optional[bool] = None
        self._translator_cls = None

    def _load(self):
        if self._loaded is not None:
            return
        try:
            from deep_translator import GoogleTranslator
            self._translator_cls = GoogleTranslator
            self._loaded = True
            print("[Trans] deep-translator (GoogleTranslator) ready.")
        except ImportError:
            print("[Trans] WARNING: deep-translator not installed.")
            print("[Trans] Run: pip install deep-translator")
            print("[Trans] Translation will return original English text.")
            self._loaded = False

    def translate(
        self,
        text: str,
        target_language: str = "Hindi",
        src_lang: str = "eng_Latn",
    ) -> str:
        # English — no translation needed
        google_code = _GOOGLE_LANG_CODES.get(target_language)
        if google_code is None:
            return text

        self._load()
        if self._loaded is False:
            return text

        try:
            # GoogleTranslator has a 5000-char limit per call; chunk if needed
            if len(text) <= 4500:
                return self._translator_cls(
                    source="en", target=google_code
                ).translate(text) or text
            else:
                return self._translate_chunked(text, google_code)
        except Exception as e:
            print(f"[Trans] Translation failed: {e} — returning English")
            return text

    def _translate_chunked(self, text: str, google_code: str) -> str:
        """Split on double-newlines, translate each chunk, rejoin."""
        chunks = text.split("\n\n")
        translated = []
        t = self._translator_cls(source="en", target=google_code)
        for chunk in chunks:
            try:
                result = t.translate(chunk) if chunk.strip() else chunk
                translated.append(result or chunk)
            except Exception:
                translated.append(chunk)
        return "\n\n".join(translated)

    @staticmethod
    def available_languages() -> list:
        return ["English"] + list(_GOOGLE_LANG_CODES.keys())

    @staticmethod
    def language_code(display_name: str) -> Optional[str]:
        return _GOOGLE_LANG_CODES.get(display_name)
