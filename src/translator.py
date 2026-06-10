"""
CropQuery — Multilingual Translation (deep-translator backend)
NOTE: Uses GoogleTranslator from deep-translator, NOT IndicTrans2.
"""
from typing import Optional

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
            print("[Trans] WARNING: deep-translator not installed. pip install deep-translator")
            self._loaded = False

    def translate(
        self,
        text: str,
        target_language: str = "Hindi",
        src_lang: str = "eng_Latn",
    ) -> str:
        google_code = _GOOGLE_LANG_CODES.get(target_language)
        if google_code is None:
            return text  # English — no translation needed
        self._load()
        if self._loaded is False:
            return text
        try:
            if len(text) <= 4500:
                return self._translator_cls(source="en", target=google_code).translate(text) or text
            else:
                return self._translate_chunked(text, google_code)
        except Exception as e:
            print(f"[Trans] Translation failed: {e} — returning English")
            return text

    def _translate_chunked(self, text: str, google_code: str) -> str:
        chunks = text.split("\n\n")
        translated = []
        t = self._translator_cls(source="en", target=google_code)
        for chunk in chunks:
            try:
                translated.append(t.translate(chunk) if chunk.strip() else chunk)
            except Exception:
                translated.append(chunk)
        return "\n\n".join(translated)

    def to_english(self, text: str) -> str:
        """
        Translate any Indian-language question to English before RAG retrieval.
        Uses source='auto' — GoogleTranslator detects the input language.
        Returns text unchanged if already English or if translation fails.
        """
        if not text or not text.strip():
            return text
        self._load()
        if self._loaded is False:
            return text
        try:
            result = self._translator_cls(source="auto", target="en").translate(text)
            print(f"[Trans] Question translated to English: {result[:80]}")
            return result or text
        except Exception as e:
            print(f"[Trans] to_english failed: {e} — using original")
            return text

    @staticmethod
    def available_languages() -> list:
        return ["English"] + list(_GOOGLE_LANG_CODES.keys())

    @staticmethod
    def language_code(display_name: str) -> Optional[str]:
        return _GOOGLE_LANG_CODES.get(display_name)
