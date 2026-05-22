"""
CropQuery — Standalone Translator Test
Run locally to test translation without loading ResNet50, LLaVA, or ChromaDB.

Two backends:
  --backend google       Uses deep-translator (no install pain, for local testing)
  --backend indictrans2  Uses IndicTrans2 (requires C++ build tools, used on Kaggle)

Install for local testing:
    pip install deep-translator

Usage:
    python translator.py --backend google
    python translator.py --backend google --lang Hindi
    python translator.py --backend google --text "Spray neem oil." --lang Telugu
"""

import sys
import types
if "transformers.onnx" not in sys.modules:
    sys.modules["transformers.onnx"] = types.ModuleType("transformers.onnx")

import argparse
import time

MODEL_ID = "ai4bharat/indictrans2-en-indic-1B"

LANGUAGE_CODES = {
    "Hindi": "hin_Deva", "Marathi": "mar_Deva", "Bengali": "ben_Beng",
    "Telugu": "tel_Telu", "Tamil": "tam_Taml", "Kannada": "kan_Knda",
    "Gujarati": "guj_Gujr", "Punjabi": "pan_Guru", "Odia": "ory_Orya",
    "Malayalam": "mal_Mlym", "English": None,
}

GOOGLE_CODES = {
    "Hindi": "hi", "Marathi": "mr", "Bengali": "bn", "Telugu": "te",
    "Tamil": "ta", "Kannada": "kn", "Gujarati": "gu", "Punjabi": "pa",
    "Odia": "or", "Malayalam": "ml", "English": None,
}

TEST_CASES = [
    {
        "label": "Cashew Leaf Miner",
        "text": (
            "1. What this is: Cashew Leaf Miner (Acrocercops syngramma), an insect pest.\n"
            "2. How serious: Moderate. Several leaves show active mining damage.\n"
            "3. Immediate action: Spray neem oil at 5 mL per litre of water today.\n"
            "4. 7-day plan: Apply Dimethoate 30 EC at 2 mL per litre after 5 days. "
            "Collect and burn fallen mined leaves."
        ),
    },
    {
        "label": "Tomato Late Blight",
        "text": (
            "1. What this is: Tomato Late Blight, Phytophthora infestans, a fungal pathogen.\n"
            "2. How serious: Severe. Over 40% of leaf area affected.\n"
            "3. Immediate action: Stop overhead irrigation. Spray Mancozeb 75 WP at 2 g per litre.\n"
            "4. 7-day plan: Repeat Mancozeb every 7 days. Remove and destroy infected material."
        ),
    },
    {
        "label": "Single sentence",
        "text": "Spray neem oil on the affected leaves immediately.",
    },
    {
        "label": "Numbers and units",
        "text": "Apply Copper oxychloride 50 WP at 3 g per litre. Spray 3 times at 15-day intervals.",
    },
]


class IndicTranslator:
    """
    Two-backend translator with identical public API.
      backend='google'      — deep-translator, local testing, no build deps
      backend='indictrans2' — IndicTrans2, Kaggle production, all bugs fixed
    """

    def __init__(self, model_id=MODEL_ID, device=None, backend="indictrans2"):
        assert backend in ("indictrans2", "google")
        self.backend = backend
        self.model_id = model_id
        self._loaded = None
        self._device_override = device

    def translate(self, text, target_language="Hindi"):
        if LANGUAGE_CODES.get(target_language) is None:
            return text
        if self.backend == "google":
            return self._google(text, target_language)
        return self._indictrans2(text, target_language)

    @staticmethod
    def available_languages():
        return list(LANGUAGE_CODES.keys())

    def _google(self, text, lang):
        code = GOOGLE_CODES.get(lang)
        if not code:
            return text
        try:
            from deep_translator import GoogleTranslator
            return GoogleTranslator(source="en", target=code).translate(text)
        except ImportError:
            print("[Trans] Run: pip install deep-translator")
            return text
        except Exception as e:
            print(f"[Trans] Google failed: {e}")
            return text

    def _indictrans2(self, text, lang):
        self._load()
        if not self._loaded:
            return text
        try:
            return self._run(text, "eng_Latn", LANGUAGE_CODES[lang])
        except Exception as e:
            print(f"[Trans] Failed: {e}")
            return text

    def _load(self):
        if self._loaded is not None:
            return
        print(f"[Trans] Loading {self.model_id} ...")
        t0 = time.time()
        try:
            import torch
            import transformers.tokenization_utils as _tu
            if not hasattr(_tu, "PreTrainedTokenizerBase"):
                from transformers.tokenization_utils_base import PreTrainedTokenizerBase
                _tu.PreTrainedTokenizerBase = PreTrainedTokenizerBase
            from IndicTransToolkit import IndicProcessor
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._ip = IndicProcessor(inference=True)
            self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, trust_remote_code=True)
            self._model.eval()
            self._device = self._device_override or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
            self._torch = torch
            self._loaded = True  # FIX: original never set this
            print(f"[Trans] Loaded in {time.time()-t0:.1f}s on {self._device}")
        except Exception as e:
            print(f"[Trans] Load failed: {e}")
            self._loaded = False

    def _run(self, text, src, tgt):
        batch = self._ip.preprocess_batch([text], src_lang=src, tgt_lang=tgt)
        inputs = self._tok(batch, truncation=True, padding="longest",
                           return_tensors="pt", max_length=512).to(self._device)
        with self._torch.no_grad():
            out = self._model.generate(**inputs, num_beams=5, max_length=512)
        # FIX: must decode inside as_target_tokenizer() context
        with self._tok.as_target_tokenizer():
            decoded = self._tok.batch_decode(out, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True)
        return self._ip.postprocess_batch(decoded, lang=tgt)[0].strip()


def run_tests(translator, languages, texts):
    passed = failed = 0
    for lang in languages:
        if lang == "English":
            continue
        print(f"\n{'='*60}\nLanguage: {lang} ({LANGUAGE_CODES[lang]})\n{'='*60}")
        for tc in texts:
            text = tc["text"]
            print(f"\n  [{tc['label']}]")
            print(f"  EN: {text[:80]}{'...' if len(text) > 80 else ''}")
            t0 = time.time()
            result = translator.translate(text, lang)
            elapsed = time.time() - t0
            if result == text:
                print(f"  FAILED [{elapsed:.1f}s]")
                failed += 1
            else:
                print(f"  OK: {result[:120]}{'...' if len(result) > 120 else ''}")
                print(f"     [{elapsed:.1f}s]")
                passed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="google", choices=["google", "indictrans2"])
    parser.add_argument("--lang",   default=None)
    parser.add_argument("--text",   default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    translator = IndicTranslator(backend=args.backend, device=args.device)
    print(f"[Trans] Backend: {args.backend}")

    if args.text:
        lang = args.lang or "Hindi"
        result = translator.translate(args.text, lang)
        print(f"\nEN:    {args.text}")
        print(f"{lang}: {result}")
        return

    if args.lang and args.lang not in LANGUAGE_CODES:
        print(f"Unknown language. Options: {list(LANGUAGE_CODES.keys())}")
        sys.exit(1)

    languages = [args.lang] if args.lang else list(LANGUAGE_CODES.keys())
    run_tests(translator, languages, TEST_CASES)


if __name__ == "__main__":
    main()