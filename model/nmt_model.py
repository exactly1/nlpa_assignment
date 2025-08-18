"""
nmt_model.py: Core NMT model logic for translation and evaluation

This module loads transformer-based models (Helsinki-NLP opus models by default) and provides
translation between supported Indian languages with optional English pivot. It also supports
basic transliteration for Eng→Indic and optional evaluation metrics (BLEU, TER, METEOR) when
reference translations are provided.
"""

from typing import Dict, Tuple, Optional, List

import os
import re
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import sacrebleu

try:
    import nltk
    from nltk.translate.meteor_score import meteor_score
    _NLTK_OK = True
except Exception:
    _NLTK_OK = False

# Supported languages for the UI
SUPPORTED_LANGUAGES = ["English", "Hindi", "Marathi"]

# Map friendly names to ISO codes
LANG_CODE = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
}

# Direct model names for some language pairs (Helsinki-NLP opus MT)
MODEL_MAP: Dict[Tuple[str, str], str] = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "mr"): "Helsinki-NLP/opus-mt-en-mr",
    ("mr", "en"): "Helsinki-NLP/opus-mt-mr-en",
    # Some direct Indic↔Indic models may exist; if not, we'll pivot via English.
    # ("hi", "mr"): "Helsinki-NLP/opus-mt-hi-mr",  # Uncomment if available
    # ("mr", "hi"): "Helsinki-NLP/opus-mt-mr-hi",
}

# Optional: allow overriding a pair with a fine-tuned model path via env var, e.g. MT_MODEL_en_hi
def _override_from_env(src: str, tgt: str) -> Optional[str]:
    key = f"MT_MODEL_{src}_{tgt}"
    return os.environ.get(key)


def _local_override_path(src: str, tgt: str) -> Optional[str]:
    """If a fine-tuned local model exists under models/local/<src>-<tgt>, use it."""
    root = Path(os.environ.get("LOCAL_MODEL_ROOT", "models/local"))
    path = root / f"{src}-{tgt}"
    if path.exists() and path.is_dir():
        # basic sanity: has config.json or pytorch_model.bin
        if any((path / f).exists() for f in ("config.json", "pytorch_model.bin", "model.safetensors")):
            return str(path)
    return None


_pipelines: Dict[Tuple[str, str], any] = {}


def _get_pipeline(src: str, tgt: str):
    key = (src, tgt)
    if key in _pipelines:
        return _pipelines[key], MODEL_MAP.get(key, "custom")
    override = _override_from_env(src, tgt)
    local_path = _local_override_path(src, tgt)
    model_name = override or local_path or MODEL_MAP.get(key)
    if model_name:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        nlp = pipeline("translation", model=mdl, tokenizer=tok)
        _pipelines[key] = nlp
        return nlp, model_name
    # Fallback to generic pipeline task if nothing mapped (least preferred)
    task = f"translation_{src}_to_{tgt}"
    nlp = pipeline(task)
    _pipelines[key] = nlp
    return nlp, task


def _pivot_translate(text: str, src: str, mid: str, tgt: str) -> Tuple[str, List[str]]:
    """Translate via a pivot language (typically English). Returns translation and list of model names used."""
    nlp1, name1 = _get_pipeline(src, mid)
    out1 = nlp1(text)
    tmp = out1[0].get("translation_text", "") if isinstance(out1, list) else out1.get("translation_text", "")

    nlp2, name2 = _get_pipeline(mid, tgt)
    out2 = nlp2(tmp)
    final = out2[0].get("translation_text", "") if isinstance(out2, list) else out2.get("translation_text", "")
    return final, [name1, name2]


def _normalize_for_transliteration(text: str) -> str:
    """Normalize common romanized variants to improve ITRANS transliteration accuracy.

    - Lowercase to avoid ITRANS interpreting capital letters as retroflex (e.g., 'Na' -> ण)
    - Fix specific frequent variants like 'namastey' -> 'namaste'.
    """
    t = text.strip()
    # Lowercase to keep dental consonants (n -> न) instead of retroflex (ṇ -> ण)
    t = t.lower()
    # Targeted fixes
    # Replace whole-word 'namastey' with 'namaste'
    t = re.sub(r"\bnamastey\b", "namaste", t)
    return t


def transliterate_english_to_script(text: str, target_lang: str) -> str:
    """Transliterate romanized input (ITRANS-like) to given Indic script."""
    norm = _normalize_for_transliteration(text)
    if target_lang == "Hindi":
        return transliterate(norm, sanscript.ITRANS, sanscript.DEVANAGARI)
    if target_lang == "Marathi":
        return transliterate(norm, sanscript.ITRANS, sanscript.DEVANAGARI)  # Marathi uses Devanagari
    return norm


# Simple romanized Hindi/Marathi hint tokens to trigger transliteration when appropriate
_ROMANIZED_HINTS = {
    "namaste", "namastey", "namaskar", "shukriya", "dhanyavad", "pranam", "kripya", "maaf", "sach", "dost",
    "pyaar", "pyar", "dil", "sab", "bhai", "behen", "pita", "maa", "matra", "aap", "hum",
}


def evaluate_translation(hypothesis: str, reference: Optional[str]) -> Dict[str, Optional[float]]:
    """Compute BLEU, TER, METEOR given a reference string (if provided)."""
    if not reference or not reference.strip():
        return {"bleu": None, "ter": None, "meteor": None}
    refs = [reference]
    bleu = float(sacrebleu.corpus_bleu([hypothesis], [refs]).score)
    ter_metric = sacrebleu.metrics.TER()
    ter = float(ter_metric.corpus_score([hypothesis], [refs]).score)
    meteor = None
    if _NLTK_OK:
        try:
            # nltk meteor expects list of reference strings
            meteor = float(meteor_score([reference], hypothesis))
        except Exception:
            meteor = None
    return {"bleu": bleu, "ter": ter, "meteor": meteor}


def translate_text(text: str, source_lang: str, target_lang: str, use_transliteration: bool = False, reference: Optional[str] = None):
    if not text.strip():
        return {"translation": "", "model_name": None, "metrics": {"bleu": None, "ter": None, "meteor": None}}
    if source_lang not in SUPPORTED_LANGUAGES or target_lang not in SUPPORTED_LANGUAGES:
        raise ValueError("Unsupported language selected.")
    if source_lang == target_lang:
        return {"translation": text, "model_name": None, "metrics": {"bleu": None, "ter": None, "meteor": None}}

    src = LANG_CODE[source_lang]
    tgt = LANG_CODE[target_lang]

    # Optional transliteration mode for Eng→Indic when input is romanized; also auto-trigger for common words
    if source_lang == "English" and target_lang in ("Hindi", "Marathi") and text.isascii():
        tokens = [t.strip(".,!?;:\"'()[]{}-").lower() for t in text.split()]
        looks_romanized = any(t in _ROMANIZED_HINTS for t in tokens)
        if use_transliteration or looks_romanized:
            translated = transliterate_english_to_script(text, target_lang)
            models_used: List[str] = ["transliteration(ITRANS)"]
        else:
            # fall through to model translation
            translated = None  # type: ignore
            models_used = []
    else:
        translated = None  # type: ignore
        models_used = []

    # Direct model or pivot via English (if not already handled by transliteration branch)
    if translated is None:
        if (src, tgt) in MODEL_MAP or _override_from_env(src, tgt) or _local_override_path(src, tgt):
            nlp, model_name = _get_pipeline(src, tgt)
            out = nlp(text)
            translated = out[0].get("translation_text", "") if isinstance(out, list) else out.get("translation_text", "")
            models_used = [model_name]
        else:
            translated, used = _pivot_translate(text, src, "en", tgt)
            models_used = used

    metrics = evaluate_translation(translated, reference)
    return {"translation": translated, "model_name": " + ".join(models_used), "metrics": metrics}
