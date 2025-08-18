"""
google_compare.py: Compare our NMT outputs with Google Translate results.

Uses googletrans (unofficial) by default. If you have an official Google Cloud
Translate API key, you can extend this to use google-cloud-translate instead.
"""
from typing import Dict

import os

try:
    # Unofficial, lightweight. May be flaky.
    from googletrans import Translator  # type: ignore
    _HAS_GOOGLETRANS = True
except Exception:
    _HAS_GOOGLETRANS = False

LANG_CODE = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
}


def translate_with_google(text: str, source_lang: str, target_lang: str) -> str:
    if not _HAS_GOOGLETRANS:
        raise RuntimeError("googletrans package not installed. Install 'googletrans==4.0.0rc1'.")
    src = LANG_CODE[source_lang]
    tgt = LANG_CODE[target_lang]
    tr = Translator()
    result = tr.translate(text, src=src, dest=tgt)
    return result.text


def compare_to_google(text: str, source_lang: str, target_lang: str, our_translation: str) -> Dict[str, str]:
    try:
        google_out = translate_with_google(text, source_lang, target_lang)
    except Exception as e:
        google_out = f"Google Translate unavailable: {e}"
    return {
        "ours": our_translation,
        "google": google_out,
    }
