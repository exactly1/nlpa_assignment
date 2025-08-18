"""
google_compare.py: Compare our NMT outputs with Google Translate results.

Uses googletrans (unofficial) by default. If you have an official Google Cloud
Translate API key, you can extend this to use google-cloud-translate instead.
"""
from typing import Dict

import os

Translator = None  # type: ignore
_HAS_GOOGLETRANS = False
_IMPORT_ERR = None
try:
    # Unofficial, lightweight. May be flaky.
    from googletrans import Translator as _GTTranslator  # type: ignore
    Translator = _GTTranslator
    _HAS_GOOGLETRANS = True
except ImportError as e:  # package not installed in current env
    _IMPORT_ERR = e
except Exception as e:  # installed but import failed (runtime compatibility)
    _IMPORT_ERR = e

LANG_CODE = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
}


def translate_with_google(text: str, source_lang: str, target_lang: str) -> str:
    # Lazy import attempt if initial import failed
    global Translator, _HAS_GOOGLETRANS
    if not _HAS_GOOGLETRANS or Translator is None:
        try:
            from googletrans import Translator as _GTTranslator  # type: ignore
            Translator = _GTTranslator
            _HAS_GOOGLETRANS = True
        except Exception as e:
            raise RuntimeError(f"googletrans unavailable: {e}. Ensure 'googletrans==4.0.0rc1' is installed in this environment.")
    src = LANG_CODE[source_lang]
    tgt = LANG_CODE[target_lang]
    try:
        tr = Translator()
        result = tr.translate(text, src=src, dest=tgt)
        return result.text
    except Exception as e:
        raise RuntimeError(f"googletrans request failed: {e}")


def compare_to_google(text: str, source_lang: str, target_lang: str, our_translation: str) -> Dict[str, str]:
    try:
        google_out = translate_with_google(text, source_lang, target_lang)
    except Exception as e:
        google_out = f"Google Translate unavailable: {e}"
    return {
        "ours": our_translation,
        "google": google_out,
    }
