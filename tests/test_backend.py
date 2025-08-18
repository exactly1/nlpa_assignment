import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.nmt_model import translate_text, SUPPORTED_LANGUAGES


def test_empty_input():
    res = translate_text("   ", "English", "Hindi")
    assert res["translation"] == ""


def test_same_language_noop():
    txt = "hello"
    res = translate_text(txt, "English", "English")
    assert res["translation"] == txt


def test_unsupported_language():
    with pytest.raises(ValueError):
        translate_text("hello", "English", "Spanish")


def test_namastey_transliteration_outputs_namaste_devnagari():
    res = translate_text("Namastey", "English", "Hindi", use_transliteration=True)
    assert res["translation"].strip() == "नमस्ते"
