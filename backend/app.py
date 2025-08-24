"""
app.py: Streamlit frontend and backend integration for Neural Machine Translation (NMT)

This script provides a web interface for real-time translation between Indian languages using a transformer-based NMT model.
It handles user input, language selection, translation, and displays results with loading indicators.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import sacrebleu
import streamlit as st

# Ensure project root is on sys.path so `model` package is importable when running from repo root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.nmt_model import translate_text, SUPPORTED_LANGUAGES
from model.google_compare import compare_to_google



# Use /tmp/data and /tmp/out as default writable directories for Streamlit Cloud
DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/data"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/tmp/out"))
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    st.error(f"Cannot create DATA_DIR at {DATA_DIR}. Please check permissions.")
try:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    st.error(f"Cannot create OUT_DIR at {OUT_DIR}. Please check permissions.")
HIST_CSV = DATA_DIR / "historical.csv"
EVAL_CSV = OUT_DIR / "eval_results.csv"


def _show_corpus_summary(df_eval: pd.DataFrame):
    # Filter rows with references
    df_ref = df_eval[df_eval['ref_text'].fillna('').astype(str).str.strip() != ''].copy()
    if df_ref.empty:
        st.info("No references found in history; corpus metrics unavailable.")
        return
    refs = [[r] for r in df_ref['ref_text'].astype(str).tolist()]

    # Our metrics
    our_hyps = df_ref['our_translation'].astype(str).tolist()
    our_bleu = sacrebleu.corpus_bleu(our_hyps, refs).score
    ter_metric = sacrebleu.metrics.TER()
    our_ter = ter_metric.corpus_score(our_hyps, refs).score

    # Google metrics
    google_hyps = df_ref['google_translation'].fillna('').astype(str).tolist()
    google_bleu = sacrebleu.corpus_bleu(google_hyps, refs).score
    google_ter = ter_metric.corpus_score(google_hyps, refs).score

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Our system (corpus):**")
        st.metric("BLEU", f"{our_bleu:.2f}")
        st.metric("TER", f"{our_ter:.2f}")
    with c2:
        st.markdown("**Google (corpus):**")
        st.metric("BLEU", f"{google_bleu:.2f}")
        st.metric("TER", f"{google_ter:.2f}")


def _run_evaluation_and_display():
    from scripts.evaluate import main as eval_main
    eval_main(str(HIST_CSV), str(EVAL_CSV))
    df_eval = pd.read_csv(EVAL_CSV)
    _show_corpus_summary(df_eval)
    st.dataframe(df_eval.head(50))
    st.download_button("Download evaluation CSV", data=df_eval.to_csv(index=False), file_name="eval_results.csv", mime="text/csv")


st.set_page_config(page_title="Indian Language NMT Translator", layout="centered")
st.title("Neural Machine Translation for Indian Languages")

# Sidebar for language selection
st.sidebar.header("Language Selection")
source_lang = st.sidebar.selectbox("Source Language", SUPPORTED_LANGUAGES)
target_lang = st.sidebar.selectbox("Target Language", SUPPORTED_LANGUAGES)
use_translit = st.sidebar.checkbox("Use transliteration for romanized input (Eng→Indic)")

st.sidebar.markdown("---")
st.sidebar.header("Evaluation")
if st.sidebar.button("Evaluate now"):
    if not HIST_CSV.exists():
        st.sidebar.error("No history CSV yet. Generate translations first.")
    else:
        with st.spinner("Evaluating…"):
            try:
                _run_evaluation_and_display()
                st.success("Evaluation complete.")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

if st.sidebar.button("View last results"):
    if not EVAL_CSV.exists():
        st.sidebar.info("No evaluation results found.")
    else:
        df_eval = pd.read_csv(EVAL_CSV)
        _show_corpus_summary(df_eval)
        st.dataframe(df_eval.head(50))
        st.download_button("Download evaluation CSV", data=df_eval.to_csv(index=False), file_name="eval_results.csv", mime="text/csv")


# Main input area
st.subheader("Enter text to translate:")
input_text = st.text_area("Source Text", height=150)
ref_text = st.text_area("Optional: Reference translation (for metrics)", height=100)

if st.button("Translate"):
    if not input_text.strip():
        st.warning("Please enter text to translate.")
    elif source_lang == target_lang:
        st.info("Source and target languages are the same. No translation needed.")
    else:
        with st.spinner("Translating..."):
            try:
                result = translate_text(input_text, source_lang, target_lang, use_transliteration=use_translit, reference=ref_text)
                st.success("Translation complete!")
                st.markdown(f"**Translated Text:**\n\n{result['translation']}")
                if result.get('model_name'):
                    st.caption(f"Model(s): {result['model_name']}")
                metrics = result.get('metrics') or {}
                if any(metrics.get(k) is not None for k in ('bleu','ter','meteor')):
                    st.markdown("**Evaluation metrics (vs reference):**")
                    cols = st.columns(3)
                    cols[0].metric("BLEU", f"{metrics.get('bleu'):.2f}" if metrics.get('bleu') is not None else "-")
                    cols[1].metric("TER", f"{metrics.get('ter'):.2f}" if metrics.get('ter') is not None else "-")
                    cols[2].metric("METEOR", f"{metrics.get('meteor'):.3f}" if metrics.get('meteor') is not None else "-")

                # Optional: side-by-side comparison with Google Translate
                with st.expander("Compare with Google Translate"):
                    comp = compare_to_google(input_text, source_lang, target_lang, result['translation'])
                    st.write("Ours:")
                    st.code(comp["ours"])
                    st.write("Google Translate:")
                    st.code(comp["google"])

                # Append to history CSV
                try:
                    df_row = pd.DataFrame([
                        {
                            "source_lang": source_lang,
                            "target_lang": target_lang,
                            "src_text": input_text,
                            "ref_text": ref_text or "",
                            "our_translation": result['translation'],
                        }
                    ])
                    if HIST_CSV.exists():
                        df_hist = pd.read_csv(HIST_CSV)
                        df_hist = pd.concat([df_hist, df_row], ignore_index=True)
                        df_hist.to_csv(HIST_CSV, index=False)
                    else:
                        df_row.to_csv(HIST_CSV, index=False)
                    st.caption(f"Logged to {HIST_CSV}")
                except Exception as log_e:
                    st.warning(f"Could not log history: {log_e}")
            except Exception as e:
                st.error(f"Translation failed: {str(e)}")


st.markdown("---")

# Evaluation section in main area
st.subheader("Batch Evaluation against History CSV")
col1, col2 = st.columns([1, 1])
with col1:
    st.write(f"History CSV path: {HIST_CSV}")
with col2:
    st.write(f"Output CSV path: {EVAL_CSV}")

if st.button("Evaluate/Get Score"):
    if not HIST_CSV.exists():
        st.error("No history CSV found yet. Generate some translations first.")
    else:
        with st.spinner("Running evaluation... this may take time on first run"):
            try:
                _run_evaluation_and_display()
                st.success("Evaluation complete.")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")


st.markdown("Developed for NLPA Assignment | Powered by Transformers")
