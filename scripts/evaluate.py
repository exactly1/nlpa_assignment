"""
Evaluate translations and compare against Google Translate.

Input CSV format (utf-8):
source_lang,target_lang,src_text,ref_text
English,Hindi,Hello world,नमस्ते दुनिया
...

Outputs a CSV with:
- Our translation + metrics (BLEU/TER/METEOR per-sentence)
- Google translation + metrics (BLEU/TER/METEOR per-sentence, when available)
Also prints corpus-level BLEU/TER and average METEOR for both Our system and Google (when available).
"""
from __future__ import annotations
import csv
import sys
from typing import List, Tuple

from model.nmt_model import translate_text
try:
    from model.google_compare import translate_with_google  # optional
    _HAS_GOOGLE = True
except Exception:
    _HAS_GOOGLE = False

import sacrebleu
try:
    from nltk.translate.meteor_score import meteor_score
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False


def main(in_path: str, out_path: str) -> None:
    rows: List[Tuple[str,str,str,str]] = []
    with open(in_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r['source_lang'], r['target_lang'], r['src_text'], r.get('ref_text','')))

    outputs: List[dict] = []
    hyps: List[str] = []
    google_hyps: List[str] = []
    refs: List[List[str]] = []
    meteors: List[float] = []
    google_meteors: List[float] = []

    for src_lang, tgt_lang, src_text, ref_text in rows:
        result = translate_text(src_text, src_lang, tgt_lang, use_transliteration=True, reference=ref_text)
        hyp = result['translation']
        hyps.append(hyp)
        refs.append([ref_text])

        if _HAS_NLTK and ref_text.strip():
            try:
                meteors.append(float(meteor_score([ref_text], hyp)))
            except Exception:
                meteors.append(float('nan'))
        else:
            meteors.append(float('nan'))

        google_out = None
        if _HAS_GOOGLE:
            try:
                g = translate_with_google(src_text, src_lang, tgt_lang)
            except Exception as e:
                g = f"unavailable: {e}"
        else:
            g = "unavailable"
        google_out = g if not g.lower().startswith("unavailable") and not g.lower().startswith("google translate unavailable") else None

        if google_out is not None:
            google_hyps.append(google_out)
            if _HAS_NLTK and ref_text.strip():
                try:
                    google_meteors.append(float(meteor_score([ref_text], google_out)))
                except Exception:
                    google_meteors.append(float('nan'))
            else:
                google_meteors.append(float('nan'))
        else:
            # keep alignment with refs for corpus scoring; use empty string placeholder
            google_hyps.append("")
            google_meteors.append(float('nan'))

        outputs.append({
            'source_lang': src_lang,
            'target_lang': tgt_lang,
            'src_text': src_text,
            'ref_text': ref_text,
            'our_translation': hyp,
            'google_translation': g,
            'bleu': result['metrics'].get('bleu'),
            'ter': result['metrics'].get('ter'),
            'meteor': result['metrics'].get('meteor'),
            # Google per-row metrics (when reference available and google_out exists)
            'google_bleu': None,
            'google_ter': None,
            'google_meteor': None,
        })

    # Corpus-level metrics
    # sacrebleu expects detok strings; we pass as-is
    has_any_ref = any(r[0].strip() for r in refs)
    corpus_bleu = sacrebleu.corpus_bleu(hyps, refs).score if has_any_ref else None
    ter_metric = sacrebleu.metrics.TER()
    corpus_ter = ter_metric.corpus_score(hyps, refs).score if has_any_ref else None
    avg_meteor = sum(x for x in meteors if x==x) / max(1, sum(1 for x in meteors if x==x))  # ignore NaNs

    # Google corpus metrics (only if we have google outputs and references)
    google_corpus_bleu = sacrebleu.corpus_bleu(google_hyps, refs).score if has_any_ref else None
    google_corpus_ter = ter_metric.corpus_score(google_hyps, refs).score if has_any_ref else None
    google_avg_meteor = sum(x for x in google_meteors if x==x) / max(1, sum(1 for x in google_meteors if x==x)) if _HAS_NLTK else None

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = list(outputs[0].keys()) if outputs else ['source_lang','target_lang','src_text','ref_text','our_translation','google_translation','bleu','ter','meteor','google_bleu','google_ter','google_meteor']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Compute per-row Google metrics when possible (needs reference and non-empty google hypothesis)
        for i, o in enumerate(outputs):
            ref = refs[i][0]
            g_hyp = google_hyps[i]
            if has_any_ref and ref.strip() and g_hyp.strip():
                try:
                    o['google_bleu'] = float(sacrebleu.corpus_bleu([g_hyp], [[ref]]).score)
                    o['google_ter'] = float(ter_metric.corpus_score([g_hyp], [[ref]]).score)
                    if _HAS_NLTK:
                        o['google_meteor'] = float(meteor_score([ref], g_hyp))
                except Exception:
                    pass
            writer.writerow(o)

    print("Our Corpus BLEU:", f"{corpus_bleu:.2f}" if corpus_bleu is not None else '-')
    print("Our Corpus TER:", f"{corpus_ter:.2f}" if corpus_ter is not None else '-')
    if _HAS_NLTK:
        print("Our Avg METEOR:", f"{avg_meteor:.3f}" if avg_meteor==avg_meteor else '-')
    else:
        print("Our Avg METEOR: NLTK not available")

    if has_any_ref:
        print("Google Corpus BLEU:", f"{google_corpus_bleu:.2f}" if google_corpus_bleu is not None else 'unavailable')
        print("Google Corpus TER:", f"{google_corpus_ter:.2f}" if google_corpus_ter is not None else 'unavailable')
        if _HAS_NLTK:
            print("Google Avg METEOR:", f"{google_avg_meteor:.3f}" if google_avg_meteor==google_avg_meteor else 'unavailable')
        else:
            print("Google Avg METEOR: NLTK not available")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python scripts/evaluate.py <input.csv> <output.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
