# Test Plan and Screenshot Guide

This document covers:
- A comprehensive list of test cases (with expected outcomes) to validate the application, including edge cases.
- A screenshot capture checklist to document the full flow for all input cases.

## 1) Functional Scope Recap

- Languages: English, Hindi, Marathi (Tamil removed).
- Features: Translation UI (Streamlit), optional transliteration for romanized English -> Indic, metrics (BLEU/TER/METEOR) when reference provided, comparison with Google, CSV history logging, batch evaluator with corpus metrics.

## 2) Test Cases

Legend:
- Pre-req: Any setup or file state needed.
- Steps: What to do in the UI.
- Expected: What should happen.

### A. Input validation and language handling

1. Empty input
- Steps: Source=English, Target=Hindi, leave text empty, click Translate.
- Expected: Warning shown: "Please enter text to translate." No history logged.

2. Same-language selection
- Steps: Source=Hindi, Target=Hindi, text="नमस्ते", click Translate.
- Expected: Info message stating no translation needed; output equals input; no model name; metrics all None; row may still log if implemented (optional), but preferred: no logging.

3. Unsupported language (guard)
- Steps: Not possible via UI dropdown; simulate via API test calling translate_text with unsupported label.
- Expected: Raises ValueError("Unsupported language selected.").

4. Long input (performance)
- Steps: Paste ~1000-character paragraph (English->Hindi) and Translate.
- Expected: Spinner shows; translation returns without crash; reasonable latency (<60s on CPU). Row appended to history.

5. Special characters and punctuation
- Steps: English->Hindi with emojis, quotes, punctuation: "Hello, world! :) — test #123".
- Expected: No crash; model preserves/handles punctuation; output present; history row saved.

### B. Romanized input and transliteration

6. Romanized Namaste (auto-detect)
- Steps: Source=English, Target=Hindi, text="Namaste" (ASCII), do NOT check transliteration, Translate.
- Expected: Auto-detection triggers; output is "नमस्ते"; model_name shows transliteration(ITRANS); history saved.

7. Romanized Namastey (edge normalization)
- Steps: Source=English, Target=Hindi, text="Namastey".
- Expected: Normalization maps to namaste; transliteration yields "नमस्ते"; regression of earlier bug where retroflex/incorrect forms appeared.

8. Force transliteration via checkbox
- Steps: Source=English, Target=Marathi, check "Use transliteration", text="aap kaise ho".
- Expected: Devanagari output; model_name=transliteration(ITRANS); no model call.

9. Mixed romanized and non-ASCII
- Steps: Source=English, Target=Hindi, text contains a Devanagari character plus ASCII tokens.
- Expected: Because not purely ASCII, transliteration path is skipped; model translation used; output non-empty.

### C. Model translation paths

10. Direct pair en->hi
- Steps: English->Hindi sentence.
- Expected: Uses Helsinki en-hi; model_name contains model id; output present; history saved.

11. Direct pair mr->en
- Steps: Marathi->English sentence.
- Expected: Uses Helsinki mr-en; output present; history saved.

12. Pivot path hi->mr (no direct model)
- Steps: Hindi->Marathi sentence.
- Expected: Two-step via English; model_name shows two components; output present; history saved.

### D. Metrics with reference

13. Metrics computed when reference provided
- Steps: Provide reference in second text box; translate.
- Expected: BLEU/TER/METEOR numbers shown (not "-"); reasonable range; history row includes ref_text.

14. No reference -> metrics suppressed
- Steps: Leave reference blank.
- Expected: Metrics section shows dashes; internal values are None.

### E. Google comparison section

15. Compare expander renders
- Steps: Translate any text; open "Compare with Google Translate" expander.
- Expected: Shows our translation and Google translation text; no crash even if Google fails; if rate-limited, Google text may be empty or error handled upstream.

### F. History logging and CSVs

16. History CSV created and appended
- Steps: Perform first translation; verify file at data/historical.csv exists with columns.
- Expected: CSV created with row containing source_lang, target_lang, src_text, ref_text, our_translation.

17. Multiple rows append
- Steps: Do 3 translations.
- Expected: historical.csv has 3 new rows.

### G. Batch evaluator and corpus summary

18. Run evaluation from sidebar
- Pre-req: historical.csv has at least 1 row with ref_text.
- Steps: Click Sidebar -> Evaluate now.
- Expected: Spinner; eval_results.csv created under out/; table shown; corpus BLEU/TER metrics shown for our system and Google.

19. View last results when file exists
- Steps: Sidebar -> View last results.
- Expected: Loads eval_results.csv; shows corpus metrics and top rows; download button available.

20. Evaluate with no history
- Steps: Ensure historical.csv absent; click Evaluate now.
- Expected: Sidebar error: No history CSV yet.

21. Main area Evaluate/Get Score button
- Steps: Click the button in main section.
- Expected: Same behavior as sidebar evaluate.

### H. Error handling and resilience

22. Google rate limit/unavailable
- Steps: Temporarily disconnect network or induce googletrans failure.
- Expected: UI does not crash; Google translation may be empty; evaluator continues and computes our metrics; corpus section handles missing Google gracefully.

23. Missing NLTK resources (METEOR)
- Steps: Uninstall or block NLTK resources.
- Expected: METEOR gracefully None; BLEU/TER still computed.

24. Dockerized run
- Steps: Run via docker-compose up --build; repeat core tests (6,10,18).
- Expected: Same behavior via http://localhost:8501; volumes create /app/data and /app/out mapped to host.

25. Local fine-tuned model override
- Pre-req: Place a model directory under models/local/en-hi with config.json.
- Steps: English->Hindi translate.
- Expected: Model name reflects local path; translation works.

## 3) Automation Hints (Optional)

- Use pytest to automate API-level tests for translate_text, including Namastey normalization and unsupported languages.
- Add a small e2e smoke check: run streamlit in a CI-friendly headless mode and call the translation endpoint via streamlit runtime or by importing app functions (advanced).

## 4) Screenshot Capture Guide

Create a folder `docs/screenshots/` and capture images named with a numeric prefix to preserve order.

Recommended shot list:

1. 01_home_page.png
- Fresh app load with title and language selectors visible.

2. 02_empty_input_warning.png
- After clicking Translate with empty input showing the warning.

3. 03_en_hi_basic_translation.png
- A simple English->Hindi translation result with translated text and model name.

4. 04_metrics_with_reference.png
- Same as above but with the reference box filled and metrics displayed.

5. 05_google_compare_expanded.png
- Expander open showing our vs Google outputs.

6. 06_transliteration_auto_namaste.png
- Input "Namaste" (no checkbox); output shows "नमस्ते" and model_name transliteration.

7. 07_transliteration_edge_namastey.png
- Input "Namastey"; output shows normalized and correct Devanagari.

8. 08_mr_en_translation.png
- Marathi->English translation example.

9. 09_hi_mr_pivot.png
- Hindi->Marathi showing two-model pivot in caption.

10. 10_history_csv_created.png
- A screenshot of the file explorer or console confirming data/historical.csv with sample contents (or show the CSV in an editor).

11. 11_sidebar_evaluate_now.png
- Clicking Evaluate now and spinner visible (or immediately after completion with success toast).

12. 12_corpus_summary_metrics.png
- The corpus BLEU/TER metrics shown for our system and Google.

13. 13_eval_table_and_download.png
- The top rows of the eval table and the download button.

14. 14_view_last_results.png
- Sidebar "View last results" path with the table displayed.

15. 15_docker_compose_up.png
- Terminal showing docker compose up --build success; browser tab http://localhost:8501.

16. 16_error_handling_google.png
- Example where Google side is empty/failed but UI still renders.

Place all images under `docs/screenshots/` and, if submitting a report, embed them in order with short captions referencing the test cases above.

## 5) Submission Checklist

- [ ] All applicable test cases executed manually or via automation.
- [ ] Screenshots captured and stored in `docs/screenshots/` with clear names.
- [ ] `data/historical.csv` contains at least a few rows for evaluation demo.
- [ ] `out/eval_results.csv` generated and downloadable from the UI.
- [ ] README and this document included in submission.
