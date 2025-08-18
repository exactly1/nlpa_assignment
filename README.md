# nlpa_assignment
Assignment of NLPA on Neural machine Translation
Neural Machine Translation

Objective:

Develop a Neural Machine Translation (NMT) application that provides real-time translation between Indian languages using a web-based interface. The system should leverage deep learning models for accurate and context-aware translation.

 

Part-A: [10 Marks]

User Interface:
Input Field: Provide a text area where users can input the source text for translation.
Language Selection: Provide dropdown menus to allow users to select the source language and target language. In this implementation we support English, Hindi, and Marathi (Tamil removed per request).
Translate Button: Once the user selects the languages and enters the text, a button should trigger the translation process.
Output Field: Display the translated text clearly on the web page.
Neural Machine Translation Model (Backend Model):
Use Transformer-based models (e.g., Google’s mT5, IndicTrans, or OpenNMT).
Train or fine-tune an NMT model on Indian language datasets (https://www.kaggle.com/datasets/mathurinache/samanantar). Please mention the languages that you have chosen.
The system should handle translation for the selected language pairs (e.g., English to Hindi, Hindi to Marathi, etc.).
The system should handle edge cases, such as empty text, unsupported languages, and text written in English (e.g., Typing "namaste" in English would be converted to "नमस्ते" in Hindi), as well as text that cannot be translated.
Real-Time Translation:
Upon clicking the "Translate" button, the system should provide the translated text instantly, simulating real-time interaction.
Display loading indicators or animation during the translation process to enhance user experience.
Multilingual Support:
The system should support translations between at least 2 language pairs (e.g., English-Hindi, Hindi-English, English-Marathi, Marathi-English).
Evaluation Metrics:
Assess translation accuracy using BLEU, METEOR, or TER scores.
Compare with existing translation systems like Google Translate.
 

Part-B: [5 Marks]

 Provide detailed documentation outlining

Introduction to NMT
What is Neural Machine Translation (NMT)?
How does NMT differ from traditional Statistical Machine Translation (SMT)?
Challenges in Translating Indian Languages
Discuss three major challenges faced in training NMT models for Indian languages.
Provide examples of Indian languages where translation is difficult and explain why.
Recent Advancements
Describe two recent advancements in NMT specific to Indian languages with references.
Mention at least one Indian government or industry initiative working on NMT.
Deliverables:

PART - A

Web Interface: The fully functional web application with frontend and backend integration.
NMT Model Integration: Clear integration with a deep learning model or API that handles the translation.
Source Code: A well-documented code Jupyter Notebook of the code and approach. Ensure all cells are executable.
Testing: Provide a list of test cases to validate various input scenarios, including edge cases.
A set of screenshots that explains the entire flow of the application is to be included in the report for all input cases.

---

CI/CD, Containerization, and Local Setup

- Dockerfile: Containerizes the Streamlit app.
- docker-compose.yml: Local dev with port mapping and Hugging Face cache volume.
- GitHub Actions: .github/workflows/ci.yml installs dependencies, runs tests, and builds the Docker image on push/PR to main.
- setup.ps1: Windows PowerShell script to create a venv and install backend requirements.

Run locally (Windows PowerShell)

```powershell
./setup.ps1
streamlit run backend/app.py
```

Build and run with Docker

```powershell
docker build -t nmt-app .
docker run -p 8501:8501 nmt-app
```

Or using docker-compose:

```powershell
docker-compose up --build
```

Open http://localhost:8501 in your browser.

Fine-tune and use a local model

1) Prepare CSVs with columns src,tgt for the language pair (e.g., en→hi):

data/train.csv
src,tgt
Hello,नमस्ते

2) Run fine-tuning (CPU/GPU):

```powershell
python training/fine_tune_mt.py --src_lang en --tgt_lang hi --train_file data/train.csv --output_dir models/local/en-hi --num_train_epochs 1 --per_device_train_batch_size 8
```

3) App will automatically prefer models in `models/local/<src>-<tgt>` if present. You can also set an explicit override via env var, e.g. `MT_MODEL_en_hi`.

Edge cases and romanized input

- If the source is English and target is Hindi/Marathi, common romanized words (e.g., "Namaste") are auto-detected and transliterated to the target script (e.g., "नमस्ते").
- You can also use the "Use transliteration" checkbox to force ITRANS-based transliteration for romanized input.

Evaluation and Google comparison

- Prepare a CSV file with columns: source_lang,target_lang,src_text,ref_text
- Run the evaluator to compute per-sentence metrics and compare with Google Translate (optional):

```powershell
python scripts/evaluate.py data/sample_eval.csv out/eval_results.csv
```

Note: The Google comparison in the UI and evaluator uses the unofficial `googletrans` package; it may be flaky or rate limited. For production, use the official Google Cloud Translate API.

Notes

- The current model wiring uses generic Hugging Face translation pipelines as placeholders. For production-quality Indian language translation, swap in IndicTrans2 or fine-tuned mBART/mT5 checkpoints and expand PIPELINE_TASKS in `model/nmt_model.py`.
- Transliteration for English→Hindi uses ITRANS→Devanagari.

Docker Hub push

- CI is configured to push to Docker Hub under `exactly1/nmt-app` on push to `main`.
- Create a GitHub Actions secret named `DOCKERHUB_TOKEN` with a Docker Hub access token for the `exactly1` account.
- To push locally after building:

```powershell
docker tag nmt-app:latest exactly1/nmt-app:latest
docker login -u exactly1
docker push exactly1/nmt-app:latest
```