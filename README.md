# HinglishSarc

Hinglish/Hindi sarcasm detection project with a FastAPI web demo and model-training scripts.

## Project Structure

- `webapp/`: FastAPI app + static frontend (`index.html`, `script.js`, `style.css`)
- `mbert_baseline_model/`: mBERT tokenizer/model artifacts (Git LFS tracked)
- `train_*.py`, `step*.py`: data preparation and training pipelines
- `*.csv`: prepared datasets used by scripts and experiments

## Quick Start (Web Demo)

1. Create and activate a virtual environment inside `webapp`:
   - `python3 -m venv webapp/venv`
   - `source webapp/venv/bin/activate`
2. Install dependencies:
   - `pip install -r webapp/requirements.txt`
3. Ensure model artifacts are available (Git LFS):
   - `git lfs install`
   - `git lfs pull`
4. Start the app:
   - `cd webapp`
   - `venv/bin/uvicorn main:app --reload`

Open: `http://127.0.0.1:8000`

## API

- `POST /predict`
  - Request: `{ "text": "your input text" }`
  - Response: `{ sarcastic, confidence, emotion, trajectory }`

## Notes

- `webapp/venv/` is intentionally ignored in Git.
- If `model.safetensors` is a tiny text pointer, run Git LFS pull again.
