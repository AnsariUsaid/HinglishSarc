import os
import torch
import math
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch.nn.functional as F

app = FastAPI(title="Sarcasm Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL LOADING ---
print("Loading Models...")
# 1. Load Emotion Model (Lightweight for fast inference)
# We use a distilbert fine-tuned on GoEmotions or similar. 
try:
    emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=False, device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    print(f"Error loading emotion model, falling back to CPU or ignoring: {e}")
    emotion_pipeline = None

# 2. Load Sarcasm mBERT Model
SARCASM_MODEL_PATH = os.path.join("..", "mbert_baseline_model")
try:
    sarcasm_tokenizer = AutoTokenizer.from_pretrained(SARCASM_MODEL_PATH)
    sarcasm_model = AutoModelForSequenceClassification.from_pretrained(SARCASM_MODEL_PATH)
    sarcasm_model = sarcasm_model.to(DEVICE)
    sarcasm_model.eval()
    SARCASM_LOADED = True
except Exception as e:
    print(f"Error loading Sarcasm model from {SARCASM_MODEL_PATH}: {e}")
    # Fallback to false if model directory isn't setup correctly yet.
    SARCASM_LOADED = False

print("Models Loaded successfully!" if SARCASM_LOADED else "Warning: Sarcasm model failed to load. Predictions unavailable.")

# Pydantic Schemas
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sarcastic: bool
    confidence: float
    emotion: str
    trajectory: list[str]

def split_text_into_three(text: str) -> list[str]:
    words = text.split()
    total_words = len(words)
    if total_words == 0:
        return ["", "", ""]
    elif total_words == 1:
        return [words[0], words[0], words[0]]
    elif total_words == 2:
        return [words[0], words[0] + " " + words[1], words[1]]
    
    chunk_size = math.ceil(total_words / 3)
    part1 = " ".join(words[:chunk_size])
    part2 = " ".join(words[chunk_size:chunk_size*2])
    part3 = " ".join(words[chunk_size*2:])
    return [part1, part2, part3]

@app.post("/predict", response_model=PredictResponse)
async def predict_sarcasm(request: PredictRequest):
    text = request.text.strip()
    
    if not text:
        return PredictResponse(sarcastic=False, confidence=0.0, emotion="neutral", trajectory=["neutral", "neutral", "neutral"])

    # 1. Calculate Emotion Trajectory & Overall Emotion (Only for Visuals)
    if emotion_pipeline:
        overall_pred = emotion_pipeline(text)[0]
        overall_emotion = overall_pred['label']
        
        parts = split_text_into_three(text)
        trajectory = []
        for p in parts:
            if p.strip():
                pred = emotion_pipeline(p)[0]
                trajectory.append(pred['label'])
            else:
                trajectory.append("neutral")
    else:
        overall_emotion = "neutral"
        trajectory = ["neutral", "neutral", "neutral"]

    if not SARCASM_LOADED:
        raise HTTPException(status_code=503, detail="Sarcasm model is not loaded on the backend.")

    # 2. Sarcasm model inference on raw input text.
    with torch.no_grad():
        inputs = sarcasm_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(DEVICE)
        attention_mask = inputs['attention_mask'].to(DEVICE)
        
        outputs = sarcasm_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        is_sarcastic = bool(pred_class == 1)
        
    return PredictResponse(
        sarcastic=is_sarcastic,
        confidence=round(confidence, 4),
        emotion=overall_emotion,
        trajectory=trajectory
    )

# Serve static files for frontend
# Ensure 'static' folder exists inside 'webapp'
app.mount("/", StaticFiles(directory="static", html=True), name="static")
