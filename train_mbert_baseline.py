"""
mBERT Clean Baseline for Hinglish Sarcasm Detection
====================================================
Phase 1: Establish clean baseline without data leakage

Dataset: sarcasm_clean_dedup.csv (8,946 unique samples)
Model: bert-base-multilingual-cased
Expected F1: ~70-78% (realistic baseline)

Usage:
    python train_mbert_baseline.py

Output:
    - mbert_baseline_model/ (trained model checkpoint)
    - results.json (test metrics)
    - training.log (detailed logs)
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────
MODEL_NAME   = 'bert-base-multilingual-cased'
MAX_LEN      = 128
BATCH_SIZE   = 32
EPOCHS       = 5
LR           = 2e-5
SEED         = 42
DATA_PATH = "hindi_dataset_clean.csv"
SAVE_DIR     = 'mbert_baseline_model'

# ────────────────────────────────────────────
# SETUP
# ────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*70)
print("mBERT BASELINE TRAINING - HINGLISH SARCASM DETECTION")
print("="*70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {device}")
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATA_PATH}")
print("="*70)

# ────────────────────────────────────────────
# LOAD DATA
# ────────────────────────────────────────────
print("\n[1/6] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Total samples: {len(df)}")
print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

# Stratified 70/15/15 split
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=SEED, stratify=df['label']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED, stratify=temp_df['label']
)

print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ────────────────────────────────────────────
# DATASET CLASS
# ────────────────────────────────────────────
class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }

# ────────────────────────────────────────────
# PREPARE DATALOADERS
# ────────────────────────────────────────────
print("\n[2/6] Preparing dataloaders...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

train_dataset = SarcasmDataset(train_df['text'], train_df['label'], tokenizer, MAX_LEN)
val_dataset = SarcasmDataset(val_df['text'], val_df['label'], tokenizer, MAX_LEN)
test_dataset = SarcasmDataset(test_df['text'], test_df['label'], tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# ────────────────────────────────────────────
# INITIALIZE MODEL
# ────────────────────────────────────────────
print("\n[3/6] Initializing model...")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"  Total training steps: {total_steps}")

# ────────────────────────────────────────────
# TRAINING FUNCTIONS
# ────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    preds_all, labels_all = [], []

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds_all.extend(torch.argmax(logits, dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average='macro')
    return avg_loss, acc, f1

def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds_all.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average='macro')
    return avg_loss, acc, f1, preds_all, labels_all

# ────────────────────────────────────────────
# TRAINING LOOP
# ────────────────────────────────────────────
print("\n[4/6] Training...")
print("-"*70)

history = {
    'train_loss': [], 'train_acc': [], 'train_f1': [],
    'val_loss': [], 'val_acc': [], 'val_f1': []
}
best_val_f1 = 0.0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc, train_f1 = train_epoch(
        model, train_loader, optimizer, scheduler, device
    )
    val_loss, val_acc, val_f1, _, _ = eval_epoch(model, val_loader, device)

    history['train_loss'].append(float(train_loss))
    history['train_acc'].append(float(train_acc))
    history['train_f1'].append(float(train_f1))
    history['val_loss'].append(float(val_loss))
    history['val_acc'].append(float(val_acc))
    history['val_f1'].append(float(val_f1))

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        os.makedirs(SAVE_DIR, exist_ok=True)
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"  ✓ Best model saved (Val F1: {best_val_f1:.4f})")

print(f"\nTraining complete! Best validation F1: {best_val_f1:.4f}")

# ────────────────────────────────────────────
# TEST EVALUATION
# ────────────────────────────────────────────
print("\n[5/6] Evaluating on test set...")
print("-"*70)

best_model = BertForSequenceClassification.from_pretrained(SAVE_DIR).to(device)
test_loss, test_acc, test_f1, test_preds, test_labels = eval_epoch(
    best_model, test_loader, device
)

print(f"\nTEST RESULTS:")
print(f"  Loss:         {test_loss:.4f}")
print(f"  Accuracy:     {test_acc:.4f}")
print(f"  Macro F1:     {test_f1:.4f}")
print(f"\nDetailed Classification Report:")
print(classification_report(
    test_labels, test_preds,
    target_names=['Not Sarcastic (0)', 'Sarcastic (1)'],
    digits=4
))

cm = confusion_matrix(test_labels, test_preds)
print("Confusion Matrix:")
print(cm)
print(f"  TN: {cm[0,0]} | FP: {cm[0,1]}")
print(f"  FN: {cm[1,0]} | TP: {cm[1,1]}")

# Per-class metrics
prec, rec, f1_per_class, support = precision_recall_fscore_support(
    test_labels, test_preds, average=None, labels=[0, 1]
)

# ────────────────────────────────────────────
# SAVE RESULTS
# ────────────────────────────────────────────
print("\n[6/6] Saving results...")

results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model': MODEL_NAME,
    'dataset': DATA_PATH,
    'dataset_stats': {
        'total_samples': len(df),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'label_0_count': int(df['label'].value_counts()[0]),
        'label_1_count': int(df['label'].value_counts()[1]),
    },
    'hyperparameters': {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'max_length': MAX_LEN,
        'seed': SEED,
    },
    'training': {
        'best_val_f1': float(best_val_f1),
        'history': history,
    },
    'test_metrics': {
        'loss': float(test_loss),
        'accuracy': float(test_acc),
        'macro_f1': float(test_f1),
        'class_0': {
            'precision': float(prec[0]),
            'recall': float(rec[0]),
            'f1': float(f1_per_class[0]),
            'support': int(support[0]),
        },
        'class_1': {
            'precision': float(prec[1]),
            'recall': float(rec[1]),
            'f1': float(f1_per_class[1]),
            'support': int(support[1]),
        },
        'confusion_matrix': cm.tolist(),
    }
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"  ✓ Results saved to results.json")
print(f"  ✓ Model saved to {SAVE_DIR}/")

print("\n" + "="*70)
print("BASELINE COMPLETE!")
print("="*70)
print(f"Final Test F1 (Macro): {test_f1:.4f}")
print(f"This is your clean baseline for Phase 2 (emotion-aware model)")
print("="*70)
