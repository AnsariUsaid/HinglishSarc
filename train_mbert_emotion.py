import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "hindi_with_emotion.csv"
MODEL_NAME = "bert-base-multilingual-cased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

df = pd.read_csv(DATA_PATH)

df = df.dropna()
df = df.drop_duplicates(subset='text')

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, emotions, tokenizer, max_len):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.emotions = emotions.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = f"{self.texts[idx]} This sentence expresses {self.emotions[idx]} emotion."

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = SarcasmDataset(train_df['text'], train_df['label'], train_df['emotion'], tokenizer, MAX_LEN)
val_dataset = SarcasmDataset(val_df['text'], val_df['label'], val_df['emotion'], tokenizer, MAX_LEN)
test_dataset = SarcasmDataset(test_df['text'], test_df['label'], test_df['emotion'], tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model = model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)

# 🔥 CLASS WEIGHTS (IMPORTANT FIX)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['label']),
    y=train_df['label']
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

def train_epoch(model, loader):
    model.train()
    preds, targets = [], []

    for batch in tqdm(loader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        targets.extend(labels.cpu().numpy())

    return f1_score(targets, preds, average='macro')


def eval_model(model, loader):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    return accuracy_score(targets, preds), f1_score(targets, preds, average='macro')


for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_f1 = train_epoch(model, train_loader)
    val_acc, val_f1 = eval_model(model, val_loader)

    print(f"Train F1: {train_f1:.4f}")
    print(f"Val F1: {val_f1:.4f}")

test_acc, test_f1 = eval_model(model, test_loader)

print("\n================ FINAL RESULT ================")
print("Test Accuracy:", test_acc)
print("Test F1 Score:", test_f1)