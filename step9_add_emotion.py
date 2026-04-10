import pandas as pd
from transformers import pipeline

df = pd.read_csv("hindi_dataset_clean_v2.csv")

classifier = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-emotion",
    top_k=1
)

def get_emotion(text):
    try:
        return classifier(str(text))[0][0]['label']
    except:
        return "neutral"

df['emotion'] = df['text'].apply(get_emotion)

df.to_csv("hindi_with_emotion.csv", index=False)

print("Emotion added successfully")