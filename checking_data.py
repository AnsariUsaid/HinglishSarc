import pandas as pd
from transformers import pipeline

df = pd.read_csv("sarcasm_hinghlish_dataset.csv")

classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

def get_emotion(text):
    try:
        result = classifier(str(text))[0][0]['label']
        return result.lower()
    except:
        return "neutral"

df['emotion'] = df['text'].apply(get_emotion)

print(df[['text', 'emotion']].head())

df.to_csv("sarcasm_with_emotion.csv", index=False)