import pandas as pd
import re

sarcastic = pd.read_csv("Sarcasm_Hindi_Tweets-SARCASTIC.csv")
non_sarcastic = pd.read_csv("Sarcasm_Hindi_Tweets-NON-SARCASTIC.csv")

sarcastic['label'] = 1
non_sarcastic['label'] = 0

df = pd.concat([sarcastic, non_sarcastic], ignore_index=True)
df = df[['text', 'label']]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"sarcasm|sarcastic|irony|कटाक्ष|kataksh", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\u0900-\u097F\s]", "", text)
    return text

df['text'] = df['text'].apply(clean_text)

df = df.dropna()
df = df.drop_duplicates(subset='text')

df.to_csv("hindi_dataset_clean_v2.csv", index=False)

print("Cleaned dataset:", df.shape)