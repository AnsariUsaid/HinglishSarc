import pandas as pd
import re

df = pd.read_csv("final_dataset.csv")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"#sarcasm|sarcasm|irony|sarcastic", "", text)
    return text

df['text'] = df['text'].apply(clean_text)

df.to_csv("cleaned_final_dataset.csv", index=False)

print("Cleaning done")