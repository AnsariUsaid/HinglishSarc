import pandas as pd

df = pd.read_csv("sarcasm_with_emotion.csv")

emotion_map = {
    'joy': 0,
    'anger': 1,
    'sadness': 2,
    'fear': 3,
    'surprise': 4,
    'neutral': 5,
    'love': 6,
    'disgust': 7,
    'admiration': 8,
    'disapproval': 9
}

df['emotion_encoded'] = df['emotion'].map(emotion_map)

df['emotion_shift'] = df['emotion_encoded'].diff().abs()
df['emotion_shift'] = df['emotion_shift'].fillna(0)

print(df.head())

df.to_csv("final_dataset.csv", index=False)