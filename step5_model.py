import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

df = pd.read_csv("final_dataset.csv")

df = df.drop_duplicates(subset='text')

X_text = df['text']
X_emotion = df[['emotion_shift']]
y = df['label']

vectorizer = TfidfVectorizer(max_features=5000)
X_text_vec = vectorizer.fit_transform(X_text)

X_combined = hstack([X_text_vec, X_emotion.values])

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))