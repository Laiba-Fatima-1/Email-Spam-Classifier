"""
Task 1: Email Spam Classification
Trains a TF-IDF + Multinomial Naive Bayes model on the Kaggle "190K+ Spam | Ham
Email Dataset for Classification" and saves it to disk so the Streamlit app can
load it instantly (no retraining on every run).
"""
import pandas as pd
import joblib
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---------- 1. Load data ----------
df = pd.read_csv("spam_emails_kaggle.csv")
df = df.dropna(subset=["text"])
df["label"] = df["label"].str.strip().str.lower()  # "Spam"/"Ham" -> "spam"/"ham"
print(f"Loaded {len(df)} labeled emails")
print(df["label"].value_counts())

# ---------- 2. Preprocess ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)      # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)             # keep only letters
    text = re.sub(r"\s+", " ", text).strip()          # collapse whitespace
    return text

df["clean_text"] = df["text"].apply(clean_text)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# ---------- 3. Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label_num"], test_size=0.2, random_state=42, stratify=df["label_num"]
)

# ---------- 4. TF-IDF feature extraction ----------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------- 5. Train model ----------
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ---------- 6. Evaluate ----------
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred).tolist()
report = classification_report(y_test, y_pred, target_names=["ham", "spam"])

print("\n=== EVALUATION RESULTS ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print("\nConfusion Matrix (rows=actual, cols=predicted) [ham, spam]:")
print(cm)
print("\n", report)

# ---------- 7. Save everything the app needs ----------
joblib.dump(model, "spam_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "confusion_matrix": cm,
    "train_size": len(X_train),
    "test_size": len(X_test),
    "total_size": len(df),
    "spam_count": int(df["label_num"].sum()),
    "ham_count": int(len(df) - df["label_num"].sum()),
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved: spam_model.joblib, vectorizer.joblib, metrics.json")
