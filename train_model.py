# ============================================================
# Sentiment Analysis Model — Training Script
# Dataset: IMDB 50k Movie Reviews
# Model: TF-IDF + Logistic Regression Pipeline
# Accuracy: ~90% on test set
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from bs4 import BeautifulSoup
import pandas as pd
import joblib

# --- Load dataset ---
print("Loading dataset...")
data = pd.read_csv("IMDB Dataset.csv")

# --- Clean HTML tags from reviews ---
print("Cleaning HTML tags...")
sentences = data["review"].apply(
    lambda x: BeautifulSoup(x, "html.parser").get_text()
)

# --- Encode labels: positive=1, negative=0 ---
labels = data["sentiment"].map({"positive": 1, "negative": 0})

# --- Train / test split (80/20) ---
X_train, X_test, y_train, y_test = train_test_split(
    sentences, labels, test_size=0.2, random_state=42
)

# --- Build pipeline: TF-IDF vectorizer + classifier ---
model = Pipeline([
    ("tfidf", TfidfVectorizer()),      # converts text to numerical features
    ("clf", LogisticRegression()),     # classifies positive or negative
])

# --- Train ---
print("Training model...")
model.fit(X_train, y_train)

# --- Evaluate ---
predictions = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, predictions):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# --- Save model ---
joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")