import numpy as np
import pandas as pd
import nltk
import re
import os
import tarfile
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_files

# Load IMDB dataset
from tensorflow.keras.utils import get_file
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
path = get_file("aclImdb_v1.tar.gz", url, untar=False, cache_dir=".")

# Extract dataset
extract_path = os.path.join(os.path.dirname(path), "aclImdb")
if not os.path.exists(extract_path):
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(os.path.dirname(path))

dataset_path = os.path.join(extract_path, "train")

# Load dataset
reviews = load_files(dataset_path, categories=["pos", "neg"], encoding="utf-8")
X, y = reviews.data, reviews.target

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

X = [preprocess_text(text) for text in X]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a text classification pipeline using Logistic Regression
model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Sentiment Analysis Accuracy: {accuracy:.2f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
