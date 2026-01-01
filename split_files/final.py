# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

print("NLTK resources downloaded!")

def read_props(config_path=None):
    if config_path is None:
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.properties'))
    props = {}
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    props[k.strip()] = v.strip()
    except FileNotFoundError:
        props = {}
    return props

props = read_props()

# Load the raw dataset
df = pd.read_csv(props.get('spam_assassin_csv', r"C:\Users\suhas\Downloads\spam_assassin.csv"))

# Rename columns if needed
df.columns = ["email_text", "label"]  # First column is text, second is spam/ham

print("Dataset loaded!")

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

print("NLTK tools initialised!")

""" Function to clean email text by removing HTML, punctuation and applying lemmatization. """
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    # Lemmatization & Stopword removal
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Apply text cleaning
df["cleaned_text"] = df["email_text"].apply(clean_text)

# Save the cleaned data for later use
df[["cleaned_text", "label"]].to_csv(props.get('spam_assassin_cleaned_csv', r"C:\Users\suhas\Downloads\spam_assassin_cleaned.csv"), index=False)

print("Dataset is cleaned and stored for further use!")

# --- Step 2: Convert Text into Numerical Representation using TF-IDF ---
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df["cleaned_text"])
y = df["label"]

print("TF-IDF done!")

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

print("Dataset split!")

# --- Step 3: Train Naïve Bayes Model ---
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

print("Naïve Bayes Model training complete!")

# --- Step 4: Make Predictions ---
y_pred = nb_model.predict(X_test)

# --- Step 5: Evaluate the Model ---
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Save the trained model for later use
joblib.dump(nb_model, props.get('spam_classifier_model', r"C:\Users\suhas\Downloads\spam_classifier_nb_model.pkl"))
joblib.dump(tfidf_vectorizer, props.get('tfidf_vectorizer', r"C:\Users\suhas\Downloads\tfidf_vectorizer.pkl"))

print("\nTrained model saved as 'spam_classifier_nb_model.pkl'")
print("TF-IDF vectorizer saved as 'tfidf_vectorizer.pkl'")

