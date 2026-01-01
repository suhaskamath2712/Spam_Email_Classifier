import os
import re
import string
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
import nltk
from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Optional: SMOTE
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False

# Ensure necessary NLTK resources are available
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Configuration loader
# `read_props()` reads a simple key=value properties file named
# `config.properties` placed next to this script. It returns a dict of keys
# and values so other parts of the program can use configurable default
# paths (model, vectorizer, intermediate CSVs, etc.). If the file is
# missing the program falls back to sensible defaults.
# ---------------------------------------------------------------------------


def read_props(config_path=None):
    if config_path is None:
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.properties'))
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


PROPS = read_props()

# ---------------------------------------------------------------------------
# Text cleaning pipeline
# `clean_text()` performs the standard preprocessing for email/text data:
#  - HTML tag removal via BeautifulSoup
#  - lowercasing
#  - URL and email address removal
#  - number and punctuation stripping
#  - tokenization, stopword removal and lemmatization
# The function returns a cleaned single-string suitable for TF-IDF.
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Clean email text: remove HTML, URLs, emails, punctuation, numbers,
    tokenize, remove stopwords and lemmatize."""
    if pd.isna(text):
        return ""
    # Remove HTML
    try:
        text = BeautifulSoup(text, "html.parser").get_text()
    except Exception:
        pass
    # Lowercase
    text = text.lower()
    # Remove URLs and emails
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S*@\S*', '', text)
    # Remove numbers and punctuation
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ---------------------------------------------------------------------------
# TF-IDF feature construction
# `build_tfidf()` accepts a CSV path containing cleaned text and returns
# the fitted `TfidfVectorizer` and a DataFrame of TF-IDF features. If
# `output_tfidf_csv` is supplied the TF-IDF matrix will be saved to disk.
# The function tries to infer the text and label columns from common names
# if they are not provided explicitly.
# ---------------------------------------------------------------------------


def build_tfidf(input_csv: str, output_tfidf_csv: str = None, text_col: str = None, max_features: int = 5000):
    """Read cleaned CSV and produce TF-IDF matrix saved as CSV and return vectorizer and dataframe.
    If text_col is None, will try common column names."""
    df = pd.read_csv(input_csv)
    # Infer text column
    if text_col is None:
        for candidate in ['cleaned_email_text', 'cleaned_text', 'email_text', 'text']:
            if candidate in df.columns:
                text_col = candidate
                break
    if text_col is None:
        raise ValueError('Could not find a text column in CSV; specify text_col')
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_col].fillna(''))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    # Try to find label column
    label_col = None
    for candidate in ['label', 'target', 'y']:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is not None:
        tfidf_df[label_col] = df[label_col].values
    if output_tfidf_csv:
        tfidf_df.to_csv(output_tfidf_csv, index=False)
    return tfidf_vectorizer, tfidf_df

# ---------------------------------------------------------------------------
# Model training helper
# `train_model_from_tfidf()` trains a Multinomial Naive Bayes classifier
# on the provided TF-IDF DataFrame and returns the trained model and a
# dictionary of evaluation metrics. The trained model and vectorizer are
# persisted using joblib to the provided paths.
# ---------------------------------------------------------------------------


def train_model_from_tfidf(tfidf_df: pd.DataFrame, label_col: str = None, save_model: str = 'spam_classifier_nb_model.pkl', save_vectorizer: str = 'tfidf_vectorizer.pkl'):
    if label_col is None:
        for candidate in ['label', 'target', 'y']:
            if candidate in tfidf_df.columns:
                label_col = candidate
                break
    if label_col is None:
        raise ValueError('Label column not found in TF-IDF DataFrame')
    X = tfidf_df.drop(columns=[label_col])
    y = tfidf_df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    joblib.dump(nb, save_model)
    print(f"Saved model to {save_model}")
    return nb, metrics

# ---------------------------------------------------------------------------
# SMOTE resampling
# `apply_smote_to_tfidf()` balances the dataset using SMOTE. It expects a
# TF-IDF CSV with a label column. The resampled DataFrame is saved to
# `output_smote_csv` so it can be used in subsequent steps.
# ---------------------------------------------------------------------------


def apply_smote_to_tfidf(input_tfidf_csv: str, output_smote_csv: str):
    if not HAS_SMOTE:
        raise RuntimeError('imblearn is not installed; cannot apply SMOTE')
    df = pd.read_csv(input_tfidf_csv)
    if 'label' not in df.columns and 'target' not in df.columns:
        raise ValueError('No label/target column found in TF-IDF CSV')
    label_col = 'label' if 'label' in df.columns else 'target'
    X = df.drop(columns=[label_col])
    y = df[label_col]
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    res_df = pd.DataFrame(X_res, columns=X.columns)
    res_df[label_col] = y_res
    res_df.to_csv(output_smote_csv, index=False)
    print(f"Saved SMOTE-resampled CSV to {output_smote_csv}")
    return res_df

# ---------------------------------------------------------------------------
# Dataset splitting
# `split_dataset()` creates train/test CSV files from a balanced dataset
# (e.g., the SMOTE output). Files are written into the directory
# specified by `out_prefix`.
# ---------------------------------------------------------------------------


def split_dataset(input_smote_csv: str, out_prefix: str = '.'):
    df = pd.read_csv(input_smote_csv)
    if 'label' not in df.columns and 'target' not in df.columns:
        raise ValueError('No label/target column found in CSV')
    label_col = 'label' if 'label' in df.columns else 'target'
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train.to_csv(os.path.join(out_prefix, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(out_prefix, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(out_prefix, 'Y_train.csv'), index=False)
    y_test.to_csv(os.path.join(out_prefix, 'Y_test.csv'), index=False)
    print('Saved split files to', out_prefix)
    return (X_train, X_test, y_train, y_test)

# ---------------------------------------------------------------------------
# Prediction helper
# `predict_email_cli()` loads a saved model and vectorizer and runs a
# single-text prediction. It prints the class and (if available) probability
# scores. This function is used by the `predict` CLI command.
# ---------------------------------------------------------------------------


def predict_email_cli(model_path: str, vectorizer_path: str, email_text: str):
    # Load vectorizer and model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f'Vectorizer not found: {vectorizer_path}')
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    cleaned = clean_text(email_text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probs = None
    try:
        probs = model.predict_proba(features)[0]
    except Exception:
        pass
    classification = 'Spam' if int(prediction) == 1 else 'Ham'
    print(f"Classification: {classification}")
    if probs is not None:
        print(f"Probabilities: {probs}")

# ---------------------------------------------------------------------------
# CSV cleaning helper
# `clean_csv_inplace()` reads a CSV, finds the text column, applies
# `clean_text()` to each row and writes the cleaned CSV either in place
# or to `out_csv` if provided.
# ---------------------------------------------------------------------------


def clean_csv_inplace(input_csv: str, text_col: str = None, out_csv: str = None):
    df = pd.read_csv(input_csv)
    if text_col is None:
        for candidate in ['text', 'email_text', 'body']:
            if candidate in df.columns:
                text_col = candidate
                break
    if text_col is None:
        raise ValueError('Could not find a text column; specify text_col')
    df['cleaned_email_text'] = df[text_col].apply(clean_text)
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"Saved cleaned CSV to {out_csv}")
    else:
        df.to_csv(input_csv, index=False)
        print(f"Overwrote {input_csv} with cleaned text")
    return df

# ---------------------------------------------------------------------------
# Command-line entrypoint
# The `main()` function defines subcommands for the pipeline steps and
# maps them to the helper functions above. Defaults for file paths are
# pulled from `PROPS` so users can centralize path configuration in
# `config.properties` instead of editing source code.
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(prog='spam_catcher')
    sub = parser.add_subparsers(dest='cmd')

    p_clean = sub.add_parser('clean', help='Clean a raw CSV of emails')
    p_clean.add_argument('input_csv')
    p_clean.add_argument('--text-col', default=None)
    p_clean.add_argument('--out', default=None)

    p_tfidf = sub.add_parser('tfidf', help='Build TF-IDF from cleaned CSV')
    p_tfidf.add_argument('input_csv')
    p_tfidf.add_argument('--out', default=PROPS.get('spam_assassin_tfidf_csv', 'spam_assassin_tfidf.csv'))
    p_tfidf.add_argument('--text-col', default=None)

    p_train = sub.add_parser('train', help='Train Naive Bayes from TF-IDF CSV')
    p_train.add_argument('tfidf_csv')
    p_train.add_argument('--label-col', default=None)
    p_train.add_argument('--save-model', default=PROPS.get('spam_classifier_model', 'spam_classifier_nb_model.pkl'))
    p_train.add_argument('--save-vectorizer', default=PROPS.get('tfidf_vectorizer', 'tfidf_vectorizer.pkl'))

    p_smote = sub.add_parser('smote', help='Apply SMOTE to TF-IDF CSV')
    p_smote.add_argument('tfidf_csv')
    p_smote.add_argument('out_csv', nargs='?', default=PROPS.get('spam_assassin_smote_csv', 'spam_assassin_smote.csv'))

    p_split = sub.add_parser('split', help='Split SMOTE CSV into train/test')
    p_split.add_argument('smote_csv')
    p_split.add_argument('--out-prefix', default='.')

    p_predict = sub.add_parser('predict', help='Predict a single email text')
    p_predict.add_argument('--model', default=PROPS.get('spam_classifier_model', 'spam_classifier_nb_model.pkl'))
    p_predict.add_argument('--vectorizer', default=PROPS.get('tfidf_vectorizer', 'tfidf_vectorizer.pkl'))
    p_predict.add_argument('--text', default=None)

    args = parser.parse_args()
    if args.cmd == 'clean':
        clean_csv_inplace(args.input_csv, text_col=args.text_col, out_csv=args.out)
    elif args.cmd == 'tfidf':
        vec, df = build_tfidf(args.input_csv, output_tfidf_csv=args.out, text_col=args.text_col)
        vec_path = args.out if args.out else PROPS.get('tfidf_vectorizer', 'tfidf_vectorizer.pkl')
        # If out is a CSV path, save vectorizer to configured vectorizer path
        if vec_path.lower().endswith('.csv'):
            vec_path = PROPS.get('tfidf_vectorizer', 'tfidf_vectorizer.pkl')
        joblib.dump(vec, vec_path)
        print(f'TF-IDF vectorizer saved to {vec_path}')
    elif args.cmd == 'train':
        df = pd.read_csv(args.tfidf_csv)
        nb, metrics = train_model_from_tfidf(df, label_col=args.label_col, save_model=args.save_model, save_vectorizer=args.save_vectorizer)
        print('Training metrics:', metrics)
    elif args.cmd == 'smote':
        if not HAS_SMOTE:
            print('imblearn not installed; cannot run SMOTE')
            sys.exit(1)
        apply_smote_to_tfidf(args.tfidf_csv, args.out_csv)
    elif args.cmd == 'split':
        split_dataset(args.smote_csv, out_prefix=args.out_prefix)
    elif args.cmd == 'predict':
        text = args.text
        if text is None:
            print('Enter email text (end with Ctrl-D):')
            text = sys.stdin.read()
        predict_email_cli(args.model, args.vectorizer, text)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
