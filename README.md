```markdown
This project was done as the dissertation for MCA at University of Mysore, Mysore.
```

## spam_catcher — Unified spam detection utility

`spam_catcher.py` combines data cleaning, TF-IDF vectorization, model training, SMOTE resampling, dataset splitting, and single-email prediction into one command-line utility.

### Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- nltk
- joblib
- beautifulsoup4
- Optional (for SMOTE): imbalanced-learn

Install with pip:

```bash
pip install -r requirements.txt
# or install individual packages, e.g.
pip install pandas scikit-learn nltk joblib beautifulsoup4
pip install imbalanced-learn  # optional: only required to run the `smote` subcommand
```

Create a simple `requirements.txt` if desired:

```
pandas
numpy
scikit-learn
nltk
joblib
beautifulsoup4
imbalanced-learn
```

### NLTK setup

The script will attempt to download NLTK resources (punkt, stopwords, wordnet) automatically the first time it runs. If you prefer to do this manually:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Usage

Run the script from the project root. Subcommands:

- clean: Clean raw CSV of emails and add a `cleaned_email_text` column.
- tfidf: Build TF-IDF features from a cleaned CSV and save the vectorizer.
- train: Train a Naïve Bayes model from a TF-IDF CSV and save the model.
- smote: (Optional) Apply SMOTE to a TF-IDF CSV to balance classes.
- split: Split a balanced CSV into training/testing CSV files.
- predict: Predict a single email text using saved model and vectorizer.

Examples:

```bash
# Clean the raw dataset (specify the column that holds raw text if needed)
python3 spam_catcher.py clean /path/to/spam_assassin.csv --text-col text --out spam_assassin_cleaned.csv

# Build TF-IDF from cleaned CSV (saves tfidf_vectorizer.pkl)
python3 spam_catcher.py tfidf spam_assassin_cleaned.csv --out spam_assassin_tfidf.csv --text-col cleaned_email_text

# Train Naive Bayes model (saves spam_classifier_nb_model.pkl)
python3 spam_catcher.py train spam_assassin_tfidf.csv --label-col label --save-model spam_classifier_nb_model.pkl

# (Optional) Apply SMOTE to TF-IDF CSV (requires imbalanced-learn)
python3 spam_catcher.py smote spam_assassin_tfidf.csv spam_assassin_smote.csv

# Split SMOTE output into train/test files
python3 spam_catcher.py split spam_assassin_smote.csv --out-prefix .

# Predict single email (pass text or pipe/interactive input)
python3 spam_catcher.py predict --model spam_classifier_nb_model.pkl --vectorizer tfidf_vectorizer.pkl --text "Free money!!!"
```

### Notes & tips

- The script tries to infer common column names (`text`, `email_text`, `cleaned_email_text`, `label`, `target`) but you can override these with flags.
- Saved model/vectorizer files are standard joblib files and can be loaded separately.
- SMOTE requires `imbalanced-learn`; if not installed the `smote` subcommand will exit with a helpful message.
- File paths in the original project used Windows paths; adapt them to your environment when running on Linux.

### Files of interest

- `spam_catcher.py` — combined CLI utility (cleaning, TF-IDF, training, SMOTE, splitting, prediction)
- Original helper scripts (kept for reference): `final.py`, `tfidf.py`, `smote.py`, `split.py`, `cli.py`, `spamemailclassfier.py`

### Attribution
This project was done as the dissertation for MCA at University of Mysore, Mysore.