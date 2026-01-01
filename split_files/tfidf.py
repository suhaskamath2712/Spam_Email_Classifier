import os
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Load cleaned dataset
df = pd.read_csv(props.get('spam_assassin_cleaned_csv', r"C:\Users\suhas\Downloads\spam_assassin_cleaned.csv"))

# Initialize the TF-IDF vectorizer
# Limit features to 5000 most important words
tfidf_vectorizer = TfidfVectorizer(max_features=5000) 

# Convert text data to TF-IDF numerical representation
tfidf_matrix = tfidf_vectorizer.fit_transform(df["cleaned_email_text"])

# Convert the sparse matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Save the transformed data to a new CSV file
tfidf_df["label"] = df["target"]  # Add the label column back
tfidf_df.to_csv(props.get('spam_assassin_tfidf_csv', r"C:\Users\suhas\Downloads\spam_assassin_tfidf.csv"), index=False)

