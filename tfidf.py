import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned dataset
df = pd.read_csv(r"C:\Users\suhas\Downloads\spam_assassin_cleaned.csv")

# Initialize the TF-IDF vectorizer
# Limit features to 5000 most important words
tfidf_vectorizer = TfidfVectorizer(max_features=5000) 

# Convert text data to TF-IDF numerical representation
tfidf_matrix = tfidf_vectorizer.fit_transform(df["cleaned_email_text"])

# Convert the sparse matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Save the transformed data to a new CSV file
tfidf_df["label"] = df["target"]  # Add the label column back
tfidf_df.to_csv(r"C:\Users\suhas\Downloads\spam_assassin_tfidf.csv", index=False)

