import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load TF-IDF transformed dataset
df = pd.read_csv(r"C:\Users\suhas\Downloads\spam_assassin_tfidf.csv")

# Separate features and labels
X = df.drop(columns=["label"])  # Features (TF-IDF vectors)
y = df["label"]  # Target (spam/ham labels)

# Split dataset into training and testing sets (before applying SMOTE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Apply SMOTE to the training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Convert back to DataFrame and save for future use
X_train_resampled_df = pd.DataFrame(X_train_resampled, columns=X.columns)
X_train_resampled_df["label"] = y_train_resampled  # Add label back

# Save the resampled dataset
X_train_resampled_df.to_csv(r"C:\Users\suhas\Downloads\spam_assassin_smote.csv", index=False)

