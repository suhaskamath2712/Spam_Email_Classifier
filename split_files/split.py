import pandas as pd
from sklearn.model_selection import train_test_split

# Load the balanced dataset (after SMOTE)
df = pd.read_csv(r"C:\Users\suhas\Downloads\spam_assassin_smote.csv")

# Separate features (X) and labels (y)
X = df.drop(columns=["label"])  # Features (TF-IDF vectors)
y = df["label"]  # Target labels (spam or ham)

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save the split datasets for later use
X_train.to_csv(r"C:\Users\suhas\Downloads\X_train.csv", index=False)
X_test.to_csv(r"C:\Users\suhas\Downloads\X_test.csv", index=False)
y_train.to_csv(r"C:\Users\suhas\Downloads\Y_train.csv", index=False)
y_test.to_csv(r"C:\Users\suhas\Downloads\Y_test.csv", index=False)

print("Data successfully split into training and testing sets.")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
