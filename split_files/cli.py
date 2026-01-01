import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load saved model and TF-IDF vectorizer
model = joblib.load(r"C:\Users\suhas\Downloads\spam_classifier_nb_model.pkl")
tfidf_vectorizer = joblib.load(r"C:\Users\suhas\Downloads\tfidf_vectorizer.pkl")

# Initialize text preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """ Clean email text by removing HTML, punctuation,
    stopwords, and applying lemmatization. """

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    # Lemmatization & Stopword removal
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words] 
    return " ".join(words)

def predict_email(email_text):
    """ Predict if an email is spam or ham and display probability. """
    # Preprocess the email text
    cleaned_text = clean_text(email_text)
    
    # Convert text to TF-IDF features
    text_features = tfidf_vectorizer.transform([cleaned_text])
    
    # Predict class (Spam/Ham)
    prediction = model.predict(text_features)
    
    # Get probability scores
    probabilities = model.predict_proba(text_features)[0]
    spam_probability = probabilities[1]  # Probability of being spam
    ham_probability = probabilities[0]  # Probability of being ham
    
    # Output prediction result
    classification = "Spam" if prediction[0] == 1 else "Ham"
    print(f"The email is classified as: {classification}")
    print(f"Probability of Spam: {spam_probability:.4f}")
    print(f"Probability of Ham: {ham_probability:.4f}")

# Example usage
email_text = input("Enter email text: ")
predict_email(email_text)

