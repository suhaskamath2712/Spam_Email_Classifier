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
# Load saved model and TF-IDF vectorizer (paths from config.properties or fall back to original)
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

model_path = props.get('spam_classifier_model', r"C:\Users\suhas\Downloads\spam_classifier_nb_model.pkl")
vectorizer_path = props.get('tfidf_vectorizer', r"C:\Users\suhas\Downloads\tfidf_vectorizer.pkl")
model = joblib.load(model_path)
tfidf_vectorizer = joblib.load(vectorizer_path)
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

