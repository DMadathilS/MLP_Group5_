# app.py
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Preprocessing function
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)       # remove hashtags & mentions
    text = re.sub(r"[^\w\s]", "", text)         # remove punctuation/emojis
    text = re.sub(r"\d+", "", text)             # remove numbers
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.set_page_config(page_title="Sentiment Classifier", layout="centered")
st.title("🔍 Mental Health Sentiment Classifier")
st.write("Enter a message to analyze its emotional sentiment (Positive, Neutral, Negative).")

# User input
user_input = st.text_area("Enter your message here:", height=150)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        cleaned = clean_text(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)
        sentiment = label_encoder.inverse_transform(prediction)[0]

        st.success(f"### 🧠 Predicted Sentiment: **{sentiment.capitalize()}**")
        st.write("Model: TF-IDF + Logistic Regression")
