import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
import re

nltk.download('stopwords')

# Load the trained model & vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Function to clean text with contraction & negation handling
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters and spaces

    # Expand common contractions
    contractions = {
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "can't": "cannot",
        "couldn't": "could not",
        "won't": "will not",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "mustn't": "must not",
        "mightn't": "might not",
    }

    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)

    # Handle negations: join 'not' with the next word
    text = re.sub(r'\bnot\s+(\w+)', r'not_\1', text)

    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("📊 Sentiment Analysis of Product Reviews")
st.write("Enter a product review below and find out if it’s **Positive** or **Negative**!")

user_input = st.text_area("Enter your review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review text!")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]

        if prediction == 1:
            st.success("✅ Positive Sentiment! 😊")
        else:
            st.error("❌ Negative Sentiment. 😞")
