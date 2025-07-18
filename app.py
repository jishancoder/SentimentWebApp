import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the trained model & vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # keep only letters
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

# Streamlit app
st.title("📊 Sentiment Analysis of Product Reviews")
st.write("Enter a product review below and find out if it’s **Positive** or **Negative**!")

user_input = st.text_area("Enter your review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        cleaned = clean_text(user_input)
        vector_input = vectorizer.transform([cleaned])
        prediction = model.predict(vector_input)[0]
        if prediction == 1:
            st.success("✅ Positive Review 🙂")
        else:
            st.error("❌ Negative Review 😡")
