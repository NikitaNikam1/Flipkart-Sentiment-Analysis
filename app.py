# Import libraries
import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load saved model and vectorizer
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("Flipkart Review Sentiment Analysis")
st.write("Type a Flipkart review and see if it is Positive or Negative!")

# Input box
user_input = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    # Clean the input
    import re, string
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
        return text

    cleaned = clean_text(user_input)
    vect = tfidf.transform([cleaned])
    prediction = model.predict(vect)[0]

    st.write("Sentiment:", prediction)


