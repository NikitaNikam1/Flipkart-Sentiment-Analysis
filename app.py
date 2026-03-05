import streamlit as st
import pandas as pd
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Flipkart Sentiment Analysis",
    page_icon="🛍️",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .stTextArea textarea {
            font-size: 16px;
        }
        .big-font {
            font-size:20px !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🛍️ Flipkart Review Sentiment Analyzer")
st.markdown("<p class='big-font'>Enter a product review below and find out the sentiment instantly!</p>", unsafe_allow_html=True)

st.divider()

# ---------------- INPUT ----------------
user_input = st.text_area("✍️ Enter your review here:", height=150)

# ---------------- BUTTON ----------------
if st.button("🔍 Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review first.")
    else:
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

        st.divider()

        # ---------------- RESULT DISPLAY ----------------
        if prediction == "Positive":
            st.success("😊 Positive Review")
            st.balloons()
        else:
            st.error("😞 Negative Review")

# ---------------- FOOTER ----------------
st.divider()
st.markdown("Developed by Nikita Nikam | NLP & Machine Learning Project 🚀")


