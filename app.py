import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

st.title("📰 Fake News Detection App")

user_input = st.text_area("Enter news text here:")

if st.button("Check News"):
    clean_text = preprocess(user_input)
    vector = vectorizer.transform([clean_text])
    prediction = model.predict(vector)[0]
    st.success(prediction.upper())
