import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os

nltk.download('punkt')
nltk.download('stopwords')

st.title("Fake Job Posting Detector")

# Sidebar settings
remove_stopwords = st.sidebar.checkbox("Remove Stopwords")
n_gram_range = st.sidebar.selectbox("N-gram Range", [(1, 1), (1, 2), (1, 3)])

@st.cache_data

def load_data():
    return pd.read_csv("fake_job_sample.csv")

def preprocess(text, remove_sw):
    tokens = word_tokenize(text.lower())
    if remove_sw:
        sw = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in sw and word.isalpha()]
    return " ".join(tokens)

def train_model(df, remove_sw, ngrams):
    df["text"] = df["title"] + " " + df["description"]
    df["text"] = df["text"].apply(lambda x: preprocess(x, remove_sw))

    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["fraudulent"], test_size=0.2, random_state=42)
    vectorizer = CountVectorizer(ngram_range=ngrams)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)

    joblib.dump(model, "fake_job_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    return acc

if not os.path.exists("fake_job_model.pkl") or not os.path.exists("vectorizer.pkl"):
    with st.spinner("Training model..."):
        df = load_data()
        accuracy = train_model(df, remove_stopwords, n_gram_range)
        st.success(f"Model trained with accuracy: {accuracy:.2f}")
else:
    st.success("Model already trained. Ready to use.")

model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.subheader("Check a Job Posting")

title_input = st.text_input("Job Title")
description_input = st.text_area("Job Description")

if st.button("Predict"):
    text_input = preprocess(title_input + " " + description_input, remove_stopwords)
    input_vec = vectorizer.transform([text_input])
    prediction = model.predict(input_vec)[0]
    if prediction == 1:
        st.error("⚠️ This job posting looks **FAKE**.")
    else:
        st.success("✅ This job posting looks **REAL**.")
