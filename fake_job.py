# fakejob.py

import streamlit as st
import joblib

# Load model
model = joblib.load("fakejob_model.pkl")

# Title
st.title("🕵️ Fake Job Posting Detector")
st.write("Enter a job title and description to check if it's potentially fake.")

# Input
title = st.text_input("Job Title")
description = st.text_area("Job Description")

# Predict
if st.button("Detect"):
    if not title or not description:
        st.warning("Please enter both title and description.")
    else:
        text = title + " " + description
        prediction = model.predict([text])[0]
        if prediction == 1:
            st.error("⚠️ This job posting looks **FAKE**.")
        else:
            st.success("✅ This job posting looks **REAL**.")
