import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
index_word = {v: k for k, v in word_index.items()}
model = load_model('simplernn_imdb.h5')


def decode_review(text):
    return ' '.join([index_word.get(i - 3, '?') for i in text])
def preprocess_review(review):
    words = review.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words]
    padded = sequence.pad_sequences([encoded], maxlen=500)
    return padded

# Prediction function
def predict_review(review):
    processed = preprocess_review(review)
    prediction = model.predict(processed)
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
    return sentiment,prediction[0][0]

# Design streamlit

import streamlit as st
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative).")

user_input = st.text_area("Movie Review", "Type your review here...")

if st.button("Predict Sentiment"):
    if user_input.strip():
        predicted_sentiment, confidence = predict_review(user_input)
        st.write(f"Predicted Sentiment: **{predicted_sentiment}**")
        st.write(f"Confidence: {confidence:.4f}")
    else:
        st.write("Please enter a valid movie review.")

