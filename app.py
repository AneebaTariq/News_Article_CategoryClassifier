import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# Load trained model and preprocessing objects
model = joblib.load(r"svm_model.pkl")
tfidf = joblib.load(r"tfidf.pkl")
scaler = joblib.load(r"scaler.pkl")

st.title("News Article Topic Classifier")

st.write(
    "Enter a news article title and its content to predict the topic."
)

title = st.text_input(
    "Article Title",
    placeholder="Enter the news article title..."
)

content = st.text_area(
    "Article Content",
    placeholder="Paste the news article content here...",
    height=250
)

predict_button = st.button("Predict Topic")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Make prediction when button is clicked
if predict_button:

    if not title.strip() or not content.strip():
        st.warning("Please enter both the article title and content.")

    else:
        # Clean title and article content
        clean_title = clean_text(title)
        clean_content = clean_text(content)

        # Combine title and content
        combined_text = clean_title + " " + clean_content

        # Convert text into TF-IDF features
        text_tfidf = tfidf.transform([combined_text])

        # Create the same engineered features used during training
        title_word_count = len(clean_title.split())
        title_char_count = len(clean_title)
        
        if title_word_count > 0:
            title_avg_word_length = (
                title_char_count / title_word_count
            )
        else:
            title_avg_word_length = 0

        title_digit_count = len(re.findall(r'\d', clean_title))
        content_word_count = len(clean_content.split())

        numeric_features = np.array([[
            title_word_count,
            title_char_count,
            title_avg_word_length,
            title_digit_count,
            content_word_count
        ]])

        # Scale the numerical features
        numeric_features_scaled = scaler.transform(numeric_features)

        # Combine TF-IDF and numerical features
        from scipy.sparse import hstack

        final_features = hstack([
            text_tfidf,
            numeric_features_scaled
        ])

        # Predict the category
        prediction = model.predict(final_features)[0]

        # Convert category code into readable topic
        category_names = {
            'b': 'Business',
            'e': 'Entertainment',
            'm': 'Health',
            't': 'Technology'
        }

        predicted_topic = category_names.get(
            prediction,
            prediction
        )

        st.success(f"Predicted Topic: {predicted_topic}")
