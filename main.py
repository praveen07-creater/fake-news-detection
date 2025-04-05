import streamlit as st
import pandas as pd
import numpy as np
import nltk
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib

nltk.download('punkt')

# Title
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article below to check if it's **Fake** or **Real**.")

# Function to load CSV from Google Drive
def load_csv_from_drive(url):
    response = requests.get(url)
    response.encoding = 'utf-8'
    csv_data = StringIO(response.text)
    return pd.read_csv(csv_data, on_bad_lines='skip')  # skips malformed lines

# Google Drive links
fake_url = "https://drive.google.com/uc?export=download&id=1rHy90tgqmnXnZk7fJzM0nNSuBpWi-ANy"
true_url = "https://drive.google.com/uc?export=download&id=1F3Sws07czVN63gkDRDdzu3p_jrzozO4e"

# Load datasets
fake = load_csv_from_drive(fake_url)
true = load_csv_from_drive(true_url)

# Add a label column
fake["label"] = 0
true["label"] = 1

# Combine datasets
df = pd.concat([fake, true])
df = df.sample(frac=1).reset_index(drop=True)

# Drop unnecessary columns
df = df[["text", "label"]]
df.dropna(inplace=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Accuracy
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)

# Input
news_input = st.text_area("🖊️ Paste the news article here:")

if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        input_tfidf = vectorizer.transform([news_input])
        prediction = model.predict(input_tfidf)[0]
        if prediction == 1:
            st.success("✅ This news is **Real**.")
        else:
            st.error("🚨 This news is **Fake**.")

# Show accuracy
st.sidebar.markdown("📊 Model Accuracy:")
st.sidebar.info(f"{acc*100:.2f}%")
