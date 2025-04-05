
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
import string
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Download NLTK data
nltk.download('stopwords')
from nltk.corpus import stopwords

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

# Function to read Google Drive files
def read_drive_csv(share_link):
    file_id = share_link.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    return pd.read_csv(StringIO(response.text))

# Load the data from Google Drive
fake_url = "https://drive.google.com/file/d/1rHy90tgqmnXnZk7fJzM0nNSuBpWi-ANy/view?usp=drive_link"
true_url = "https://drive.google.com/file/d/1F3Sws07czVN63gkDRDdzu3p_jrzozO4e/view?usp=drive_link"

fake = read_drive_csv(fake_url)
true = read_drive_csv(true_url)

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine and shuffle
data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

# Preprocess text
data["text"] = data["text"].apply(preprocess_text)

# Split data
X = data["text"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
acc = accuracy_score(y_test, model.predict(X_test_vec))

# Streamlit App
st.title("📰 Fake News Detection App")
st.markdown(f"**Model Accuracy:** {acc * 100:.2f}%")

user_input = st.text_area("Enter a news article text to verify:")

if st.button("Predict"):
    if user_input.strip() != "":
        processed_input = preprocess_text(user_input)
        input_vec = vectorizer.transform([processed_input])
        prediction = model.predict(input_vec)[0]
        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.success(f"Prediction: {label}")
    else:
        st.warning("Please enter some news content first.")
