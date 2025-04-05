import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

nltk.download('stopwords')
from nltk.corpus import stopwords

# --- Google Drive CSV Download ---
fake_url = 'https://drive.google.com/uc?id=1rHy90tgqmnXnZk7fJzM0nNSuBpWi-ANy'
true_url = 'https://drive.google.com/uc?id=1F3Sws07czVN63gkDRDdzu3p_jrzozO4e'

gdown.download(fake_url, 'Fake.csv', quiet=False)
gdown.download(true_url, 'True.csv', quiet=False)

# --- Read CSVs ---
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

# --- Preprocess ---
fake['label'] = 0
true['label'] = 1
data = pd.concat([fake, true], axis=0)
data.reset_index(drop=True, inplace=True)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

data['text'] = data['text'].apply(clean_text)

X = data['text']
y = data['label']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = PassiveAggressiveClassifier()
model.fit(X_train, y_train)

# --- Streamlit UI ---
st.title("📰 Fake News Detection App")

user_input = st.text_area("Enter News Text to Check:")

if st.button("Check"):
    cleaned_input = clean_text(user_input)
    input_vec = vectorizer.transform([cleaned_input])
    prediction = model.predict(input_vec)

    if prediction[0] == 0:
        st.error("❌ This news seems **Fake**.")
    else:
        st.success("✅ This news seems **Real**.")
