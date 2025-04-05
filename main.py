import streamlit as st
import pandas as pd
import numpy as np
import nltk
import requests
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import string

nltk.download('stopwords')

# 🔽 Google Drive CSV Links (Exportable)
fake_url = "https://drive.google.com/uc?export=download&id=1rHy90tgqmnXnZk7fJzM0nNSuBpWi-ANy"
true_url = "https://drive.google.com/uc?export=download&id=1F3Sws07czVN63gkDRDdzu3p_jrzozO4e"

# 🔽 Load files using requests
def load_csv_from_drive(url):
    response = requests.get(url)
    csv_data = StringIO(response.text)
    return pd.read_csv(csv_data)

fake = load_csv_from_drive(fake_url)
true = load_csv_from_drive(true_url)

# 🔽 Labeling and combining
fake["label"] = 0
true["label"] = 1
data = pd.concat([fake, true], axis=0)
data = data.drop(['title', 'subject', 'date'], axis=1)
data = data.sample(frac=1)

# 🔽 Preprocessing
ps = PorterStemmer()
stop_words = stopwords.words('english')

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

data["text"] = data["text"].apply(clean_text)

# 🔽 Split and train
X = data["text"]
y = data["label"]
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# 🔽 Streamlit UI
st.title("📰 Fake News Detection App")
st.write(f"🔍 **Model Accuracy**: {acc * 100:.2f}%")

user_input = st.text_area("Enter a news article to check if it's fake or real:")

if st.button("Check"):
    input_cleaned = clean_text(user_input)
    input_vectorized = vectorizer.transform([input_cleaned])
    prediction = model.predict(input_vectorized)[0]
    if prediction == 1:
        st.success("✅ This looks like **Real News**.")
    else:
        st.error("🚫 This seems like **Fake News**.")
