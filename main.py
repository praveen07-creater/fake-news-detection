import streamlit as st
import pandas as pd
import numpy as np
import gdown
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download datasets from Google Drive
fake_url = 'https://drive.google.com/uc?id=1rHy90tgqmnXnZk7fJzM0nNSuBpWi-ANy'
true_url = 'https://drive.google.com/uc?id=1F3Sws07czVN63gkDRDdzu3p_jrzozO4e'

gdown.download(fake_url, 'Fake.csv', quiet=False)
gdown.download(true_url, 'True.csv', quiet=False)

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine and shuffle
data = pd.concat([fake, true], axis=0)
data = data.drop(['title', 'subject', 'date'], axis=1)
data = data.sample(frac=1)

# Preprocessing
nltk.download('stopwords')
ps = PorterStemmer()

def stemming(content):
    content = content.lower()
    content = content.split()
    content = [ps.stem(word) for word in content if word not in stopwords.words('english')]
    return " ".join(content)

data['text'] = data['text'].apply(stemming)

# Train-test split
X = data['text']
y = data['label']
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# Streamlit App
st.title("📰 Fake News Detection App")
st.write(f"Model Accuracy: **{accuracy * 100:.2f}%**")

user_input = st.text_area("Enter the news article text:")
if st.button("Check if it's Fake or Real"):
    transformed_input = vectorizer.transform([stemming(user_input)])
    prediction = model.predict(transformed_input)
    result = "✅ Real News" if prediction[0] == 1 else "❌ Fake News"
    st.subheader(result)
