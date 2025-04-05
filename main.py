import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("📰 Fake News Detection Web App")

# Sidebar
st.sidebar.title("🔍 Upload Dataset")
true_file = st.sidebar.file_uploader("Upload True News CSV", type="csv")
fake_file = st.sidebar.file_uploader("Upload Fake News CSV", type="csv")

# Load and show datasets
def load_dataset():
    if true_file and fake_file:
        try:
            true_df = pd.read_csv(true_file)
            fake_df = pd.read_csv(fake_file)
            return true_df, fake_df
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
    return None, None

true_df, fake_df = load_dataset()

if true_df is not None and fake_df is not None:
    st.success("✅ Successfully loaded both datasets!")
    
    st.subheader("🟢 True News Sample")
    st.write(true_df.head())

    st.subheader("🔴 Fake News Sample")
    st.write(fake_df.head())

    # Combine and prepare data
    true_df["label"] = "REAL"
    fake_df["label"] = "FAKE"
    df = pd.concat([true_df, fake_df]).reset_index(drop=True)

    # Assume 'text' column exists
    if 'text' not in df.columns:
        st.error("❌ 'text' column not found in datasets.")
    else:
        # Split data
        X = df['text']
        y = df['label']

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        X_vectorized = vectorizer.fit_transform(X)

        # Train model
        model = PassiveAggressiveClassifier(max_iter=50)
        model.fit(X_vectorized, y)

        st.success("✅ Model trained successfully!")

        st.subheader("🧠 Try Your Own News")
        user_input = st.text_area("Enter news text here...", height=200)

        if st.button("Predict"):
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]
            if prediction == "FAKE":
                st.error("🛑 This news is **FAKE**.")
            else:
                st.success("✅ This news is **REAL**.")
else:
    st.warning("⚠️ Please upload both True and Fake News CSV files.")
