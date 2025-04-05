import pandas as pd
import streamlit as st

st.title("📰 Fake News Detection Dataset Viewer")

def load_csv_from_url(url):
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return None

true_url = "https://docs.google.com/spreadsheets/d/1Dxn5o-xKX4Sc5tkMf86hXOHFmb-l1e-TRe8mSgFDixw/export?format=csv"
fake_url = "https://docs.google.com/spreadsheets/d/1kAH1iwC4r4hRXjc8rXIkXv8dLYLB0NVtcEoCthkxowY/export?format=csv"

true_df = load_csv_from_url(true_url)
fake_df = load_csv_from_url(fake_url)

if true_df is not None and fake_df is not None:
    st.success("✅ Successfully loaded both datasets!")
    st.subheader("🟢 True News Sample")
    st.dataframe(true_df.head())

    st.subheader("🔴 Fake News Sample")
    st.dataframe(fake_df.head())
else:
    st.warning("⚠️ Failed to load one or both datasets.")
