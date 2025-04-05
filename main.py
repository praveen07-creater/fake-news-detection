import streamlit as st
import pandas as pd

st.set_page_config(page_title="Fake News Detection Viewer", layout="centered")
st.title("📰 Fake News Detection Dataset Viewer")

def load_csv_from_google_sheet(sheet_url):
    # Convert Google Sheet link to CSV export format
    if "edit?usp=sharing" in sheet_url:
        csv_url = sheet_url.replace("/edit?usp=sharing", "/export?format=csv")
    elif "edit" in sheet_url:
        csv_url = sheet_url.replace("/edit", "/export?format=csv")
    else:
        st.error("❌ Invalid Google Sheet URL")
        return None
    
    try:
        df = pd.read_csv(csv_url, on_bad_lines='skip', engine='python')
        return df
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return None

# 🟢 Paste your Google Sheet links here
true_sheet_url = "https://docs.google.com/spreadsheets/d/1Dxn5o-xKX4Sc5tkMf86hXOHFmb-l1e-TRe8mSgFDixw/edit?usp=sharing"
fake_sheet_url = "https://docs.google.com/spreadsheets/d/1kAH1iwC4r4hRXjc8rXIkXv8dLYLB0NVtcEoCthkxowY/edit?usp=sharing"

# Load datasets
true_df = load_csv_from_google_sheet(true_sheet_url)
fake_df = load_csv_from_google_sheet(fake_sheet_url)

if true_df is not None and fake_df is not None:
    st.success("✅ Datasets loaded successfully!")

    st.subheader("🟢 True News Dataset Sample")
    st.dataframe(true_df.head())

    st.subheader("🔴 Fake News Dataset Sample")
    st.dataframe(fake_df.head())
else:
    st.warning("⚠️ Failed to load one or both datasets.")
