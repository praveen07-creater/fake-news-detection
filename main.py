import streamlit as st
import pandas as pd
import requests
from io import StringIO

# Use your working Google Drive file URL
fake_url = "https://drive.google.com/uc?export=download&id=1F3Sws07czVN63gkDRDdzu3p_jrzozO4e"

def load_csv_from_drive(drive_url):
    response = requests.get(drive_url)
    if response.status_code != 200:
        st.error(f"❌ Failed to download file. Status code: {response.status_code}")
        return None
    try:
        csv_data = StringIO(response.text)
        return pd.read_csv(csv_data)
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
        return None

st.set_page_config(page_title="Fake News Viewer", layout="wide")
st.title("📰 Fake News Detection Dataset Viewer")

# Load only fake.csv from your link
fake_df = load_csv_from_drive(fake_url)

if fake_df is not None:
    st.success("✅ Fake News Dataset Loaded Successfully!")
    st.dataframe(fake_df.head())
else:
    st.warning("⚠️ Could not load the dataset. Please check the file format and sharing settings.")
