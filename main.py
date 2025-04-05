import pandas as pd
import requests
from io import StringIO
import streamlit as st

st.title("📰 Fake News Detection Dataset Viewer")

def load_csv_from_gdrive(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"❌ Failed to download file. Status code: {response.status_code}")
            return None
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data, on_bad_lines='skip', engine='python')  # ✅ new way to skip bad lines
        return df
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
        return None

# ✅ Google Drive direct download links
true_url = "https://drive.google.com/uc?export=download&id=1q6B_iSEivL2JwKJX14ZQPUH6MmLJ9l6i"
fake_url = "https://drive.google.com/uc?export=download&id=1ZNFliz7vxLePuJIA48-Hoc3C7m9byoGt"

true_df = load_csv_from_gdrive(true_url)
fake_df = load_csv_from_gdrive(fake_url)

if true_df is not None and fake_df is not None:
    st.success("✅ Successfully loaded both datasets!")
    st.subheader("🟢 True News Sample")
    st.dataframe(true_df.head())

    st.subheader("🔴 Fake News Sample")
    st.dataframe(fake_df.head())
else:
    st.warning("⚠️ Could not load the datasets. Please check file format and sharing settings.")
