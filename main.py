import streamlit as st
import pandas as pd
import requests
from io import StringIO

st.title("📰 Fake News Detection")

# Google Drive File IDs
FAKE_ID = "1F3Sws07czVN63gkDRDdzu3p_jrzozO4e"
TRUE_ID = "1BMK4RzPxXK6EtFeDlq1j5G0keV_HpQIE"

# Convert Google Drive ID to direct download URL
def get_drive_url(file_id):
    return f"https://drive.google.com/uc?export=download&id={file_id}"

# Download and read CSV
@st.cache_data
def load_csv_from_drive(file_id):
    download_url = get_drive_url(file_id)
    response = requests.get(download_url)
    if response.status_code != 200:
        st.error(f"Failed to download file from Google Drive. Status code: {response.status_code}")
        return pd.DataFrame()

    # Try reading the CSV data
    try:
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame()

# Load both datasets
fake = load_csv_from_drive(FAKE_ID)
true = load_csv_from_drive(TRUE_ID)

# Display some preview
if not fake.empty and not true.empty:
    st.subheader("📊 Sample Fake News")
    st.write(fake.head())

    st.subheader("📊 Sample True News")
    st.write(true.head())
else:
    st.error("Failed to load one or both CSV files. Please check your file permissions or link.")
