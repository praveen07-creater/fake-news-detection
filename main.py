import streamlit as st
import pandas as pd

st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("📰 Fake News Detection Dataset Viewer")

# Google Drive direct CSV URLs
true_url = "https://drive.google.com/uc?export=download&id=1q6B_iSEivL2JwKJX14ZQPUH6MmLJ9l6i"
fake_url = "https://drive.google.com/uc?export=download&id=1ZNFliz7vxLePuJIA48-Hoc3C7m9byoGt"

# Load CSVs with error handling
@st.cache_data
def load_data(url):
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
        return None


true_df = pd.read_csv("true_clean.csv")
fake_df = pd.read_csv("fake_clean.csv")


if true_df is not None and fake_df is not None:
    st.success("✅ Datasets loaded successfully!")
    
    st.subheader("✅ True News Sample")
    st.dataframe(true_df.head(10), use_container_width=True)
    
    st.subheader("🚫 Fake News Sample")
    st.dataframe(fake_df.head(10), use_container_width=True)
else:
    st.warning("⚠️ Could not load the datasets. Please check file format and sharing settings.")
