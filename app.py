import streamlit as st
import pandas as pd
import os
from extract_data import process_invoices

# UI Configuration
st.title("Automated Invoice Processing System")
st.sidebar.header("Upload Options")

# Folder for invoice storage
UPLOAD_FOLDER = "data/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# File uploader
uploaded_files = st.sidebar.file_uploader("Upload Invoice Images", accept_multiple_files=True, type=["png", "jpg", "jpeg", "pdf"])

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.sidebar.success("Files uploaded successfully!")

# Process button
if st.sidebar.button("Process Invoices"):
    results = process_invoices(UPLOAD_FOLDER)
    
    if results:
        st.write("### Extracted Invoice Data")
        df = pd.DataFrame(results)
        st.dataframe(df)
    else:
        st.warning("No valid invoices found.")
