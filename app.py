import streamlit as st
import os
import requests
from config import INPUT_DIR, OUTPUT_DIR
from processor import process_invoices
from logger import logger

# Streamlit app
st.title("Invoice Processing System")
st.write("Upload images or provide a drive/github link to process invoices.")

# Input option
input_option = st.radio("Select input option:", ["Upload Images", "Provide Link"])

if input_option == "Upload Images":
    uploaded_files = st.file_uploader("Upload invoice images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        try:
            for file in uploaded_files:
                with open(os.path.join(INPUT_DIR, file.name), "wb") as f:
                    f.write(file.getbuffer())
            st.success("Files uploaded successfully!")
        except Exception as e:
            logger.error(f"Error uploading files: {e}")
            st.error(f"Failed to upload files: {e}")

elif input_option == "Provide Link":
    link = st.text_input("Enter drive/github link:")
    if link:
        try:
            response = requests.get(link)
            if response.status_code == 200:
                file_name = link.split("/")[-1]
                with open(os.path.join(INPUT_DIR, file_name), "wb") as f:
                    f.write(response.content)
                st.success("File downloaded successfully!")
            else:
                logger.error(f"Failed to download file from {link}. Status code: {response.status_code}")
                st.error("Failed to download file.")
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            st.error(f"Error: {e}")

# Process invoices
if st.button("Process Invoices"):
    try:
        with st.spinner("Processing invoices..."):
            results = process_invoices(INPUT_DIR, OUTPUT_DIR)
            if results:
                st.success("Invoices processed successfully!")
                for vendor, invoices in results.items():
                    st.write(f"Vendor: {vendor}")
                    st.json(invoices)
            else:
                st.error("No invoices processed.")
    except Exception as e:
        logger.error(f"Error processing invoices: {e}")
        st.error(f"An error occurred while processing invoices: {e}")