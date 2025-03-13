import streamlit as st
import os
import requests
from config import INPUT_DIR, OUTPUT_DIR
from processor import process_invoices
from logger import logger

# Streamlit app
st.title("Invoice Processing System")

# Test logging
logger.info("Application started.")


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
            logger.info(f"Uploaded {len(uploaded_files)} files.")
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
                logger.info(f"Downloaded file: {file_name}")
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
            df = process_invoices(INPUT_DIR, OUTPUT_DIR)
            if df is not None and not df.empty:
                st.success("Invoices processed successfully!")
                
                # Sort the dataframe by vendor name
                df_sorted = df.sort_values(by="vendor_name")
                
                # Display the sorted dataframe
                st.write("Processed Invoices (Sorted by Vendor Name):")
                st.dataframe(df_sorted)
                
                # Save the sorted dataframe to a CSV file
                output_file = os.path.join(OUTPUT_DIR, "invoices_sorted.csv")
                df_sorted.to_csv(output_file, index=False)
                st.success(f"Sorted invoices saved to {output_file}")
            else:
                st.error("No invoices processed.")
    except Exception as e:
        st.error(f"An error occurred while processing invoices: {e}")