import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import os
import pandas as pd

# Load Model
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME).to(device)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Function to extract text from invoice image
def extract_invoice_data(image_path):
    image = Image.open(image_path).convert("RGB")
    
    prompt = ("Extract the following details from this invoice: "
              "Invoice Number, Date, Total Amount, Vendor Name. "
              "Provide output in JSON format.")

    inputs = processor(image, text=prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    extracted_text = processor.batch_decode(output, skip_special_tokens=True)[0]

    return extracted_text

# Process all images in the data folder
def process_invoices(folder_path):
    results = []
    for img_file in os.listdir(folder_path):
        if img_file.endswith((".png", ".jpg", ".jpeg", ".pdf")):
            image_path = os.path.join(folder_path, img_file)
            extracted_data = extract_invoice_data(image_path)
            results.append({"file_name": img_file, "extracted_data": extracted_data})

    return results
