import os
import json
import re
import torch
import pandas as pd
import numpy as np
from pathlib import Path as P
from config import INPUT_DIR, OUTPUT_DIR
from logger import logger
from qwen_vl_utils import process_vision_info
from torch_snippets.adapters import np_2_b64
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image, UnidentifiedImageError
from torch_snippets import (
    read,
    resize,
    Info,
    in_debug_mode,
    show,
    P,
    np,
    PIL,
    Warn,
    ifnone,
)

torch.classes.__path__ = [] 

# Load the model and tokenizer
model_name = "Qwen/Qwen2-VL-2B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map=device)
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels )
except Exception as e:
    logger.error(f"Failed to load model or tokenizer: {e}")
    raise

def path_2_b64(path, image_size=None):
    """
    Convert an image path or PIL image to a base64-encoded string.
    """
    try:
        
        if isinstance(path, (str, P)):
            image = read(path)
            image_type = f"image/{P(path).extn}"
        elif isinstance(path, PIL.Image.Image):
            image = np.array(path)
            image_type = f"image/jpeg"
        else:
            raise NotImplementedError(f"Yet to implement for {type(path)}")
        if image_size:
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            image = resize(image, ("at-most", image_size))
        if in_debug_mode():
            Info(f"{image.shape=}")
            show(image)
        return np_2_b64(image), image_type
   
    except Exception as e:
        logger.error(f"Error in path_2_b64: {e}")
        raise


def clean_and_validate_data(invoice_data):
    """
    Clean and validate the extracted invoice data.
    """
    cleaned_data = {}
    
    # Clean and validate invoice number
    if "invoice_number" in invoice_data:
        cleaned_data["invoice_number"] = re.sub(r'[\\/*?:"<>|]', "", invoice_data["invoice_number"]).strip()
        # Remove trailing commas and dots
        cleaned_data["invoice_number"] = cleaned_data["invoice_number"].rstrip(".,")
    else:
        cleaned_data["invoice_number"] = "N/A"
    
    # Clean and validate date
    if "date" in invoice_data:
        cleaned_data["date"] = re.sub(r'[\\/*?:"<>|]', "", invoice_data["date"]).strip()
        # Remove trailing commas and dots
        cleaned_data["date"] = cleaned_data["date"].rstrip(".,")
    else:
        cleaned_data["date"] = "N/A"
    
    # Clean and validate total amount
    if "total_amount" in invoice_data:
        cleaned_data["total_amount"] = re.sub(r'[^0-9.]', "", invoice_data["total_amount"]).strip()
        # Remove trailing commas and dots
        cleaned_data["total_amount"] = cleaned_data["total_amount"].rstrip(".,")
    else:
        cleaned_data["total_amount"] = "N/A"
    
    # Clean and validate vendor name
    if "vendor_name" in invoice_data:
        cleaned_data["vendor_name"] = re.sub(r'[\\/*?:"<>|]', "", invoice_data["vendor_name"]).strip()
        # Remove trailing commas and dots
        cleaned_data["vendor_name"] = cleaned_data["vendor_name"].rstrip(".,")
    else:
        cleaned_data["vendor_name"] = "Unknown Vendor"
    
    return cleaned_data


def predict(image, prompt, max_new_tokens=1024):
    """
    Generate text from an image using the Qwen model.
    """
    try:
        img_b64_str, image_type = path_2_b64(image)
        logger.info("Image converted to array")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:{image_type};base64,{img_b64_str}",
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs =processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        )
        
        inputs = inputs.to(device)
        # Inference: Generation of the output
        generated_ids= model.generate(**inputs, max_new_tokens=max_new_tokens)
        logger.info("Ids generated")
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]
    
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        raise


def extract_invoice_data(image_path):
    """
    Extract invoice data from an image using the Qwen model.
    """
    try:
        # Define the prompt for invoice extraction
        prompt = """Extract the following details from the invoice:
- Invoice Number
- Date
- Total Amount
- Vendor Name
Return the details in JSON format."""
        
        # Generate text from the image
        extracted_text = predict(image_path, prompt)
        
        # Parse the extracted text for key fields
        invoice_data = {
            "invoice_number": None,
            "date": None,
            "total_amount": None,
            "vendor_name": None,
        }
        
        # Example parsing logic (customize based on your needs)
        for line in extracted_text.split("\n"):
            if "Invoice Number" in line:
                invoice_data["invoice_number"] = line.split(":")[-1].strip()
            elif "Date" in line:
                invoice_data["date"] = line.split(":")[-1].strip()
            elif "Total Amount" in line:
                invoice_data["total_amount"] = line.split(":")[-1].strip()
            elif "Vendor Name" in line:
                invoice_data["vendor_name"] = line.split(":")[-1].strip()
        
        logger.info(f"Extracted data from {image_path}: {invoice_data}")
        return invoice_data
    
    except UnidentifiedImageError:
        logger.error(f"Invalid image file: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None


def process_invoices(input_dir, output_dir):
    """
    Process all invoices in the input directory and save results to a dataframe.
    """
    results = []
    try:
        if not os.path.exists(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            return None
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(input_dir, filename)
                invoice_data = extract_invoice_data(image_path)
                if invoice_data:
                    cleaned_data = clean_and_validate_data(invoice_data)
                    results.append(cleaned_data)
        
        # Create a dataframe from the results
        df = pd.DataFrame(results)
        
        # Save the dataframe to a CSV file
        output_file = os.path.join(output_dir, "invoices.csv")
        df.to_csv(output_file, index=False)
        
        logger.info(f"Processed {len(df)} invoices.")
        return df
    
    except Exception as e:
        logger.error(f"Error processing invoices: {e}")
        return None