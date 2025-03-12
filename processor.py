import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image, UnidentifiedImageError
import os
import json
import numpy as np
from pathlib import Path as P
from config import INPUT_DIR, OUTPUT_DIR
from logger import logger
from qwen_vl_utils import process_vision_info
from torch_snippets import P, np

# Load the model and tokenizer
model_name = "Qwen/Qwen2-VL-2B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
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
            image = Image.open(path)
            image_type = f"image/{P(path).suffix[1:]}"  # Extract file extension
        elif isinstance(path, Image.Image):
            image = path
            image_type = "image/jpeg"
        else:
            raise NotImplementedError(f"Unsupported input type: {type(path)}")
        
        # Resize image if required
        if image_size:
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            image = image.resize(image_size)
        
        # Convert image to base64
        import base64
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format=image_type.split("/")[-1])
        img_b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_b64_str, image_type
    
    except Exception as e:
        logger.error(f"Error in path_2_b64: {e}")
        raise

def predict(image, prompt, max_new_tokens=1024):
    """
    Generate text from an image using the Qwen model.
    """
    try:
        img_b64_str, image_type = path_2_b64(image)
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
        messages, tokenize=False, add_generation_prompt=True    )
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
    Process all invoices in the input directory and save results.
    """
    results = {}
    try:
        if not os.path.exists(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            return None
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(input_dir, filename)
                invoice_data = extract_invoice_data(image_path)
                if invoice_data and invoice_data["vendor_name"]:
                    vendor = invoice_data["vendor_name"]
                    if vendor not in results:
                        results[vendor] = []
                    results[vendor].append(invoice_data)
        
        # Save results to output directory
        for vendor, invoices in results.items():
            vendor_dir = os.path.join(output_dir, vendor)
            os.makedirs(vendor_dir, exist_ok=True)
            with open(os.path.join(vendor_dir, "invoices.json"), "w") as f:
                json.dump(invoices, f, indent=4)
        
        logger.info(f"Processed {len(results)} vendors.")
        return results
    
    except Exception as e:
        logger.error(f"Error processing invoices: {e}")
        return None