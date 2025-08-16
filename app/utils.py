import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import json
import re

# --- OCR Function ---

@st.cache_resource
def load_ocr_reader():
    """Loads the EasyOCR reader into cache to avoid reloading on each run."""
    return easyocr.Reader(['en'])

def extract_text_from_image(image_bytes):
    """
    Uses EasyOCR to extract text from an image provided as bytes.

    Args:
        image_bytes: The image file in bytes.

    Returns:
        str: The extracted text, concatenated into a single string.
    """
    reader = load_ocr_reader()
    try:
        image = Image.open(image_bytes).convert("RGB")
        result = reader.readtext(np.array(image))
        extracted_text = " ".join([text for (_, text, _) in result])
        return extracted_text
    except Exception as e:
        st.error(f"Error during OCR processing: {e}")
        return ""

# --- LLM Inference Function ---

@st.cache_resource
def load_finetuned_model():
    """
    Loads the base model and the fine-tuned LoRA adapter.
    Caches the model to avoid reloading on every app rerun.
    """
    base_model_name = "NousResearch/Llama-2-7b-chat-hf"
    adapter_path = "../models/output/llama-2-7b-lab-notes"

    # Load the base model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)

    # Create a text generation pipeline
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return text_gen_pipeline

def get_json_from_llm(text):
    """
    Sends the extracted text to the fine-tuned LLM and gets a JSON response.

    Args:
        text (str): The text extracted from the OCR process.

    Returns:
        dict: A dictionary parsed from the LLM's JSON output.
    """
    pipe = load_finetuned_model()
    
    # Construct the prompt using the same format as in fine-tuning
    prompt = f"""### Instruction:
Extract the parameters for a projectile motion problem from the text and provide them in JSON format. If any information is missing, ask a clarifying question in the JSON response.

### Input:
{text}

### Response:"""

    sequences = pipe(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=pipe.tokenizer.eos_token_id,
        max_new_tokens=100, # Limit the number of new tokens
    )
    
    generated_text = sequences[0]['generated_text']
    
    # Extract the JSON part of the response
    # The model should output the JSON after "### Response:"
    response_part = generated_text.split("### Response:")[1].strip()

    # Clean up the JSON string - find the first '{' and the last '}'
    try:
        json_match = re.search(r'\{.*\}', response_part, re.DOTALL)
        if json_match:
            json_string = json_match.group(0)
            parsed_json = json.loads(json_string)
            return parsed_json
        else:
            return {"error": "No valid JSON object found in the response."}
    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON from the model's response.", "raw_response": response_part}
    except Exception as e:
        return {"error": str(e), "raw_response": response_part}

