import streamlit as st
from PIL import Image
import json

# Import helper functions from other modules
from utils import extract_text_from_image, get_json_from_llm
from simulation.projectile import create_animation

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Lab Notes Analyzer",
    page_icon="🔬",
    layout="wide"
)

# --- Main Application UI ---
st.title("🔬 Messy Lab Notes Analyzer")
st.markdown("Upload an image of your handwritten lab notes for a projectile motion problem. The app will use OCR to read the text, an LLM to extract the parameters, and then run a simulation.")

# --- Sidebar for Upload and Controls ---
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader(
        "Upload your lab note image",
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file:
        st.image(uploaded_file, caption="Your Uploaded Image", use_column_width=True)

# --- Main Content Area ---
if uploaded_file is not None:
    # Create two columns for displaying results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Extracted Text (from OCR)")
        with st.spinner("Reading your notes..."):
            # Get the bytes of the uploaded file
            image_bytes = uploaded_file.getvalue()
            # Extract text using the OCR utility function
            extracted_text = extract_text_from_image(image_bytes)
            
            if extracted_text:
                st.text_area("OCR Output", extracted_text, height=200)
            else:
                st.warning("Could not extract any text from the image.")

    with col2:
        st.subheader("2. Structured Data (from LLM)")
        if extracted_text:
            with st.spinner("Interpreting text with the fine-tuned LLM..."):
                # Get the structured JSON from the LLM
                json_output = get_json_from_llm(extracted_text)
                
                if "error" in json_output:
                    st.error(f"LLM Error: {json_output.get('error')}")
                    st.json(json_output.get('raw_response', '{}'))
                elif "clarification_question" in json_output:
                    st.warning(f"LLM needs more info: {json_output['clarification_question']}")
                    st.json(json_output)
                else:
                    st.success("Successfully extracted parameters!")
                    st.json(json_output)
        else:
            st.info("Waiting for extracted text to process with LLM.")

    # --- Simulation Section (Full Width) ---
    if 'json_output' in locals() and "error" not in json_output and "clarification_question" not in json_output:
        st.divider()
        st.subheader("3. Physics Simulation")
        
        try:
            # Check if all required keys are present for the simulation
            required_keys = ["initial_velocity", "angle", "initial_height"]
            if all(key in json_output for key in required_keys):
                with st.spinner("Running the projectile motion simulation..."):
                    # Create the animation using the simulation module
                    animation_figure = create_animation(json_output)
                    st.pyplot(animation_figure)
            else:
                st.warning("The JSON output from the LLM is missing one or more required keys for the simulation (initial_velocity, angle, initial_height).")

        except Exception as e:
            st.error(f"An error occurred during the simulation: {e}")

else:
    st.info("Please upload an image in the sidebar to get started.")

