import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import os
from dotenv import load_dotenv # Import load_dotenv

load_dotenv() ## Load all our environment variables from .env file

# --- Function to initialize and get the Gemini model ---
def get_gemini_model():
    """
    Configures the Google Generative AI model with the API key from environment variables.
    Handles missing API key gracefully.
    Uses 'gemini-pro-vision' for image understanding.
    """
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable in your .env file.")
        st.stop() # Stop the Streamlit app if the key is missing

    try:
        genai.configure(api_key=api_key)
        # Using 'gemini-pro-vision' for multimodal (image + text) input
        # 'gemini-2.0-flash' primarily for text, 'gemini-pro-vision' for vision tasks.
        model = genai.GenerativeModel('gemini-2.0-flash')
        return model
    except Exception as e:
        st.error(f"Error loading Gemini model. Please check your API key and model name. Details: {e}")
        st.stop() # Stop if the model cannot be loaded

# --- Function to send image and prompt to Gemini ---
def get_diagnosis_result(image, prompt_text):
    """
    Sends the image and prompt to the Gemini model and returns the response text.
    """
    model = get_gemini_model() # Get the configured Gemini model
    try:
        # The generate_content method can take a list of parts for multimodal input
        response = model.generate_content([prompt_text, image])
        return response.text
    except Exception as e:
        st.error(f"Error generating content from Gemini API. Details: {e}")
        print(f"DEBUG: Gemini API Error: {type(e).__name__}: {e}") # Print to terminal for debugging
        return None # Return None to indicate an error

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Dental AI Analyzer",
    page_icon="🦷",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🦷 Dental AI Analyzer: Teeth Condition Diagnosis")
st.markdown("""
    Capture an image of your teeth using your webcam. Gemini AI will then analyze the image
    to provide a preliminary diagnosis of their condition.

    **Disclaimer:** This tool is for informational purposes only and should not replace professional medical advice.
""")

# --- Camera Input ---
st.header("1. Capture Your Teeth Image")
captured_image_bytes = st.camera_input("Take a clear picture of your teeth")

# --- AI Analysis ---
if captured_image_bytes:
    st.image(captured_image_bytes, caption="Captured Image", use_container_width=True)

    # Convert bytes from camera_input to PIL Image format
    # FIX: Use .read() to get the bytes from the UploadedFile object
    image = Image.open(io.BytesIO(captured_image_bytes.read()))

    st.header("2. Get AI Diagnosis")
    if st.button("Analyze Teeth Condition"):
        with st.spinner("Analyzing image with AI... This may take a moment..."):
            # Define the prompt for Gemini
            diagnosis_prompt = """
            Analyze this image of human teeth and provide a concise diagnosis of their condition.
            Focus on general health, presence of cavities, gum health, and any other visible issues.
            If the image is not clearly of teeth or is of poor quality, please state that.
            Provide the diagnosis in a clear, easy-to-understand format.
            """

            # Get the diagnosis result from Gemini
            diagnosis_text = get_diagnosis_result(image, diagnosis_prompt)

            if diagnosis_text:
                st.success("Analysis Complete!")
                st.markdown("---")
                st.write("### AI Diagnosis:")
                st.write(diagnosis_text)
            else:
                st.error("Could not retrieve AI diagnosis. Please check the console for errors or try again.")
else:
    st.info("Please use the camera above to capture an image of your teeth for analysis.")

st.markdown("---")
st.caption("Powered by Google Gemini AI and Streamlit")

