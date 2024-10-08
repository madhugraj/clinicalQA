import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
from google.oauth2 import service_account
from google.cloud import aiplatform
from google.auth import credentials

# Path to your service account JSON file
credentials_path = 'client_secret_1018231518277-j045b226q3pbgncpfu3mg33h2eofr14k.apps.googleusercontent.com.json'

# Initialize the client
aiplatform.init(credentials=credentials.Credentials.from_service_account_file(credentials_path))

# Load credentials and initialize Vertex AI
def init_vertex_ai():
    try:
        # Load service account credentials
        credentials = service_account.Credentials.from_service_account_file(
            'mlai-rnd-aiml-f785c0229f8d.json'  # Make sure the path is correct
        )
        # Initialize Vertex AI
        vertexai.init(project="mlai-rnd-aiml", location="us-central1", credentials=credentials)
        st.write("Vertex AI initialized successfully.")
    except Exception as e:
        st.error(f"Error initializing Vertex AI: {str(e)}")

# Function to generate text using the Gemini model
def generate_text(prompt):
    try:
        model = GenerativeModel("gemini-1.5-flash-001")
        responses = model.generate_content(
            [prompt],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True,
        )

        generated_text = ""
        for response in responses:
            generated_text += response.text

        return generated_text
    except Exception as e:
        st.error(f"Error generating text: {str(e)}")
        return ""

# Streamlit app interface
def main():
    st.title("Text Generation with Vertex AI")

    # Initialize Vertex AI
    init_vertex_ai()

    # User input for text generation
    prompt = st.text_input("Enter your prompt:", "Write a story about a cat and dog")

    if st.button("Generate"):
        st.write("Generating text...")
        generated_text = generate_text(prompt)
        
        if generated_text:
            st.write("Generated Text:")
            st.text_area("Generated Output", generated_text, height=300)

# Generation configuration and safety settings
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.OFF,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.OFF,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.OFF,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.OFF,
}

if __name__ == "__main__":
    main()
