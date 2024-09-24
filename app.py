import json
import streamlit as st
from google.oauth2 import service_account
import vertexai

# Initialize the Vertex AI and Gemini model with debugging
def init_vertex_ai():
    json_file_path = 'mlai-rnd-aiml-f785c0229f8d.json'
    
    # Check if the file exists and can be read
    try:
        with open(json_file_path, 'r') as f:
            json_content = json.load(f)
        st.write("JSON file loaded successfully.")
    except FileNotFoundError:
        st.error(f"JSON file '{json_file_path}' not found.")
        return
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON file: {e}")
        return
    
    # Attempt to load credentials
    try:
        credentials = service_account.Credentials.from_service_account_file(json_file_path)
        st.write("Service account credentials loaded successfully.")
    except Exception as e:
        st.error(f"Error loading service account credentials: {e}")
        return

    # Initialize Vertex AI with the loaded credentials
    try:
        vertexai.init(project="mlai-rnd-aiml", location="us-central1", credentials=credentials)
        st.write("Vertex AI initialized successfully.")
    except Exception as e:
        st.error(f"Error initializing Vertex AI: {e}")



def main():
    st.title("Vertex AI Initialization Debug")

    # Initialize Vertex AI and check for errors in loading the credentials
    init_vertex_ai()

    # You can include the rest of your app logic here once the credentials are validated
    st.write("Continue with the rest of the app...")

if __name__ == "__main__":
    main()


