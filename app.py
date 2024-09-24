import streamlit as st
from google.oauth2 import service_account
import vertexai
import PyPDF2
import base64

# Function to initialize Vertex AI
def init_vertex_ai(): 
    credentials = service_account.Credentials.from_service_account_file(
        'mlai-rnd-aiml-f785c0229f8d.json'  
    )
    vertexai.init(project="mlai-rnd-aiml", location="us-central1", credentials=credentials)

# Function to upload PDF files
def upload_pdfs():
    st.write("Please upload your PDFs (you can select multiple files)")
    uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)
    return uploaded_files

# Function to extract text from uploaded PDFs
def extract_text_from_pdfs(uploaded_files):
    extracted_text = ""
    for uploaded_file in uploaded_files:
        with PyPDF2.PdfReader(uploaded_file) as pdf_reader:
            for page in pdf_reader.pages:
                extracted_text += page.extract_text() + "\n"  # Add extracted text from each page
    return extracted_text

# Function to query the chatbot
def query_chatbot(extracted_text):
    model = vertexai.generative_models.GenerativeModel("gemini-1.5-flash-001")
    chat = model.start_chat()

    # Send the extracted text to the chatbot
    response = chat.send_message(
        [extracted_text, "OCR the document"],
        generation_config={
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
        },
        safety_settings=[
            vertexai.generative_models.SafetySetting(
                category=vertexai.generative_models.SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=vertexai.generative_models.SafetySetting.HarmBlockThreshold.OFF
            ),
            # Add other safety settings as needed...
        ]
    )
    
    return response

# Initialize Vertex AI
init_vertex_ai()

# Streamlit app layout
st.title("PDF Chatbot Integration")
uploaded_files = upload_pdfs()

if uploaded_files:
    extracted_text = extract_text_from_pdfs(uploaded_files)
    st.text_area("Extracted Text", value=extracted_text, height=300)

    # Query chatbot button
    if st.button("Query Chatbot"):
        response = query_chatbot(extracted_text)
        st.text_area("Chatbot Response", value=response, height=300)
