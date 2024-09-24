import base64
import os
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
from google.oauth2 import service_account

# Initialize Vertex AI and the model
def init_vertex_ai():
    credentials = service_account.Credentials.from_service_account_file(
        'path/to/your/service_account.json'  # Update with the path to your credentials
    )
    vertexai.init(project="mlai-rnd-aiml", location="us-central1", credentials=credentials)

# Function to upload files in Streamlit
def upload_pdfs():
    st.write("Please upload your PDFs (you can select multiple files)")
    uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)
    return uploaded_files

# Function to merge PDFs
def merge_pdfs(pdf_files):
    pdf_writer = PdfWriter()
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page_num])

    merged_pdf_path = '/content/merged_output.pdf'
    with open(merged_pdf_path, 'wb') as output_pdf:
        pdf_writer.write(output_pdf)

    return merged_pdf_path

# Function to generate text from the merged PDF using Gemini
def generate_text_from_pdf(merged_pdf_path):
    with open(merged_pdf_path, 'rb') as pdf_file:
        pdf_bytes = pdf_file.read()
    document_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

    model = GenerativeModel("gemini-1.5-flash-001")
    document_part = Part.from_data(
        mime_type="application/pdf",
        data=base64.b64decode(document_base64)
    )

    responses = model.generate_content(
        [document_part, "Extract the text from the provided PDF."],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    extracted_text = ""
    for response in responses:
        extracted_text += response.text

    return extracted_text

# Function to interact with the chatbot
def chat_with_gemini(extracted_text, question):
    model = GenerativeModel("gemini-1.5-flash-001")
    chat = model.start_chat()

    # Send the extracted text to the chatbot
    response = chat.send_message(
        [extracted_text, "OCR the document"],
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    # Ask a question based on the extracted text
    response += chat.send_message(
        [question],
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    return response

# Streamlit app layout
st.title("PDF Chatbot Integration")
init_vertex_ai()

# Upload PDFs
pdf_files = upload_pdfs()

if pdf_files and len(pdf_files) == 2:
    merged_pdf_path = merge_pdfs(pdf_files)

    # Generate text from the merged PDF
    st.write("Extracting text from the merged PDF...")
    extracted_text = generate_text_from_pdf(merged_pdf_path)
    st.write("Extracted Text:")
    st.text_area("Extracted Text", value=extracted_text, height=300)

    # Chatbot interaction
    question = st.text_input("Ask a question about the extracted text:")
    if st.button("Submit"):
        if question:
            response = chat_with_gemini(extracted_text, question)
            st.write("Chatbot Response:")
            st.write(response)
        else:
            st.warning("Please enter a question.")

else:
    st.warning("Please upload exactly 2 PDF files.")

# Configuration for generation
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]
