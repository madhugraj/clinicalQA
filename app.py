import base64
import os
from PyPDF2 import PdfReader, PdfWriter
from google.colab import files
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
from google.oauth2 import service_account
from google.auth.transport.requests import Request  # Import Request
import streamlit as st  # Import Streamlit for the app

# Initialize the Vertex AI and Gemini model
def init_vertex_ai():
    try:
        SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
        credentials = service_account.Credentials.from_service_account_file(
            'mlai-rnd-aiml-f785c0229f8d.json',
            scopes=SCOPES
        )
        
        vertexai.init(project="mlai-rnd-aiml", location="us-central1", credentials=credentials)
        st.success("Vertex AI initialized successfully.")

        # Test authentication
        credentials.refresh(Request())
        st.success("Authentication successful.")
    except Exception as e:
        st.error(f"Error initializing Vertex AI: {e}")

# Function to upload files in Colab
def upload_pdfs():
    st.write("Please upload your PDFs (you can select multiple files)")
    uploaded = files.upload()  # Allow multiple files to be selected in one go
    file_paths = []
    for file_name in uploaded.keys():
        file_paths.append(f"/content/{file_name}")
    st.success("PDFs uploaded successfully!")
    return file_paths

# Function to merge two PDFs
def merge_pdfs(pdf_files):
    pdf_writer = PdfWriter()
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page_num])

    # Write to a bytes object instead of a file
    merged_pdf_bytes = pdf_writer.write_to_bytes()
    return merged_pdf_bytes

# Function to read PDF as base64
def read_pdf_as_base64(pdf_bytes):
    return base64.b64encode(pdf_bytes).decode('utf-8')

# Function to generate text from the merged PDF using Gemini
def generate_text_from_pdf(document_base64):
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

# Function to interact with the chatbot in a live session
def live_chat_with_bot(extracted_text):
    model = GenerativeModel("gemini-1.5-flash-001")
    chat = model.start_chat()

    # Send the extracted text to the chatbot once
    chat.send_message(
        [extracted_text, "Use this information to answer questions about the document."],
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    st.write("You can start asking questions. Type 'exit' to end the chat.")
    
    while True:
        user_question = st.text_input("You: ")
        if user_question.lower() == "exit":
            st.write("Ending chat session.")
            break

        response = chat.send_message(
            [user_question],
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        st.write(f"Bot: {response.text}")

# Main function to upload, merge, extract text, and query the chatbot
def main():
    # Initialize Vertex AI
    init_vertex_ai()

    # Upload the PDF files
    pdf_files = upload_pdfs()

    if len(pdf_files) == 2:
        # Merge the uploaded PDFs
        merged_pdf_bytes = merge_pdfs(pdf_files)

        # Read the merged PDF as base64
        merged_pdf_base64 = read_pdf_as_base64(merged_pdf_bytes)

        # Generate text from the merged PDF using Gemini
        st.write("Extracting text from the merged PDF...")
        extracted_text = generate_text_from_pdf(merged_pdf_base64)

        st.write("Extracted Text from Merged PDF:")
        st.text_area("Extracted Text", value=extracted_text, height=300)

        # Start live chat with the chatbot using the extracted text
        live_chat_with_bot(extracted_text)
    else:
        st.error("Please upload exactly 2 PDF files.")

# Execute the main function
if __name__ == "__main__":
    main()
