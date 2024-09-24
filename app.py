import streamlit as st
import base64
import os
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting

# Function to upload PDF files
def upload_pdfs():
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    return uploaded_files

# Function to merge PDFs
def merge_pdfs(pdf_files):
    pdf_writer = PdfWriter()
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page_num])
    
    # Use BytesIO to hold the merged PDF in memory
    merged_pdf_stream = BytesIO()
    pdf_writer.write(merged_pdf_stream)
    merged_pdf_stream.seek(0)  # Move to the beginning of the stream
    return merged_pdf_stream

# Function to generate text from PDF using Gemini
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

# Function to initialize the Vertex AI and Gemini model
def init_vertex_ai(credentials_path):
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    vertexai.init(project="mlai-rnd-aiml", location="us-central1", credentials=credentials)

# Streamlit app
def main():
    st.title("PDF Merger and Chatbot")
    
    # Upload the service account JSON file
    credentials_file = st.file_uploader("mlai-rnd-aiml-f785c0229f8d.json", type=["json"])
    
    if credentials_file is not None:
        with open("service_account.json", "wb") as f:
            f.write(credentials_file.getbuffer())

        # Initialize Vertex AI
        init_vertex_ai("service_account.json")

        pdf_files = upload_pdfs()
        
        if st.button("Merge and Extract Text"):
            if len(pdf_files) == 2:
                # Merge PDFs and extract text
                merged_pdf_stream = merge_pdfs(pdf_files)

                # Read merged PDF as base64
                merged_pdf_base64 = base64.b64encode(merged_pdf_stream.read()).decode('utf-8')

                st.write("Extracting text from the merged PDF...")
                extracted_text = generate_text_from_pdf(merged_pdf_base64)
                st.subheader("Extracted Text:")
                st.write(extracted_text)

                # Set up chat
                chat = GenerativeModel("gemini-1.5-flash-001").start_chat()
                question = st.text_input("Ask a question based on the extracted text:")
                if st.button("Send"):
                    response = chat.send_message(
                        [extracted_text, question],
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    st.subheader("Chatbot Response:")
                    st.write(response)
            else:
                st.warning("Please upload exactly 2 PDF files.")
    else:
        st.warning("Please upload your Service Account JSON file.")

# Safety and generation configurations
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

if __name__ == "__main__":
    main()
