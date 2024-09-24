import base64
import io
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
from google.oauth2 import service_account

# Initialize the Vertex AI and Gemini model
def init_vertex_ai():
    credentials = service_account.Credentials.from_service_account_file(
        'mlai-rnd-aiml-f785c0229f8d.json'  
    )
    vertexai.init(project="mlai-rnd-aiml", location="us-central1", credentials=credentials)

# Function to upload PDF files in Streamlit
def upload_pdfs():
    st.write("Please upload your PDFs (you can select multiple files)")
    uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)
    return uploaded_files

# Function to merge two PDFs
def merge_pdfs(pdf_files):
    pdf_writer = PdfWriter()
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page_num])
    
    # Save merged PDF to a BytesIO object
    merged_pdf_stream = io.BytesIO()
    pdf_writer.write(merged_pdf_stream)
    merged_pdf_stream.seek(0)  # Reset stream position
    return merged_pdf_stream

# Function to read PDF as base64
def read_pdf_as_base64(pdf_stream):
    pdf_bytes = pdf_stream.read()
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

    # Send the extracted text to the chatbot
    chat.send_message(
        [extracted_text, "Use this information to answer questions about the document."],
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    st.write("You can start asking questions. Type 'exit' to end the chat.")

    # Create a text input for user questions
    user_question = st.text_input("Your question:", "")
    if user_question:
        response = chat.send_message(
            [user_question],
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        st.write(f"Bot: {response.text}")

# Main function to run the Streamlit app
def main():
    # Initialize Vertex AI
    init_vertex_ai()

    # Upload the PDF files
    pdf_files = upload_pdfs()

    if len(pdf_files) == 2:
        # Merge the uploaded PDFs
        merged_pdf_stream = merge_pdfs(pdf_files)

        # Read the merged PDF as base64
        merged_pdf_base64 = read_pdf_as_base64(merged_pdf_stream)

        # Generate text from the merged PDF using Gemini
        st.write("Extracting text from the merged PDF...")
        extracted_text = generate_text_from_pdf(merged_pdf_base64)

        st.write("Extracted Text from Merged PDF:")
        st.write(extracted_text)

        # Start live chat with the chatbot using the extracted text
        live_chat_with_bot(extracted_text)
    else:
        st.warning("Please upload exactly 2 PDF files.")

# Execute the main function
if __name__ == "__main__":
    main()
