import streamlit as st
import base64
from PyPDF2 import PdfReader, PdfWriter
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
from google.oauth2 import service_account
import io

# Configuration settings for generation and safety
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

# Initialize Vertex AI and Gemini model
def init_vertex_ai():
    try:
        credentials = service_account.Credentials.from_service_account_file(
            'mlai-rnd-aiml-f785c0229f8d.json'
        )
        vertexai.init(project="mlai-rnd-aiml", location="us-central1", credentials=credentials)
        st.write("Vertex AI Initialization Debug")
        st.write("JSON file loaded successfully.")
        st.write("Service account credentials loaded successfully.")
        st.write("Vertex AI initialized successfully.")
    except Exception as e:
        st.error(f"Failed to initialize Vertex AI: {e}")

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

# Function to start a chatbot session and query based on the extracted text
def live_chat_with_bot(extracted_text, user_question):
    model = GenerativeModel("gemini-1.5-flash-001")
    chat = model.start_chat()

    chat.send_message(
        [extracted_text, "Use this information to answer questions about the document."],
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    response = chat.send_message(
        [user_question],
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    return response.text

# Main Streamlit app function
def main():
    st.title("PDF Upload and Text Extraction with Vertex AI")

    # Step 1: Initialize Vertex AI
    init_vertex_ai()

    # Step 2: PDF Upload
    st.write("Upload exactly 2 PDF files to merge and extract text.")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) == 2:
        st.write("PDFs uploaded successfully!")

        # Step 3: Merge PDFs (in-memory, no need to write to disk)
        pdf_writer = PdfWriter()
        for uploaded_file in uploaded_files:
            pdf_reader = PdfReader(uploaded_file)
            for page_num in range(len(pdf_reader.pages)):
                pdf_writer.add_page(pdf_reader.pages[page_num])

        # Convert merged PDF to bytes
        pdf_bytes = io.BytesIO()
        pdf_writer.write(pdf_bytes)
        pdf_bytes.seek(0)

        # Step 4: Extract Text using Vertex AI
        st.write("Extracting text from the merged PDF...")
        merged_pdf_base64 = base64.b64encode(pdf_bytes.read()).decode('utf-8')
        extracted_text = generate_text_from_pdf(merged_pdf_base64)

        st.write("**Extracted Text from Merged PDF:**")
        st.text(extracted_text)

        # Step 5: Start Chatbot Interaction
        st.write("You can now ask questions based on the extracted text.")
        user_question = st.text_input("Your question:")
        if user_question:
            response = live_chat_with_bot(extracted_text, user_question)
            st.write(f"**Bot:** {response}")
    else:
        st.write("Please upload exactly 2 PDF files to proceed.")

# Call the main function
if __name__ == "__main__":
    main()
