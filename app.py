import base64
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

    # Store merged PDF in memory without writing to disk
    merged_pdf_base64 = base64.b64encode(pdf_writer.write_to_bytes()).decode('utf-8')
    return merged_pdf_base64

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

    st.write("Start asking questions:")
    
    while True:
        user_question = st.text_input("You:", key="user_question")
        if user_question.lower() == "exit":
            st.write("Chat session ended.")
            break

        if user_question:
            response = chat.send_message(
                [user_question],
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            st.write(f"Bot: {response.text}")

# Main function to upload, merge, extract text, and query the chatbot
def main():
    st.title("PDF Merging and Chatbot Interaction with Vertex AI")
    
    # Initialize Vertex AI
    init_vertex_ai()

    # Upload the PDF files
    pdf_files = upload_pdfs()

    if len(pdf_files) == 2:
        # Merge the uploaded PDFs
        merged_pdf_base64 = merge_pdfs(pdf_files)

        # Generate text from the merged PDF using Gemini
        st.write("Extracting text from the merged PDF...")
        extracted_text = generate_text_from_pdf(merged_pdf_base64)

        st.write("\nExtracted Text from Merged PDF:")
        st.text(extracted_text)

        # Start live chat with the chatbot using the extracted text
        live_chat_with_bot(extracted_text)
    else:
        st.write("Please upload exactly 2 PDF files.")

# Configuration for generation and safety settings
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

# Execute the main function
if __name__ == "__main__":
    main()
