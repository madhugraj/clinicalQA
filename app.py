import base64
import io
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
from google.oauth2 import service_account

# Initialize the Vertex AI and Gemini model
def init_vertex_ai():
    try:
        credentials = service_account.Credentials.from_service_account_file(
            'mlai-rnd-aiml-f785c0229f8d.json'  # Update with your JSON file path
        )
        vertexai.init(project="mlai-rnd-aiml", location="us-central1", credentials=credentials)
        st.write("Vertex AI initialized successfully.")
    except Exception as e:
        st.error(f"Error initializing Vertex AI: {str(e)}")
        return False
    return True

# Function to merge two PDFs in memory
def merge_pdfs(pdf_files):
    pdf_writer = PdfWriter()
    
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(io.BytesIO(pdf_file))
        for page in pdf_reader.pages:
            pdf_writer.add_page(page)

    # Write the merged PDF to a byte array
    merged_pdf_bytes = io.BytesIO()
    pdf_writer.write(merged_pdf_bytes)
    merged_pdf_bytes.seek(0)  # Go back to the start of the BytesIO object
    return merged_pdf_bytes.getvalue()

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

# Streamlit app interface
def main():
    st.title("PDF Merger and Text Extractor")

    if init_vertex_ai():
        st.write("Upload exactly 2 PDF files to merge and extract text.")
        
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        
        if uploaded_files and len(uploaded_files) == 2:
            st.write("PDFs uploaded successfully!")

            # Merge the uploaded PDFs
            merged_pdf_bytes = merge_pdfs([file.read() for file in uploaded_files])

            # Read the merged PDF as base64
            merged_pdf_base64 = read_pdf_as_base64(merged_pdf_bytes)

            # Generate text from the merged PDF using Gemini
            st.write("Extracting text from the merged PDF...")
            extracted_text = generate_text_from_pdf(merged_pdf_base64)

            st.write("\nExtracted Text from Merged PDF:")
            st.text_area("Extracted Text", extracted_text, height=300)
        else:
            st.warning("Please upload exactly 2 PDF files.")

# Generation configuration and safety settings
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
