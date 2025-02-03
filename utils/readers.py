from PyPDF2 import PdfReader

def get_pdf_text(docs):
    # '''
    # Reads the text from the uploaded document(s).
    # Returns extracted_text.
    # '''
    text = ""
    for doc in docs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text