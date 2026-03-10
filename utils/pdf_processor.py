import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += f"\n[Page {page_num + 1}]\n"
        text += page.get_text()
    pdf_document.close()
    return text


def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)