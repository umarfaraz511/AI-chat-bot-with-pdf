from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def search_similar_chunks(vector_store, query, k=4):
    return vector_store.similarity_search(query, k=k)