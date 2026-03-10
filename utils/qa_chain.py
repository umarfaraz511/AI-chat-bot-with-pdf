from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
import os


def create_qa_chain(vector_store):
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and latest user question, reformulate the question to be standalone."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm,
        vector_store.as_retriever(search_kwargs={"k": 4}),
        contextualize_prompt
    )

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer the question using only the context below.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    return rag_chain