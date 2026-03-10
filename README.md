# 🤖 AI Chat with Your PDF

> RAG-Powered Document Intelligence System — Ask anything about your PDF using AI

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-red) ![LangChain](https://img.shields.io/badge/LangChain-RAG-green) ![Groq](https://img.shields.io/badge/Groq-Free%20LLM-orange) ![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-purple)

---

## 📌 Overview

**AI Chat with Your PDF** is a Retrieval-Augmented Generation (RAG) application that lets you upload any PDF and ask questions about it in natural language. The system retrieves the most relevant sections and uses a large language model to generate precise, context-aware answers with source citations.

Built as a portfolio project demonstrating end-to-end AI application development.

---

## ⚙️ How It Works

```
PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Index
                                                              ↓
User Question → Embed Query → Similarity Search → Top-4 Chunks
                                                              ↓
                                              Groq LLM → Answer + Sources
```

1. **PDF Parsing** — PyMuPDF extracts text page by page
2. **Chunking** — Text split into 1000-char chunks with 200-char overlap
3. **Embeddings** — HuggingFace `all-MiniLM-L6-v2` converts chunks to vectors (runs locally)
4. **Vector Store** — FAISS indexes all vectors for fast similarity search
5. **Retrieval** — User question is embedded, top-4 similar chunks retrieved
6. **Generation** — Groq `llama-3.3-70b-versatile` generates answer from context
7. **Memory** — Full chat history maintained for natural follow-up questions

---

## 🛠️ Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| PDF Parsing | PyMuPDF (fitz) | Fast, accurate text extraction |
| Text Splitting | LangChain Text Splitters | Recursive character splitting |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Runs locally — FREE |
| Vector Store | FAISS | Millisecond similarity search |
| LLM | Groq llama-3.3-70b-versatile | FREE tier API |
| Framework | LangChain Classic | RAG chain orchestration |
| UI | Streamlit | Custom dark-themed interface |

---

## 📁 Project Structure

```
AI Chat with Your PDF/
├── app.py                  # Main Streamlit app — UI and routing
├── .env                    # API keys (GROQ_API_KEY)
├── requirements.txt        # Python dependencies
└── utils/
    ├── pdf_processor.py    # PDF extraction and chunking
    ├── vector_store.py     # FAISS embeddings and search
    └── qa_chain.py         # Groq LLM + RAG chain setup
```

---

## 🚀 Installation & Setup

### 1. Create Project Folder & Virtual Environment

```powershell
mkdir "AI Chat with Your PDF"
cd "AI Chat with Your PDF"
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get your **free** Groq API key at https://console.groq.com/keys

### 4. Run the App

```powershell
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## 📦 requirements.txt

```
streamlit
langchain
langchain-community
langchain-classic
langchain-groq
langchain-text-splitters
faiss-cpu
pymupdf
python-dotenv
tiktoken
sentence-transformers
```

---

## ✨ Features

- 📄 Upload any PDF — CV, research paper, book, contract, report
- 🔍 Semantic search — finds relevant content by meaning, not just keywords
- 💬 Conversational AI — ask follow-up questions with full chat memory
- 📚 Source citations — see exactly which chunks each answer came from
- 💸 100% Free — Groq free tier + local HuggingFace embeddings
- 🎨 Clean dark UI — modern responsive Streamlit interface
- ⚡ Fast — FAISS enables millisecond-level vector search

---

## 💬 Usage

1. **Upload PDF** — Select any PDF on the home screen
2. **Process** — Click **🚀 Process & Index PDF** (first run downloads ~90MB embedding model)
3. **Ask questions** — Type anything in the chat input and click Send
4. **View sources** — Click **📚 Sources** under any answer to see retrieved chunks

**Example questions:**
- *"What is the main topic of this document?"*
- *"Summarize the key points"*
- *"What skills are mentioned in this CV?"*
- *"Write a cover letter based on this resume"*

---

## 🔑 API Info — Groq Free Tier

| Limit | Value |
|-------|-------|
| Requests/day | 14,400 |
| Requests/minute | 30 |
| Model | llama-3.3-70b-versatile |
| Cost | FREE |

---

## 🔧 Troubleshooting

**ModuleNotFoundError** — activate venv and reinstall:
```powershell
pip install langchain-classic langchain-text-splitters
```

**Groq 401 Error** — check your `.env` file:
```env
GROQ_API_KEY=gsk_your_actual_key_here
```

**Model decommissioned** — update `utils/qa_chain.py`:
```python
model_name="llama-3.3-70b-versatile"
```

**First run is slow** — HuggingFace downloads the embedding model (~90MB) once. All future runs are instant.

---

## 🎯 Skills Demonstrated

- RAG pipeline design and implementation
- Vector embeddings and semantic similarity search
- LLM integration (Groq API)
- LangChain framework for AI orchestration
- Full-stack AI app development with Streamlit
- FAISS vector database
- Conversational memory management
- Custom CSS UI theming

---

*Built with Python · LangChain · Groq · FAISS · Streamlit*