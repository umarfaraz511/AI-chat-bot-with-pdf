import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from utils.pdf_processor import extract_text_from_pdf, split_text_into_chunks
from utils.vector_store import create_vector_store
from utils.qa_chain import create_qa_chain

load_dotenv()

st.set_page_config(
    page_title="AI Chat with Your PDF",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif !important; }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .navbar {
        display: flex; align-items: center;
        justify-content: space-between;
        padding: 16px 32px;
        background: rgba(15,52,96,0.3);
        border-bottom: 1px solid rgba(233,69,96,0.2);
        border-radius: 0 0 16px 16px;
        margin-bottom: 32px;
    }
    .navbar-brand {
        font-size: 1.4rem; font-weight: 800;
        background: linear-gradient(135deg, #e94560, #a855f7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .navbar-subtitle { font-size: 0.78rem; color: #6b7280; }
    .navbar-badge {
        background: linear-gradient(135deg, #e94560, #0f3460);
        color: white; padding: 6px 14px; border-radius: 20px;
        font-size: 0.72rem; font-weight: 600;
    }

    .steps-row {
        display: grid; grid-template-columns: repeat(3, 1fr);
        gap: 16px; margin: 28px 0;
    }
    .step-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 24px 20px; text-align: center;
    }
    .step-num {
        font-size: 0.7rem; font-weight: 700;
        letter-spacing: 2px; text-transform: uppercase;
        color: #e94560; margin-bottom: 10px;
    }
    .step-icon { font-size: 2rem; margin-bottom: 10px; }
    .step-title { font-size: 1rem; font-weight: 700; color: #e0e0ff; margin-bottom: 6px; }
    .step-desc { font-size: 0.8rem; color: #6b7280; line-height: 1.6; }

    .section-divider {
        display: flex; align-items: center;
        gap: 12px; margin: 8px 0 20px;
    }
    .divider-line { flex: 1; height: 1px; background: rgba(255,255,255,0.07); }
    .divider-text {
        font-size: 0.7rem; color: #4b5563;
        text-transform: uppercase; letter-spacing: 2px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(15,52,96,0.3) !important;
        border: 2px dashed rgba(233,69,96,0.4) !important;
        border-radius: 16px !important;
        padding: 30px !important;
    }
    [data-testid="stFileUploaderDropzone"] * { color: #a0a0cc !important; }

    .file-card {
        background: linear-gradient(135deg, rgba(15,52,96,0.5), rgba(83,52,131,0.3));
        border: 1px solid rgba(233,69,96,0.3);
        border-radius: 14px; padding: 16px 20px;
        display: flex; align-items: center;
        gap: 14px; margin-bottom: 16px;
    }
    .file-name { font-size: 0.95rem; font-weight: 600; color: #e0e0ff; }
    .file-size { font-size: 0.78rem; color: #6b7280; margin-top: 2px; }

    .stButton > button {
        background: linear-gradient(135deg, #e94560, #7c3aed) !important;
        color: white !important; border: none !important;
        border-radius: 14px !important; padding: 12px 20px !important;
        font-weight: 700 !important; font-size: 0.95rem !important;
        width: 100% !important; transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(233,69,96,0.5) !important;
    }

    .status-bar {
        background: linear-gradient(135deg, rgba(15,52,96,0.6), rgba(83,52,131,0.3));
        border: 1px solid rgba(233,69,96,0.3);
        border-radius: 14px; padding: 14px 20px;
        display: flex; align-items: center;
        justify-content: space-between;
        margin-bottom: 16px; flex-wrap: wrap; gap: 10px;
    }
    .status-doc { display: flex; align-items: center; gap: 10px; }
    .status-dot {
        width: 10px; height: 10px; background: #22c55e;
        border-radius: 50%; box-shadow: 0 0 8px #22c55e;
        animation: pulse 2s infinite; flex-shrink: 0;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
    .status-filename { font-size: 0.9rem; font-weight: 600; color: #e0e0ff; }
    .status-meta { font-size: 0.75rem; color: #6b7280; margin-top: 2px; }
    .status-stats { display: flex; gap: 20px; }
    .stat-item { text-align: center; }
    .stat-val { font-size: 1.3rem; font-weight: 800; color: #e94560; line-height: 1; }
    .stat-lbl { font-size: 0.65rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; }

    .msg-user { display: flex; justify-content: flex-end; margin: 16px 0; }
    .msg-user-bubble {
        background: linear-gradient(135deg, #0f3460, #533483);
        padding: 14px 18px;
        border-radius: 18px 18px 4px 18px;
        max-width: 65%; color: white;
        font-size: 0.92rem; line-height: 1.6;
        box-shadow: 0 4px 15px rgba(83,52,131,0.4);
    }
    .msg-ai {
        display: flex; justify-content: flex-start;
        margin: 16px 0; gap: 10px; align-items: flex-start;
    }
    .msg-ai-avatar {
        width: 34px; height: 34px;
        background: linear-gradient(135deg, #e94560, #7c3aed);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 16px; flex-shrink: 0; margin-top: 2px;
    }
    .msg-ai-bubble {
        background: linear-gradient(135deg, rgba(15,52,96,0.6), rgba(26,26,46,0.8));
        border: 1px solid rgba(233,69,96,0.2);
        padding: 14px 18px;
        border-radius: 4px 18px 18px 18px;
        max-width: 70%; color: #e0e0ff;
        font-size: 0.92rem; line-height: 1.7;
    }

    .source-box {
        background: rgba(15,52,96,0.3);
        border-left: 3px solid #e94560;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        font-size: 0.78rem; color: #a0a0cc;
        margin: 6px 0; line-height: 1.6;
    }

    /* ── EXPANDER: hide all default Streamlit elements, show only book icon ── */
    [data-testid="stExpander"] {
        background: rgba(15,52,96,0.2) !important;
        border: 1px solid rgba(15,52,96,0.5) !important;
        border-radius: 10px !important;
        margin-top: 4px !important;
        margin-bottom: 20px !important;
    }
    [data-testid="stExpander"] summary {
        padding: 8px 12px !important;
    }
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary div {
        display: none !important;
    }
    [data-testid="stExpander"] summary svg {
        display: none !important;
    }
    [data-testid="stExpander"] summary::after {
        content: '📚 Sources' !important;
        color: #6b7280 !important;
        font-size: 0.8rem !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
        padding: 10px 14px !important;
    }

    .input-spacer { height: 20px; clear: both; display: block; }

    .stTextInput label { display: none !important; }
    .stTextInput > div {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    .stTextInput > div > div {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
    .stTextInput > div > div > input {
        background: rgba(15,52,96,0.5) !important;
        border: 1.5px solid rgba(233,69,96,0.35) !important;
        border-radius: 14px !important;
        color: #ffffff !important;
        padding: 14px 20px !important;
        font-size: 0.92rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s !important;
        box-shadow: none !important;
        outline: none !important;
        width: 100% !important;
        -webkit-appearance: none !important;
        appearance: none !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #e94560 !important;
        box-shadow: 0 0 0 3px rgba(233,69,96,0.15) !important;
        background: rgba(15,52,96,0.7) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #4b5563 !important;
        font-size: 0.88rem !important;
    }
    .stTextInput [data-baseweb="input"] svg { display: none !important; }
    .stTextInput [data-baseweb="input-container"] { background: transparent !important; }

    .stSuccess {
        background: rgba(34,197,94,0.1) !important;
        border: 1px solid rgba(34,197,94,0.3) !important;
        border-radius: 10px !important; color: #22c55e !important;
    }
    .stSpinner > div { border-top-color: #e94560 !important; }
    hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.2rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

# ── NAVBAR ─────────────────────────────────────────────────────
st.markdown("""
<div class='navbar'>
    <div>
        <div class='navbar-brand'>🤖 AI Chat with Your PDF</div>
        <div class='navbar-subtitle'>Powered by RAG · LangChain · Groq · FAISS</div>
    </div>
    <div class='navbar-badge'>⚡ RAG Pipeline Active</div>
</div>
""", unsafe_allow_html=True)

# ── UPLOAD SCREEN ──────────────────────────────────────────────
if not st.session_state.pdf_processed:
    st.markdown("""
    <div class='steps-row'>
        <div class='step-card'>
            <div class='step-num'>Step 01</div>
            <div class='step-icon'>📤</div>
            <div class='step-title'>Upload PDF</div>
            <div class='step-desc'>Select any PDF — CV, research paper, contract, or book.</div>
        </div>
        <div class='step-card'>
            <div class='step-num'>Step 02</div>
            <div class='step-icon'>🧠</div>
            <div class='step-title'>AI Indexes It</div>
            <div class='step-desc'>RAG pipeline chunks text, creates embeddings, builds a search index.</div>
        </div>
        <div class='step-card'>
            <div class='step-num'>Step 03</div>
            <div class='step-icon'>💬</div>
            <div class='step-title'>Chat Freely</div>
            <div class='step-desc'>Ask anything in plain language and get answers with source citations.</div>
        </div>
    </div>
    <div class='section-divider'>
        <div class='divider-line'></div>
        <div class='divider-text'>Upload your document to begin</div>
        <div class='divider-line'></div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        uploaded_file = st.file_uploader("Drop PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded_file:
            st.markdown(f"""
            <div class='file-card'>
                <div style='font-size:2rem'>📄</div>
                <div>
                    <div class='file-name'>{uploaded_file.name}</div>
                    <div class='file-size'>{uploaded_file.size / 1024:.1f} KB · PDF Document</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🚀 Process & Index PDF"):
                with st.spinner("📖 Reading PDF..."):
                    raw_text = extract_text_from_pdf(uploaded_file)
                with st.spinner("✂️ Splitting into chunks..."):
                    chunks = split_text_into_chunks(raw_text)
                    st.session_state.total_chunks = len(chunks)
                with st.spinner("🧠 Creating embeddings..."):
                    vector_store = create_vector_store(chunks)
                with st.spinner("⚡ Building AI chain..."):
                    st.session_state.qa_chain = create_qa_chain(vector_store)
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []
                st.success("✅ PDF Ready!")
                st.rerun()

# ── CHAT SCREEN ────────────────────────────────────────────────
else:
    st.markdown(f"""
    <div class='status-bar'>
        <div class='status-doc'>
            <div class='status-dot'></div>
            <div>
                <div class='status-filename'>📄 {st.session_state.pdf_name}</div>
                <div class='status-meta'>Document indexed and ready</div>
            </div>
        </div>
        <div class='status-stats'>
            <div class='stat-item'>
                <div class='stat-val'>{st.session_state.total_chunks}</div>
                <div class='stat-lbl'>Chunks</div>
            </div>
            <div class='stat-item'>
                <div class='stat-val'>{len(st.session_state.chat_history)}</div>
                <div class='stat-lbl'>Messages</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([4, 1, 1])
    with col_b:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    with col_c:
        if st.button("📂 New PDF"):
            st.session_state.pdf_processed = False
            st.session_state.chat_history = []
            st.session_state.qa_chain = None
            st.rerun()

    st.markdown("---")

    # ── MESSAGES ──
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class='msg-user'>
                <div class='msg-user-bubble'>{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='msg-ai'>
                <div class='msg-ai-avatar'>🤖</div>
                <div class='msg-ai-bubble'>{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
            # ✅ BUG FIX: correct indentation — expander is inside the if block
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(
                            f"<div class='source-box'><b>Source {i+1}:</b> {source[:300]}...</div>",
                            unsafe_allow_html=True
                        )

    # ── SPACER ──
    st.markdown("<div class='input-spacer'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── INPUT ROW ──
    col1, col2 = st.columns([6, 1])
    with col1:
        user_question = st.text_input(
            "msg",
            placeholder="Ask anything about your document...",
            label_visibility="collapsed",
            key="user_input"
        )
    with col2:
        send_button = st.button("Send 🚀")

    # ── HANDLE SEND ──
    if send_button and user_question:
        st.session_state.chat_history.append({
            "role": "user", "content": user_question
        })
        lc_history = []
        for msg in st.session_state.chat_history[:-1]:
            if msg["role"] == "user":
                lc_history.append(HumanMessage(content=msg["content"]))
            else:
                lc_history.append(AIMessage(content=msg["content"]))

        with st.spinner("🤔 Thinking..."):
            response = st.session_state.qa_chain.invoke({
                "input": user_question,
                "chat_history": lc_history
            })
            answer = response["answer"]
            sources = [doc.page_content for doc in response.get("context", [])]

        st.session_state.chat_history.append({
            "role": "assistant", "content": answer, "sources": sources
        })
        st.rerun()