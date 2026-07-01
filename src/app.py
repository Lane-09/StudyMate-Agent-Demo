from pathlib import Path

import streamlit as st

from graph import run_agent


# =========================
# Page configuration
# =========================

st.set_page_config(
    page_title="StudyMate Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PAGES_DIR = PROJECT_ROOT / "data" / "processed" / "pdf_pages"


# =========================
# Custom CSS
# =========================

st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7fb;
        color: #111827;
    }

    .block-container {
        max-width: 1160px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 750 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }

    section[data-testid="stSidebar"] div.stButton > button {
        background: #1e293b !important;
        border: 1px solid #475569 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
    }

    section[data-testid="stSidebar"] div.stButton > button:hover {
        background: #334155 !important;
    }

    /* Hero */
    .hero {
        background: #111827;
        color: white;
        padding: 2rem 2.2rem;
        border-radius: 22px;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.05rem;
        font-weight: 780;
        letter-spacing: -0.03em;
        margin-bottom: 0.45rem;
    }

    .hero-text {
        color: #d1d5db;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 900px;
        margin-bottom: 0.2rem;
    }

    .tag {
        display: inline-block;
        background: #1f2937;
        color: #e5e7eb;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        border: 1px solid #374151;
        margin-right: 0.35rem;
        margin-top: 0.85rem;
    }

    .section-title {
        font-size: 1.22rem;
        font-weight: 760;
        color: #111827;
        margin-top: 1.3rem;
        margin-bottom: 0.7rem;
    }

    .section-note {
        color: #6b7280;
        font-size: 0.92rem;
        margin-bottom: 0.8rem;
    }

    .source-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }

    .source-title {
        font-weight: 720;
        color: #111827;
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
    }

    .source-meta {
        color: #6b7280;
        font-size: 0.85rem;
        margin-bottom: 0.7rem;
    }

    .meta-pill {
        display: inline-block;
        background: #eef2ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .history-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }

    .history-question {
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.35rem;
    }

    .history-answer {
        color: #374151;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    .history-meta {
        color: #6b7280;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }

    .sidebar-history-item {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 0.65rem 0.75rem;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
        line-height: 1.45;
    }

    div.stButton > button {
        border-radius: 12px;
        border: 1px solid #d1d5db;
        background: #ffffff;
        color: #111827;
        font-weight: 620;
        padding: 0.55rem 0.9rem;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: #111827;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.12);
    }

    textarea {
        background: #ffffff !important;
        color: #111827 !important;
        border-radius: 16px !important;
        border: 1px solid #d1d5db !important;
        font-size: 1rem !important;
    }

    textarea::placeholder {
        color: #9ca3af !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Session state
# =========================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_question" not in st.session_state:
    st.session_state.current_question = ""


# =========================
# Helper functions
# =========================

def set_question(question: str):
    st.session_state.current_question = question


def safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def shorten_text(text: str, max_length: int = 160) -> str:
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length].strip() + "..."


def resolve_page_image(source_file: str, page_number):
    """
    Find PDF page image generated by pdf_processor.py.
    Expected format:
    Lecture2_Multimodal_Embedding_page_13.png
    """
    page_number = safe_int(page_number)

    if not source_file or page_number is None:
        return None

    stem = Path(source_file).stem

    candidates = [
        PDF_PAGES_DIR / f"{stem}_page_{page_number}.png",
        PDF_PAGES_DIR / f"{stem}_page_{page_number}.jpg",
        PDF_PAGES_DIR / f"{stem}_page_{page_number}.jpeg",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for file in PDF_PAGES_DIR.glob("*"):
        name = file.stem.lower()
        if stem.lower() in name and f"page_{page_number}" in name:
            return file

    return None


def resolve_raw_image(metadata: dict):
    """
    Display standalone images such as Transformer_Structure.png.
    """
    source_path = metadata.get("source_path", "")

    if source_path and Path(source_path).exists():
        return Path(source_path)

    return None


def unique_retrieved_docs(retrieved_docs):
    """
    Avoid showing duplicate source pages.
    """
    seen = set()
    unique_docs = []

    for doc in retrieved_docs:
        metadata = doc.get("metadata", {})
        source_file = metadata.get("source_file", "")
        page_number = metadata.get("page_number", "")
        source_path = metadata.get("source_path", "")

        key = (source_file, page_number, source_path)

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs


def render_source_images(retrieved_docs, max_sources=5):
    """
    Show source evidence directly as PDF page images or raw images.
    """
    if not retrieved_docs:
        st.info("No retrieved source pages available.")
        return

    docs = unique_retrieved_docs(retrieved_docs)[:max_sources]

    for i, doc in enumerate(docs, start=1):
        metadata = doc.get("metadata", {})

        source_file = metadata.get("source_file", "Unknown source")
        page_number = metadata.get("page_number", "")
        modality = metadata.get("modality", "unknown")

        page_image = resolve_page_image(source_file, page_number)
        raw_image = resolve_raw_image(metadata)

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-title">Source {i}</div>
                <div class="source-meta">
                    {source_file}
                    {' | page ' + str(page_number) if page_number else ''}
                    | modality: {modality}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if page_image:
            st.image(
                str(page_image),
                caption=f"{source_file} - page {page_number}",
                use_container_width=True
            )
        elif raw_image:
            st.image(
                str(raw_image),
                caption=f"{source_file}",
                use_container_width=True
            )
        else:
            st.warning(f"Source page image not found for {source_file}, page {page_number}.")
            with st.expander("Show retrieved text instead"):
                st.text(doc.get("text", "")[:1500])


def render_agent_trace(latest):
    st.write("**Selected route**")
    st.code(latest.get("route", ""))

    st.write("**Tool calls**")
    tool_calls = latest.get("tool_calls", [])
    if tool_calls:
        for tool in tool_calls:
            st.write(f"- {tool}")
    else:
        st.write("No tool calls recorded.")

    st.write("**Trace steps**")
    trace_steps = latest.get("trace_steps", [])
    if trace_steps:
        for step in trace_steps:
            st.write(f"- {step}")
    else:
        st.write("No trace steps recorded.")


def render_sidebar_history():
    """
    Show recent conversation history in the sidebar.
    """
    history = st.session_state.chat_history

    if not history:
        st.caption("No history yet.")
        return

    recent_items = list(reversed(history[-5:]))

    for index, item in enumerate(recent_items, start=1):
        question = shorten_text(item.get("query", ""), 70)
        answer = shorten_text(item.get("answer", ""), 90)

        st.markdown(
            f"""
            <div class="sidebar-history-item">
                <strong>Q{len(history) - index + 1}</strong><br>
                {question}<br>
                <span style="color:#9ca3af;">{answer}</span>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_main_history():
    """
    Show full conversation history in the main page.
    """
    history = st.session_state.chat_history

    if not history:
        st.info("No conversation history yet.")
        return

    for i, item in enumerate(reversed(history), start=1):
        number = len(history) - i + 1
        question = item.get("query", "")
        answer = item.get("answer", "")
        sources = item.get("sources", [])

        with st.expander(f"Conversation {number}: {shorten_text(question, 90)}", expanded=(i == 1)):
            st.markdown(
                f"""
                <div class="history-card">
                    <div class="history-question">Question</div>
                    <div class="history-answer">{question}</div>
                    <div class="history-question" style="margin-top: 0.8rem;">Answer</div>
                    <div class="history-answer">{answer}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if sources:
                st.write("**Sources**")
                for source in sources:
                    st.markdown(
                        f'<span class="meta-pill">{source}</span>',
                        unsafe_allow_html=True
                    )


# =========================
# Sidebar
# =========================

with st.sidebar:
    st.markdown("## 🎓 StudyMate Agent")
    st.markdown(
        """
        Personalised multimodal study assistant for **INFS4205/7205**.
        """
    )

    st.markdown("---")

    st.markdown("### How to use")
    st.markdown(
        """
        1. Choose an example or type a question.  

        2. Click **Run StudyMate Agent**.  

        3. Read the generated answer.  

        4. Check source pages for detailed reference.  
        """
    )

    st.markdown("---")

    st.markdown("### What it uses")
    st.markdown(
        """
        - Lecture 1–11 PDFs  
        - PDF text + page OCR  
        - ChromaDB retrieval  
        - LangGraph agent  
        - Memory + trace log  
        """
    )

    st.markdown("---")

    st.markdown("### Recent History")
    render_sidebar_history()

    st.markdown("---")

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")


# =========================
# Main page
# =========================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">StudyMate Agent</div>
        <div class="hero-text">
            Ask questions about your lecture materials. The agent retrieves relevant evidence
            from your knowledge base and displays source pages for detailed reference.
        </div>
        <span class="tag">Lecture 1–11</span>
        <span class="tag">PDF + OCR</span>
        <span class="tag">ChromaDB</span>
        <span class="tag">LangGraph</span>
        <span class="tag">Source Pages</span>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# Example questions
# =========================

with st.expander("Example questions", expanded=True):
    st.markdown(
        '<div class="section-note">Click one example to fill the question box.</div>',
        unsafe_allow_html=True
    )

    e1, e2, e3 = st.columns(3)

    with e1:
        if st.button("What is Transformer?"):
            set_question("What is a Transformer according to Lecture 2?")

        if st.button("Explain PEFT"):
            set_question("Explain PEFT and LoRA in beginner-friendly language.")

    with e2:
        if st.button("Why Multimodal RAG?"):
            set_question("Why does RAG help make MLLMs more reliable?")

        if st.button("Explain source image"):
            set_question("Explain the Transformer structure image, including encoder, decoder, and attention.")

    with e3:
        if st.button("What is an agent?"):
            set_question("What is an agent according to Lecture 10?")

        if st.button("Beginner explanation"):
            set_question("Explain multimodal alignment step by step for a beginner.")


# =========================
# Question input
# =========================

st.markdown('<div class="section-title">Ask your question</div>', unsafe_allow_html=True)

query = st.text_area(
    label="Question",
    value=st.session_state.current_question,
    placeholder="Example: What is LoRA according to Lecture 11?",
    height=120,
    label_visibility="collapsed"
)

run_button = st.button("Run StudyMate Agent")


# =========================
# Run agent
# =========================

if run_button:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving lecture evidence and generating answer..."):
            result = run_agent(query)

        st.session_state.chat_history.append(
            {
                "query": query,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "route": result.get("route", ""),
                "retrieved_docs": result.get("retrieved_docs", []),
                "tool_calls": result.get("tool_calls", []),
                "trace_steps": result.get("trace_steps", [])
            }
        )


# =========================
# Output area
# =========================

if st.session_state.chat_history:
    latest = st.session_state.chat_history[-1]

    st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(latest["answer"])

    st.markdown('<div class="section-title">Source pages</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Retrieved PDF source pages or images are shown below for detailed reference.</div>',
        unsafe_allow_html=True
    )

    render_source_images(latest["retrieved_docs"], max_sources=5)

    st.markdown('<div class="section-title">Conversation history</div>', unsafe_allow_html=True)
    render_main_history()

    tabs = st.tabs(["Retrieved text", "Agent trace"])

    with tabs[0]:
        st.markdown("### Retrieved text evidence")
        retrieved_docs = latest.get("retrieved_docs", [])

        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs, start=1):
                metadata = doc.get("metadata", {})
                source_file = metadata.get("source_file", "Unknown source")
                page_number = metadata.get("page_number", "")
                modality = metadata.get("modality", "unknown")

                with st.expander(f"Evidence {i}: {source_file} | page {page_number}"):
                    st.write("**Source file:**", source_file)
                    st.write("**Page number:**", page_number)
                    st.write("**Modality:**", modality)
                    st.write("**Distance:**", doc.get("distance"))
                    st.text(doc.get("text", "")[:1800])
        else:
            st.write("No retrieved text evidence.")

    with tabs[1]:
        st.markdown("### Agent trace")
        render_agent_trace(latest)

else:
    st.info("Ask a question or choose an example to begin.")