import os
import re
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from question_safety import REFUSAL_MESSAGE, is_disallowed
from web_fallback import search_public_sources, summarize_public_sources
from chat_history import load_history, save_history
from rag_pipeline import (
    VECTOR_DIR,
    EmbeddingManager,
    RAGRetriever,
    VectorStore,
    ask,
    create_llm,
)

st.set_page_config(
    page_title="LegalRAG for Supreme Court Judgments",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

NO_CONTEXT_MESSAGE = "No relevant context found in the indexed judgments."
NO_WEB_MESSAGE = "No reliable related information was found from public sources at this time."


def config(key, default=""):
    try:
        return str(st.secrets.get(key) or os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


@st.cache_resource(show_spinner=False)
def load_retriever():
    store = VectorStore(persist_directory=VECTOR_DIR)
    if store.collection.count() == 0:
        raise RuntimeError("Index is empty. Run python prepare_index.py first.")
    return RAGRetriever(store, EmbeddingManager())


@st.cache_resource(show_spinner=False)
def load_llm():
    return create_llm(
        api_key=config("GROQ_API_KEY"),
        model=config("GROQ_MODEL", "openai/gpt-oss-120b"),
    )


def escape_markdown(value):
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", str(value))


def render_web_sources(web_sources):
    if not web_sources:
        st.info(NO_WEB_MESSAGE)
        return
    st.markdown("**Related information from public web sources**")
    st.caption("These results are from the internet, not your indexed judgment PDFs. Verify the original decisions before relying on them.")
    for i, item in enumerate(web_sources, start=1):
        title = escape_markdown(item["title"])
        st.markdown(f"{i}. **{title}** — [{item['domain']}]({item['url']})")
        st.caption(item["snippet"])


def render_assistant_message(message):
    status = message.get("status")
    if status == "no_context":
        st.error(NO_CONTEXT_MESSAGE)
        if message.get("content"):
            st.markdown(message["content"])
        render_web_sources(message.get("web_sources", []))
    elif status == "blocked":
        st.warning(REFUSAL_MESSAGE)
    elif status == "error":
        st.error(message.get("content", "The request could not be processed."))
    else:
        st.markdown(message.get("content", ""))
        if message.get("sources"):
            with st.expander("Judgment sources"):
                for source in message["sources"]:
                    st.write(source)


st.title("LegalRAG - Supreme Court Judgments")
st.caption("Ask questions grounded in Indian Supreme Court judgments.")
st.caption("**Disclaimer:** This project is developed solely for educational and academic purposes. Any unintended, inaccurate, or inappropriate responses do not reflect the views or intentions of the developer. The application is not intended as a substitute for professional legal advice.")

chat_id = st.query_params.get("chat", "")
if not isinstance(chat_id, str) or not re.fullmatch(r"[0-9a-f]{32}", chat_id):
    chat_id = uuid.uuid4().hex
    st.query_params["chat"] = chat_id

if st.session_state.get("chat_id") != chat_id:
    st.session_state.chat_id = chat_id
    st.session_state.messages = load_history(chat_id)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message)
        else:
            st.write(message["content"])

SUGGESTED_QUESTIONS = [
    "What factors does the Supreme Court consider when granting bail?",
    "How has the Supreme Court interpreted the right to a fair trial?",
    "What principles govern the admissibility of evidence in court?",
    "When can the Supreme Court intervene in a High Court decision?",
]

suggested_question = None
if not st.session_state.messages:
    st.markdown("**Suggested questions**")
    left, right = st.columns(2)
    for i, suggestion in enumerate(SUGGESTED_QUESTIONS):
        with (left if i % 2 == 0 else right):
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                suggested_question = suggestion

typed_question = st.chat_input("Ask a question about a Supreme Court judgment...")
question = typed_question or suggested_question

if question and question.strip():
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        save_history(st.session_state.chat_id, st.session_state.messages)
    except Exception:
        st.warning("This conversation could not be saved.")
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            llm = load_llm()
            if is_disallowed(question, llm):
                message = {"role": "assistant", "status": "blocked", "content": REFUSAL_MESSAGE}
            else:
                with st.spinner("Searching Supreme Court judgments..."):
                    retriever = load_retriever()
                    result = ask(question, rag_retriever=retriever, llm=llm)

                if result.get("status") == "no_context":
                    
                    with st.spinner("Looking for related public legal information..."):
                        web_sources = search_public_sources(question)
                        overview = summarize_public_sources(question, web_sources, llm)
                    message = {
                        "role": "assistant", "status": "no_context",
                        "content": overview, "web_sources": web_sources,
                    }
                else:
                    message = {
                        "role": "assistant",
                        "status": result.get("status", "answered"),
                        "content": result.get("answer", ""),
                        "sources": list(dict.fromkeys(result.get("source", [])))
                            if result.get("status") == "answered" else [],
                    }
        except Exception:
            message = {
                "role": "assistant", "status": "error",
                "content": "The request could not be processed. Please try again later.",
            }

        render_assistant_message(message)
        st.session_state.messages.append(message)
        try:
            save_history(st.session_state.chat_id, st.session_state.messages)
        except Exception:
            st.warning("This conversation could not be saved.")
