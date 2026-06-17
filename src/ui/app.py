"""
app.py — Streamlit UI
AI Research Agent
The single entry point users interact with. Wires together session_manager
(memory + graph execution) into a chat-style research interface.

Run with: streamlit run src/ui/app.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import setup_logging, validate_config, APP_NAME
from llm import check_llm_connections
from memory.session_manager import (
    create_session,
    run_and_log_research,
    get_session_history,
    list_all_sessions,
    delete_session_history,
)
from tools.retrieval_tool import ingest_text, get_collection_stats, clear_collection
from guardrails.validators import validate_user_query

logger = logging.getLogger(__name__)


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── One-time app initialisation ───────────────────────────────────────────
if "initialised" not in st.session_state:
    setup_logging()
    st.session_state.initialised = True
    st.session_state.config_status = validate_config()
    st.session_state.current_session_id = None
    st.session_state.llm_status_checked = False


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔎 " + APP_NAME)
    st.caption("Multi-agent research, powered by LangGraph")

    st.divider()

    # New session button
    if st.button("➕ New Research Session", use_container_width=True):
        st.session_state.current_session_id = create_session()
        st.rerun()

    # Session switcher
    sessions = list_all_sessions()
    if sessions:
        st.subheader("Past Sessions")
        session_labels = {
            s["session_id"]: f"{s['title'][:35]}" for s in sessions
        }
        selected = st.radio(
            "Select a session",
            options=list(session_labels.keys()),
            format_func=lambda sid: session_labels[sid],
            index=0 if st.session_state.current_session_id is None else (
                list(session_labels.keys()).index(st.session_state.current_session_id)
                if st.session_state.current_session_id in session_labels else 0
            ),
            label_visibility="collapsed",
        )
        if selected != st.session_state.current_session_id:
            st.session_state.current_session_id = selected
            st.rerun()

        if st.session_state.current_session_id:
            if st.button("🗑️ Clear This Session", use_container_width=True):
                delete_session_history(st.session_state.current_session_id)
                st.rerun()
    else:
        st.caption("No sessions yet — start a new one above.")

    st.divider()

    # System health panel
    with st.expander("⚙️ System Status", expanded=False):
        cfg = st.session_state.config_status
        if cfg["valid"]:
            st.success("All required API keys configured.")
        else:
            st.error(f"Missing keys: {', '.join(cfg['missing_keys'])}")
        for w in cfg["warnings"]:
            st.warning(w)

        if st.button("Test LLM connections"):
            with st.spinner("Pinging Groq and Gemini..."):
                results = check_llm_connections()
            for provider, info in results.items():
                if info["status"] == "ok":
                    st.success(f"{provider.upper()}: connected")
                else:
                    st.error(f"{provider.upper()}: {info['message'][:100]}")

        stats = get_collection_stats()
        st.caption(f"📚 Knowledge base: {stats['total_chunks']} chunks stored")
        if stats["total_chunks"] > 0 and st.button("Clear knowledge base"):
            clear_collection()
            st.rerun()

    st.divider()

    # Document upload for RAG
    with st.expander("📄 Add Documents to Knowledge Base", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload .txt or .md files",
            type=["txt", "md"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Ingest Documents"):
            with st.spinner("Ingesting documents..."):
                total_chunks = 0
                for f in uploaded_files:
                    text = f.read().decode("utf-8", errors="ignore")
                    count = ingest_text(text, source=f.name)
                    total_chunks += count
            st.success(f"Ingested {total_chunks} chunks from {len(uploaded_files)} file(s).")


# ── Main area ─────────────────────────────────────────────────────────────
st.title("AI Research Agent")
st.caption(
    "Ask a research question. The agent will plan, search, retrieve, "
    "synthesise, and validate its findings before answering."
)

# Ensure a session exists
if st.session_state.current_session_id is None:
    st.session_state.current_session_id = create_session()

session_id = st.session_state.current_session_id

# Display conversation history
history = get_session_history(session_id)

if not history:
    st.info("No messages yet in this session. Ask a research question below to get started.")
else:
    for msg in history:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])


# ── Chat input ─────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a research question...")

if user_input:
    # Validate the query before doing anything else (prompt injection / sanity check)
    check = validate_user_query(user_input)

    if not check.passed:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.error(f"This request couldn't be processed: {check.reason}")
    else:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            status_box = st.status("Researching...", expanded=True)

            try:
                status_box.write("📋 Planning research steps...")
                result = run_and_log_research(topic=user_input, session_id=session_id)

                final_state = result["final_state"]
                plan = final_state.get("plan", [])

                if plan:
                    status_box.write(f"🔍 Executed {len(plan)} research step(s):")
                    for step in plan:
                        status_box.write(f"   • {step['description']}")

                status_box.write("✍️ Synthesising report...")
                status_box.write("✅ Validating output...")

                if result["validation_passed"]:
                    status_box.update(label="Research complete ✅", state="complete")
                else:
                    status_box.update(label="Research complete (with warnings) ⚠️", state="complete")

                st.markdown(result["report_markdown"])

                # Show validation details for transparency
                if result["validation_messages"]:
                    with st.expander("🛡️ Guardrail details"):
                        for vm in result["validation_messages"]:
                            st.caption(vm)

                # Download button for the saved report
                if result["report_path"]:
                    report_path = Path(result["report_path"])
                    if report_path.exists():
                        st.download_button(
                            label="📥 Download report (.md)",
                            data=report_path.read_text(encoding="utf-8"),
                            file_name=report_path.name,
                            mime="text/markdown",
                        )

            except Exception as exc:
                logger.error("UI: research run failed: %s", exc)
                status_box.update(label="Research failed ❌", state="error")
                st.error(f"Something went wrong while researching this topic: {exc}")

    st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"{APP_NAME} • Session: `{session_id}` • "
    f"Powered by LangGraph, Groq, Gemini, Tavily & ChromaDB — all free tier."
)