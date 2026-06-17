"""
session_manager.py — Connects LangGraph Runs to ConversationMemory
AI Research Agent
This is the integration layer between the agent pipeline (graph.py) and the
persistence layer (conversation_memory.py). The Streamlit UI calls into this
module, not directly into graph.py or ConversationMemory — keeping the UI
simple and the logging behavior centralised in one place.
"""

import logging
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.graph import run_research
from agents.state import ResearchState
from memory.conversation_memory import ConversationMemory, list_sessions

logger = logging.getLogger(__name__)


def create_session(title: str = "") -> str:
    """
    Creates a new research session and returns its session_id.

    Args:
        title : Optional human-readable title. Defaults to a timestamp-based label.

    Returns:
        The new session_id string.
    """
    session_id = f"session_{uuid.uuid4().hex[:12]}"

    if not title:
        title = f"Research session — {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    # ConversationMemory's constructor ensures the session row exists in SQLite
    ConversationMemory(session_id=session_id, title=title)
    logger.info("Created new session | id: %s | title: '%s'", session_id, title)
    return session_id


def run_and_log_research(topic: str, session_id: Optional[str] = None) -> dict:
    """
    Runs the full LangGraph research pipeline for a topic, then logs the
    user query and assistant response into ConversationMemory so it persists
    and is retrievable as conversation history.

    Args:
        topic      : The research question/topic.
        session_id : Existing session to log into. If None, a new session is created.

    Returns:
        {
            "session_id": str,
            "final_state": ResearchState,
            "report_markdown": str,
            "report_path": str,
            "validation_passed": bool,
            "validation_messages": list[str],
        }
    """
    if not session_id:
        session_id = create_session(title=topic[:60])

    memory = ConversationMemory(session_id=session_id)

    # Log the user's research request as a conversation turn
    memory.add_message("user", topic)
    logger.info("Logged user query to memory | session: %s", session_id)

    # Run the actual multi-agent pipeline
    final_state: ResearchState = run_research(topic=topic, session_id=session_id)

    # Log the assistant's response (the report) as a conversation turn
    report_md = final_state.get("report_markdown", "")
    memory.add_message("assistant", report_md)
    logger.info("Logged assistant report to memory | session: %s | report length: %d chars",
                session_id, len(report_md))

    # Record the topic itself as an entity, so follow-up questions in this
    # session ("tell me more about X") have context to draw on
    memory.add_entity(entity=topic, context=final_state.get("report_markdown", "")[:200])

    return {
        "session_id": session_id,
        "final_state": final_state,
        "report_markdown": report_md,
        "report_path": final_state.get("report_path", ""),
        "validation_passed": final_state.get("validation_passed", False),
        "validation_messages": final_state.get("validation_messages", []),
    }


def get_session_history(session_id: str) -> list[dict]:
    """
    Retrieves the full conversation history for a session, formatted for
    display in the Streamlit UI's chat view.

    Args:
        session_id : The session to retrieve.

    Returns:
        List of dicts: [{"role": ..., "content": ..., "timestamp": ...}]
    """
    memory   = ConversationMemory(session_id=session_id)
    messages = memory.get_history()
    return [{"role": m.role, "content": m.content, "timestamp": m.timestamp} for m in messages]


def list_all_sessions() -> list[dict]:
    """
    Lists every research session stored in the database, most recent first.
    Used by the Streamlit sidebar to populate the session switcher.

    Returns:
        List of dicts: [{"session_id": ..., "title": ..., "created_at": ...}]
    """
    return list_sessions()


def delete_session_history(session_id: str) -> None:
    """
    Clears all messages and entities for a session (used by a "clear chat"
    button in the UI). The session row itself is kept so it still appears
    in the sidebar, just empty.

    Args:
        session_id : The session to clear.
    """
    memory = ConversationMemory(session_id=session_id)
    memory.clear()
    logger.info("Cleared session history | session: %s", session_id)


# ── Quick self-test (run: python -m memory.session_manager from src/) ────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    print("\n" + "=" * 55)
    print("  Session Manager Test")
    print("=" * 55)

    print("\n[1] Creating a new session...")
    sid = create_session(title="Test: LangGraph capabilities")
    print(f"    Session created: {sid}")

    print("\n[2] Listing all sessions...")
    sessions = list_all_sessions()
    print(f"    Total sessions found: {len(sessions)}")
    for s in sessions[:5]:
        print(f"      • {s['session_id']} | {s['title']}")

    print("\n[3] Manually logging messages (no LLM calls, simulating a run)...")
    memory = ConversationMemory(session_id=sid)
    memory.add_message("user", "What is LangGraph?")
    memory.add_message("assistant", "# LangGraph Report\n\nLangGraph is a multi-agent orchestration library.")
    memory.add_entity("LangGraph", context="Multi-agent orchestration library")
    print("    2 messages + 1 entity logged.")

    print("\n[4] Retrieving session history...")
    history = get_session_history(sid)
    for h in history:
        print(f"      [{h['role']}] {h['content'][:60]}")

    print("\n[5] Clearing session history...")
    delete_session_history(sid)
    history = get_session_history(sid)
    print(f"    History after clear: {len(history)} messages")

    print("\n" + "=" * 55)
    print("Session manager test complete.")
    print("=" * 55 + "\n")
    print("NOTE: run_and_log_research() was not tested here since it calls")
    print("the full LLM pipeline — test that manually once Groq's daily quota resets.")
    print()