"""
conversation_memory.py — Memory-Enabled Conversation State
AI Research Agent
Maintains conversational context across a research session:
  - Rolling message history (token-bounded)
  - Entity memory (topics/terms mentioned, for follow-up question context)
  - SQLite persistence so sessions survive app restarts
"""

import logging
import sys
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SQLITE_DB_PATH, MEMORY_MAX_TOKENS, ENTITY_MEMORY_K

logger = logging.getLogger(__name__)


# ── Message container ─────────────────────────────────────────────────────────
@dataclass
class Message:
    """A single turn in the conversation."""
    role:      str                 # "user" | "assistant"
    content:   str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ── SQLite schema setup ───────────────────────────────────────────────────────
def _get_connection() -> sqlite3.Connection:
    """Opens a SQLite connection, creating the DB file/tables if needed."""
    Path(SQLITE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Creates the required tables if they don't already exist:
      - sessions       : one row per research session
      - messages        : conversation turns, linked to a session
      - entities        : extracted topics/terms per session, for context recall
    """
    conn = _get_connection()
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            title      TEXT DEFAULT ''
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            timestamp  TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            entity     TEXT NOT NULL,
            context    TEXT DEFAULT '',
            timestamp  TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    """)

    conn.commit()
    conn.close()
    logger.info("SQLite database initialised at: %s", SQLITE_DB_PATH)


# ── ConversationMemory class ──────────────────────────────────────────────────
class ConversationMemory:
    """
    Manages conversation history and entity memory for a single research session.
    Backed by SQLite for persistence across app restarts.
    """

    def __init__(self, session_id: str, title: str = ""):
        """
        Args:
            session_id : Unique identifier for this conversation session.
            title      : Optional human-readable session title.
        """
        self.session_id = session_id
        init_db()
        self._ensure_session(title)
        logger.info("ConversationMemory initialised for session: %s", session_id)

    def _ensure_session(self, title: str) -> None:
        """Creates the session row if it doesn't already exist."""
        conn = _get_connection()
        cur  = conn.cursor()
        cur.execute("SELECT session_id FROM sessions WHERE session_id = ?", (self.session_id,))
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO sessions (session_id, created_at, title) VALUES (?, ?, ?)",
                (self.session_id, datetime.now().isoformat(), title),
            )
            conn.commit()
            logger.info("New session created: %s", self.session_id)
        conn.close()

    # ── Message history ────────────────────────────────────────────────────
    def add_message(self, role: str, content: str) -> None:
        """
        Appends a message to the conversation history.

        Args:
            role    : "user" or "assistant".
            content : The message text.
        """
        conn = _get_connection()
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (self.session_id, role, content, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
        logger.info("Message added | session: %s | role: %s | len: %d chars", self.session_id, role, len(content))

    def get_history(self, limit: Optional[int] = None) -> list[Message]:
        """
        Retrieves conversation history for this session, oldest first.

        Args:
            limit : Optional max number of most-recent messages to return.

        Returns:
            List of Message objects.
        """
        conn = _get_connection()
        cur  = conn.cursor()

        if limit:
            cur.execute(
                "SELECT role, content, timestamp FROM messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (self.session_id, limit),
            )
            rows = list(reversed(cur.fetchall()))
        else:
            cur.execute(
                "SELECT role, content, timestamp FROM messages "
                "WHERE session_id = ? ORDER BY id ASC",
                (self.session_id,),
            )
            rows = cur.fetchall()

        conn.close()
        return [Message(role=r["role"], content=r["content"], timestamp=r["timestamp"]) for r in rows]

    def get_history_as_text(self, limit: Optional[int] = None, max_chars: int = MEMORY_MAX_TOKENS * 4) -> str:
        """
        Returns conversation history formatted as plain text, suitable for
        injecting into an LLM prompt as context. Truncates from the oldest
        messages if it exceeds max_chars.

        Args:
            limit     : Optional max number of recent messages to include.
            max_chars : Rough character budget (approx 4 chars/token).

        Returns:
            Formatted conversation text, most recent messages prioritised.
        """
        messages = self.get_history(limit=limit)
        if not messages:
            return ""

        lines = [f"{m.role.upper()}: {m.content}" for m in messages]
        text  = "\n".join(lines)

        # Truncate from the front if too long, keeping the most recent context
        if len(text) > max_chars:
            text = "...[earlier conversation truncated]...\n" + text[-max_chars:]

        return text

    # ── Entity memory ───────────────────────────────────────────────────────
    def add_entity(self, entity: str, context: str = "") -> None:
        """
        Records a topic/entity mentioned in the conversation, used to maintain
        context for follow-up questions like "tell me more about that".

        Args:
            entity  : The topic/term/entity name.
            context : Optional short context snippet about the entity.
        """
        conn = _get_connection()
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO entities (session_id, entity, context, timestamp) VALUES (?, ?, ?, ?)",
            (self.session_id, entity, context, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
        logger.info("Entity recorded | session: %s | entity: '%s'", self.session_id, entity)

    def get_recent_entities(self, k: int = ENTITY_MEMORY_K) -> list[dict]:
        """
        Returns the most recently mentioned entities for this session.

        Args:
            k : Max number of entities to return.

        Returns:
            List of dicts: [{"entity": ..., "context": ..., "timestamp": ...}]
        """
        conn = _get_connection()
        cur  = conn.cursor()
        cur.execute(
            "SELECT entity, context, timestamp FROM entities "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (self.session_id, k),
        )
        rows = cur.fetchall()
        conn.close()
        return [{"entity": r["entity"], "context": r["context"], "timestamp": r["timestamp"]} for r in rows]

    # ── Session management ─────────────────────────────────────────────────
    def clear(self) -> None:
        """Deletes all messages and entities for this session (keeps the session row)."""
        conn = _get_connection()
        cur  = conn.cursor()
        cur.execute("DELETE FROM messages WHERE session_id = ?", (self.session_id,))
        cur.execute("DELETE FROM entities WHERE session_id = ?", (self.session_id,))
        conn.commit()
        conn.close()
        logger.info("Session cleared: %s", self.session_id)

    def get_summary_stats(self) -> dict:
        """Returns basic stats about this session's memory."""
        messages = self.get_history()
        entities = self.get_recent_entities(k=1000)
        return {
            "session_id":     self.session_id,
            "message_count":  len(messages),
            "entity_count":   len(entities),
        }


# ── Session listing (across all sessions) ────────────────────────────────────
def list_sessions() -> list[dict]:
    """
    Lists all research sessions stored in the database, most recent first.
    Used by the Streamlit UI sidebar to show session history.

    Returns:
        List of dicts: [{"session_id": ..., "title": ..., "created_at": ...}]
    """
    init_db()
    conn = _get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT session_id, title, created_at FROM sessions ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [{"session_id": r["session_id"], "title": r["title"], "created_at": r["created_at"]} for r in rows]


# ── Quick self-test (run: python src/memory/conversation_memory.py) ──────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    print("\n" + "=" * 55)
    print("  Conversation Memory Test")
    print("=" * 55)

    test_session_id = "test_session_001"

    print(f"\n[1] Initialising memory for session: {test_session_id}")
    memory = ConversationMemory(session_id=test_session_id, title="Test Research Session")

    print("\n[2] Adding messages...")
    memory.add_message("user", "What is LangGraph?")
    memory.add_message("assistant", "LangGraph is a library for building stateful multi-agent LLM applications.")
    memory.add_message("user", "How does it compare to plain LangChain?")
    memory.add_message("assistant", "LangGraph adds graph-based orchestration with cycles, which LangChain alone doesn't provide.")
    print("    4 messages added.")

    print("\n[3] Retrieving full history:")
    history = memory.get_history()
    for m in history:
        print(f"    [{m.role}] {m.content[:60]}")

    print("\n[4] Retrieving history as text (for LLM context):")
    text = memory.get_history_as_text()
    print(f"    {text[:200]}...")

    print("\n[5] Adding entities...")
    memory.add_entity("LangGraph", context="A library for stateful multi-agent LLM apps")
    memory.add_entity("LangChain", context="Base framework LangGraph extends")
    print("    2 entities added.")

    print("\n[6] Retrieving recent entities:")
    entities = memory.get_recent_entities()
    for e in entities:
        print(f"    • {e['entity']}: {e['context']}")

    print("\n[7] Session summary stats:")
    stats = memory.get_summary_stats()
    print(f"    {stats}")

    print("\n[8] Listing all sessions:")
    sessions = list_sessions()
    for s in sessions:
        print(f"    • {s['session_id']} | {s['title']} | {s['created_at']}")

    print("\n[9] Clearing test session...")
    memory.clear()
    stats = memory.get_summary_stats()
    print(f"    Stats after clear: {stats}")

    print("\n" + "=" * 55)
    print("Conversation memory test complete.")
    print("=" * 55 + "\n")