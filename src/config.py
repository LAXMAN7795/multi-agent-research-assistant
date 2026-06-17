"""
config.py — Central Configuration Loader
AI Research Agent
Loads all environment variables and exposes typed constants used across the project.

Works in two environments with the same code:
  - Local development : reads from .env via python-dotenv
  - Streamlit Cloud    : reads from st.secrets (pasted into the dashboard's
                          "Secrets" field at deploy time — .env is never
                          uploaded since it's gitignored)
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root (no-op on Streamlit Cloud — file won't exist) ─
BASE_DIR = Path(__file__).resolve().parent.parent   # D:\AI_Research_Agent\
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def _get_secret(key: str, default: str = "") -> str:
    """
    Reads a config value with the following precedence:
      1. st.secrets[key]   — Streamlit Cloud's secrets manager (production)
      2. os.getenv(key)    — .env file via python-dotenv (local development)
      3. default

    Importing streamlit here is intentionally lazy/guarded — this module is
    also imported by plain scripts (tests, CLI runs) that never call
    `streamlit run`, where st.secrets would raise if no secrets.toml exists.
    """
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass   # not running under Streamlit, or no secrets.toml present — fall through

    return os.getenv(key, default)


# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY: str        = _get_secret("GROQ_API_KEY")
GOOGLE_API_KEY: str      = _get_secret("GOOGLE_API_KEY")
TAVILY_API_KEY: str      = _get_secret("TAVILY_API_KEY")


# ── LLM Settings ─────────────────────────────────────────────────────────────
GROQ_MODEL: str          = "llama-3.3-70b-versatile"    # primary reasoning model
GEMINI_MODEL: str        = "gemini-2.5-flash"            # long-context summarisation
LLM_TEMPERATURE: float   = 0.2                          # low temp = focused output
LLM_MAX_TOKENS: int      = 4096


# ── Paths ─────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH: str      = os.getenv("CHROMA_DB_PATH",  str(BASE_DIR / "data" / "chroma_db"))
REPORTS_PATH: str        = os.getenv("REPORTS_PATH",    str(BASE_DIR / "data" / "reports"))
UPLOADS_PATH: str        = str(BASE_DIR / "data" / "uploads")
SQLITE_DB_PATH: str      = os.getenv("SQLITE_DB_PATH",  str(BASE_DIR / "data" / "research_history.db"))
LOGS_PATH: str           = str(BASE_DIR / "logs" / "agent.log")


# ── Search Settings ───────────────────────────────────────────────────────────
TAVILY_MAX_RESULTS: int  = 5          # results per Tavily search call
DDG_MAX_RESULTS: int     = 5          # results per DuckDuckGo fallback call
SEARCH_TIMEOUT: int      = 10         # seconds before search times out


# ── RAG / Vector Store Settings ───────────────────────────────────────────────
EMBEDDING_MODEL: str     = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION: str   = "research_docs"
RAG_TOP_K: int           = 5          # number of chunks to retrieve per query
CHUNK_SIZE: int          = 1000       # characters per document chunk
CHUNK_OVERLAP: int       = 200        # overlap between consecutive chunks


# ── Agent Settings ────────────────────────────────────────────────────────────
MAX_ITERATIONS: int      = 3          # max LangGraph loop iterations (kept low for free-tier rate limits)
AGENT_TIMEOUT: int       = 120        # seconds before agent run times out
PLANNER_MAX_STEPS: int   = 5         # max sub-tasks the planner can create


# ── Memory Settings ───────────────────────────────────────────────────────────
MEMORY_MAX_TOKENS: int   = 2000       # max tokens kept in conversation buffer
ENTITY_MEMORY_K: int     = 10         # entity memory window size


# ── App Settings ──────────────────────────────────────────────────────────────
APP_NAME: str            = os.getenv("APP_NAME", "AI Research Agent")
LOG_LEVEL: str           = os.getenv("LOG_LEVEL", "INFO")


# ── Logging Setup ─────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    """
    Configure root logger to write to both console and log file.
    Call this once at application startup.
    """
    log_dir = Path(LOGS_PATH).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),                          # console
            logging.FileHandler(LOGS_PATH, encoding="utf-8") # log file
        ]
    )
    logger = logging.getLogger(APP_NAME)
    logger.info("Logging initialised — level: %s", LOG_LEVEL)
    return logger


# ── Validation ────────────────────────────────────────────────────────────────
def validate_config() -> dict:
    """
    Check that all required API keys and paths are present.
    Returns a dict with status per key so the UI can surface missing items.

    Returns:
        {
            "valid": bool,          # True only if all required keys present
            "missing_keys": list,   # names of missing/empty env vars
            "warnings": list        # non-fatal issues (e.g. fallback will be used)
        }
    """
    missing_keys = []
    warnings     = []

    # Required keys — agent won't start without these
    required = {
        "GROQ_API_KEY":   GROQ_API_KEY,
        "GOOGLE_API_KEY": GOOGLE_API_KEY,
    }
    for name, value in required.items():
        if not value:
            missing_keys.append(name)

    # Optional but recommended — fallback exists
    if not TAVILY_API_KEY:
        warnings.append(
            "TAVILY_API_KEY not set — DuckDuckGo fallback will be used for web search."
        )

    # Ensure data directories exist
    for path_str in [CHROMA_DB_PATH, REPORTS_PATH, UPLOADS_PATH]:
        Path(path_str).mkdir(parents=True, exist_ok=True)

    return {
        "valid":        len(missing_keys) == 0,
        "missing_keys": missing_keys,
        "warnings":     warnings,
    }


# ── Quick self-test (run: python src/config.py) ───────────────────────────────
if __name__ == "__main__":
    logger = setup_logging()
    result = validate_config()

    print("\n" + "=" * 50)
    print(f"  {APP_NAME} — Config Check")
    print("=" * 50)

    if result["valid"]:
        print("✅  All required API keys are present.")
    else:
        print("❌  Missing required keys:")
        for k in result["missing_keys"]:
            print(f"    • {k}")

    if result["warnings"]:
        print("\n⚠️   Warnings:")
        for w in result["warnings"]:
            print(f"    • {w}")

    print(f"\n📁  BASE_DIR       : {BASE_DIR}")
    print(f"📁  CHROMA_DB_PATH : {CHROMA_DB_PATH}")
    print(f"📁  REPORTS_PATH   : {REPORTS_PATH}")
    print(f"📁  SQLITE_DB_PATH : {SQLITE_DB_PATH}")
    print(f"🤖  GROQ_MODEL     : {GROQ_MODEL}")
    print(f"🤖  GEMINI_MODEL   : {GEMINI_MODEL}")
    print("=" * 50 + "\n")