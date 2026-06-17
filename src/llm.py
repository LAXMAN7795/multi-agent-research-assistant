"""
llm.py — LLM Factory
AI Research Agent
Initialises and returns Groq (LLaMA 3.1 70B) and Gemini clients.
All agent modules import from here — never instantiate LLMs elsewhere.
"""

import logging
from functools import lru_cache
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

from config import (
    GROQ_API_KEY,
    GOOGLE_API_KEY,
    GROQ_MODEL,
    GEMINI_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ── Groq — Primary Reasoning LLM ─────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_groq_llm(
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int    = LLM_MAX_TOKENS,
) -> ChatGroq:
    """
    Returns a cached ChatGroq instance (LLaMA 3.1 70B).
    Used for: planning, reasoning, tool calling, report drafting.

    Args:
        temperature : Sampling temperature (0.0 = deterministic, 1.0 = creative).
        max_tokens  : Max tokens in the model response.

    Returns:
        ChatGroq instance ready for invoke / stream calls.

    Raises:
        ValueError : If GROQ_API_KEY is missing.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file.\n"
            "Get a free key at: https://console.groq.com"
        )

    logger.info("Initialising Groq LLM — model: %s | temp: %s", GROQ_MODEL, temperature)

    return ChatGroq(
        api_key     = GROQ_API_KEY,
        model       = GROQ_MODEL,
        temperature = temperature,
        max_tokens  = max_tokens,
    )


# ── Gemini — Long-Context Summarisation LLM ──────────────────────────────────
@lru_cache(maxsize=1)
def get_gemini_llm(
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int    = LLM_MAX_TOKENS,
) -> ChatGoogleGenerativeAI:
    """
    Returns a cached ChatGoogleGenerativeAI instance (Gemini 1.5 Flash).
    Used for: long-document summarisation, final report synthesis.

    Args:
        temperature : Sampling temperature.
        max_tokens  : Max tokens in the model response.

    Returns:
        ChatGoogleGenerativeAI instance ready for invoke / stream calls.

    Raises:
        ValueError : If GOOGLE_API_KEY is missing.
    """
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Add it to your .env file.\n"
            "Get a free key at: https://aistudio.google.com"
        )

    logger.info("Initialising Gemini LLM — model: %s | temp: %s", GEMINI_MODEL, temperature)

    return ChatGoogleGenerativeAI(
        google_api_key = GOOGLE_API_KEY,
        model          = GEMINI_MODEL,
        temperature    = temperature,
        max_tokens     = max_tokens,
    )


# ── Role-based selector ───────────────────────────────────────────────────────
def get_llm(role: str = "reasoning") -> BaseChatModel:
    """
    Returns the appropriate LLM for a given agent role.
    Central switch — change model assignments here without touching agents.

    Args:
        role : One of "reasoning" | "summarisation" | "planning" | "validation"

    Returns:
        A LangChain BaseChatModel instance.

    Role → Model mapping:
        reasoning    → Groq  (fast, structured tool calling)
        planning     → Groq  (step decomposition needs speed)
        validation   → Groq  (short, fast checks)
        summarisation→ Gemini (handles large document context)
    """
    role = role.lower().strip()

    role_map = {
        "reasoning"    : get_groq_llm,
        "planning"     : get_groq_llm,
        "validation"   : get_groq_llm,
        "summarisation": get_gemini_llm,
    }

    factory = role_map.get(role)

    if factory is None:
        logger.warning("Unknown role '%s' — defaulting to Groq.", role)
        factory = get_groq_llm

    return factory()


# ── Connection health check ───────────────────────────────────────────────────
def check_llm_connections() -> dict:
    """
    Sends a minimal test message to both LLMs and reports status.
    Used by config validation and the Streamlit sidebar health check.

    Returns:
        {
            "groq"  : {"status": "ok" | "error", "message": str},
            "gemini": {"status": "ok" | "error", "message": str},
        }
    """
    results = {}
    test_prompt = "Reply with exactly one word: Ready"

    # Test Groq
    try:
        llm  = get_groq_llm()
        resp = llm.invoke(test_prompt)
        results["groq"] = {"status": "ok", "message": resp.content.strip()}
        logger.info("Groq connection — OK | response: %s", resp.content.strip())
    except Exception as exc:
        results["groq"] = {"status": "error", "message": str(exc)}
        logger.error("Groq connection — FAILED | %s", exc)

    # Test Gemini
    try:
        llm  = get_gemini_llm()
        resp = llm.invoke(test_prompt)
        results["gemini"] = {"status": "ok", "message": resp.content.strip()}
        logger.info("Gemini connection — OK | response: %s", resp.content.strip())
    except Exception as exc:
        results["gemini"] = {"status": "error", "message": str(exc)}
        logger.error("Gemini connection — FAILED | %s", exc)

    return results


# ── Quick self-test (run: python src/llm.py) ──────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    setup_logging()

    print("\n" + "=" * 50)
    print("  AI Research Agent — LLM Connection Test")
    print("=" * 50)

    results = check_llm_connections()

    for provider, info in results.items():
        icon = "✅" if info["status"] == "ok" else "❌"
        print(f"{icon}  {provider.upper():8s} → {info['message']}")

    all_ok = all(v["status"] == "ok" for v in results.values())
    print("=" * 50)
    print("All LLMs ready!" if all_ok else "Some LLMs failed — check keys above.")
    print("=" * 50 + "\n")