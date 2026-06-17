"""
search_tool.py — Web Search Tool
AI Research Agent
Primary: Tavily Search API (structured, AI-optimised results)
Fallback: DuckDuckGo (no API key required)
"""

import logging
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from tavily import TavilyClient
from ddgs import DDGS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    TAVILY_API_KEY,
    TAVILY_MAX_RESULTS,
    DDG_MAX_RESULTS,
    SEARCH_TIMEOUT,
)

logger = logging.getLogger(__name__)


# ── Raw search functions (used internally + by tests) ─────────────────────────

def _tavily_search(query: str, max_results: int = TAVILY_MAX_RESULTS) -> list[dict]:
    """
    Search the web using Tavily API.
    Returns a list of result dicts with keys: title, url, content, score.

    Args:
        query       : Search query string.
        max_results : Max number of results to return.

    Returns:
        List of dicts: [{"title": ..., "url": ..., "content": ..., "score": ...}]

    Raises:
        Exception : Propagates Tavily API errors so caller can fallback.
    """
    client  = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(
        query              = query,
        max_results        = max_results,
        search_depth       = "advanced",   # deeper crawl for research tasks
        include_answer     = True,         # Tavily's own summary of top results
        include_raw_content= False,        # raw HTML not needed
    )

    results = []
    for r in response.get("results", []):
        results.append({
            "title"  : r.get("title", ""),
            "url"    : r.get("url", ""),
            "content": r.get("content", ""),
            "score"  : round(r.get("score", 0.0), 3),
            "source" : "tavily",
        })

    # Tavily sometimes returns a direct answer — prepend it as a result
    answer = response.get("answer", "")
    if answer:
        results.insert(0, {
            "title"  : "Tavily Direct Answer",
            "url"    : "",
            "content": answer,
            "score"  : 1.0,
            "source" : "tavily_answer",
        })

    logger.info("Tavily search | query: '%s' | results: %d", query, len(results))
    return results


def _ddg_search(query: str, max_results: int = DDG_MAX_RESULTS) -> list[dict]:
    """
    Search the web using DuckDuckGo (no API key required).
    Used as fallback when Tavily is unavailable or rate-limited.

    Args:
        query       : Search query string.
        max_results : Max number of results to return.

    Returns:
        List of dicts: [{"title": ..., "url": ..., "content": ..., "score": ...}]
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title"  : r.get("title", ""),
                "url"    : r.get("href", ""),
                "content": r.get("body", ""),
                "score"  : 0.5,       # DDG doesn't return relevance scores
                "source" : "duckduckgo",
            })

    logger.info("DuckDuckGo search | query: '%s' | results: %d", query, len(results))
    return results


def _format_results_as_text(results: list[dict]) -> str:
    """
    Converts a list of search result dicts into a readable text block
    that can be injected into an LLM prompt.

    Args:
        results : List of result dicts from _tavily_search or _ddg_search.

    Returns:
        Formatted string ready for LLM consumption.
    """
    if not results:
        return "No search results found."

    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r['title']}")
        if r["url"]:
            lines.append(f"    URL: {r['url']}")
        lines.append(f"    {r['content'][:500]}")   # cap per-result content
        lines.append("")

    return "\n".join(lines).strip()


# ── LangChain Tool (used by agents via tool calling) ─────────────────────────

@tool
def web_search(query: str) -> str:
    """
    Search the web for current information on a given topic.
    Uses Tavily as the primary search engine with DuckDuckGo as fallback.
    Returns a formatted text block of the top search results.

    Args:
        query : The search query string. Be specific for better results.

    Returns:
        Formatted string of search results including titles, URLs, and snippets.
    """
    logger.info("web_search tool called | query: '%s'", query)

    # Try Tavily first
    if TAVILY_API_KEY:
        try:
            results = _tavily_search(query)
            return _format_results_as_text(results)
        except Exception as exc:
            logger.warning("Tavily failed — falling back to DuckDuckGo | error: %s", exc)

    # Fallback: DuckDuckGo
    try:
        results = _ddg_search(query)
        return _format_results_as_text(results)
    except Exception as exc:
        logger.error("DuckDuckGo also failed | error: %s", exc)
        return f"Search failed for query '{query}'. Error: {exc}"


@tool
def multi_search(queries: list[str]) -> dict[str, Any]:
    """
    Run multiple search queries in sequence and return all results.
    Used by the Planner agent to gather information on several sub-topics at once.

    Args:
        queries : List of search query strings (max 5 recommended).

    Returns:
        Dict mapping each query to its formatted search results string.
    """
    logger.info("multi_search tool called | %d queries", len(queries))

    output = {}
    for q in queries[:5]:        # hard cap at 5 to avoid rate limits
        output[q] = web_search.invoke(q)

    return output


# ── Quick self-test (run: python src/tools/search_tool.py) ───────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(levelname)-8s | %(message)s",
    )

    TEST_QUERY = "LangGraph multi-agent AI research 2024"

    print("\n" + "=" * 55)
    print("  Search Tool Test")
    print("=" * 55)

    # Test 1: raw Tavily
    if TAVILY_API_KEY:
        print(f"\n[1] Tavily search → '{TEST_QUERY}'")
        try:
            results = _tavily_search(TEST_QUERY, max_results=3)
            for r in results:
                print(f"  • [{r['source']}] {r['title'][:60]}")
                print(f"    {r['url']}")
        except Exception as e:
            print(f"  ❌ Tavily error: {e}")
    else:
        print("\n[1] Tavily — skipped (no API key)")

    # Test 2: raw DuckDuckGo
    print(f"\n[2] DuckDuckGo search → '{TEST_QUERY}'")
    try:
        results = _ddg_search(TEST_QUERY, max_results=3)
        for r in results:
            print(f"  • {r['title'][:60]}")
            print(f"    {r['url']}")
    except Exception as e:
        print(f"  ❌ DuckDuckGo error: {e}")

    # Test 3: LangChain tool (auto-selects best source)
    print(f"\n[3] web_search tool (LangChain) → '{TEST_QUERY}'")
    result = web_search.invoke(TEST_QUERY)
    print(result[:600])

    print("\n" + "=" * 55)
    print("Search tool test complete.")
    print("=" * 55 + "\n")