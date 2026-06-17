"""
report_tool.py — Structured Report Generation
AI Research Agent
Generates a structured research report (Pydantic-validated) from gathered
findings, then renders it to Markdown and saves it to disk.
"""

import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPORTS_PATH
from llm import get_llm

logger = logging.getLogger(__name__)


# ── Structured report schema ──────────────────────────────────────────────────
class ReportSection(BaseModel):
    """A single section within the research report."""
    heading: str = Field(description="Section heading, concise and descriptive")
    content: str = Field(description="Section body text, well-organised prose")


class ResearchReport(BaseModel):
    """
    Full structured research report.
    The LLM is prompted to fill this schema exactly — Pydantic validates
    the output before it's accepted, catching malformed/incomplete reports.
    """
    title:    str            = Field(description="Report title summarising the research topic")
    summary:  str            = Field(description="2-4 sentence executive summary of key findings")
    sections: list[ReportSection] = Field(description="Main body sections of the report")
    sources:  list[str]      = Field(default_factory=list, description="List of source URLs or references used")
    key_takeaways: list[str] = Field(default_factory=list, description="3-5 bullet point key takeaways")


# ── Report generation ─────────────────────────────────────────────────────────
_parser = PydanticOutputParser(pydantic_object=ResearchReport)

_REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research report writer. Given a topic and a set of raw findings "
     "(from web search and/or document retrieval), produce a well-structured, "
     "factually grounded research report.\n\n"
     "STRICT RULE: every claim in your report must be directly traceable to the "
     "raw findings below. Do not add specifics, examples, applications, statistics, "
     "or claims that are not explicitly stated in the findings, even if they sound "
     "plausible or are generally true of the topic. If the findings don't cover "
     "something, omit it rather than filling the gap from general knowledge.\n\n"
     "{format_instructions}"),
    ("human",
     "Research topic: {topic}\n\n"
     "Raw findings:\n{findings}\n\n"
     "{feedback_block}"
     "Produce the structured report now, using ONLY information present in the findings above."),
])


def generate_report(topic: str, findings: str, previous_feedback: str = "") -> ResearchReport:
    """
    Uses the LLM to synthesise raw findings into a structured ResearchReport.

    Args:
        topic             : The research topic/question.
        findings          : Raw text of gathered findings (from search + retrieval tools).
        previous_feedback : Optional groundedness/quality feedback from a prior failed
                             validation pass, injected into the prompt so the retry
                             actually corrects the specific flagged issue instead of
                             regenerating the same hallucination again.

    Returns:
        A validated ResearchReport instance.

    Raises:
        ValueError : If the LLM output cannot be parsed into the schema after retry.
    """
    llm    = get_llm(role="reasoning")
    chain  = _REPORT_PROMPT | llm | _parser

    feedback_block = ""
    if previous_feedback:
        feedback_block = (
            f"IMPORTANT — your previous attempt at this report was rejected for this reason:\n"
            f"\"{previous_feedback}\"\n"
            f"Rewrite the report to fix this specific issue. Remove or rephrase any claim "
            f"that isn't directly supported by the findings above.\n\n"
        )

    logger.info("Generating report for topic: '%s' | findings length: %d chars | retry_feedback: %s",
                topic, len(findings), bool(previous_feedback))

    try:
        report = chain.invoke({
            "topic": topic,
            "findings": findings,
            "feedback_block": feedback_block,
            "format_instructions": _parser.get_format_instructions(),
        })
        logger.info("Report generated successfully | sections: %d", len(report.sections))
        return report

    except Exception as exc:
        logger.error("Report generation/parsing failed: %s", exc)
        raise ValueError(f"Failed to generate structured report: {exc}") from exc


# ── Markdown rendering ─────────────────────────────────────────────────────────
def render_markdown(report: ResearchReport) -> str:
    """
    Converts a ResearchReport into a clean Markdown string.

    Args:
        report : A ResearchReport instance.

    Returns:
        Markdown-formatted string ready to save or display.
    """
    lines = [f"# {report.title}", ""]
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(report.summary)
    lines.append("")

    for section in report.sections:
        lines.append(f"## {section.heading}")
        lines.append(section.content)
        lines.append("")

    if report.key_takeaways:
        lines.append("## Key Takeaways")
        for point in report.key_takeaways:
            lines.append(f"- {point}")
        lines.append("")

    if report.sources:
        lines.append("## Sources")
        for src in report.sources:
            lines.append(f"- {src}")
        lines.append("")

    return "\n".join(lines).strip()


# ── Save to disk ────────────────────────────────────────────────────────────
def save_report(report: ResearchReport, filename: Optional[str] = None) -> str:
    """
    Saves a report as both Markdown (.md) and JSON (.json) to the reports directory.

    Args:
        report   : A ResearchReport instance.
        filename : Optional base filename (without extension). Auto-generated if omitted.

    Returns:
        Path to the saved .md file.
    """
    Path(REPORTS_PATH).mkdir(parents=True, exist_ok=True)

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in report.title)
        safe_title = safe_title.strip().replace(" ", "_")[:50]
        filename = f"{timestamp}_{safe_title or 'report'}"

    md_path   = Path(REPORTS_PATH) / f"{filename}.md"
    json_path = Path(REPORTS_PATH) / f"{filename}.json"

    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    logger.info("Report saved | md: %s | json: %s", md_path, json_path)
    return str(md_path)


# ── LangChain Tool (used by agents) ───────────────────────────────────────────
@tool
def create_research_report(topic: str, findings: str, previous_feedback: str = "") -> str:
    """
    Generate a structured, well-organised research report from gathered findings
    and save it to disk as Markdown. Use this as the final step of a research
    workflow, after web_search and/or retrieve_documents have gathered information.

    Args:
        topic             : The research topic or question being answered.
        findings          : All raw findings text gathered so far (search results,
                            retrieved documents, agent notes) to be synthesised into the report.
        previous_feedback : Optional feedback from a prior failed validation pass,
                            used to correct specific issues on retry.

    Returns:
        Confirmation message with the saved file path and a preview of the report.
    """
    logger.info("create_research_report tool called | topic: '%s'", topic)

    try:
        report   = generate_report(topic, findings, previous_feedback=previous_feedback)
        path     = save_report(report)
        markdown = render_markdown(report)

        preview = markdown[:300] + ("..." if len(markdown) > 300 else "")
        return f"Report successfully created and saved to: {path}\n\nPreview:\n{preview}"

    except Exception as exc:
        logger.error("create_research_report failed: %s", exc)
        return f"Failed to create report: {exc}"


# ── Quick self-test (run: python src/tools/report_tool.py) ───────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(levelname)-8s | %(message)s",
    )

    print("\n" + "=" * 55)
    print("  Report Tool Test")
    print("=" * 55)

    test_topic = "Benefits of LangGraph for multi-agent systems"
    test_findings = """
    [1] LangGraph and Research Agents - Pinecone
        URL: https://www.pinecone.io/learn/langgraph-research-agent
        LangGraph allows developers to define agent workflows as graphs with
        nodes and edges, supporting cycles for iterative reasoning loops.

    [2] Top 5 LangGraph Agents in Production 2024 - LangChain
        URL: https://www.langchain.com/blog/top-5-langgraph-agents-in-production-2024
        Companies like LinkedIn, Elastic, and Replit use LangGraph in production
        for its fault tolerance and transparent cognitive architecture.

    [3] Multi-Agent System Tutorial with LangGraph
        URL: https://blog.futuresmart.ai/multi-agent-system-with-langgraph
        LangGraph supports human-in-the-loop checkpointing and streaming of
        intermediate agent steps, useful for debugging complex workflows.
    """

    print(f"\n[1] Generating report for: '{test_topic}'")
    report = generate_report(test_topic, test_findings)
    print(f"    Title: {report.title}")
    print(f"    Sections: {len(report.sections)}")
    print(f"    Takeaways: {len(report.key_takeaways)}")

    print("\n[2] Rendering markdown...")
    md = render_markdown(report)
    print(md[:500])
    print("...")

    print("\n[3] Saving report to disk...")
    path = save_report(report)
    print(f"    Saved to: {path}")

    print("\n[4] Testing LangChain tool wrapper...")
    result = create_research_report.invoke({"topic": test_topic, "findings": test_findings})
    print(result[:400])

    print("\n" + "=" * 55)
    print("Report tool test complete.")
    print("=" * 55 + "\n")