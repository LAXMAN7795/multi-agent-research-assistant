"""
nodes.py — Agent Node Functions
AI Research Agent
Each function here is a LangGraph NODE — it receives the current ResearchState,
performs one agent's job, and returns a partial state update dict.

Pipeline:  planner -> researcher -> synthesizer -> validator -> (end | back to researcher)
"""

import logging
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import get_llm
from config import PLANNER_MAX_STEPS, MAX_ITERATIONS
from tools.search_tool import web_search
from tools.retrieval_tool import retrieve_documents
from tools.report_tool import generate_report, render_markdown, save_report
from guardrails.validators import (
    validate_tool_output,
    run_output_guardrails,
)
from agents.state import ResearchState, PlanStep, reset_list

logger = logging.getLogger(__name__)


def _is_quota_exhausted(exc: Exception) -> bool:
    """
    Detects whether an exception represents a hard quota/rate-limit error
    that retrying within this run cannot fix (e.g. Groq's daily token limit,
    as opposed to a transient per-minute limit the SDK already retries).
    Used to short-circuit the graph straight to END instead of burning
    through MAX_ITERATIONS against a wall that won't move for minutes/hours.
    """
    msg = str(exc).lower()
    return any(marker in msg for marker in [
        "rate_limit_exceeded", "tokens per day", "tpd", "429", "quota",
    ])


# ── 1. PLANNER NODE ────────────────────────────────────────────────────────
_PLANNER_PROMPT = """You are a research planning agent. Break down the following research \
topic into {max_steps} or fewer concrete, searchable sub-tasks.

Topic: {topic}

For each sub-task, decide which tool is most appropriate:
- "web_search"  : for general/current information from the internet
- "retrieve_documents" : for information that may already be in the local knowledge base

Respond with ONLY a JSON array, no other text, in this exact format:
[
  {{"step_number": 1, "description": "...", "tool_to_use": "web_search"}},
  {{"step_number": 2, "description": "...", "tool_to_use": "web_search"}}
]
"""

def planner_node(state: ResearchState) -> dict:
    """
    Decomposes the research topic into a list of concrete sub-tasks (PlanStep).
    Uses the Groq reasoning LLM with a strict JSON-only prompt.

    Args:
        state : Current ResearchState (reads `topic`).

    Returns:
        Partial state update: {"plan": [...], "next_action": "search"}
    """
    logger.info("PLANNER node running | topic: '%s'", state["topic"])

    llm = get_llm(role="planning")
    prompt = _PLANNER_PROMPT.format(topic=state["topic"], max_steps=PLANNER_MAX_STEPS)

    try:
        response = llm.invoke(prompt).content.strip()

        # Strip markdown code fences if the LLM added them despite instructions
        if response.startswith("```"):
            response = response.strip("`")
            response = response.replace("json", "", 1).strip()

        raw_steps = json.loads(response)

        plan: list[PlanStep] = [
            PlanStep(
                step_number=s["step_number"],
                description=s["description"],
                tool_to_use=s.get("tool_to_use", "web_search"),
                completed=False,
            )
            for s in raw_steps[:PLANNER_MAX_STEPS]
        ]

        logger.info("PLANNER created %d steps", len(plan))
        return {"plan": plan, "current_step_index": 0, "next_action": "search"}

    except Exception as exc:
        logger.error("PLANNER failed to produce a valid plan: %s", exc)
        # Fallback: a single generic search step so the pipeline can still proceed
        fallback_plan = [
            PlanStep(step_number=1, description=state["topic"], tool_to_use="web_search", completed=False)
        ]
        return {
            "plan": fallback_plan,
            "current_step_index": 0,
            "next_action": "search",
            "error": f"Planner fallback used due to error: {exc}",
        }


# ── 2. RESEARCHER NODE ─────────────────────────────────────────────────────
def researcher_node(state: ResearchState) -> dict:
    """
    Executes the current plan step using the appropriate tool
    (web_search and/or retrieve_documents), validates the tool output,
    and accumulates results into `findings` / `sources`.

    Args:
        state : Current ResearchState (reads `plan`, `current_step_index`).

    Returns:
        Partial state update with new findings/sources appended, and either
        advances to the next step or moves to "synthesize" when the plan is done.
    """
    idx  = state["current_step_index"]
    plan = state["plan"]

    if idx >= len(plan):
        logger.info("RESEARCHER: all plan steps complete -> synthesize")
        return {"next_action": "synthesize"}

    step = plan[idx]
    logger.info("RESEARCHER node running | step %d/%d: '%s' (tool: %s)",
                idx + 1, len(plan), step["description"], step["tool_to_use"])

    new_findings: list[str] = []
    new_sources:  list[str] = []
    validation_msgs: list[str] = []

    tools_to_run = (
        ["web_search", "retrieve_documents"] if step["tool_to_use"] == "both"
        else [step["tool_to_use"]]
    )

    for tool_name in tools_to_run:
        try:
            if tool_name == "web_search":
                result = web_search.invoke(step["description"])
            elif tool_name == "retrieve_documents":
                result = retrieve_documents.invoke(step["description"])
            else:
                logger.warning("Unknown tool_to_use: '%s' — skipping.", tool_name)
                continue

            check = validate_tool_output(result, tool_name)
            if not check.passed:
                validation_msgs.append(f"Step {idx+1} ({tool_name}): {check.reason}")
                logger.warning("Tool output validation failed: %s", check.reason)

            new_findings.append(f"[Step {idx+1} - {tool_name}] {result}")
            new_sources.append(f"Step {idx+1}: {step['description']} (via {tool_name})")

        except Exception as exc:
            logger.error("RESEARCHER tool call failed | tool: %s | error: %s", tool_name, exc)
            validation_msgs.append(f"Step {idx+1} ({tool_name}) failed: {exc}")

    # Mark step completed and advance
    updated_plan = list(plan)
    updated_plan[idx] = {**step, "completed": True}

    next_idx = idx + 1
    next_action = "search" if next_idx < len(plan) else "synthesize"

    return {
        "plan": updated_plan,
        "current_step_index": next_idx,
        "findings": new_findings,
        "sources": new_sources,
        "validation_messages": validation_msgs,
        "next_action": next_action,
    }


# ── 3. SYNTHESIZER NODE ────────────────────────────────────────────────────
def synthesizer_node(state: ResearchState) -> dict:
    """
    Combines all accumulated findings into a structured report using the
    report_tool, then renders and saves it.

    Args:
        state : Current ResearchState (reads `topic`, `findings`).

    Returns:
        Partial state update: {"report_markdown": ..., "report_path": ..., "next_action": "validate"}
    """
    logger.info("SYNTHESIZER node running | findings count: %d", len(state["findings"]))

    combined_findings = "\n\n".join(state["findings"]) if state["findings"] else "No findings gathered."

    # Cap findings sent to the LLM so we stay well under Groq's free-tier
    # TPM limit (12000 tokens/min ≈ ~36000 chars budget for prompt+response
    # combined). Truncating here is the actual fix; the validator-side reset
    # (see validator_node) prevents this from growing across retry loops.
    MAX_FINDINGS_CHARS = 8000
    if len(combined_findings) > MAX_FINDINGS_CHARS:
        logger.warning(
            "Findings length %d exceeds cap %d — truncating.",
            len(combined_findings), MAX_FINDINGS_CHARS,
        )
        combined_findings = combined_findings[:MAX_FINDINGS_CHARS] + "\n\n...[truncated for length]"

    try:
        report   = generate_report(
            state["topic"],
            combined_findings,
            previous_feedback=state.get("last_validation_feedback", ""),
        )
        markdown = render_markdown(report)
        path     = save_report(report)

        logger.info("SYNTHESIZER produced report | sections: %d | saved to: %s", len(report.sections), path)

        return {
            "report_markdown": markdown,
            "report_path": path,
            "next_action": "validate",
        }

    except Exception as exc:
        logger.error("SYNTHESIZER failed: %s", exc)

        if _is_quota_exhausted(exc):
            logger.error("SYNTHESIZER: LLM quota/rate limit exhausted — ending run early instead of retrying.")
            return {
                "report_markdown": (
                    "# Report Generation Paused\n\n"
                    "The LLM provider's free-tier quota was reached during this run.\n\n"
                    f"Details: {exc}\n\n"
                    "Please wait for the quota to reset (daily limits typically reset every 24h, "
                    "per-minute limits within a few minutes) and try again."
                ),
                "report_path": "",
                "next_action": "end",
                "validation_passed": False,
                "error": f"quota_exhausted: {exc}",
            }

        return {
            "report_markdown": f"# Report Generation Failed\n\nError: {exc}",
            "report_path": "",
            "next_action": "validate",
            "error": str(exc),
        }


# ── 4. VALIDATOR NODE ──────────────────────────────────────────────────────
def validator_node(state: ResearchState) -> dict:
    """
    Runs output guardrails (groundedness + structural quality) on the final
    report. If checks fail AND we haven't hit MAX_ITERATIONS, routes back to
    the researcher for another pass; otherwise ends the run.

    Args:
        state : Current ResearchState (reads `report_markdown`, `findings`, `iteration_count`).

    Returns:
        Partial state update: {"validation_passed": ..., "next_action": "end" | "search", ...}
    """
    logger.info("VALIDATOR node running | iteration: %d", state["iteration_count"])

    combined_findings = "\n\n".join(state["findings"])
    results = run_output_guardrails(state["report_markdown"], combined_findings)

    quality_msg      = results["quality"].reason
    groundedness_msg = results["groundedness"].reason
    new_iteration     = state["iteration_count"] + 1

    validation_msgs = [f"Quality: {quality_msg}", f"Groundedness: {groundedness_msg}"]

    if results["all_passed"] or new_iteration >= MAX_ITERATIONS:
        if not results["all_passed"]:
            logger.warning("VALIDATOR: max iterations reached, ending despite unresolved issues.")
        logger.info("VALIDATOR: pipeline ending | passed=%s", results["all_passed"])
        return {
            "validation_passed": results["all_passed"],
            "validation_messages": validation_msgs,
            "iteration_count": new_iteration,
            "next_action": "end",
        }

    # Failed validation and we have iterations left — loop back to research.
    # Reset findings/sources and the plan's `completed` flags so the next pass
    # starts clean instead of appending onto an ever-growing findings list
    # (which would blow past the LLM's token-per-minute limit after a few loops).
    reset_plan = [{**step, "completed": False} for step in state["plan"]]

    logger.info("VALIDATOR: validation failed, routing back to researcher for another pass.")
    return {
        "validation_passed": False,
        "validation_messages": validation_msgs,
        "last_validation_feedback": groundedness_msg if not results["groundedness"].passed else quality_msg,
        "iteration_count": new_iteration,
        "plan": reset_plan,
        "current_step_index": 0,
        "findings": reset_list(),
        "sources": reset_list(),
        "next_action": "search",
    }


# ── Quick self-test (run: python -m agents.nodes from src/) ──────────────────
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    from agents.state import create_initial_state

    print("\n" + "=" * 55)
    print("  Agent Nodes Test (Planner -> Researcher -> Synthesizer -> Validator)")
    print("=" * 55)

    state = create_initial_state(
        topic="What are the key features of LangGraph?",
        session_id="test_nodes_session",
    )

    print("\n[1] Running planner_node...")
    update = planner_node(state)
    state.update(update)
    print(f"    Plan steps: {len(state['plan'])}")
    for s in state["plan"]:
        print(f"      {s['step_number']}. {s['description']} ({s['tool_to_use']})")

    print("\n[2] Running researcher_node (looping through all plan steps)...")
    safety_counter = 0
    while state["next_action"] == "search" and safety_counter < 10:
        update = researcher_node(state)
        state.update(update)
        safety_counter += 1
    print(f"    Findings gathered: {len(state['findings'])}")
    print(f"    Next action: {state['next_action']}")

    print("\n[3] Running synthesizer_node...")
    update = synthesizer_node(state)
    state.update(update)
    print(f"    Report path: {state['report_path']}")
    print(f"    Report preview:\n{state['report_markdown'][:300]}...")

    print("\n[4] Running validator_node...")
    update = validator_node(state)
    state.update(update)
    print(f"    Validation passed: {state['validation_passed']}")
    print(f"    Next action: {state['next_action']}")
    for msg in state["validation_messages"]:
        print(f"      • {msg}")

    print("\n" + "=" * 55)
    print("Agent nodes test complete.")
    print("=" * 55 + "\n")