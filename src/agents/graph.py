"""
graph.py — LangGraph StateGraph Assembly
AI Research Agent
Wires planner -> researcher -> synthesizer -> validator into a real
LangGraph StateGraph with conditional routing and cycle support.

This is the file that turns the individual node functions (nodes.py) into
an actual multi-agent orchestration pipeline.
"""

import logging
import sys
from pathlib import Path

from langgraph.graph import StateGraph, END

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.state import ResearchState, create_initial_state
from agents.nodes import (
    planner_node,
    researcher_node,
    synthesizer_node,
    validator_node,
)

logger = logging.getLogger(__name__)


# ── Conditional routing functions ─────────────────────────────────────────────
# These read state["next_action"] (set by each node) and tell LangGraph which
# node to execute next. This is what makes cycles possible — the validator
# can route back to "researcher" instead of always going to END.

def route_after_researcher(state: ResearchState) -> str:
    """
    After the researcher node runs, decide whether to loop back to itself
    (more plan steps remain) or move on to synthesis.
    """
    action = state.get("next_action", "synthesize")
    logger.info("Routing after researcher -> %s", action)
    return "researcher" if action == "search" else "synthesizer"


def route_after_synthesizer(state: ResearchState) -> str:
    """
    After the synthesizer node runs, normally proceeds to validation — but
    if report generation hit an unrecoverable error (e.g. LLM quota
    exhausted), the synthesizer sets next_action="end" so we skip wasting
    another LLM call on validating a failure message.
    """
    action = state.get("next_action", "validate")
    logger.info("Routing after synthesizer -> %s", action)
    return "end" if action == "end" else "validator"


def route_after_validator(state: ResearchState) -> str:
    """
    After the validator node runs, decide whether to end the run or loop
    back to the researcher for another pass (e.g. groundedness failed).
    """
    action = state.get("next_action", "end")
    logger.info("Routing after validator -> %s", action)
    return "researcher" if action == "search" else "end"


# ── Graph builder ──────────────────────────────────────────────────────────
def build_graph():
    """
    Constructs and compiles the LangGraph StateGraph for the research pipeline.

    Graph shape:

        planner -> researcher --(more steps?)--> researcher (loop)
                       |
                       v (plan complete)
                  synthesizer --(quota exhausted?)--> END
                       |
                       v (normal)
                  validator --(failed + iterations left?)--> researcher (loop)
                       |
                       v (passed or max iterations)
                      END

    Returns:
        A compiled LangGraph graph object, ready for .invoke() or .stream().
    """
    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("validator", validator_node)

    # Entry point
    graph.set_entry_point("planner")

    # planner always proceeds to researcher
    graph.add_edge("planner", "researcher")

    # researcher loops on itself until plan is exhausted, then -> synthesizer
    graph.add_conditional_edges(
        "researcher",
        route_after_researcher,
        {
            "researcher": "researcher",
            "synthesizer": "synthesizer",
        },
    )

    # synthesizer normally proceeds to validator, but can short-circuit to END
    # if LLM quota was exhausted (no point validating a failure message)
    graph.add_conditional_edges(
        "synthesizer",
        route_after_synthesizer,
        {
            "validator": "validator",
            "end": END,
        },
    )

    # validator either ends the run or loops back to researcher
    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "researcher": "researcher",
            "end": END,
        },
    )

    compiled = graph.compile()
    logger.info("LangGraph compiled successfully.")
    return compiled


# ── High-level entry point ────────────────────────────────────────────────────
def run_research(topic: str, session_id: str) -> ResearchState:
    """
    Runs the full multi-agent research pipeline end-to-end for a given topic.

    Args:
        topic      : The research question/topic.
        session_id : Conversation session ID (for memory linkage).

    Returns:
        The final ResearchState after the graph run completes.
    """
    logger.info("Starting research run | topic: '%s' | session: %s", topic, session_id)

    app   = build_graph()
    state = create_initial_state(topic=topic, session_id=session_id)

    # recursion_limit guards against runaway cycles beyond MAX_ITERATIONS
    final_state = app.invoke(state, config={"recursion_limit": 50})

    logger.info(
        "Research run complete | validation_passed=%s | report_path=%s",
        final_state.get("validation_passed"),
        final_state.get("report_path"),
    )
    return final_state


# ── Quick self-test (run: python -m agents.graph from src/) ──────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    print("\n" + "=" * 55)
    print("  LangGraph Full Pipeline Test")
    print("=" * 55)

    print("\n[1] Building graph...")
    app = build_graph()
    print("    Graph compiled successfully.")

    # Optional: print a text representation of the graph structure
    try:
        print("\n[2] Graph structure:")
        print(app.get_graph().draw_ascii())
    except Exception:
        print("    (ASCII graph rendering unavailable — skipping, not critical)")

    test_topic = "What are the main benefits of using LangGraph for AI agents?"
    print(f"\n[3] Running full pipeline for topic:\n    '{test_topic}'")
    print("    This will call Groq + Tavily + ChromaDB multiple times — please wait...\n")

    final_state = run_research(topic=test_topic, session_id="graph_test_session")

    print("\n[4] Final state summary:")
    print(f"    Plan steps executed   : {len(final_state['plan'])}")
    print(f"    Findings accumulated  : {len(final_state['findings'])}")
    print(f"    Iterations run        : {final_state['iteration_count']}")
    print(f"    Validation passed     : {final_state['validation_passed']}")
    print(f"    Report saved to       : {final_state['report_path']}")

    print("\n[5] Validation messages:")
    for msg in final_state["validation_messages"]:
        print(f"    • {msg}")

    print("\n[6] Final report preview:")
    print(final_state["report_markdown"][:500])
    print("...")

    print("\n" + "=" * 55)
    print("Full pipeline test complete.")
    print("=" * 55 + "\n")