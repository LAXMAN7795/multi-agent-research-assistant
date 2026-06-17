"""
state.py — LangGraph State Schema
AI Research Agent
Defines the shared state object that flows through every node in the graph.
Every agent node reads from and writes to this single state — this is how
LangGraph maintains context across the planning -> search -> retrieval ->
synthesis -> validation pipeline.
"""

from typing import TypedDict, Annotated, Optional
import operator


def _append_or_reset(existing: list, update: list) -> list:
    """
    Custom LangGraph reducer for `findings` / `sources`.

    Behaves like operator.add (append) for normal updates, but if a node
    sends a special reset sentinel (a list starting with "__RESET__"),
    it replaces the list instead of appending — used by the validator
    when looping back to research, so findings don't grow unbounded
    across retry cycles (which previously blew past the LLM's TPM limit).
    """
    if update and update[0] == "__RESET__":
        return update[1:]
    return existing + update


def reset_list(new_items: Optional[list] = None) -> list:
    """Wraps a list with the reset sentinel so _append_or_reset replaces instead of appends."""
    return ["__RESET__"] + (new_items or [])


class PlanStep(TypedDict):
    """A single sub-task created by the Planner agent."""
    step_number: int
    description: str
    tool_to_use: str          # "web_search" | "retrieve_documents" | "both"
    completed:   bool


class ResearchState(TypedDict):
    """
    The full shared state object passed between all LangGraph nodes.

    Fields use `Annotated[..., operator.add]` where LangGraph should
    APPEND new values (e.g. accumulating findings across multiple search
    steps) rather than overwrite the previous value.
    """

    # ── Input ──────────────────────────────────────────────────────────────
    topic: str                              # the original user research query
    session_id: str                         # links this run to ConversationMemory

    # ── Planning ───────────────────────────────────────────────────────────
    plan: list[PlanStep]                    # sub-tasks created by the Planner
    current_step_index: int                 # which plan step we're executing

    # ── Findings (accumulated across steps, reset on validator retry) ──────
    findings: Annotated[list[str], _append_or_reset]      # raw text chunks gathered
    sources:  Annotated[list[str], _append_or_reset]      # URLs/source names collected

    # ── Synthesis output ──────────────────────────────────────────────────
    report_markdown: str                    # final rendered report
    report_path: str                        # saved file path

    # ── Guardrails ─────────────────────────────────────────────────────────
    validation_passed: bool                 # overall pass/fail from guardrails
    validation_messages: Annotated[list[str], operator.add]  # warnings/blocks logged
    last_validation_feedback: str           # most recent failure reason, fed back into synthesizer on retry

    # ── Control flow ───────────────────────────────────────────────────────
    iteration_count: int                    # safety counter to prevent infinite loops
    next_action: str                        # routing signal: "plan" | "search" | "retrieve" | "synthesize" | "validate" | "end"
    error: Optional[str]                    # set if a node fails, used to short-circuit gracefully


def create_initial_state(topic: str, session_id: str) -> ResearchState:
    """
    Builds a fresh ResearchState dict to kick off a new graph run.

    Args:
        topic      : The user's research question/topic.
        session_id : The conversation session this run belongs to.

    Returns:
        A fully-initialised ResearchState ready to pass into graph.invoke().
    """
    return ResearchState(
        topic=topic,
        session_id=session_id,
        plan=[],
        current_step_index=0,
        findings=[],
        sources=[],
        report_markdown="",
        report_path="",
        validation_passed=False,
        validation_messages=[],
        last_validation_feedback="",
        iteration_count=0,
        next_action="plan",
        error=None,
    )


# ── Quick self-test (run: python src/agents/state.py) ─────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  ResearchState Schema Test")
    print("=" * 55)

    state = create_initial_state(
        topic="Benefits of LangGraph for multi-agent systems",
        session_id="test_session_001",
    )

    print("\n[1] Initial state created:")
    for key, value in state.items():
        print(f"    {key:22s} = {value!r}")

    print("\n[2] Simulating plan creation...")
    state["plan"] = [
        PlanStep(step_number=1, description="Search for LangGraph overview", tool_to_use="web_search", completed=False),
        PlanStep(step_number=2, description="Search for production use cases", tool_to_use="web_search", completed=False),
    ]
    state["next_action"] = "search"
    print(f"    Plan created with {len(state['plan'])} steps.")
    for step in state["plan"]:
        print(f"      Step {step['step_number']}: {step['description']} (tool: {step['tool_to_use']})")

    print("\n[3] Simulating findings accumulation (Annotated + operator.add)...")
    # In a real graph, LangGraph automatically merges these via operator.add
    # when multiple nodes return partial state updates with the same key.
    partial_update_1 = {"findings": ["Finding from search step 1..."]}
    partial_update_2 = {"findings": ["Finding from search step 2..."]}

    merged_findings = state["findings"] + partial_update_1["findings"] + partial_update_2["findings"]
    print(f"    Merged findings count: {len(merged_findings)}")
    for f in merged_findings:
        print(f"      • {f}")

    print("\n[4] Validating TypedDict structure (all required keys present)...")
    required_keys = ResearchState.__annotations__.keys()
    missing = [k for k in required_keys if k not in state]
    if missing:
        print(f"    ❌ Missing keys: {missing}")
    else:
        print(f"    ✅ All {len(required_keys)} required keys present.")

    print("\n" + "=" * 55)
    print("State schema test complete.")
    print("=" * 55 + "\n")