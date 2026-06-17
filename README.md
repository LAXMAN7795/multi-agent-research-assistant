# LangGraph Research Agent

**🔗 Live demo:** https://multi-agent-research-assistant-esohnhn9bvmcdrdffqovkz.streamlit.app/

A multi-agent AI research assistant that plans, searches, retrieves, synthesizes, and validates its own output before answering. Built entirely on free-tier infrastructure — no paid API keys required.

Ask it a research question, and it doesn't just generate an answer from memory. It breaks the question into sub-tasks, searches the live web and a local document store for each one, writes a structured report from what it actually found, then runs that report back through an LLM-as-judge to check whether every claim is actually supported by the evidence — looping back to re-research if it isn't.

---

## Why this exists

Most "AI agent" demos are a single LLM call with a system prompt. This project is an attempt to build something closer to how a real research workflow should work: decompose the problem, gather evidence from multiple sources, write from that evidence rather than from the model's own priors, and check the output for hallucination before calling it done. The guardrails layer isn't decorative — in testing, it caught and rejected several genuinely fabricated claims (invented statistics, invented use-cases) that a naive single-shot pipeline would have shipped silently.

---

## What it does

1. **Plans** — breaks a research topic into up to 5 concrete sub-tasks, each assigned to the most appropriate tool.
2. **Researches** — executes each sub-task via live web search (Tavily, with a DuckDuckGo fallback) and/or retrieval from a local vector knowledge base (ChromaDB), accumulating raw findings.
3. **Synthesizes** — feeds all accumulated findings into an LLM with a strict "don't invent anything not in these findings" instruction, producing a structured report (title, summary, sections, key takeaways, sources) validated against a Pydantic schema.
4. **Validates** — runs two guardrail checks on the report: a structural quality check (has a title, has sections, isn't suspiciously short) and a groundedness check, where a second LLM call acts as a fact-checking judge, comparing every claim in the report against the original findings.
5. **Loops or ends** — if validation fails and a retry budget remains, the pipeline routes back to the researcher with specific feedback about what was wrong, so the next attempt corrects the actual flagged issue rather than blindly regenerating. If validation passes, or the retry budget is exhausted, the run ends and the report is saved.

Every step above is a real LangGraph node and a real conditional edge — this is a cyclic graph, not a fixed linear pipeline, which is what lets step 5 actually loop back instead of always marching forward regardless of quality.

---

## Architecture

```
                    ┌─────────────┐
                    │   PLANNER   │  Decomposes topic into ≤5 sub-tasks
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
              ┌────▶│  RESEARCHER │  Executes one plan step per call
              │     └──────┬──────┘  (web_search / retrieve_documents)
              │            │
              │   more steps remain?
              │            │
              └────────────┤
                            │ plan complete
                            ▼
                    ┌─────────────┐
                    │ SYNTHESIZER │  Generates structured report from findings
                    └──────┬──────┘  (or short-circuits to END on quota errors)
                           │
                           ▼
                    ┌─────────────┐
              ┌────▶│  VALIDATOR  │  Groundedness + quality checks (LLM-as-judge)
              │     └──────┬──────┘
              │            │
              │   failed AND retries left?
              │            │
              └────────────┤
                            │ passed, or retries exhausted
                            ▼
                          END
```

The shared state object (`ResearchState`) flows through every node. Two fields — `findings` and `sources` — use a custom reducer that normally appends across steps within a single pass, but resets cleanly when the validator loops back for a retry, so findings don't grow unbounded across multiple validation attempts (an early version of this project hit exactly that bug, and it's the reason that reducer exists).

---

## Tech stack

Every component below has a free tier generous enough to build and demo this project without a credit card.

| Layer | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph** | Cyclic graph execution with conditional routing — the actual "agentic" part of this project |
| Agent framework | **LangChain** | Prompt templates, tool-calling interface, output parsers |
| Reasoning LLM | **Groq** (`llama-3.3-70b-versatile`) | Free tier, extremely fast inference, used for planning, research-step execution, and the groundedness judge |
| Long-context LLM | **Google Gemini** (`gemini-2.5-flash`) | Free tier, available as a second model for long-context summarization work |
| Web search | **Tavily** (primary) + **DuckDuckGo** (fallback) | Tavily is purpose-built for AI agents and returns clean, structured results with an AI-generated direct answer; DuckDuckGo requires no API key at all and kicks in automatically if Tavily is unavailable |
| Vector store | **ChromaDB** | Local, persistent, zero-config — no hosted vector DB needed |
| Embeddings | **HuggingFace** `sentence-transformers/all-MiniLM-L6-v2` | Runs entirely locally on CPU, no API calls or cost |
| Structured output | **Pydantic** | Enforces the report schema — a malformed LLM response is rejected before it reaches the user |
| Guardrails | Custom validators + LLM-as-judge | Prompt-injection detection, tool-output sanity checks, structural report checks, and a groundedness/hallucination check |
| Memory | **SQLite** | Persists conversation history and entity memory across app restarts — sessions survive a page refresh or redeploy |
| UI | **Streamlit** | Chat interface, session switching, document upload, live system-status panel |
| Deployment | **Streamlit Community Cloud** | Free hosting, deploys directly from this GitHub repo |

---

## Project structure

```
ai-research-agent/
├── src/
│   ├── config.py                  # Central config — reads from .env locally, st.secrets on Cloud
│   ├── llm.py                     # Groq + Gemini client factory, role-based model selection
│   ├── agents/
│   │   ├── state.py               # Shared LangGraph state schema (ResearchState)
│   │   ├── nodes.py                # Planner, Researcher, Synthesizer, Validator node functions
│   │   └── graph.py                # Wires nodes into the actual LangGraph StateGraph
│   ├── tools/
│   │   ├── search_tool.py          # Tavily + DuckDuckGo web search, exposed as LangChain tools
│   │   ├── retrieval_tool.py       # ChromaDB ingestion + retrieval (RAG pipeline)
│   │   └── report_tool.py          # Structured report generation, markdown rendering, disk save
│   ├── guardrails/
│   │   └── validators.py           # Prompt-injection detection, output validation, groundedness judge
│   ├── memory/
│   │   ├── conversation_memory.py  # SQLite-backed message + entity history
│   │   └── session_manager.py      # Bridges graph runs into conversation memory
│   └── ui/
│       └── app.py                  # Streamlit application — the entry point
├── data/
│   ├── chroma_db/                  # Vector store (gitignored, local-only)
│   ├── reports/                    # Saved .md / .json reports (gitignored)
│   └── uploads/                    # User-uploaded documents for RAG ingestion
├── .streamlit/
│   └── config.toml                 # Streamlit runtime config (committed)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11+
- Free API keys from:
  - [Groq Console](https://console.groq.com) — required
  - [Google AI Studio](https://aistudio.google.com) — required
  - [Tavily](https://app.tavily.com) — optional (DuckDuckGo fallback works without it)

### Local installation

```bash
git clone https://github.com/YOUR_USERNAME/ai-research-agent.git
cd ai-research-agent

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_key_here
TAVILY_API_KEY=your_tavily_key_here
```

Run it:

```bash
streamlit run src/ui/app.py
```

### Deploying to Streamlit Community Cloud

This repo is structured to deploy as-is. The `.env` file is gitignored and never uploaded — instead, paste the same key-value pairs into the app's **Advanced Settings → Secrets** field at deploy time:

```toml
GROQ_API_KEY = "your_groq_key_here"
GOOGLE_API_KEY = "your_google_key_here"
TAVILY_API_KEY = "your_tavily_key_here"
```

`config.py` checks `st.secrets` first and falls back to `.env`/environment variables, so the identical codebase runs correctly in both environments without any code changes between local and deployed.

**Known limitation:** Streamlit Cloud's filesystem is ephemeral. The ChromaDB knowledge base, saved reports, and SQLite conversation history all reset on every redeploy or app restart. For a persistent production deployment, these would need to move to a hosted store (e.g. a managed Postgres instance or a hosted vector DB) — out of scope for this free-tier portfolio build, but a known next step.

---

## Guardrails in detail

This is the part of the project I'd point to first. Three layers run automatically, with no manual triggering:

**Input validation** — every user query is checked against a set of prompt-injection patterns ("ignore previous instructions", "reveal your system prompt", "DAN mode", etc.) before it reaches any LLM call. Caught queries are blocked with a clear message rather than silently sanitized.

**Tool-output validation** — every web search and retrieval call is checked for empty responses, suspiciously short output, or leaked error strings before that output is allowed into the findings pool the report gets built from.

**Groundedness validation (LLM-as-judge)** — after the report is generated, a second LLM call compares every claim in the report against the raw findings it was supposed to be built from, and returns either `GROUNDED` or a specific explanation of which claim isn't supported. If it fails, the validator routes the graph back to the researcher with that exact feedback injected into the next synthesis attempt — so retries actually correct the flagged issue instead of regenerating the same hallucination. This check fails *open* (doesn't block the pipeline) if the judge call itself errors, since a guardrail-infrastructure failure shouldn't be allowed to take down the whole system — but it's surfaced to the user as "unverified" rather than silently reported as a pass.

---

## What I'd build next

- PDF ingestion for the RAG pipeline (currently `.txt`/`.md` only)
- A sidebar browser for past saved reports (currently saved to disk but not surfaced in the UI)
- Streaming the report text into the UI as it generates, rather than waiting for the full response
- Persistent storage for the vector DB and conversation history on Cloud deployments

---

## Acknowledgments

Built using LangGraph and LangChain (LangChain AI), Groq, Google Gemini, Tavily, ChromaDB, and Streamlit — all under their respective free tiers.
