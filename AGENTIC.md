# BrickFin Agent v3 — True Agentic Orchestration

The 7 Jul 2026 course-correction: move from a **hardcoded phase state machine**
(`agent/graph_agent.py`) to a **true agentic ReAct loop** (`agent/agentic_agent.py`)
where the LLM decides the next action each turn.

> **The one idea:** the old design made the *state machine the brain and the LLM a
> sidecar*. v3 makes the *LLM the brain, deterministic checks the guardrails, and
> MCP tools the agent's hands.*

## The agent loop (per turn)

```
OBSERVE  → gather state; call MCP get_application_status FIRST (LOS = source of truth)
REASON   → LLM picks the next action from goal + policy + live state (not a script)
ACT      → call a tool: validate / update field / check eligibility / answer Q / handoff
REFLECT  → validate + guardrail + trace; loop until reply_to_customer ends the turn
```

## How it meets BrickFin's expectations

| Expectation | Where |
|---|---|
| 1. Agentic orchestration | `AgenticAgent` ReAct loop — LLM selects tools/actions |
| 2. Cross-platform resume | `get_application_status` called first every turn; resumes at real LOS stage |
| 3. Intelligent Q&A | `AgentTools.answer_question` → Vector Search + `AnswerJudge` two-gate LLM-as-judge |
| 4. Flexibility via prompts | System prompt + policy + `journey` fields + KB are DATA (edit, no redeploy) |
| 5. Cost optimization | Model tiering in `llm.endpoints` (small/strong/judge); validators are ~0-cost |
| 6. Agent eval | MLflow traces per turn; scorers on orchestration/grounding/empathy/compliance |
| 7. Governed | AI Gateway endpoints, PII masked before writes, Unity Catalog |

## Files

- `agent/agentic_agent.py` — the agent: `AgenticAgent`, `AgentTools`, `Validators`, `AnswerJudge`
- `agent/graph_agent.py` — legacy state machine (kept for reference / A-B)
- `agent/app.py` — selects agent via `AGENT_MODE` env or `agent.mode` in config
- `config/agent_config.yaml` — `agent.mode`, `llm.endpoints` (tiering), `qa.judge` thresholds, `qa.vector_search`

## Run / switch modes

```bash
# Agentic (default)
AGENT_MODE=agentic uvicorn app:app --port 8000
# Legacy state machine (comparison)
AGENT_MODE=legacy   uvicorn app:app --port 8000
```

`GET /health` reports the active `mode` and LLM tiers.

## Notes

- If `langgraph` + `databricks-langchain` are installed AND workspace creds are
  present, the real LangGraph `create_react_agent` runs. Otherwise it falls back
  to an equivalent built-in ReAct loop so the pattern is demonstrable anywhere.
- Deterministic validators (`Validators`) run before any MCP write — invalid data
  is never saved. This is the "guardrail" layer, kept from the original design.
- `answer_question` never invents answers: both judge gates (retrieval relevance,
  groundedness) must pass or the agent sends a graceful "let me follow up" and
  logs a KB gap for the human/KB loop.
