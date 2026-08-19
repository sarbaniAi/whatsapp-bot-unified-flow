# WhatsApp Loan Agent — Step-by-Step + Unified Flow (A/B demo)

A reproducible reference demo of a conversational **personal-loan WhatsApp agent** on
Databricks, showing **two journeys in one codebase** for A/B testing:

- **Journey A — Unified Flow** (`/`): a WhatsApp Flows-style **form card** presents a whole
  stage's fields at once → **structured submit** → the **agent takes over in chat** (confirms,
  always asks "any questions?", answers them, sends the next form on "continue").
- **Journey B — Step-by-step** (`/chat`): the fully **conversational** agent collects one field
  per turn.

> Uses a fictitious brand, **BrickFin Financial Services**. It is a **browser simulation** of
> WhatsApp (and of a WhatsApp Flow) — no real WABA/BSP needed to run it. Swapping to a live
> Kaleyra WABA is a config change (`BSP_PROVIDER=kaleyra` + creds), no agent code changes.

## What's inside
| Area | Files |
|---|---|
| Agentic orchestration (ReAct loop, tools, two-gate judge, general-intelligence fallback) | `agent/agentic_agent.py` |
| App + routes (`/`, `/chat`, `/api/flow/*`, `/webhook/whatsapp`) | `agent/app.py` |
| Unified-flow form spec + structured submit (canonical PAN-after-submit) | `agent/flow_forms.py` |
| Unified-flow UI (form card + between-forms chat) | `agent/flow_ui.py` |
| Hybrid glue — transition + "any questions?" gate + intent classify | `agent/hybrid.py` |
| BSP layer (simulator / kaleyra / twilio) | `agent/bsp.py` |
| Mock MCP (LOS with staged personas) | `mcp_server/server.py` |
| Config-driven journey, prompts, model tiering, KB | `config/agent_config.yaml` |
| MLflow eval (golden set + scorers) | `eval/` |

## Architecture (both journeys share this)
```
Customer (WhatsApp / simulator)
   -> Databricks App (FastAPI)  [BSP-agnostic]
       -> Agent (LangGraph ReAct): route -> act -> compose
            - FMAPI (tiered: gpt-oss-120b / claude-haiku / llama-8b)
            - Vector Search KB  (two-gate judge; general-intelligence fallback)
            - MCP client -> mock LOS (status-first resume, validate, update)
       -> Unified Flow: /api/flow/submit (structured, validated) + /api/flow/turn (agent gate)
```
**Design principle:** the Flow is deterministic bulk data entry; the **agent is the conversation
glue** between forms (transition, Q&A, "ready?" gate). Field validation, PAN, and stage ORDER
stay deterministic; the agent never invents figures.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r agent/requirements.txt
# needs a Databricks profile with FMAPI (+ Vector Search for grounded KB Q&A)
DATABRICKS_CONFIG_PROFILE=<profile> MCP_SERVER_URL=in-process AGENT_MODE=agentic \
BSP_PROVIDER=simulator uvicorn app:app --app-dir agent --port 8099
```
- `http://localhost:8099/`     → Journey A (unified flow)
- `http://localhost:8099/chat` → Journey B (step-by-step)

Personas (mock LOS): `+919910175907` fresh · `+919876543210` cross-channel resume ·
`+919876543212` PAN mismatch.

## Knowledge base
Grounded Q&A uses a Databricks Vector Search index (BrickFin policy docs). See
`scripts/setup_kb.py` to provision the Delta table + endpoint + index on your workspace,
then set `qa.vector_search.endpoint` / `index` in `config/agent_config.yaml`.
