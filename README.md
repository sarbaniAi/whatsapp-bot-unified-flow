# WhatsApp Loan Agent — Step-by-Step + Unified Flow (A/B demo)

A reproducible reference demo of a conversational **personal-loan WhatsApp agent** on
Databricks, with **two journeys in one codebase** for A/B testing:

- **Journey A — Unified Flow** (`/`): a WhatsApp Flows-style **form card** shows a whole stage's
  fields at once → **structured submit** → the **agent takes over in chat** (confirms, always asks
  "any questions?", answers them, sends the next form on "continue").
- **Journey B — Step-by-step** (`/chat`): the fully **conversational** agent collects one field per turn.

Uses a fictitious brand, **BrickFin Financial Services**. It runs as a **browser simulation** of
WhatsApp (and of a WhatsApp Flow) — no real WABA/BSP needed. Going live on a real Kaleyra WABA is a
config change (`BSP_PROVIDER=kaleyra` + creds), not a code change.

**Reference deployment:** a Databricks App on the FE workspace (`whatsapp-unified-flow`) —
SSO-gated, for internal reference.

---

## Prerequisites
- A **Databricks workspace** with:
  - **Foundation Model APIs** (chat + embeddings). This demo uses `databricks-gpt-oss-120b`,
    `databricks-claude-haiku-4-5`, `databricks-meta-llama-3-1-8b-instruct`, `databricks-gte-large-en`
    — **swap these for models available in your region** (see "Values to change").
  - **Vector Search** (for grounded KB Q&A).
  - A **SQL warehouse** (used once to create the KB table).
- **Databricks CLI** authenticated: `databricks auth login --host <your-workspace-url>`
- **Python 3.10+**

---

## Option A — Run locally (fastest, ~5 min)

```bash
git clone https://github.com/sarbaniAi/whatsapp-bot-unified-flow.git
cd whatsapp-bot-unified-flow
python -m venv .venv && source .venv/bin/activate
pip install -r agent/requirements.txt

# authenticate to your Databricks workspace (FMAPI is called at runtime)
databricks auth login --host https://<your-workspace>.cloud.databricks.com --profile myws

DATABRICKS_CONFIG_PROFILE=myws MCP_SERVER_URL=in-process AGENT_MODE=agentic \
BSP_PROVIDER=simulator uvicorn app:app --app-dir agent --port 8099
```
- `http://localhost:8099/`     → **Journey A** (unified flow)
- `http://localhost:8099/chat` → **Journey B** (step-by-step)

Field collection, off-topic decline, and general Q&A work immediately.
**Grounded product Q&A needs the KB** (Option C) — until then those questions defer gracefully.

Demo personas (mock LOS): `+919910175907` fresh · `+919876543210` cross-channel resume ·
`+919876543212` PAN mismatch.

---

## Option B — Deploy as a Databricks App

```bash
# 1. create the app (provisions compute; ~2-3 min)
databricks apps create whatsapp-unified-flow -p myws

# 2. sync the source to your workspace
databricks workspace import-dir . \
  /Workspace/Users/<you>@databricks.com/whatsapp-unified-flow --overwrite -p myws

# 3. deploy
databricks apps deploy whatsapp-unified-flow \
  --source-code-path /Workspace/Users/<you>@databricks.com/whatsapp-unified-flow -p myws
```
Then grant the app's **service principal** (shown in `databricks apps get whatsapp-unified-flow`)
query + KB access (see "Grants" below). The app URL is in the `apps get` output; it is SSO-gated.

---

## Option C — Provision the Knowledge Base (for grounded Q&A)

```bash
# edit scripts/setup_kb.py: set CATALOG, SCHEMA, WAREHOUSE_ID, VS_ENDPOINT, EMBED_ENDPOINT
PROFILE=myws python scripts/setup_kb.py
```
This creates the Delta table (BrickFin policy docs), a Vector Search endpoint, and a Delta Sync
index. **The index takes ~10-15 min to become `ready`** ("pending endpoint provisioning" is normal);
until then grounded questions defer. Then set `qa.vector_search.endpoint` / `index` in
`config/agent_config.yaml` to match.

### Grants (app SP needs these once)
```sql
GRANT USE CATALOG ON CATALOG <catalog> TO `<app-sp-client-id>`;
GRANT USE SCHEMA  ON SCHEMA  <catalog>.<schema> TO `<app-sp-client-id>`;
GRANT SELECT ON TABLE <catalog>.<schema>.kb_docs  TO `<app-sp-client-id>`;
GRANT SELECT ON TABLE <catalog>.<schema>.kb_index TO `<app-sp-client-id>`;
```
FMAPI pay-per-token endpoints are usually queryable by default; grant `CAN_QUERY` if your workspace restricts them.

---

## Values to change for your workspace

| Value | File(s) | Current (this repo) |
|---|---|---|
| KB catalog / schema | `scripts/setup_kb.py` (`CATALOG`,`SCHEMA`), `config/agent_config.yaml` (`qa.vector_search.index`) | `serverless_stable_v41mwb_catalog.whatsapp_agent` |
| SQL warehouse id | `scripts/setup_kb.py` (`WAREHOUSE_ID`, or `WAREHOUSE_ID` env) | `87b872956b927e71` |
| Vector Search endpoint | `config/agent_config.yaml` (`qa.vector_search.endpoint`), `app.yaml` (`VS_ENDPOINT`), `scripts/setup_kb.py` | `whatsapp-agent-vs` |
| Embedding endpoint | `scripts/setup_kb.py` (`EMBED_ENDPOINT`) | `databricks-gte-large-en` |
| Chat model endpoints | `config/agent_config.yaml` (`llm.endpoints`) | `gpt-oss-120b` / `claude-haiku-4-5` / `llama-3-1-8b` |
| MLflow experiment id | `app.yaml` (`MLFLOW_EXPERIMENT_ID`) | `1154190824408236` (create your own, or remove) |
| Workspace fallback path | `agent/config_loader.py` | `/Workspace/Users/sarbani.maiti@databricks.com/...` (only a fallback; local load wins) |

---

## Architecture (both journeys share this)
```
Customer (WhatsApp / simulator)
  -> Databricks App (FastAPI) [BSP-agnostic: simulator | kaleyra | twilio]
      -> Agent (LangGraph ReAct): route -> act -> compose
           - FMAPI (tiered: gpt-oss-120b / claude-haiku / llama-8b)
           - Vector Search KB (two-gate judge; general-intelligence fallback)
           - MCP client -> mock LOS (status-first resume, validate, update)
      -> Unified Flow: /api/flow/submit (structured, validated) + /api/flow/turn (agent gate)
```
**Design principle:** the Flow is deterministic bulk data entry; the **agent is the conversation
glue** between forms (transition, Q&A, "ready?" gate). Field validation, PAN, and stage ORDER stay
deterministic; the agent never invents figures.

## Layout
```
agent/agentic_agent.py   ReAct agent, two-gate judge, general-intelligence fallback
agent/app.py             FastAPI app + routes (/ , /chat , /api/flow/* , /webhook/whatsapp)
agent/flow_forms.py      unified-flow form spec + structured submit (canonical PAN-after-submit)
agent/flow_ui.py         unified-flow UI (form card + between-forms chat)
agent/hybrid.py          transition + "any questions?" gate + intent classify
agent/bsp.py             channel layer (simulator | kaleyra | twilio)
mcp_server/server.py     mock LOS with staged personas
config/agent_config.yaml journey, prompts, model tiering, KB config
scripts/setup_kb.py      provision KB table + Vector Search index
eval/                    MLflow golden set + scorers
```

## Known gaps / reproducibility notes
- **Config is workspace-specific and spread across 3 files** (`setup_kb.py`, `agent_config.yaml`,
  `app.yaml`) — no single `.env`/params file yet. Edit the "Values to change" entries.
- **KB is a manual prerequisite** (`setup_kb.py`) and the index takes ~10-15 min to sync; grounded
  Q&A defers until it is `ready`.
- **App SP grants are manual** (no script) — see Grants.
- **FMAPI + embedding endpoint names are region-specific** — swap for what's available in your region.
- `VectorSearchClient` in `setup_kb.py` needs an explicit workspace URL + token (it doesn't read the
  CLI profile) — already handled in the script.
- `eval/run_eval.py` imports the 10-scorer `eval/scorers.py`; the fuller 31-scorer panel is in
  `eval/scorers_full.py` (wire it in if you run eval).
- Databricks Apps are **SSO-gated**; external API calls need a bearer token — the simulator UI is the
  easy path.
- It is a **WhatsApp simulation**; production needs a WhatsApp Flow published on the WABA + Kaleyra
  Flow support (send + `nfm_reply` webhook).
