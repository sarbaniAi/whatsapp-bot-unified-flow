"""BrickFin WhatsApp AI Agent — Databricks App (Reusable Framework).

Config-driven agent with:
- MCP client for external customer data
- LLM via Databricks FMAPI
- MLflow tracing
- Twilio/Kaleyra WhatsApp integration
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from threading import Thread

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load Config ---
from config_loader import load_config, get_nudge_template
from mcp_client import MCPClient

CONFIG = load_config()
LLM_CFG = CONFIG.get("llm", {})
LLM_ENDPOINT = LLM_CFG.get("endpoint", "databricks-gpt-oss-120b")
# Model tiering: strong for reasoning/answers, small for simple turns, judge for scoring
ENDPOINTS = LLM_CFG.get("endpoints", {})

# --- LLM (tier-aware for cost optimization) ---
_wc = None


def call_llm(messages: list[dict], tier: str = "strong") -> str:
    """Call an FMAPI endpoint chosen by tier ('strong' | 'small' | 'judge')."""
    global _wc
    if _wc is None:
        from databricks.sdk import WorkspaceClient
        _wc = WorkspaceClient()
    endpoint = ENDPOINTS.get(tier, LLM_ENDPOINT)
    try:
        response = _wc.api_client.do("POST", f"/serving-endpoints/{endpoint}/invocations",
            body={"messages": messages, "max_tokens": LLM_CFG.get("max_tokens", 500),
                  "temperature": LLM_CFG.get("temperature", 0.4)})
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Reasoning models return a list of parts; keep only the text parts.
        if isinstance(content, list):
            texts = [item.get("text", "") for item in content
                     if isinstance(item, dict) and item.get("type") == "text"]
            return "\n".join(t for t in texts if t)
        return content if isinstance(content, str) else (str(content) if content else "")
    except Exception as e:
        logger.error(f"LLM error ({tier}/{endpoint}): {e}")
        return ""


def call_llm_tools(messages: list[dict], tools: list = None, tier: str = "orchestrator") -> dict:
    """Native function-calling call. Returns the assistant message dict
    ({content, tool_calls}) so the agent can run a real tool-calling ReAct loop."""
    global _wc
    if _wc is None:
        from databricks.sdk import WorkspaceClient
        _wc = WorkspaceClient()
    endpoint = ENDPOINTS.get(tier, LLM_ENDPOINT)
    body = {"messages": messages, "max_tokens": LLM_CFG.get("max_tokens", 1500),
            "temperature": LLM_CFG.get("temperature", 0.4)}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    try:
        response = _wc.api_client.do("POST", f"/serving-endpoints/{endpoint}/invocations", body=body)
        msg = response.get("choices", [{}])[0].get("message", {}) or {}
        content = msg.get("content", "")
        if isinstance(content, list):  # reasoning models -> keep text parts
            content = "\n".join(i.get("text", "") for i in content
                                if isinstance(i, dict) and i.get("type") == "text")
        return {"content": content or "", "tool_calls": msg.get("tool_calls") or []}
    except Exception as e:
        logger.error(f"LLM tools error ({tier}/{endpoint}): {e}")
        return {"content": "", "tool_calls": []}


# --- Vector Search KB (grounded Q&A) ---
_vs_index = None


def kb_search(query: str, k: int = 3) -> list[str]:
    """Retrieve top-k passages from the Databricks Vector Search KB index."""
    global _vs_index
    vs_cfg = CONFIG.get("qa", {}).get("vector_search", {})
    index_name = vs_cfg.get("index")
    endpoint = vs_cfg.get("endpoint")
    text_col = vs_cfg.get("text_column", "content")
    if not (index_name and endpoint):
        return []
    try:
        if _vs_index is None:
            from databricks.vector_search.client import VectorSearchClient
            # In a Databricks App the SP OAuth is injected as env vars; pass it
            # explicitly (auto-detect fails). Locally, fall back to default creds.
            cid = os.environ.get("DATABRICKS_CLIENT_ID")
            csec = os.environ.get("DATABRICKS_CLIENT_SECRET")
            host = os.environ.get("DATABRICKS_HOST", "")
            if host and not host.startswith("http"):
                host = "https://" + host  # app env sets host without scheme
            if cid and csec:
                vsc = VectorSearchClient(workspace_url=host, service_principal_client_id=cid,
                                         service_principal_client_secret=csec, disable_notice=True)
            else:
                vsc = VectorSearchClient(disable_notice=True)
            _vs_index = vsc.get_index(endpoint_name=endpoint, index_name=index_name)
        res = _vs_index.similarity_search(query_text=query, columns=[text_col], num_results=k)
        rows = res.get("result", {}).get("data_array", [])
        return [r[0] for r in rows if r]
    except Exception as e:
        global _kb_last_error
        _kb_last_error = f"{type(e).__name__}: {e}"
        logger.error(f"Vector Search error: {e}")
        return []


_kb_last_error = None


# --- Initialize Agent (agentic ReAct loop or legacy state machine) ---
# Empty / "in-process" MCP_SERVER_URL => in-process simulated MCP (default).
# A real URL => network transport (tunnel'd mock, or BrickFin's real MCP).
MCP_URL = os.environ.get("MCP_SERVER_URL", CONFIG.get("mcp", {}).get("url", ""))
if MCP_URL.startswith("${"):  # unresolved ${...} placeholder => treat as in-process
    MCP_URL = ""
mcp_client = MCPClient(server_url=MCP_URL)
AGENT_MODE = os.environ.get("AGENT_MODE", CONFIG.get("agent", {}).get("mode", "agentic"))

if AGENT_MODE == "agentic":
    from agentic_agent import AgenticAgent
    agent = AgenticAgent(config=CONFIG, mcp_client=mcp_client, llm_fn=call_llm,
                         kb_search=kb_search, llm_tools_fn=call_llm_tools)
    logger.info("Agent mode: AGENTIC (LangGraph ReAct tool-calling loop)")
else:
    from graph_agent import WhatsAppAgent
    # legacy state machine expects llm_fn(messages) with no tier arg
    agent = WhatsAppAgent(config=CONFIG, mcp_client=mcp_client,
                          llm_fn=lambda m: call_llm(m, tier="strong"))
    logger.info("Agent mode: LEGACY (phase state machine)")

# --- MLflow Tracing ---
# Tracing setup must NEVER crash the app. On Databricks Apps, an unconfigured
# tracking URI resolves to the local file store, which MLflow 3.x rejects with an
# exception (not ImportError) — so catch everything and degrade gracefully.
# We log to the experiment given by MLFLOW_EXPERIMENT_ID (app.yaml env); the app SP
# has CAN_EDIT on it. A missing experiment_id is why traces were previously dropped.
TRACE_ENABLED = False
try:
    import mlflow
    exp_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
    exp_path = CONFIG.get("eval", {}).get("mlflow", {}).get("experiment_name", "")
    if os.environ.get("MLFLOW_TRACKING_URI") or os.environ.get("DATABRICKS_CLIENT_ID"):
        mlflow.set_tracking_uri("databricks")
        if exp_id:
            mlflow.set_experiment(experiment_id=exp_id)
            TRACE_ENABLED = True
        elif exp_path and exp_path.startswith("/"):
            mlflow.set_experiment(exp_path)
            TRACE_ENABLED = True
        if TRACE_ENABLED:
            logger.info(f"MLflow tracing ON (experiment_id={exp_id or exp_path})")
except Exception as e:
    TRACE_ENABLED = False
    logger.warning(f"MLflow tracing disabled: {e}")


def trace_turn_span(phone, message, reply, meta):
    """Emit one MLflow trace per WhatsApp turn (inputs, outputs, tools, stage, latency)."""
    if not TRACE_ENABLED:
        return
    try:
        with mlflow.start_span(name="whatsapp_turn") as span:
            span.set_inputs({"phone": phone, "message": message[:300]})
            span.set_outputs({"reply": reply[:400]})
            span.set_attributes(meta or {})
    except Exception as e:
        logger.warning(f"trace_turn_span failed: {e}")


# --- WhatsApp BSP (provider-agnostic: simulator | kaleyra | twilio) ---
from bsp import normalize_inbound, get_bsp, verify_kaleyra_signature, SIMULATOR_HTML

BSP = get_bsp(CONFIG)
logger.info(f"BSP provider: {BSP.provider}")


def send_whatsapp(to_phone: str, message: str):
    return BSP.send(to_phone, message)


# --- FastAPI ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info(f"Agent: {CONFIG.get('agent', {}).get('name', 'WhatsApp Agent')}")
    logger.info(f"MCP: {MCP_URL or 'disabled'}")
    logger.info(f"LLM: {LLM_ENDPOINT}")
    if mcp_client and mcp_client.is_available():
        try:
            tools = mcp_client.get_tools()
            logger.info(f"MCP tools: {[t['name'] for t in tools]}")
        except Exception as e:
            logger.error(f"MCP connection failed: {e}")
    yield


app = FastAPI(title=CONFIG.get("agent", {}).get("name", "WhatsApp Agent"), lifespan=lifespan)


@app.get("/health")
async def health():
    mcp_ok = False
    if mcp_client and mcp_client.is_available():
        try:
            tools = mcp_client.get_tools()
            mcp_ok = len(tools) > 0
        except Exception:
            pass
    return {
        "status": "ok",
        "agent": CONFIG.get("agent", {}).get("name", ""),
        "version": CONFIG.get("agent", {}).get("version", ""),
        "mode": AGENT_MODE,
        "bsp": BSP.provider,
        "mcp": {"transport": mcp_client.transport if mcp_client else "none", "connected": mcp_ok},
        "kb": {"endpoint": CONFIG.get("qa", {}).get("vector_search", {}).get("endpoint", ""),
               "index": CONFIG.get("qa", {}).get("vector_search", {}).get("index", "")},
        "llm": {"default": LLM_ENDPOINT, "tiers": ENDPOINTS},
    }


@app.get("/", response_class=HTMLResponse)
async def root_ui():
    """This is the UNIFIED-FLOW demo: root shows Journey A (form card)."""
    return HTMLResponse(FLOW_HTML)


@app.get("/chat", response_class=HTMLResponse)
async def simulator_ui():
    """WhatsApp-style simulator chat UI (Journey B — step-by-step)."""
    return HTMLResponse(SIMULATOR_HTML)


# --- Journey A: Unified Flow (WhatsApp Flows-style form card) ---------------
from flow_ui import FLOW_HTML          # noqa: E402
from flow_forms import build_form, process_submit  # noqa: E402
from agentic_agent import Validators   # noqa: E402
from hybrid import compose_transition, classify_turn  # noqa: E402
_FLOW_VALIDATORS = Validators(CONFIG)


@app.get("/flow", response_class=HTMLResponse)
async def flow_ui():
    """Unified-flow demo: all fields of a stage in one form card (Journey A)."""
    return HTMLResponse(FLOW_HTML)


@app.get("/api/flow/form")
async def flow_form(stage: str = "BASIC_DETAILS"):
    """Form spec for a stage, built from the same journey config."""
    return build_form(CONFIG, stage)


@app.post("/api/flow/submit")
async def flow_submit(request: Request):
    """Structured whole-form submit → validate every field + canonical processing.
    Mirrors a WhatsApp Flow `nfm_reply`; reuses the same validators as Journey B.
    On success the AGENT composes the transition (always asks 'any questions?')."""
    body = await request.json()
    res = process_submit(CONFIG, _FLOW_VALIDATORS,
                         body.get("stage", "BASIC_DETAILS"), body.get("values", {}))
    if res["ok"]:
        res["chat_message"] = compose_transition(call_llm, res["stage"], res["next_stage"])
    return res


@app.post("/api/flow/turn")
async def flow_turn(request: Request):
    """The conversational gate BETWEEN forms. The agent classifies the reply and
    either answers a question (KB + general intelligence) and re-offers, or advances
    to the next form. Stage ORDER stays deterministic; the agent runs the talk."""
    body = await request.json()
    phone = body.get("phone", "+919000000200")
    message = (body.get("message") or "").strip()
    next_stage = body.get("next_stage")
    intent = classify_turn(call_llm, message)
    if intent == "proceed":
        if not next_stage or next_stage == "COMPLETED":
            return {"action": "complete",
                    "reply": "🎉 That's everything for now — our team will take it forward from here. Thank you!"}
        return {"action": "next_form", "stage": next_stage, "form": build_form(CONFIG, next_stage),
                "reply": f"Great — let's continue with your {next_stage.lower().replace('_', ' ')}."}
    if intent == "stop":
        return {"action": "answer",
                "reply": "No problem — reply *continue* whenever you're ready to resume. 👍"}
    # question / other → agentic Q&A. Grounded KB answers don't self-steer, so add a
    # single gentle re-offer; the general/off-topic fallback already ends with one
    # gracefully, so we don't double it.
    obs = agent.tools.answer_question(phone, message) if hasattr(agent, "tools") else {}
    ans = (obs.get("answer") or obs.get("fallback") or "").strip()
    if ans and obs.get("grounded"):
        ans += " Any other questions, or shall we continue? 😊"
    return {"action": "answer", "reply": ans or
            "Happy to help — could you rephrase that? Or shall we continue? 😊"}


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    """Receive WhatsApp messages — Twilio (form) or Kaleyra/simulator (JSON).
    Same normalized path for all providers."""
    is_json = request.headers.get("content-type", "").startswith("application/json")
    raw = await request.body()
    payload = (await request.json()) if is_json else dict(await request.form())

    # Kaleyra HMAC verification (no-op unless a secret is configured)
    if payload.get("_provider") == "kaleyra" and not payload.get("_simulator"):
        secret = os.environ.get("KALEYRA_WEBHOOK_SECRET", "")
        if not verify_kaleyra_signature(raw, request.headers.get("x-kaleyra-signature", ""), secret):
            raise HTTPException(401, "bad signature")

    msg = normalize_inbound(payload, dict(request.headers))
    phone = msg["phone"]
    if not phone:
        return {"status": "ignored"}

    logger.info(f"[{msg['provider']}] from {phone}: {msg['text'][:80]} | media={bool(msg['media_url'])}")

    start = time.time()
    if msg["media_url"]:
        reply = agent.process_message(phone, msg["text"], media_url=msg["media_url"], media_type=msg["media_type"])
    else:
        reply = agent.process_message(phone, msg["text"])
    elapsed = time.time() - start

    turn_trace = agent.last_trace(phone) if hasattr(agent, "last_trace") else {}
    trace_turn_span(phone, msg["text"], reply, {
        "provider": msg["provider"],
        "tools": ",".join(turn_trace.get("tools", []) or []),
        "current_step": turn_trace.get("current_step") or "",
        "latency_ms": int(elapsed * 1000)})
    logger.info(f"Reply ({elapsed:.1f}s): {reply[:100]}")

    # Simulator wants the reply inline (+ trace); Kaleyra push happens async in prod.
    if payload.get("_simulator"):
        return JSONResponse({"reply": reply, "trace": {
            "tools": turn_trace.get("tools", []),
            "current_step": turn_trace.get("current_step"),
            "latency_ms": int(elapsed * 1000)}})
    if msg["provider"] == "twilio":
        from xml.etree.ElementTree import Element, tostring
        resp = Element("Response"); m = Element("Message"); m.text = reply; resp.append(m)
        return HTMLResponse(content=f'<?xml version="1.0" encoding="UTF-8"?>{tostring(resp, encoding="unicode")}',
                            media_type="application/xml")
    # Real Kaleyra: send outbound via REST, ack the webhook.
    BSP.send(phone, reply)
    return {"status": "sent"}


@app.post("/api/nudge")
async def trigger_nudge(request: Request):
    """Send outbound nudge to a customer."""
    body = await request.json()
    phone = body.get("phone", "+919910175907")
    lang = body.get("language", "hi")

    # Lookup via MCP
    if mcp_client and mcp_client.is_available():
        lookup = mcp_client.lookup_customer(phone)
        if not lookup.get("found"):
            return {"status": "not_found"}
        customer = lookup["customer"]
        app_data = mcp_client.get_app_status(customer["id"])
        step = app_data.get("application", {}).get("current_step", "OTP_VERIFIED") if app_data.get("found") else "OTP_VERIFIED"
    else:
        return {"status": "mcp_unavailable"}

    name = customer["name"].split()[0]
    template = get_nudge_template(CONFIG, step, lang)
    nudge_msg = template.replace("{name}", name) if template else f"Namaste {name}! Aapka loan application complete karte hain! Reply YES."

    send_whatsapp(phone, nudge_msg)
    return {"status": "sent", "customer": customer["name"], "step": step, "message": nudge_msg}


@app.post("/api/simulate")
async def simulate(request: Request):
    """Simulate a WhatsApp message (for testing without Twilio)."""
    body = await request.json()
    phone = body.get("phone", "+919910175907")
    message = body.get("message", "")
    if not message:
        raise HTTPException(400, "message required")
    reply = agent.process_message(phone, message)
    return {"reply": reply}


@app.post("/api/reset")
async def reset_session(request: Request):
    body = await request.json()
    phone = body.get("phone", "+919910175907")
    agent.reset_session(phone)
    return {"status": "reset"}


@app.get("/api/config")
async def get_config():
    """Return current agent config (excluding secrets)."""
    safe = {k: v for k, v in CONFIG.items() if k not in ("whatsapp",)}
    return safe
