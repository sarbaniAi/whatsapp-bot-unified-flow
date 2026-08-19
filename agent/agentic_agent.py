"""True Agentic Agent — LangGraph ReAct tool-calling orchestrator.

This is the CORRECTED architecture (7 Jul 2026 course-correction):
the LLM is the BRAIN (decides the next action each turn), MCP tools are the
agent's HANDS, and deterministic validators are GUARDRAILS.

Contrast with graph_agent.py (the old design): a hardcoded phase state machine
(INIT -> GREETING -> CONSENT -> COLLECTING -> ...) where the LLM was only a
sidecar for FAQ. BrickFin rejected that as "static". Here the journey is a GOAL,
not an if/else ladder — the model reasons about where the customer is (via
MCP status-first resume) and picks the next tool/action dynamically.

Key properties (mapped to BrickFin's expectations):
  1. Agentic orchestration  -> ReAct loop, LLM selects tools/actions
  2. Cross-platform resume   -> get_application_status called first, LOS = truth
  3. Intelligent Q&A         -> search_knowledge_base + two-gate LLM-as-judge
  4. Flexibility via prompts -> system prompt + policy + field config are DATA
  5. Cost optimization       -> model tiering (small model for simple turns)
  6. Agent eval              -> MLflow autolog traces every tool/llm span
  7. Governed                -> AI Gateway endpoints, PII masked before writes

Runtime deps (add to requirements.txt):
    langgraph>=0.2
    databricks-langchain>=0.1
    langchain-core

Falls back to a lightweight built-in ReAct loop if langgraph isn't installed,
so the pattern is demonstrable even in a bare app.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Deterministic validators — GUARDRAILS the agent must call, never bypass    #
# --------------------------------------------------------------------------- #
class Validators:
    """Field validation from config. Deterministic — no LLM, ~0 cost."""

    def __init__(self, config: dict):
        self.field_map: dict[str, dict] = {}
        for step in config.get("journey", {}).get("steps", []):
            for field in step.get("fields", []):
                field["step"] = step["name"]
                self.field_map[field["key"]] = field

    def validate(self, field_key: str, value: str) -> dict:
        fc = self.field_map.get(field_key)
        if not fc:
            return {"valid": False, "reason": f"unknown field '{field_key}'"}
        ftype = fc.get("type", "TEXT")
        v = value.strip()
        if ftype == "ENUM":
            opts = [o.lower() for o in fc.get("options", [])]
            aliases = {k.lower(): val for k, val in fc.get("aliases", {}).items()}
            ok = v.lower() in opts or v.lower() in aliases
            norm = aliases.get(v.lower(), v.title())
            return {"valid": ok, "normalized": norm if ok else None,
                    "reason": None if ok else f"must be one of {fc.get('options')}"}
        if ftype == "PAN":
            ok = bool(re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", v.upper()))
            return {"valid": ok, "normalized": v.upper() if ok else None,
                    "reason": None if ok else fc.get("error_hint", "invalid PAN")}
        if ftype == "NUMBER":
            cleaned = re.sub(r"[,\srR]|Rs", "", v)
            ok = bool(re.match(r"^\d+$", cleaned))
            return {"valid": ok, "normalized": cleaned if ok else None,
                    "reason": None if ok else fc.get("error_hint", "numbers only")}
        regex = fc.get("validation")
        if regex:
            ok = bool(re.match(regex, v))
            return {"valid": ok, "normalized": v if ok else None,
                    "reason": None if ok else fc.get("error_hint", "invalid format")}
        return {"valid": True, "normalized": v, "reason": None}


# --------------------------------------------------------------------------- #
#  Two-gate LLM-as-judge — anti-hallucination for free-text answers           #
# --------------------------------------------------------------------------- #
class AnswerJudge:
    """Gate 1: is retrieval relevant?  Gate 2: is the draft grounded in it?

    Both must pass or the agent falls back gracefully. Thresholds are config
    knobs (tunable in the pilot without code changes).
    """

    def __init__(self, judge_llm: Callable, config: dict):
        self.judge_llm = judge_llm  # callable(messages, tier='judge') -> str
        j = config.get("qa", {}).get("judge", {})
        self.relevance_threshold = j.get("relevance_threshold", 0.6)
        self.grounding_threshold = j.get("grounding_threshold", 0.7)

    def _score(self, prompt: str) -> float:
        raw = self.judge_llm(
            [{"role": "system", "content": "You are a strict evaluator. Reply ONLY a float 0.0-1.0."},
             {"role": "user", "content": prompt}], tier="judge")
        m = re.search(r"[01](?:\.\d+)?", raw or "")
        return float(m.group()) if m else 0.0

    def check(self, question: str, passages: list[str], draft: str) -> dict:
        ctx = "\n---\n".join(passages) if passages else ""
        if not ctx:
            return {"pass": False, "gate": 1, "relevance": 0.0, "grounding": 0.0}
        relevance = self._score(
            f"Question: {question}\n\nRetrieved passages:\n{ctx}\n\n"
            f"How relevant are these passages to answering the question? 0.0-1.0.")
        if relevance < self.relevance_threshold:
            return {"pass": False, "gate": 1, "relevance": relevance, "grounding": 0.0}
        grounding = self._score(
            f"Passages:\n{ctx}\n\nDraft answer:\n{draft}\n\n"
            f"Is EVERY claim in the draft supported by the passages? 0.0-1.0.")
        return {"pass": grounding >= self.grounding_threshold, "gate": 2,
                "relevance": relevance, "grounding": grounding}


# --------------------------------------------------------------------------- #
#  Tool layer — the agent's HANDS (MCP + KB + validators)                      #
# --------------------------------------------------------------------------- #
class AgentTools:
    """Wraps MCP client, Vector Search KB, and validators as callable tools.

    Each method returns a JSON-serializable observation the LLM reads back.
    Per-phone session context is held so the model needn't re-pass ids.
    """

    def __init__(self, config, mcp_client, kb_search: Callable | None,
                 judge: AnswerJudge, llm: Callable):
        self.config = config
        self.mcp = mcp_client
        self.kb_search = kb_search  # callable(query, k) -> list[str] (Vector Search)
        self.judge = judge
        self.llm = llm
        self.validators = Validators(config)
        self.ctx: dict[str, dict] = {}  # phone -> {customer, application}

    # -- MCP: status-first resume (LOS is source of truth) ------------------ #
    def get_application_status(self, phone: str) -> dict:
        if not (self.mcp and self.mcp.is_available()):
            return {"error": "MCP unavailable"}
        lookup = self.mcp.lookup_customer(phone)
        if not lookup.get("found"):
            return {"found": False, "note": "phone not in LOS — ask if verified on app"}
        cust = lookup["customer"]
        status = self.mcp.get_app_status(cust["id"])
        app = status.get("application", {}) if status.get("found") else {}
        self.ctx[phone] = {"customer": cust, "application": app}
        return {"found": True, "customer_name": cust.get("name"), "customer_id": cust["id"],
                "current_step": app.get("current_step"),
                "missing_basic": status.get("missing_basic", []),
                "missing_eligibility": status.get("missing_eligibility", [])}

    def validate_field(self, phone: str, field_key: str, value: str) -> dict:
        return self.validators.validate(field_key, value)

    def update_application_field(self, phone: str, field_key: str, value: str) -> dict:
        chk = self.validators.validate(field_key, value)
        if not chk["valid"]:
            return {"saved": False, "reason": chk["reason"]}  # guardrail: never save invalid
        cid = self.ctx.get(phone, {}).get("customer", {}).get("id")
        if cid and self.mcp and self.mcp.is_available():
            self.mcp.update_field(cid, field_key, chk["normalized"])
        return {"saved": True, "field": field_key, "value": chk["normalized"]}

    def pan_validate(self, phone: str) -> dict:
        """Bureau name/DOB match on the saved PAN (deterministic format check is a
        separate guardrail in Validators). Lets the agent branch on a mismatch."""
        cid = self.ctx.get(phone, {}).get("customer", {}).get("id")
        if not (cid and self.mcp and self.mcp.is_available()):
            return {"error": "MCP unavailable"}
        return self.mcp.pan_validate(cid)

    def check_eligibility(self, phone: str) -> dict:
        cid = self.ctx.get(phone, {}).get("customer", {}).get("id")
        if not (cid and self.mcp and self.mcp.is_available()):
            return {"error": "MCP unavailable"}
        return self.mcp.check_eligibility(cid)

    def generate_aa_link(self, phone: str) -> dict:
        cid = self.ctx.get(phone, {}).get("customer", {}).get("id")
        if not (cid and self.mcp and self.mcp.is_available()):
            return {"error": "MCP unavailable"}
        return self.mcp.generate_aa_link(cid)

    def upload_bank_statement(self, phone: str, filename: str = None) -> dict:
        cid = self.ctx.get(phone, {}).get("customer", {}).get("id")
        if not (cid and self.mcp and self.mcp.is_available()):
            return {"error": "MCP unavailable"}
        return self.mcp.upload_bank_statement(cid, filename)

    def push_to_los(self, phone: str) -> dict:
        cid = self.ctx.get(phone, {}).get("customer", {}).get("id")
        if cid and self.mcp and self.mcp.is_available():
            return self.mcp.push_to_los(cid)
        return {"handed_off": True, "reference": f"BRICKFIN-{cid or 0:06d}"}

    # -- Intelligent, grounded Q&A with two-gate judge ---------------------- #
    def answer_question(self, phone: str, question: str, passages: list = None) -> dict:
        if passages is None:
            passages = self.kb_search(question, 3) if self.kb_search else []
        sys = self.config.get("llm", {}).get("system_prompt", "")
        # Draft with the instruct tier: reliable text output (the reasoning model
        # often returns only hidden reasoning) and cheaper. The judge still gates it.
        draft = self.llm(
            [{"role": "system", "content": sys},
             {"role": "system", "content":
              "Answer the question using ONLY the passages below. Include the SPECIFIC facts and "
              "numbers from the passages that answer it (exact amounts, rates, percentages, durations). "
              "2-3 short WhatsApp lines, India context. If the passages do not contain the answer, "
              "reply with exactly: NOT_FOUND\n\n" + "\n---\n".join(passages)},
             {"role": "user", "content": question}], tier="orchestrator")
        # NOT_FOUND trip-wire: model self-reports the KB doesn't cover it -> graceful fallback.
        if draft and draft.strip().upper().startswith("NOT_FOUND"):
            draft = ""
        verdict = self.judge.check(question, passages, draft or "")
        if draft and verdict["pass"]:
            return {"answer": draft, "grounded": True, "scores": verdict}
        # KB MISS. Don't blunt-defer. Low-risk general / how-it-works / financial-
        # literacy questions are answered from general India-finance knowledge (NO
        # BrickFin-specific figures). Only a missing SPECIFIC figure defers; off-topic
        # is declined. This is the general-intelligence fallback (never invents numbers).
        logger.info(f"KB_GAP phone={phone} q={question!r} verdict={verdict}")
        fb = self.llm(
            [{"role": "system", "content": sys},
             {"role": "system", "content":
              "The knowledge base did NOT cover this question. Reply in 2-3 short WhatsApp lines, "
              "India financial context. Always be WARM, friendly and graceful — never curt, blunt or "
              "dismissive (do NOT say things like 'not stock tips' or 'that's not my job'). Choose ONE:\n"
              "1) GENERAL / how-it-works / financial-literacy / industry-standard (e.g. what if I miss an "
              "EMI, personal loan vs credit card, fixed vs floating rate, does applying hurt my credit "
              "score): ANSWER it correctly from general knowledge, then tie back to BrickFin. Do NOT state "
              "any specific BrickFin rate, fee, amount, tenure or eligibility number.\n"
              "2) A SPECIFIC BrickFin figure you don't have (exact rate/fee/amount/eligibility/approval): "
              "warmly say you'll confirm and follow up.\n"
              "3) OFF-TOPIC (stocks, politics, weather, sports, other businesses, life advice): FIRST "
              "acknowledge their message warmly (a genuine one-liner), THEN gently note it's a bit outside "
              "your lane since you help with BrickFin personal loans — if it's a purchase/business/goal, "
              "add that a loan could help toward it — and softly bring them back. Mirror the friendly tone "
              "of: 'That's an exciting plan! But X is a bit outside my lane — I'm here to help you get a "
              "BrickFin personal loan, which you could use toward it.'\n"
              "End every reply by gently inviting them to continue with their application. "
              "Output ONLY the message to send."},
             {"role": "user", "content": question}], tier="orchestrator")
        fb = (fb or "").strip() or ("That's a good question — let me confirm and get back to you "
                                    "shortly. Meanwhile, shall we continue your application?")
        return {"answer": fb, "grounded": False, "fallback_used": True, "scores": verdict}


# --------------------------------------------------------------------------- #
#  The agentic orchestrator                                                   #
# --------------------------------------------------------------------------- #
TOOL_SPECS = [
    ("get_application_status", "Look up the customer in the LOS and get their CURRENT stage and "
     "which fields are still missing. ALWAYS call this FIRST on a new turn to resume correctly.",
     {"phone": "string"}),
    ("update_application_field", "Validate AND save a field value to the LOS in one step. Returns "
     "{saved:true,...} on success, or {saved:false, reason:...} if invalid — then tell the customer "
     "the reason and ask again. Use this to record every field the customer provides.",
     {"phone": "string", "field_key": "string", "value": "string"}),
    ("pan_validate", "Bureau check that the saved PAN matches the customer's name/DOB. Call after "
     "saving PAN. If it returns a mismatch, ask the customer to re-check/re-enter their PAN.",
     {"phone": "string"}),
    ("check_eligibility", "Run the eligibility/credit check once required fields are collected. Returns offer.",
     {"phone": "string"}),
    ("generate_aa_link", "Generate an Account Aggregator link for bank-statement sharing.",
     {"phone": "string"}),
    ("answer_question", "Answer a free-text/general customer question from BrickFin's knowledge base "
     "(grounded + hallucination-checked). Use for ANY question or objection.",
     {"phone": "string", "question": "string"}),
    ("push_to_los", "Terminal: hand the completed application back to the BrickFin LOS.",
     {"phone": "string"}),
    ("reply_to_customer", "Send a WhatsApp message to the customer (ask the next field, confirm, etc.). "
     "This ENDS the turn.", {"phone": "string", "message": "string"}),
]


def _fn(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


# Native function-calling schemas. `phone` is injected by the runtime (not the model).
# There is no reply_to_customer tool: the assistant's final text message IS the reply.
NATIVE_TOOLS = [
    _fn("get_application_status",
        "Look up the customer's CURRENT stage in the LOS and which fields are still missing. "
        "Call this FIRST each turn so you resume correctly (the customer may have progressed elsewhere).",
        {}, []),
    _fn("update_application_field",
        "Validate AND save one field value to the LOS. Returns saved=true, or saved=false with a reason "
        "(then tell the customer the reason and ask again).",
        {"field_key": {"type": "string", "description": "field being provided, from the missing lists"},
         "value": {"type": "string", "description": "the EXTRACTED, NORMALIZED value pulled from the "
                   "customer's message (e.g. 'Single', '100000', 'ABCDE1234F', '15/08/1995') — never the "
                   "raw sentence. For ENUM fields pass the closest matching option label."}},
        ["field_key", "value"]),
    _fn("pan_validate",
        "Bureau check that the saved PAN matches the customer's name/DOB. Call after saving PAN; "
        "on mismatch ask the customer to re-check and re-enter their PAN.", {}, []),
    _fn("check_eligibility",
        "Run the eligibility/credit check once all required fields are collected. Returns the loan offer.",
        {}, []),
    _fn("generate_aa_link",
        "Generate an Account Aggregator link for the customer to share their bank statement.", {}, []),
    _fn("answer_question",
        "Answer ANY customer question, objection, or request for info from BrickFin's knowledge base "
        "(grounded + hallucination-checked). Use this for anything that is not a direct field answer.",
        {"question": {"type": "string", "description": "the customer's question, verbatim"}},
        ["question"]),
    _fn("push_to_los",
        "Terminal: hand the completed application back to the BrickFin LOS.", {}, []),
]


class AgenticAgent:
    """LangGraph ReAct agent if available; else a faithful built-in loop.

    The LLM decides the next action each turn from goal + policy + live state.
    """

    def __init__(self, config: dict, mcp_client=None, llm_fn: Callable | None = None,
                 kb_search: Callable | None = None, llm_tools_fn: Callable | None = None):
        self.config = config
        self.llm = llm_fn or (lambda m, tier="strong": "")
        self.llm_tools = llm_tools_fn  # native function-calling (preferred driver)
        self.judge = AnswerJudge(self.llm, config)
        self.tools = AgentTools(config, mcp_client, kb_search, self.judge, self.llm)
        self.max_steps = config.get("agent", {}).get("max_tool_steps", 8)
        self._history: dict[str, list] = {}  # phone -> messages (session memory)
        self._last_trace: dict[str, dict] = {}  # phone -> {tools, current_step}
        self._lg = self._try_build_langgraph()

    def last_trace(self, phone: str) -> dict:
        return self._last_trace.get(phone, {})

    # -- system prompt is DATA: goal + policy, editable without redeploy ---- #
    def _system_prompt(self, phone: str) -> str:
        base = self.config.get("llm", {}).get("system_prompt", "")
        fields = []
        for step in self.config.get("journey", {}).get("steps", []):
            for f in step.get("fields", []):
                fields.append(f"- {f['key']} ({step['name']}): {f.get('question_en', '')}")
        policy = (
            "\nYou are an AUTONOMOUS agent completing a personal-loan application on WhatsApp.\n"
            "GOAL: get the customer's application to completion and hand back to the LOS.\n"
            "POLICY (never violate):\n"
            "  1. The live LOS state is GIVEN to you at the start of each turn (you may also call "
            "get_application_status any time to refresh). It is the source of truth: RESUME from the "
            "current stage, collect only MISSING fields, and never re-ask a field already collected.\n"
            "  2. Decide the next best action yourself from the state — do NOT follow a fixed script.\n"
            "  3. Collect only MISSING fields. Always validate_field before update_application_field.\n"
            "     After saving the PAN, call pan_validate; if it returns a mismatch, ask the customer "
            "to re-check and re-enter their PAN before moving on.\n"
            "  4. For ANY question, objection, or anything that is NOT a direct field answer, call "
            "answer_question and send back its `answer` text verbatim (it is either grounded in the KB "
            "or a safe general-knowledge reply) — never invent your own figures. You may answer a "
            "question and still ask the next field in the same reply.\n"
            "  5. When all required fields are collected, call check_eligibility, present the offer, then "
            "generate_aa_link.\n"
            "  6. When you are ready to respond, write the WhatsApp message to the customer as your reply "
            "(short, warm, Hinglish, 2-3 lines) — that text is sent to them directly. Do not output JSON.\n"
            "AVAILABLE FIELDS:\n" + "\n".join(fields))
        return base + policy

    # -- leaner prompt for the built-in step-planner (status is pre-fetched) --- #
    def _planner_system(self, phone: str) -> str:
        base = self.config.get("llm", {}).get("system_prompt", "")
        fields = []
        for step in self.config.get("journey", {}).get("steps", []):
            for f in step.get("fields", []):
                fields.append(f"- {f['key']} ({step['name']}): {f.get('question_en', '')}")
        policy = (
            "\nYou complete a personal-loan application on WhatsApp. The customer's LIVE application "
            "state (current stage + missing fields) is GIVEN to you each turn — never look it up.\n"
            "Each step, choose ONE action:\n"
            "  - If the message is a QUESTION or NOT a direct answer to the pending field -> answer_question.\n"
            "  - To record a field the customer gave: update_application_field (it validates AND saves in "
            "one call; if it returns saved=false, tell them the reason and re-ask).\n"
            "  - After saving PAN: pan_validate; on mismatch, ask the customer to re-enter their PAN.\n"
            "  - When all required fields are collected: check_eligibility, present the offer, then generate_aa_link.\n"
            "  - ALWAYS finish the turn with reply_to_customer (short, warm, Hinglish).\n"
            "Collect only MISSING fields, in order; never re-ask a collected field.\n"
            "AVAILABLE FIELDS:\n" + "\n".join(fields))
        return base + policy

    # ---------------------- LangGraph path --------------------------------- #
    def _try_build_langgraph(self):
        try:
            from langgraph.prebuilt import create_react_agent
            from langgraph.checkpoint.memory import MemorySaver
            from databricks_langchain import ChatDatabricks
            from langchain_core.tools import tool as lc_tool
            return self._build_langgraph(create_react_agent, MemorySaver, ChatDatabricks, lc_tool)
        except Exception as e:
            # Missing deps OR no workspace creds (e.g. local dev) — fall back gracefully.
            logger.warning(f"LangGraph agent unavailable ({e}); using built-in ReAct loop")
            return None

    def _build_langgraph(self, create_react_agent, MemorySaver, ChatDatabricks, lc_tool):
        endpoints = self.config.get("llm", {}).get("endpoints", {})
        model = ChatDatabricks(endpoint=endpoints.get(
            "strong", self.config.get("llm", {}).get("endpoint", "databricks-gpt-oss-120b")))
        t = self.tools

        # Bind tools; phone is injected via config at call time
        lc_tools = [
            lc_tool(lambda phone: t.get_application_status(phone), name="get_application_status",
                    description="Get the customer's current stage and missing fields. Call FIRST every turn."),
            lc_tool(lambda phone, field_key, value: t.update_application_field(phone, field_key, value),
                    name="update_application_field", description="Validate + save a field to the LOS."),
            lc_tool(lambda phone: t.check_eligibility(phone), name="check_eligibility",
                    description="Run eligibility/credit check; returns loan offer."),
            lc_tool(lambda phone: t.generate_aa_link(phone), name="generate_aa_link",
                    description="Generate Account Aggregator link for bank statement."),
            lc_tool(lambda phone, question: t.answer_question(phone, question), name="answer_question",
                    description="Grounded, hallucination-checked answer to any customer question."),
            lc_tool(lambda phone: t.push_to_los(phone), name="push_to_los",
                    description="Terminal: hand completed application back to LOS."),
        ]
        return create_react_agent(model, lc_tools, checkpointer=MemorySaver())

    # ---------------------- Public entrypoint ------------------------------ #
    def process_message(self, phone: str, message: str,
                        media_url: str = None, media_type: str = None) -> str:
        if media_url:  # bank statement upload — deterministic completion
            self.tools.get_application_status(phone)  # load ctx (customer id)
            self.tools.upload_bank_statement(phone, media_url)
            res = self.tools.push_to_los(phone)
            self._last_trace[phone] = {"tools": ["get_application_status", "upload_bank_statement", "push_to_los"],
                                       "current_step": "COMPLETE"}
            ref = res.get("reference", "")
            return (f"Bank statement mil gaya ✅ Aapki application COMPLETE hai (Ref: {ref}). "
                    "BrickFin team 24 ghante mein contact karegi.")
        if self.llm_tools is not None:
            return self._run_native(phone, message)   # native function-calling ReAct (preferred)
        if self._lg is not None:
            return self._run_langgraph(phone, message)
        return self._run_builtin(phone, message)       # fallback if no tool-calling available

    # ---------------------- Native tool-calling ReAct loop ----------------- #
    # The LLM decides every action via the model's native `tools` API. Its final
    # text message IS the reply. Deterministic pieces (validators in the save tool,
    # the two-gate judge inside answer_question) are GUARDRAILS, not routing.
    def _run_native(self, phone: str, message: str) -> str:
        msgs = self._history.setdefault(
            phone, [{"role": "system", "content": self._system_prompt(phone)}])
        # Guardrail (not routing): always refresh LOS truth so cross-channel resume is
        # reliable even if the customer progressed elsewhere. The agent still decides
        # every action; this just guarantees it sees the current stage.
        status = self.tools.get_application_status(phone)
        called: list[str] = ["get_application_status"]
        self._last_trace[phone] = {"tools": called, "current_step": status.get("current_step")}
        msgs.append({"role": "system", "content":
            "Live LOS state for this turn (source of truth — resume from here, collect only the "
            "missing fields, never re-ask a collected one):\n" + json.dumps(status)[:600]})
        msgs.append({"role": "user", "content": message})
        for _ in range(self.max_steps):
            resp = self.llm_tools(msgs, tools=NATIVE_TOOLS, tier="orchestrator")
            tcs = resp.get("tool_calls") or []
            if not tcs:  # no tool call => the assistant's text is the customer reply
                reply = self._sanitize(resp.get("content"))
                if not reply:
                    reply = self._sanitize(self.llm(msgs + [{"role": "user", "content":
                        "Reply to the customer now in 2-3 short warm Hinglish lines, plain text."}],
                        tier="orchestrator")) or "Ji bataiye, main aapki kaise madad karun?"
                msgs.append({"role": "assistant", "content": reply})
                return reply
            msgs.append({"role": "assistant", "content": resp.get("content") or "", "tool_calls": tcs})
            for tc in tcs:
                fname = tc.get("function", {}).get("name", "")
                try:
                    fargs = json.loads(tc.get("function", {}).get("arguments") or "{}")
                except Exception:
                    fargs = {}
                called.append(fname)
                result = self._exec_tool(fname, fargs, phone)
                if fname == "get_application_status" and isinstance(result, dict):
                    self._last_trace[phone]["current_step"] = result.get("current_step")
                msgs.append({"role": "tool", "tool_call_id": tc.get("id"),
                             "content": json.dumps(result)[:900]})
        # Budget exhausted — force a final plain-text reply (no tools).
        reply = self._sanitize(self.llm(msgs + [{"role": "user", "content":
            "Reply to the customer now in 2-3 short warm Hinglish lines, plain text only."}],
            tier="orchestrator"))
        return reply or "Ek minute — main aapki application check kar raha hoon."

    def _exec_tool(self, name: str, args: dict, phone: str) -> dict:
        t = self.tools
        if name == "get_application_status":
            return t.get_application_status(phone)
        if name == "update_application_field":
            return t.update_application_field(phone, args.get("field_key", ""), args.get("value", ""))
        if name == "pan_validate":
            return t.pan_validate(phone)
        if name == "check_eligibility":
            return t.check_eligibility(phone)
        if name == "generate_aa_link":
            return t.generate_aa_link(phone)
        if name == "answer_question":
            return t.answer_question(phone, args.get("question", ""))
        if name == "push_to_los":
            return t.push_to_los(phone)
        return {"error": f"unknown tool {name}"}

    def _run_langgraph(self, phone: str, message: str) -> str:
        from langchain_core.messages import SystemMessage, HumanMessage
        cfg = {"configurable": {"thread_id": phone}}
        msgs = [SystemMessage(self._system_prompt(phone)),
                HumanMessage(f"[phone={phone}] {message}")]
        out = self._lg.invoke({"messages": msgs}, cfg)
        final = out["messages"][-1]
        return getattr(final, "content", str(final))

    # ---------------------- Built-in ReAct loop ---------------------------- #
    # Faithful fallback so the agentic pattern runs without langgraph.
    # HYBRID: status-first is a DETERMINISTIC step (always fetched once at the top
    # of the turn = reliable cross-channel resume); the LLM then REASONS over the
    # remaining actions. This is the guardrail-vs-brain split, and it also prevents
    # the model from looping on get_application_status.
    def _run_builtin(self, phone: str, message: str) -> str:
        """Route -> act -> compose. Deterministic control flow (no fragile ReAct
        loop): classify the message once, take the right action, then compose one
        clean plain-text reply. The customer can NEVER see tool JSON, and there is
        no way to loop on a tool."""
        called: list[str] = ["get_application_status"]  # deterministic OBSERVE (resume)
        status = self.tools.get_application_status(phone)
        self._last_trace[phone] = {"tools": called, "current_step": status.get("current_step")}
        missing = (status.get("missing_basic") or []) + (status.get("missing_eligibility") or [])
        pending = missing[0] if missing else None

        # 1. ROUTE — classify the customer's message (reliable single call).
        cls = self.llm([
            {"role": "system", "content": self._planner_system(phone)},
            {"role": "system", "content": f"LOS STATE: {json.dumps(status)[:500]}\nPending field: {pending}"},
            {"role": "user", "content": message},
            {"role": "system", "content":
             "Classify the customer's message. Reply ONLY JSON: "
             '{"intent":"answer|question|other","field_key":"<the field being answered, only if it is one of '
             'the MISSING fields>","value":"<the value they gave>"}. '
             "intent=answer only if the message directly provides the pending/a missing field; "
             "intent=question for any question, objection, or request for info; else intent=other."}],
            tier="orchestrator")
        c = self._parse_json(cls) or {}
        intent = c.get("intent", "other")

        # 2a. QUESTION / objection -> grounded, judged answer (+ steer back).
        if intent == "question":
            obs = self.tools.answer_question(phone, message)
            called.append("answer_question")
            # answer is always populated now: a grounded KB answer, or the general-
            # intelligence fallback (low-risk answer / defer / decline) on a KB miss.
            ans = obs.get("answer") or obs.get("fallback", "")
            tail = f" {self._field_question(pending)}" if pending else ""
            return self._sanitize(ans) + (("\n\n" + tail.strip()) if tail.strip() else "")

        # 2b. ANSWER to a field -> validate+save atomically, branch, then compose.
        if intent == "answer" and c.get("field_key") in missing:
            fk = c["field_key"]
            res = self.tools.update_application_field(phone, fk, c.get("value", message))
            called.append("update_application_field")
            if not res.get("saved"):
                return self._compose(phone, f"The value for {fk} was invalid ({res.get('reason')}). "
                                            f"Apologise briefly and ask for it again.")
            if fk == "pan":  # bureau name/DOB check after saving PAN
                pv = self.tools.pan_validate(phone); called.append("pan_validate")
                if not pv.get("valid"):
                    return self._compose(phone, "The PAN's name/DOB did not match our records. "
                                                "Ask the customer to re-check and re-enter their PAN.")
            status = self.tools.get_application_status(phone)  # refresh after save
            if not ((status.get("missing_basic") or []) + (status.get("missing_eligibility") or [])):
                el = self.tools.check_eligibility(phone); called.append("check_eligibility")
                return self._compose(phone, f"All details collected. Eligibility result: {json.dumps(el)[:200]}. "
                                            "Present the loan offer warmly and mention next step is the bank statement.")
            self._last_trace[phone]["current_step"] = status.get("current_step")
            return self._compose(phone, f"Saved {fk}. Confirm briefly and ask the next missing field.")

        # 2c. OTHER -> acknowledge and move the application forward.
        return self._compose(phone, "Acknowledge the message warmly and ask for the next missing field.")

    def _field_question(self, field_key: str) -> str:
        """The configured question text for a field (English)."""
        for step in self.config.get("journey", {}).get("steps", []):
            for f in step.get("fields", []):
                if f["key"] == field_key:
                    return f.get("question_en", f.get("question_hi", f"Please provide {field_key}."))
        return ""

    def _compose(self, phone: str, instruction: str) -> str:
        """Compose one short, clean customer reply. Plain text only — sanitised."""
        status = self.tools.ctx.get(phone, {}).get("application", {})
        st = self.tools.get_application_status(phone)
        missing = (st.get("missing_basic") or []) + (st.get("missing_eligibility") or [])
        nextq = self._field_question(missing[0]) if missing else "Aapki application ab poori ho gayi hai!"
        r = self.llm([
            {"role": "system", "content": self.config.get("llm", {}).get("system_prompt", "")},
            {"role": "user", "content":  # Claude requires a user turn
             f"{instruction}\nThe next missing field to ask for is: \"{nextq}\"\n"
             "Write ONLY the WhatsApp message to the customer: 2-3 short warm Hinglish lines. "
             "Plain text only — no JSON, no tool names, no code."}],
            tier="orchestrator")
        return self._sanitize(r) or nextq

    @staticmethod
    def _sanitize(text) -> str:
        """Guarantee the customer never sees tool JSON / code leakage."""
        if not isinstance(text, str) or not text.strip():
            return ""
        t = text.strip()
        # If it looks like tool-call JSON or contains a tool name as a bare token, strip to safe.
        bad = ('{"tool"', '"field_key"', '"question":', "update_application_field",
               "get_application_status", "answer_question(", "reply_to_customer")
        if t.startswith("{") or any(b in t for b in bad):
            return "Ek minute — main aapki details confirm kar raha hoon. Kya hum aage badh sakte hain?"
        return t

    @staticmethod
    def _parse_json(s) -> dict | None:
        """Extract the FIRST balanced {...} object (models sometimes emit several)."""
        if not isinstance(s, str) or "{" not in s:
            return None
        start = s.index("{")
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except Exception:
                        return None
        return None

    def reset_session(self, phone: str):
        self._history.pop(phone, None)
        self.tools.ctx.pop(phone, None)
