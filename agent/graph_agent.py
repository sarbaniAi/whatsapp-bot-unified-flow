"""LangGraph Agent — Config-driven, MCP-connected WhatsApp AI agent.

State machine with tool-calling:
INIT → GREETING → CONSENT → COLLECTING → ELIGIBILITY → OFFER → BANK_STATEMENT → DONE

Uses MCP server for live customer data and LLM for conversational intelligence.
"""

import json
import logging
import os
import re
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class WhatsAppAgent:
    """Config-driven WhatsApp agent with MCP integration."""

    def __init__(self, config: dict, mcp_client=None, llm_fn=None):
        self.config = config
        self.mcp = mcp_client
        self.llm_fn = llm_fn  # callable(messages) -> str
        self._sessions = {}  # phone -> session state
        self._build_field_index()

    def _build_field_index(self):
        """Build field lookup from config."""
        self.all_fields = []
        self.field_map = {}  # key -> field config
        self.step_fields = {}  # step_name -> [field configs]
        for step in self.config.get("journey", {}).get("steps", []):
            step_name = step["name"]
            self.step_fields[step_name] = step.get("fields", [])
            for field in step.get("fields", []):
                field["step"] = step_name
                self.all_fields.append(field)
                self.field_map[field["key"]] = field

    def process_message(self, phone: str, message: str, media_url: str = None, media_type: str = None) -> str:
        """Process an incoming WhatsApp message. Returns reply text."""
        session = self._get_session(phone)

        # Handle media (PDF upload)
        if media_url:
            return self._handle_media(session, media_url, media_type)

        phase = session.get("phase", "INIT")

        if phase == "INIT":
            return self._handle_init(session, phone, message)
        elif phase == "GREETING":
            return self._handle_greeting(session, message)
        elif phase == "CONSENT":
            return self._handle_consent(session, message)
        elif phase == "COLLECTING":
            return self._handle_collecting(session, message)
        elif phase == "OFFER_PENDING":
            return self._handle_offer(session, message)
        elif phase == "BANK_STATEMENT":
            return self._handle_bank_statement(session, message)
        elif phase == "DONE":
            return "Aapka application complete hai! BrickFin team jaldi contact karegi."
        else:
            session["phase"] = "INIT"
            return self._handle_init(session, phone, message)

    def _get_session(self, phone: str) -> dict:
        if phone not in self._sessions:
            self._sessions[phone] = {"phone": phone, "phase": "INIT", "customer": None, "app": None}
        return self._sessions[phone]

    def reset_session(self, phone: str):
        self._sessions.pop(phone, None)

    # --- Phase Handlers ---

    def _handle_init(self, session, phone, message):
        """Look up customer via MCP and generate greeting."""
        if self.mcp and self.mcp.is_available():
            # Fetch from MCP
            lookup = self.mcp.lookup_customer(phone)
            if not lookup.get("found"):
                return "Namaste! Aapka phone number humare records mein nahi mila. Kya aapne BrickFin app par mobile verify kiya hai?"
            session["customer"] = lookup["customer"]
            cust_id = lookup["customer"]["id"]

            # Get application status
            app_data = self.mcp.get_app_status(cust_id)
            if app_data.get("found"):
                session["app"] = app_data["application"]
                session["missing_basic"] = app_data.get("missing_basic", [])
                session["missing_elig"] = app_data.get("missing_eligibility", [])
        else:
            return "Namaste! System temporarily unavailable. Please try again."

        first_name = session["customer"]["name"].split()[0]
        step = session["app"]["current_step"]

        step_msgs = {
            "OTP_VERIFIED": "aapka loan application shuru hua tha! Bas kuch details chahiye.",
            "BASIC_DETAILS": "basic details almost done! Thoda aur chahiye.",
            "ELIGIBILITY_DETAILS": "basic details mil gaye! Ab eligibility ke liye income details chahiye.",
            "LOAN_OFFER": "aapka loan offer ready hai!",
        }
        context = step_msgs.get(step, "aapka application complete karte hain.")

        session["phase"] = "CONSENT"
        return (f"Namaste {first_name}! Main BrickFin Finance ki AI assistant hoon.\n\n"
                f"{first_name}, {context}\n\n"
                f"Kya hum abhi shuru karein? (YES reply karein)")

    def _handle_greeting(self, session, message):
        return self._handle_consent(session, message)

    def _handle_consent(self, session, message):
        lower = message.lower().strip()
        first_name = session["customer"]["name"].split()[0]

        yes_words = {"yes", "haan", "ha", "ok", "sure", "ji", "chalo", "start", "y"}
        no_words = {"no", "nahi", "later", "baad", "cancel"}

        if any(w in lower for w in yes_words):
            session["phase"] = "COLLECTING"
            # Determine what to collect
            missing = self._get_next_missing_field(session)
            if missing:
                session["current_field"] = missing["key"]
                q = missing.get("question_hi", missing.get("question_en", f"Please provide {missing['key']}"))
                return f"Dhanyavaad! Chaliye shuru karte hain.\n\n{q}"
            # Nothing missing — check eligibility
            return self._transition_to_eligibility(session, first_name)

        if any(w in lower for w in no_words):
            session["phase"] = "INIT"
            return f"Koi baat nahi {first_name}! Jab bhi ready hon, 'Hi' bhej dijiye."

        return "YES ya NO reply karein. Kya hum aage badhein?"

    def _handle_collecting(self, session, message):
        first_name = session["customer"]["name"].split()[0]

        # Check if question/objection — use LLM
        if self._is_question(message):
            llm_reply = self._call_llm_contextual(session, message)
            if llm_reply:
                current = session.get("current_field")
                if current and current in self.field_map:
                    q = self.field_map[current].get("question_hi", "")
                    return f"{llm_reply}\n\n---\nChaliye continue karte hain:\n{q}"
                return llm_reply

        current_field = session.get("current_field")
        if not current_field:
            missing = self._get_next_missing_field(session)
            if not missing:
                return self._transition_to_eligibility(session, first_name)
            current_field = missing["key"]
            session["current_field"] = current_field

        field_config = self.field_map.get(current_field, {})

        # Validate
        if not self._validate_field(field_config, message):
            hint = field_config.get("error_hint", "Check karein")
            return f"Format sahi nahi hai. {hint}"

        # Normalize
        value = self._normalize_field(field_config, message)

        # Save via MCP
        cust_id = session["customer"]["id"]
        if self.mcp and self.mcp.is_available():
            self.mcp.update_field(cust_id, current_field, value)

        # Update local state
        session["app"][current_field] = value
        if current_field in session.get("missing_basic", []):
            session["missing_basic"].remove(current_field)
        if current_field in session.get("missing_elig", []):
            session["missing_elig"].remove(current_field)

        # Next field
        missing = self._get_next_missing_field(session)
        if missing:
            session["current_field"] = missing["key"]
            q = missing.get("question_hi", missing.get("question_en", f"Please provide {missing['key']}"))
            total = len([f for f in self.all_fields if f["step"] == missing["step"]])
            step_fields = self.step_fields.get(missing["step"], [])
            done = total - len([f for f in step_fields if f["key"] in session.get("missing_basic", []) + session.get("missing_elig", [])])
            progress = f"({done}/{total} done) " if done > 0 else ""
            return f"Noted! {progress}\n\n{q}"

        # All fields collected
        return self._transition_to_eligibility(session, first_name)

    def _handle_offer(self, session, message):
        lower = message.lower().strip()
        first_name = session["customer"]["name"].split()[0]

        if any(w in lower for w in ["yes", "haan", "accept", "ok", "sure"]):
            session["phase"] = "BANK_STATEMENT"
            cust_id = session["customer"]["id"]
            if self.mcp and self.mcp.is_available():
                link_data = self.mcp.generate_aa_link(cust_id)
                link = link_data.get("link", f"https://brickfin.com/aa-link/{cust_id}")
            else:
                link = f"https://brickfin.com/aa-link/{cust_id}"
            return (f"Bahut accha {first_name}! Last step: Bank statement.\n\n"
                    f"Account Aggregator link:\n{link}\n\n"
                    f"Ya WhatsApp par PDF bhej sakte hain.")

        if any(w in lower for w in ["no", "nahi", "decline"]):
            session["phase"] = "INIT"
            return f"Koi baat nahi {first_name}. Mann badle toh 'Hi' bhejein."

        # Question about the offer — use LLM
        if self._is_question(message):
            return self._call_llm_contextual(session, message) or "Loan offer accept karne ke liye YES bhejein."

        return "Loan offer accept karne ke liye YES, decline ke liye NO bhejein."

    def _handle_bank_statement(self, session, message):
        first_name = session["customer"]["name"].split()[0]
        lower = message.lower().strip()

        if any(w in lower for w in ["done", "ho gaya", "submitted", "sent", "bhej diya"]):
            session["phase"] = "DONE"
            app_id = session["app"]["id"]
            return (f"Bahut badhiya {first_name}! Application COMPLETE!\n"
                    f"BrickFin team 24 ghante mein contact karegi.\n"
                    f"Reference: BRICKFIN-{app_id:06d}")

        if self._is_question(message):
            reply = self._call_llm_contextual(session, message)
            return f"{reply}\n\nBank statement ke liye AA link use karein ya PDF bhejein." if reply else "AA link use karein ya WhatsApp par PDF bhejein."

        return "Bank statement submit karne ke liye AA link use karein ya yahan PDF bhejein."

    def _handle_media(self, session, media_url, media_type):
        first_name = session.get("customer", {}).get("name", "").split()[0] or "Customer"
        is_pdf = "pdf" in (media_type or "").lower()
        is_image = any(t in (media_type or "").lower() for t in ["image", "jpg", "jpeg", "png"])

        if is_pdf or is_image:
            session["phase"] = "DONE"
            app_id = session.get("app", {}).get("id", 0)
            doc_type = "PDF" if is_pdf else "image"
            return (f"Dhanyavaad {first_name}! Bank statement ({doc_type}) mil gaya.\n\n"
                    f"Application COMPLETE!\n"
                    f"BrickFin team 24 ghante mein contact karegi.\n"
                    f"Reference: BRICKFIN-{app_id:06d}")

        return "Yeh file type support nahi hai. PDF ya image mein bank statement bhejein."

    # --- Helpers ---

    def _get_next_missing_field(self, session) -> dict | None:
        """Get next field to collect based on missing fields from MCP."""
        # Check basic fields first
        for field in self.all_fields:
            if field["key"] in session.get("missing_basic", []):
                return field
        # Then eligibility fields
        for field in self.all_fields:
            if field["key"] in session.get("missing_elig", []):
                return field
        return None

    def _validate_field(self, field_config, value):
        regex = field_config.get("validation")
        if not regex:
            return True
        field_type = field_config.get("type", "TEXT")
        if field_type == "ENUM":
            options = [o.lower() for o in field_config.get("options", [])]
            aliases = {k.lower(): v for k, v in field_config.get("aliases", {}).items()}
            return value.strip().lower() in options or value.strip().lower() in aliases
        if field_type == "PAN":
            return bool(re.match(regex, value.strip().upper()))
        if field_type == "NUMBER":
            cleaned = value.strip().replace(",", "").replace("rs", "").replace("Rs", "").replace(" ", "")
            return bool(re.match(regex, cleaned))
        return bool(re.match(regex, value.strip()))

    def _normalize_field(self, field_config, value):
        field_type = field_config.get("type", "TEXT")
        if field_type == "ENUM":
            aliases = field_config.get("aliases", {})
            return aliases.get(value.strip().lower(), aliases.get(value.strip(), value.strip().title()))
        if field_type == "PAN":
            return value.strip().upper()
        if field_type == "NUMBER":
            return value.strip().replace(",", "").replace("rs", "").replace("Rs", "").replace(" ", "")
        if field_config.get("normalize") == "uppercase":
            return value.strip().upper()
        return value.strip()

    def _transition_to_eligibility(self, session, first_name):
        cust_id = session["customer"]["id"]
        if self.mcp and self.mcp.is_available():
            result = self.mcp.check_eligibility(cust_id)
        else:
            result = {"eligible": False, "error": "MCP unavailable"}

        if result.get("eligible"):
            session["phase"] = "OFFER_PENDING"
            amt = f"{int(result['amount']):,}"
            return (f"Congratulations {first_name}! Aap eligible hain!\n\n"
                    f"Loan: Rs {amt}\nRate: {result['rate']}% p.a.\n"
                    f"Tenure: {result['tenure']} months\nEMI: Rs {int(result['emi']):,}/month\n\n"
                    f"Accept karein? (YES/NO)")
        else:
            session["phase"] = "INIT"
            reason = result.get("reason", result.get("error", ""))
            return f"Sorry {first_name}, abhi eligible nahi hain. {reason}"

    def _is_question(self, message):
        lower = message.lower().strip()
        return any(w in lower for w in [
            "?", "kya", "kaise", "kitna", "kyun", "why", "how", "what", "when",
            "interest", "rate", "emi", "tenure", "processing", "fee", "time",
            "safe", "secure", "nahi chahiye", "not interested", "later", "busy",
            "sochna", "think", "trust", "fraud", "kyon", "zarurat",
        ])

    def _call_llm_contextual(self, session, message):
        if not self.llm_fn:
            return ""
        system = self.config.get("llm", {}).get("system_prompt", "You are a helpful assistant.")
        ctx = f"Customer: {session.get('customer', {}).get('name', '')} | Step: {session.get('app', {}).get('current_step', '')}"
        messages = [
            {"role": "system", "content": system},
            {"role": "system", "content": f"Context: {ctx}"},
            {"role": "user", "content": message},
        ]
        return self.llm_fn(messages)
