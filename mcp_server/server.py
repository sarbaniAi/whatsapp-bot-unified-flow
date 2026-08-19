"""BrickFin MCP Server (SIMULATED) — stands in for BrickFin's real MCP server.

In production this is BrickFin's MCP server inside their VPC (OAuth + PrivateLink),
fronting the LOS. Here it is a faithful STAND-IN with the same wire contract so
the agent code is identical against the mock and the real thing:

    MCPClient  --POST /call {tool, arguments}-->  MCP server  -->  {result: ...}

Two transport modes (see mcp_client.py):
  • in-process : the agent imports this module and calls `call_tool()` directly
                 — no network, no tunnel, no IP ACL. Default for demos.
  • network    : run this file as an HTTP server (`python server.py`) and point
                 MCP_SERVER_URL at it (optionally via a Cloudflare tunnel) to
                 demonstrate the real over-the-wire MCP-client path.

Tools (mirroring the intent of BrickFin's 13-tool set):
  lookup_by_mobile · get_application_status · pan_validate · soft_credit_pull ·
  hard_credit_pull · update_application · check_eligibility · generate_aa_link ·
  upload_bank_statement · notify_los_stage_completed · push_to_los

Demo personas are seeded at DIFFERENT stages so cross-channel RESUME is provable:
a customer who finished Basic Details on BrickFin web then messages on WhatsApp
must be resumed at Eligibility — never re-asked their PAN.
"""

import json
import logging
import random
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | MCP | %(message)s")
logger = logging.getLogger(__name__)

PORT = 8090

CUSTOMERS: dict[str, dict] = {}
APPLICATIONS: dict[int, dict] = {}

BASIC_FIELDS = ["dob", "gender", "pan", "address_line1", "address_city", "address_pincode"]
ELIG_FIELDS = ["marital_status", "employment_type", "monthly_income", "company_name", "loan_amount_requested"]

# Stage progression the LOS reports back (source of truth for resume)
STAGE_ORDER = ["OTP_VERIFIED", "BASIC_DETAILS", "ELIGIBILITY_DETAILS",
               "ELIGIBILITY_CHECK", "LOAN_OFFER", "BANK_STATEMENT", "COMPLETE"]


def _norm_phone(phone: str) -> str:
    phone = str(phone).replace("whatsapp:", "").strip()
    if not phone.startswith("+"):
        phone = f"+91{phone}" if not phone.startswith("91") else f"+{phone}"
    return phone


def _blank_app(cid: int) -> dict:
    return {"id": cid, "customer_id": cid, "current_step": "OTP_VERIFIED",
            "dob": None, "gender": None, "pan": None,
            "address_line1": None, "address_city": None, "address_pincode": None,
            "marital_status": None, "employment_type": None, "monthly_income": None,
            "company_name": None, "loan_amount_requested": None,
            "loan_amount_offered": None, "interest_rate": None, "tenure_months": None,
            "eligibility_status": "PENDING", "bank_statement_status": "PENDING",
            "created_at": (datetime.now() - timedelta(days=random.randint(1, 14))).isoformat()}


def _fill(app: dict, fields: list[str], city: str):
    samples = {
        "dob": f"{random.randint(1,28):02d}/{random.randint(1,12):02d}/{random.randint(1985,2000)}",
        "gender": random.choice(["Male", "Female"]),
        "pan": f"{''.join(chr(random.randint(65,90)) for _ in range(5))}{random.randint(1000,9999)}{chr(random.randint(65,90))}",
        "address_line1": f"{random.randint(1,500)}, Sector {random.randint(1,50)}",
        "address_city": city, "address_pincode": str(random.randint(100000, 999999)),
        "marital_status": random.choice(["Single", "Married"]),
        "employment_type": random.choice(["Salaried", "Self-Employed"]),
        "monthly_income": random.choice([35000, 50000, 75000, 100000]),
        "company_name": random.choice(["TCS", "Infosys", "HDFC", "Self"]),
        "loan_amount_requested": 500000,
    }
    for f in fields:
        app[f] = samples[f]


def seed_data():
    """Seed demo personas at different journey stages (for resume/rejection demos)."""
    CUSTOMERS.clear(); APPLICATIONS.clear()
    # (phone, name, city, lang, stage-reached, demo-note)
    personas = [
        ("+919910175907", "Sarbani Maiti", "Bengaluru", "hi", "OTP_VERIFIED",
         "FRESH — nothing collected; full journey from the top"),
        ("+919876543210", "Rajesh Kumar", "Delhi", "hi", "BASIC_DETAILS",
         "RESUME — finished Basic Details on BrickFin WEB; returns on WhatsApp -> must resume at Eligibility"),
        ("+919876543211", "Priya Sharma", "Mumbai", "mr", "ELIGIBILITY_DETAILS",
         "READY-FOR-OFFER — all details in; next action is eligibility check + offer"),
        ("+919876543212", "Amit Patel", "Ahmedabad", "gu", "BASIC_DETAILS",
         "PAN-MISMATCH — PAN on file fails name/DOB match at the bureau (agentic branch)"),
        ("+919876543213", "Sunita Devi", "Jaipur", "hi", "OTP_VERIFIED", "FRESH"),
    ]
    for i, (phone, name, city, lang, stage, note) in enumerate(personas, 1):
        CUSTOMERS[phone] = {"id": i, "name": name, "phone": phone, "city": city,
                            "language": lang, "email": f"{name.split()[0].lower()}@gmail.com",
                            "otp_verified": True, "_demo": note}
        app = _blank_app(i)
        app["current_step"] = stage
        reached = STAGE_ORDER.index(stage)
        if reached >= STAGE_ORDER.index("BASIC_DETAILS"):
            _fill(app, BASIC_FIELDS, city)
        if reached >= STAGE_ORDER.index("ELIGIBILITY_DETAILS"):
            _fill(app, ELIG_FIELDS, city)
        # Amit: seed a PAN that will fail bureau match
        if name == "Amit Patel":
            app["pan"] = "ZZZZZ9999Z"
        APPLICATIONS[i] = app
    logger.info(f"Seeded {len(CUSTOMERS)} personas: " +
                ", ".join(f"{c['phone']}={a['current_step']}" for c, a in zip(CUSTOMERS.values(), APPLICATIONS.values())))


def _advance(app: dict):
    """Recompute current_step from what's collected (LOS-style state)."""
    have_basic = all(app.get(f) for f in BASIC_FIELDS)
    have_elig = all(app.get(f) for f in ELIG_FIELDS)
    if app["bank_statement_status"] == "RECEIVED":
        app["current_step"] = "COMPLETE"
    elif app["eligibility_status"] == "ELIGIBLE":
        app["current_step"] = "LOAN_OFFER"
    elif have_basic and have_elig:
        app["current_step"] = "ELIGIBILITY_CHECK"
    elif have_basic:
        app["current_step"] = "ELIGIBILITY_DETAILS"
    elif app["current_step"] == "OTP_VERIFIED" and any(app.get(f) for f in BASIC_FIELDS):
        app["current_step"] = "BASIC_DETAILS"


# --- Tools -----------------------------------------------------------------
def lookup_by_mobile(phone):
    cust = CUSTOMERS.get(_norm_phone(phone))
    if not cust:
        return {"found": False, "error": f"No customer found for {_norm_phone(phone)}"}
    return {"found": True, "customer": {k: v for k, v in cust.items() if not k.startswith("_")}}


def get_application_status(customer_id):
    """Source of truth for cross-channel resume: current stage + what's missing."""
    app = APPLICATIONS.get(int(customer_id))
    if not app:
        return {"found": False, "error": f"No application for customer {customer_id}"}
    _advance(app)
    return {"found": True, "application": app,
            "current_step": app["current_step"],
            "missing_basic": [f for f in BASIC_FIELDS if not app.get(f)],
            "missing_eligibility": [f for f in ELIG_FIELDS if not app.get(f)]}


def pan_validate(customer_id, pan=None):
    """Bureau check: does the PAN match the name/DOB on file? (name/DOB-mismatch demo)"""
    app = APPLICATIONS.get(int(customer_id))
    if not app:
        return {"valid": False, "error": "Application not found"}
    pan = (pan or app.get("pan") or "").upper()
    # Simulated bureau rule: PANs starting 'ZZZZZ' fail the name/DOB match
    if pan.startswith("ZZZZZ"):
        return {"valid": False, "reason": "NAME_DOB_MISMATCH",
                "message": "Name/DOB on PAN do not match the details provided"}
    return {"valid": True, "pan": pan}


def soft_credit_pull(customer_id):
    return {"passed": True, "bureau_score_band": random.choice(["GOOD", "EXCELLENT"])}


def hard_credit_pull(customer_id):
    return {"passed": True, "cibil": random.randint(720, 810)}


def update_application(customer_id, field, value):
    app = APPLICATIONS.get(int(customer_id))
    if not app:
        return {"success": False, "error": "Application not found"}
    if field not in app:
        return {"success": False, "error": f"Unknown field: {field}"}
    app[field] = value
    _advance(app)
    return {"success": True, "field": field, "value": value, "current_step": app["current_step"]}


def check_eligibility(customer_id):
    app = APPLICATIONS.get(int(customer_id))
    if not app:
        return {"eligible": False, "error": "Application not found"}
    income = float(app.get("monthly_income") or 0)
    requested = float(app.get("loan_amount_requested") or 0)
    if not income or not requested:
        return {"eligible": False, "error": "Income or loan amount not provided"}
    offered = min(requested, income * 20)
    if offered < 50000:
        return {"eligible": False, "reason": "Income too low for minimum loan amount"}
    rate = round(random.uniform(10.49, 16.5), 2)
    tenure = 36
    r = rate / 100 / 12
    emi = round(offered * r / (1 - (1 + r) ** (-tenure)))
    app.update({"loan_amount_offered": offered, "interest_rate": rate, "tenure_months": tenure,
                "eligibility_status": "ELIGIBLE", "current_step": "LOAN_OFFER"})
    return {"eligible": True, "amount": offered, "rate": rate, "tenure": tenure, "emi": emi}


def generate_aa_link(customer_id):
    return {"link": f"https://brickfin.com/aa/{int(customer_id):06d}", "expires_in": "24 hours"}


def upload_bank_statement(customer_id, filename=None):
    app = APPLICATIONS.get(int(customer_id))
    if not app:
        return {"accepted": False, "error": "Application not found"}
    app["bank_statement_status"] = "RECEIVED"
    _advance(app)
    return {"accepted": True, "current_step": app["current_step"]}


def notify_los_stage_completed(customer_id, stage):
    """Reverse callback (fire-and-forget in real life): LOS learns furthest stage."""
    logger.info(f"notify_los_stage_completed(cid={customer_id}, stage={stage})")
    return {"acknowledged": True, "stage": stage}


def push_to_los(customer_id):
    app = APPLICATIONS.get(int(customer_id))
    ref = f"BRICKFIN-{int(customer_id):06d}"
    if app:
        app["current_step"] = "COMPLETE"
    return {"handed_off": True, "reference": ref, "status": "Pending_Credit_Review"}


TOOLS = {
    "lookup_by_mobile": {"description": "Find a customer by mobile number; returns profile.",
                         "parameters": {"phone": {"type": "string"}}, "handler": lookup_by_mobile},
    "get_application_status": {"description": "Current stage + missing fields (LOS source of truth). Call FIRST every turn.",
                               "parameters": {"customer_id": {"type": "integer"}}, "handler": get_application_status},
    "pan_validate": {"description": "Bureau check that PAN matches name/DOB on file.",
                     "parameters": {"customer_id": {"type": "integer"}, "pan": {"type": "string"}}, "handler": pan_validate},
    "soft_credit_pull": {"description": "Soft bureau pull (no score impact).",
                         "parameters": {"customer_id": {"type": "integer"}}, "handler": soft_credit_pull},
    "hard_credit_pull": {"description": "Hard bureau pull; returns CIBIL.",
                         "parameters": {"customer_id": {"type": "integer"}}, "handler": hard_credit_pull},
    "update_application": {"description": "Save a single validated field to the LOS.",
                           "parameters": {"customer_id": {"type": "integer"}, "field": {"type": "string"}, "value": {"type": "string"}},
                           "handler": update_application},
    "check_eligibility": {"description": "Run eligibility/credit check; returns loan offer.",
                          "parameters": {"customer_id": {"type": "integer"}}, "handler": check_eligibility},
    "generate_aa_link": {"description": "Account Aggregator link for bank-statement sharing.",
                         "parameters": {"customer_id": {"type": "integer"}}, "handler": generate_aa_link},
    "upload_bank_statement": {"description": "Register a received bank statement.",
                              "parameters": {"customer_id": {"type": "integer"}, "filename": {"type": "string"}}, "handler": upload_bank_statement},
    "notify_los_stage_completed": {"description": "Reverse callback so LOS knows the furthest completed stage.",
                                   "parameters": {"customer_id": {"type": "integer"}, "stage": {"type": "string"}}, "handler": notify_los_stage_completed},
    "push_to_los": {"description": "Terminal: hand the completed application back to the LOS.",
                    "parameters": {"customer_id": {"type": "integer"}}, "handler": push_to_los},
}


def call_tool(tool_name: str, arguments: dict) -> dict:
    """Dispatch used by BOTH the in-process client and the HTTP handler."""
    if tool_name not in TOOLS:
        return {"error": f"Tool '{tool_name}' not found"}
    try:
        result = TOOLS[tool_name]["handler"](**(arguments or {}))
        logger.info(f"call {tool_name}({arguments}) -> {json.dumps(result)[:180]}")
        return result
    except Exception as e:
        logger.error(f"error {tool_name}({arguments}) -> {e}")
        return {"error": str(e)}


def tools_list() -> list[dict]:
    return [{"name": n, "description": t["description"],
             "inputSchema": {"type": "object", "properties": t["parameters"]}}
            for n, t in TOOLS.items()]


# Seed on import so in-process mode works without running the server.
seed_data()


# --- HTTP server (network mode) -------------------------------------------
class MCPHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        if self.path in ("/tools", "/mcp/tools"):
            return self._respond(200, {"tools": tools_list()})
        if self.path in ("/call", "/mcp/call"):
            name = body.get("tool", body.get("name", ""))
            args = body.get("arguments", body.get("args", body.get("input", {})))
            if name not in TOOLS:
                return self._respond(404, {"error": f"Tool '{name}' not found"})
            return self._respond(200, {"result": call_tool(name, args)})
        if self.path == "/health":
            return self._respond(200, {"status": "ok", "tools": list(TOOLS), "customers": len(CUSTOMERS)})
        self._respond(404, {"error": "Use /tools, /call, or /health"})

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok", "tools": list(TOOLS), "customers": len(CUSTOMERS)})
        elif self.path == "/tools":
            self._respond(200, {"tools": tools_list()})
        else:
            self._respond(200, {"message": "BrickFin MCP Server (simulated)", "endpoints": ["/health", "/tools", "/call"]})

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), MCPHandler)
    logger.info(f"MCP server (network mode) on http://0.0.0.0:{PORT} | tools={list(TOOLS)}")
    logger.info(f"Expose externally: cloudflared tunnel --url http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
