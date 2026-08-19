"""MCP Client — dual transport.

Same interface regardless of transport, so the agent never changes:
  • in-process : imports the simulated MCP server module and calls call_tool()
                 directly. No network / tunnel / IP ACL. Selected when
                 MCP_SERVER_URL is empty, "in-process", or "local".
  • network    : HTTP POST /call to a remote MCP server (the simulated one behind
                 a tunnel, or — in production — BrickFin's real MCP over PrivateLink).

Swapping to BrickFin's real MCP = set MCP_SERVER_URL to their endpoint and add the
OAuth header in _http_call(). Nothing else changes.
"""

import json
import logging
import os
import ssl
import urllib.request

logger = logging.getLogger(__name__)

_INPROC_VALUES = {"", "in-process", "inprocess", "local", "mock"}


class MCPClient:
    def __init__(self, server_url: str = None, timeout: int = 10):
        self.server_url = (server_url if server_url is not None
                           else os.environ.get("MCP_SERVER_URL", "")).strip()
        self.timeout = timeout
        self.in_process = self.server_url.lower() in _INPROC_VALUES
        self._ctx = ssl.create_default_context()
        self._tools_cache = None
        self._mock = None
        if self.in_process:
            # Import the simulated server as a library (seeds data on import).
            import importlib, sys, pathlib
            root = pathlib.Path(__file__).resolve().parent.parent
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            self._mock = importlib.import_module("mcp_server.server")
            logger.info("MCP transport: IN-PROCESS (simulated BrickFin MCP)")
        else:
            logger.info(f"MCP transport: NETWORK ({self.server_url})")

    def is_available(self) -> bool:
        return self.in_process or bool(self.server_url)

    @property
    def transport(self) -> str:
        return "in-process" if self.in_process else "network"

    # -- transport primitives ------------------------------------------------
    def _http_call(self, tool_name: str, arguments: dict) -> dict:
        body = json.dumps({"tool": tool_name, "arguments": arguments}).encode()
        headers = {"Content-Type": "application/json"}
        # Production: headers["Authorization"] = f"Bearer {oauth_token()}"
        req = urllib.request.Request(f"{self.server_url}/call", data=body,
                                     headers=headers, method="POST")
        resp = urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx)
        data = json.loads(resp.read())
        return data.get("result", data)

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        try:
            if self.in_process:
                result = self._mock.call_tool(tool_name, arguments)
            else:
                result = self._http_call(tool_name, arguments)
            logger.info(f"MCP {tool_name}({arguments}) -> {json.dumps(result)[:180]}")
            return result
        except Exception as e:
            logger.error(f"MCP call error: {tool_name}({arguments}) -> {e}")
            return {"error": str(e)}

    def get_tools(self) -> list[dict]:
        if self._tools_cache is not None:
            return self._tools_cache
        try:
            if self.in_process:
                self._tools_cache = self._mock.tools_list()
            else:
                req = urllib.request.Request(f"{self.server_url}/tools",
                    headers={"Content-Type": "application/json"}, method="POST", data=b"{}")
                resp = urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx)
                self._tools_cache = json.loads(resp.read()).get("tools", [])
            return self._tools_cache
        except Exception as e:
            logger.error(f"MCP get_tools error: {e}")
            return []

    # -- typed convenience wrappers (used by the agent) ----------------------
    def lookup_customer(self, phone: str) -> dict:
        return self.call_tool("lookup_by_mobile", {"phone": phone})

    def get_app_status(self, customer_id: int) -> dict:
        return self.call_tool("get_application_status", {"customer_id": customer_id})

    def pan_validate(self, customer_id: int, pan: str = None) -> dict:
        return self.call_tool("pan_validate", {"customer_id": customer_id, "pan": pan})

    def soft_credit_pull(self, customer_id: int) -> dict:
        return self.call_tool("soft_credit_pull", {"customer_id": customer_id})

    def hard_credit_pull(self, customer_id: int) -> dict:
        return self.call_tool("hard_credit_pull", {"customer_id": customer_id})

    def update_field(self, customer_id: int, field: str, value: str) -> dict:
        return self.call_tool("update_application", {"customer_id": customer_id, "field": field, "value": value})

    def check_eligibility(self, customer_id: int) -> dict:
        return self.call_tool("check_eligibility", {"customer_id": customer_id})

    def generate_aa_link(self, customer_id: int) -> dict:
        return self.call_tool("generate_aa_link", {"customer_id": customer_id})

    def upload_bank_statement(self, customer_id: int, filename: str = None) -> dict:
        return self.call_tool("upload_bank_statement", {"customer_id": customer_id, "filename": filename})

    def notify_los_stage_completed(self, customer_id: int, stage: str) -> dict:
        return self.call_tool("notify_los_stage_completed", {"customer_id": customer_id, "stage": stage})

    def push_to_los(self, customer_id: int) -> dict:
        return self.call_tool("push_to_los", {"customer_id": customer_id})
