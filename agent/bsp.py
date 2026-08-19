"""WhatsApp BSP layer — provider-agnostic, so Kaleyra / Twilio / simulator all
hit the SAME agent through one normalized message shape.

Inbound  : normalize_inbound(payload, headers) -> {phone, text, media_url, media_type, message_id}
Outbound : get_bsp(config).send(phone, text)

The simulator sends Kaleyra's inbound payload shape, so the parsing/verification
path exercised in the demo is exactly the code that runs against real Kaleyra —
only the sender is faked. Switching to real Kaleyra = set whatsapp.provider:
kaleyra + credentials; nothing in the agent changes.
"""

import base64
import hashlib
import hmac
import logging
import os
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)


def _norm_phone(raw: str) -> str:
    p = str(raw or "").replace("whatsapp:", "").strip()
    if not p:
        return ""
    if not p.startswith("+"):
        p = f"+{p}" if p.startswith("91") else f"+91{p}"
    return p


def normalize_inbound(payload: dict, headers: dict = None) -> dict:
    """Map Twilio / Kaleyra / simulator inbound shapes to one dict."""
    headers = headers or {}
    # Twilio: form fields From/Body/MediaUrl0/NumMedia
    if "From" in payload or "Body" in payload:
        return {"phone": _norm_phone(payload.get("From")),
                "text": str(payload.get("Body", "")),
                "media_url": payload.get("MediaUrl0", ""),
                "media_type": payload.get("MediaContentType0", ""),
                "message_id": payload.get("MessageSid", ""),
                "provider": "twilio"}
    # Kaleyra / simulator: {from, text|text.body|message, media:{url,content_type}, message_id}
    text = payload.get("message", payload.get("body", ""))
    t = payload.get("text")
    if isinstance(t, dict):
        text = t.get("body", text)
    elif isinstance(t, str):
        text = t or text
    media = payload.get("media", {}) or {}
    return {"phone": _norm_phone(payload.get("from", payload.get("phone", ""))),
            "text": str(text or ""),
            "media_url": media.get("url", payload.get("media_url", "")),
            "media_type": media.get("content_type", payload.get("media_type", "")),
            "message_id": payload.get("message_id", ""),
            "provider": payload.get("_provider", "kaleyra")}


def verify_kaleyra_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    """HMAC-SHA256 webhook verification (same check used against real Kaleyra)."""
    if not secret:
        return True  # not enforced in simulator/demo
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")


# --- Outbound adapters -----------------------------------------------------
class SimulatorBSP:
    """No real send; the reply is returned inline to the chat UI."""
    provider = "simulator"

    def send(self, phone: str, text: str) -> dict:
        logger.info(f"[simulator] -> {phone}: {text[:80]}")
        return {"sent": True, "channel": "simulator"}


class TwilioBSP:
    provider = "twilio"

    def __init__(self, cfg: dict):
        self.sid = os.environ.get("TWILIO_SID", cfg.get("account_sid", ""))
        self.token = os.environ.get("TWILIO_TOKEN", cfg.get("auth_token", ""))
        self.frm = os.environ.get("TWILIO_WHATSAPP_FROM", cfg.get("whatsapp_from", "whatsapp:+14155238886"))

    def send(self, phone: str, text: str) -> dict:
        try:
            auth = base64.b64encode(f"{self.sid}:{self.token}".encode()).decode()
            data = urllib.parse.urlencode({"From": self.frm, "To": f"whatsapp:{phone}", "Body": text}).encode()
            req = urllib.request.Request(
                f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Messages.json",
                data=data, headers={"Authorization": f"Basic {auth}"}, method="POST")
            urllib.request.urlopen(req, timeout=10)
            return {"sent": True, "channel": "twilio"}
        except Exception as e:
            logger.error(f"Twilio send error: {e}")
            return {"sent": False, "error": str(e)}


class KaleyraBSP:
    """Real Kaleyra WhatsApp send (shape only; used when provider=kaleyra)."""
    provider = "kaleyra"

    def __init__(self, cfg: dict):
        self.sid = os.environ.get("KALEYRA_SID", cfg.get("sid", ""))
        self.api_key = os.environ.get("KALEYRA_API_KEY", cfg.get("api_key", ""))
        self.frm = os.environ.get("KALEYRA_WABA_NUMBER", cfg.get("waba_number", ""))
        self.base = cfg.get("base_url", "https://api.kaleyra.io/v1")

    def send(self, phone: str, text: str) -> dict:
        try:
            url = f"{self.base}/{self.sid}/messages"
            data = urllib.parse.urlencode({
                "to": phone.lstrip("+"), "from": self.frm, "channel": "whatsapp",
                "type": "text", "body": text}).encode()
            req = urllib.request.Request(url, data=data,
                headers={"api-key": self.api_key, "Content-Type": "application/x-www-form-urlencoded"},
                method="POST")
            urllib.request.urlopen(req, timeout=10)
            return {"sent": True, "channel": "kaleyra"}
        except Exception as e:
            logger.error(f"Kaleyra send error: {e}")
            return {"sent": False, "error": str(e)}


def get_bsp(config: dict):
    wa = config.get("whatsapp", {})
    provider = os.environ.get("BSP_PROVIDER", wa.get("provider", "simulator")).lower()
    if provider == "twilio":
        return TwilioBSP(wa.get("twilio", {}))
    if provider == "kaleyra":
        return KaleyraBSP(wa.get("kaleyra", {}))
    return SimulatorBSP()


# --- Simulator chat UI (served at GET /) -----------------------------------
SIMULATOR_HTML = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>BrickFin WhatsApp Agent — Simulator</title>
<style>
 :root{--wa:#075e54;--wamsg:#dcf8c6;--bg:#e5ddd5}
 *{box-sizing:border-box;font-family:-apple-system,Segoe UI,Roboto,sans-serif}
 body{margin:0;background:#111;display:flex;justify-content:center;padding:16px}
 .wrap{width:100%;max-width:460px}
 .phone{background:var(--bg);border-radius:14px;overflow:hidden;box-shadow:0 8px 40px #0008;height:78vh;display:flex;flex-direction:column}
 .top{background:var(--wa);color:#fff;padding:10px 14px;display:flex;align-items:center;gap:10px}
 .top .a{width:38px;height:38px;border-radius:50%;background:#fff3;display:flex;align-items:center;justify-content:center;font-weight:700}
 .top .n{font-weight:600}.top .s{font-size:11px;opacity:.8}
 .log{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:8px}
 .m{max-width:80%;padding:7px 10px;border-radius:8px;font-size:14px;white-space:pre-wrap;line-height:1.35;box-shadow:0 1px 1px #0001}
 .out{align-self:flex-end;background:var(--wamsg)}.in{align-self:flex-start;background:#fff}
 .sys{align-self:center;background:#ffeecc;color:#7a5c00;font-size:11px;border-radius:10px;padding:4px 10px}
 .bar{display:flex;gap:6px;padding:8px;background:#f0f0f0}
 .bar input{flex:1;border:0;border-radius:20px;padding:10px 14px;font-size:14px}
 .bar button{border:0;background:var(--wa);color:#fff;border-radius:50%;width:42px;height:42px;font-size:18px;cursor:pointer}
 .ctl{display:flex;gap:6px;margin-bottom:8px}
 .ctl select,.ctl button{padding:8px;border-radius:8px;border:1px solid #333;background:#1c1c1c;color:#eee;font-size:12px}
 .ctl button{cursor:pointer}
 .hint{color:#888;font-size:11px;margin:6px 2px}
</style></head><body><div class=wrap>
 <div class=ctl>
   <select id=persona></select>
   <button onclick=reset()>Reset</button>
 </div>
 <div class=phone>
   <div class=top><div class=a>BF</div><div><div class=n>BrickFin</div><div class=s id=sub>business account</div></div></div>
   <div class=log id=log></div>
   <div class=bar><input id=inp placeholder="Type a message" onkeydown="if(event.key=='Enter')send()"><button onclick=send()>&#10148;</button></div>
 </div>
 <div class=hint>Simulated Kaleyra WABA → POST /webhook/whatsapp (Kaleyra payload shape). Same handler as production.</div>
</div>
<script>
const personas=[
 ["+919910175907","Sarbani Maiti — FRESH (full journey)"],
 ["+919876543210","Rajesh Kumar — RESUME (did Basic Details on web)"],
 ["+919876543211","Priya Sharma — READY FOR OFFER"],
 ["+919876543212","Amit Patel — PAN mismatch branch"],
];
const sel=document.getElementById('persona');
personas.forEach(([p,n])=>{let o=document.createElement('option');o.value=p;o.textContent=n;sel.appendChild(o)});
const log=document.getElementById('log');
function add(cls,txt){let d=document.createElement('div');d.className='m '+cls;d.textContent=txt;log.appendChild(d);log.scrollTop=log.scrollHeight}
function sys(txt){let d=document.createElement('div');d.className='sys';d.textContent=txt;log.appendChild(d);log.scrollTop=log.scrollHeight}
async function reset(){log.innerHTML='';await fetch('/api/reset',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({phone:sel.value})});sys('Session reset for '+sel.options[sel.selectedIndex].text)}
sel.onchange=reset;
async function send(){
 const inp=document.getElementById('inp');const t=inp.value.trim();if(!t)return;inp.value='';add('out',t);
 const payload={from:sel.value,text:{body:t},message_id:'sim-'+Date.now(),_provider:'kaleyra',_simulator:true};
 try{
  const r=await fetch('/webhook/whatsapp',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)});
  const j=await r.json();add('in',j.reply||'(no reply)');
  if(j.trace){sys('stage: '+(j.trace.current_step||'?')+' · tools: '+(j.trace.tools||[]).join(', ')+' · '+(j.trace.latency_ms||0)+'ms')}
 }catch(e){sys('error: '+e)}
}
window.onload=()=>{sys('Pick a persona and say "Hi". FRESH starts fresh; RESUME picks up where BrickFin web left off.')}
</script></body></html>"""
