"""Unified-flow (hybrid) demo UI — a WhatsApp Flows-style form card (all fields at
once) with a structured submit, then the AGENT takes over in chat: it confirms,
ALWAYS asks 'any questions?', answers them, and sends the next form on 'continue'.
Self-contained HTML/JS; served at GET /."""

FLOW_HTML = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>BrickFin — Unified Flow (WhatsApp Flows) Demo</title>
<style>
:root{--wa:#075e54;--wa2:#128c7e;--bg:#ece5dd;--card:#fff;--line:#e2e2e2;--mut:#667781;--ok:#047857;--no:#b91c1c;--accent:#4338ca}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#f4f4f6;color:#111}
.wrap{display:flex;gap:24px;flex-wrap:wrap;padding:22px;max-width:1100px;margin:0 auto}
.phone{width:380px;background:var(--bg);border-radius:22px;overflow:hidden;box-shadow:0 8px 30px rgba(0,0,0,.15);border:1px solid #ddd}
.top{background:var(--wa);color:#fff;padding:12px 16px;display:flex;align-items:center;gap:10px}
.av{width:38px;height:38px;border-radius:50%;background:#25d366;display:flex;align-items:center;justify-content:center;font-weight:700;color:#053d34}
.top .n{font-weight:600}.top .s{font-size:12px;opacity:.85}
.body{padding:16px;min-height:440px}
.flowcard{background:var(--card);border-radius:12px;box-shadow:0 1px 2px rgba(0,0,0,.1);overflow:hidden}
.fhead{background:var(--wa2);color:#fff;padding:12px 14px;font-weight:600}
.fhead .sub{font-weight:400;font-size:12px;opacity:.9}
.fbody{padding:14px}
.fld{margin-bottom:12px}.fld label{display:block;font-size:12.5px;color:var(--mut);margin-bottom:4px;font-weight:600}
.fld input,.fld select{width:100%;padding:9px 10px;border:1px solid var(--line);border-radius:8px;font-size:14px;background:#fff}
.fld.err input,.fld.err select{border-color:var(--no);background:#fef2f2}
.fld .msg{font-size:11.5px;margin-top:3px}.fld.err .msg{color:var(--no)}.fld.ok .msg{color:var(--ok)}
.submit{width:100%;padding:11px;border:0;border-radius:8px;background:var(--wa2);color:#fff;font-size:15px;font-weight:600;cursor:pointer;margin-top:4px}
.submit:disabled{opacity:.6}
.result{margin-top:10px;font-size:12.5px}
.pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;margin:2px 3px 2px 0}
.pill.ok{background:#ecfdf5;color:var(--ok)}.pill.no{background:#fef2f2;color:var(--no)}
.bub{max-width:85%;padding:8px 11px;border-radius:10px;margin:6px 0;font-size:13.5px;white-space:pre-wrap;clear:both}
.bub.bot{background:#fff;float:left}.bub.me{background:#dcf8c6;float:right}
.chatrow{display:flex;gap:6px;margin-top:8px;clear:both}
.chatrow input{flex:1;padding:9px;border:1px solid var(--line);border-radius:8px;font-size:14px}
.side{flex:1;min-width:320px}
.side h1{font-size:20px;margin:0 0 6px}.side h2{font-size:14px;margin:18px 0 6px;color:var(--accent)}
.side p,.side li{font-size:13.5px;color:#333}.side code{background:#f0f0f4;padding:1px 5px;border-radius:4px;font-size:12.5px}
.cohort{background:#eef2ff;border:1px solid #c7d2fe;border-radius:10px;padding:10px 12px;font-size:13px;margin:10px 0}
.toggle{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;margin-bottom:10px}
.toggle a{padding:7px 12px;font-size:13px;text-decoration:none;color:#333;background:#fff}
.toggle a.on{background:var(--wa2);color:#fff;font-weight:600}
.trace{font-size:11.5px;color:var(--mut);margin-top:8px;white-space:pre-wrap}
</style></head><body>
<div class=wrap>
  <div class=phone>
    <div class=top><div class=av>BF</div><div><div class=n>BrickFin</div><div class=s id=topsub>Journey A · Unified Flow</div></div></div>
    <div class=body>
      <div class=flowcard id=card>
        <div class=fhead>Complete your loan details<div class=sub id=stagelabel>Loading…</div></div>
        <div class=fbody>
          <form id=flowform><div id=fields></div>
          <button class=submit type=submit id=sb>Submit all details</button></form>
          <div id=result></div>
        </div>
      </div>
      <div id=chat></div>
      <div class=trace id=trace></div>
    </div>
  </div>
  <div class=side>
    <div class=toggle><a class=on href="/">Journey A · Unified Flow</a><a href="/chat">Journey B · Step-by-step</a></div>
    <h1>Unified Flow — hybrid (form + agent)</h1>
    <p>A stage's fields are shown at once as a <b>structured form card</b> (a WhatsApp Flow). On submit the app gets <b>one structured payload</b>. Then the <b>agent takes over in chat</b>: it confirms, <b>always asks if you have questions</b>, answers them, and sends the next form when you say continue.</p>
    <div class=cohort id=cohort>Cohort: …</div>
    <h2>Where the intelligence sits</h2>
    <ul>
      <li><b>Flow</b> = deterministic bulk data entry (validated by rules).</li>
      <li><b>Agent</b> = the conversation between forms: transition, Q&A, "ready to continue?" gate.</li>
      <li>Stage <b>order</b> stays deterministic; the agent never invents figures.</li>
    </ul>
    <p style="font-size:12px;color:#888">Browser simulation of a WhatsApp Flow. In production the Flow is published on the WABA and sent via Kaleyra; the submit arrives as <code>nfm_reply</code> and hits the same endpoint.</p>
  </div>
</div>
<script>
const PHONE="+919000000200";
function bucket(p){let h=0;for(const c of p)h=(h*31+c.charCodeAt(0))>>>0;const r=h%100;return r<20?"CONTROL":(r<60?"TREATMENT_A (Unified Flow)":"TREATMENT_B (Step-by-step)");}
document.getElementById('cohort').textContent="Cohort for "+PHONE+": "+bucket(PHONE)+"  (deterministic 20/40/40)";
let STAGE="BASIC_DETAILS", NEXT=null, SPEC=null;
const $=id=>document.getElementById(id);
function tr(t){$('trace').textContent=t;}
function chatBub(txt,who){const d=document.createElement('div');d.className='bub '+who;d.textContent=txt;$('chat').appendChild(d);$('chat').scrollTop=1e9;}

function renderForm(spec){
  SPEC=spec;STAGE=spec.stage;
  $('stagelabel').textContent=spec.label+" · "+spec.fields.length+" fields, one card";
  const box=$('fields');box.innerHTML='';
  for(const f of spec.fields){
    const d=document.createElement('div');d.className='fld';d.id='fld-'+f.key;
    let input;
    if(f.type==='ENUM'&&f.options){input=`<select name="${f.key}"><option value="">Select…</option>`+f.options.map(o=>`<option>${o}</option>`).join('')+`</select>`;}
    else if(f.type==='DATE'){input=`<input name="${f.key}" placeholder="DD/MM/YYYY">`;}
    else if(f.type==='NUMBER'){input=`<input name="${f.key}" inputmode="numeric" placeholder="e.g. 75000">`;}
    else{input=`<input name="${f.key}" placeholder="${f.hint||''}">`;}
    d.innerHTML=`<label>${(f.label||f.key).replace(/\\n/g,' ')}</label>${input}<div class=msg></div>`;
    box.appendChild(d);
  }
  $('result').innerHTML='';$('sb').disabled=false;$('card').style.display='';
}

async function loadForm(stage){const r=await fetch('/api/flow/form?stage='+stage);renderForm(await r.json());}

$('flowform').addEventListener('submit',async(e)=>{
  e.preventDefault();$('sb').disabled=true;
  const vals={};new FormData(e.target).forEach((v,k)=>{if(v)vals[k]=v;});
  const t0=performance.now();
  const r=await fetch('/api/flow/submit',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({phone:PHONE,stage:STAGE,values:vals})});
  const j=await r.json();const ms=Math.round(performance.now()-t0);
  SPEC.fields.forEach(f=>{const el=$('fld-'+f.key);if(!el)return;const m=el.querySelector('.msg');el.classList.remove('err','ok');m.textContent='';
    if(j.errors&&j.errors[f.key]){el.classList.add('err');m.textContent='✕ '+j.errors[f.key];}
    else if(j.saved&&j.saved[f.key]!==undefined){el.classList.add('ok');m.textContent='✓ '+j.saved[f.key];}});
  tr(`POST /api/flow/submit · one structured payload · ${Object.keys(vals).length} fields · ${ms}ms`);
  if(!j.ok){$('result').innerHTML=`<span class=pill no>${j.message}</span>`;$('sb').disabled=false;return;}
  // success -> hand off to the agent in chat
  $('result').innerHTML=`<span class=pill ok>${STAGE} captured in one submit</span>`+(j.pan_status?`<span class=pill ok>PAN validated server-side</span>`:'');
  NEXT=j.next_stage;$('card').style.display='none';
  chatBub(j.chat_message||'✅ Saved. Any questions before we continue?','bot');
  ensureChatInput();
});

let chatInputAdded=false;
function ensureChatInput(){
  if(chatInputAdded)return;chatInputAdded=true;
  const row=document.createElement('div');row.className='chatrow';
  row.innerHTML=`<input id=chatin placeholder="Ask a question or type 'continue'…"><button class=submit id=chatsend style="width:auto;margin:0;padding:9px 14px">Send</button>`;
  $('chat').appendChild(row);
  $('chatsend').onclick=sendTurn;$('chatin').addEventListener('keydown',e=>{if(e.key==='Enter')sendTurn();});
}
async function sendTurn(){
  const box=$('chatin');const msg=box.value.trim();if(!msg)return;box.value='';chatBub(msg,'me');$('chatsend').disabled=true;
  const t0=performance.now();
  const r=await fetch('/api/flow/turn',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({phone:PHONE,message:msg,next_stage:NEXT})});
  const j=await r.json();tr(`POST /api/flow/turn · ${Math.round(performance.now()-t0)}ms · action=${j.action}`);
  if(j.reply)chatBub(j.reply,'bot');
  $('chatsend').disabled=false;
  if(j.action==='next_form'&&j.form){chatInputAdded=false;$('chat').innerHTML='';renderForm(j.form);}
  else if(j.action==='complete'){const ci=$('chatin');if(ci)ci.closest('.chatrow').remove();chatInputAdded=false;}
}
loadForm('BASIC_DETAILS');
</script></body></html>"""
