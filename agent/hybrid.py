"""Hybrid orchestrator — the conversational glue AROUND the unified Flow forms.

Division of labor (see the Architecture POV §12):
  * Flow form = deterministic bulk data entry (validated by rules).
  * Agent (here) = the connective conversation: post-submit transition, the
    "any questions?" gate (ALWAYS asked), Q&A between forms, and narrating the
    deterministic next step. It NEVER decides validity, stage order, or figures.

Stateless helpers driven by `call_llm` (the app's FMAPI caller) + `answer_question`
(the existing KB two-gate + general-knowledge fallback). Per-session state is a
tiny dict kept in the app.
"""

from __future__ import annotations

_STAGE_LABEL = {
    "BASIC_DETAILS": "basic details",
    "ELIGIBILITY_DETAILS": "employment & eligibility details",
    "LOAN_SELECTION": "loan selection",
    "BANK_DETAILS": "bank details",
}


def compose_transition(call_llm, saved_stage: str, next_stage: str | None) -> str:
    """Agent composes the post-submit transition: acknowledge + ALWAYS ask if the
    customer has questions before moving on. Intelligence = tone/language, not order."""
    done = _STAGE_LABEL.get(saved_stage, saved_stage.lower().replace("_", " "))
    nxt = _STAGE_LABEL.get(next_stage, (next_stage or "").lower().replace("_", " "))
    nxt_clause = (f"the next step is {nxt}" if next_stage and next_stage != "COMPLETED"
                  else "we're almost done")
    msg = call_llm(
        [{"role": "system", "content":
          "You are BrickFin's warm WhatsApp loan assistant. The customer just submitted a form and "
          "it was saved successfully. Write ONE short WhatsApp message (2 lines, India context, at most "
          "one emoji) that: (a) confirms their " + done + " are saved, and (b) ALWAYS asks if they have "
          "any questions before continuing, or if they're ready to proceed to " + nxt_clause + ". "
          "Do not invent any figures."},
         {"role": "user", "content": "Compose the transition message."}], tier="orchestrator")
    return (msg or "").strip() or (
        f"✅ Your {done} are saved. Any questions before we continue, or shall we move to {nxt}?")


def classify_turn(call_llm, message: str) -> str:
    """Classify the between-forms reply: 'proceed' | 'question' | 'stop' | 'other'.
    Agentic (instruct tier), no hardcoded keyword list."""
    out = call_llm(
        [{"role": "system", "content":
          "Classify the customer's reply during a loan chat, right after we asked 'any questions before "
          "we continue?'. Reply with ONE word only:\n"
          "proceed = they want to move on / no questions (yes, continue, next, go ahead, chalo, aage badho).\n"
          "question = they are asking something or raising an objection.\n"
          "stop = they want to stop / pause / not now.\n"
          "other = greeting or unclear.\n"
          "Output only the label."},
         {"role": "user", "content": message}], tier="orchestrator")
    lab = (out or "").strip().lower().split()[0] if out else "other"
    return lab if lab in ("proceed", "question", "stop", "other") else "other"
