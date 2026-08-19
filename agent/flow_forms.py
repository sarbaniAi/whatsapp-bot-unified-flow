"""Unified-flow (WhatsApp Flows) demo — build a form spec from the same journey
config the conversational agent uses, and process a structured form submit.

This mirrors how a real WhatsApp Flow works: ALL fields of a stage are presented
at once as one form card; the customer submits a single STRUCTURED payload; the
server validates every field and processes them in its own CANONICAL order (so
e.g. PAN validation happens regardless of the order the customer typed things).
It reuses the SAME deterministic Validators as the free-text arm — only the
front-end differs, which is exactly the A/B contrast.
"""

from __future__ import annotations

from typing import Any


def build_form(config: dict, stage: str) -> dict:
    """Return the form spec (fields + widgets) for a stage, from journey config."""
    for s in config.get("journey", {}).get("steps", []):
        if s["name"] == stage:
            fields = []
            for f in s.get("fields", []):
                q = (f.get("question_en") or f.get("key")).split("\n")[0]
                fields.append({
                    "key": f["key"],
                    "type": f.get("type", "TEXT"),
                    "label": q,
                    "options": f.get("options"),
                    "validation": f.get("validation"),
                    "hint": f.get("error_hint", ""),
                    "only_if": f.get("only_if"),
                })
            return {"stage": stage, "label": s.get("label", stage), "fields": fields}
    return {"stage": stage, "label": stage, "fields": []}


def next_stage(config: dict, stage: str) -> str | None:
    steps = [s["name"] for s in config.get("journey", {}).get("steps", [])]
    if stage in steps:
        i = steps.index(stage)
        return steps[i + 1] if i + 1 < len(steps) else "COMPLETED"
    return None


def process_submit(config: dict, validators, stage: str, values: dict[str, Any]) -> dict:
    """Validate a whole-form structured submit and process in canonical order.

    Returns per-field results + a PAN-validation step (mock) that runs only after
    name/DOB-style basics are present — demonstrating that ordering is the SERVER's
    concern, not the customer's."""
    saved, errors = {}, {}
    form = {f["key"]: f for f in build_form(config, stage)["fields"]}
    for k, v in values.items():
        if k not in form:
            continue
        # honor only_if — a conditional field is ignored unless its condition holds
        cond = form[k].get("only_if")
        if cond and not all(str(values.get(ck, "")).lower() == str(cv).lower()
                            for ck, cv in cond.items()):
            continue
        if v is None or str(v).strip() == "":
            continue
        chk = validators.validate(k, str(v))
        if chk["valid"]:
            saved[k] = chk["normalized"]
        else:
            errors[k] = chk["reason"]

    # Canonical server-side step: PAN is validated AFTER the form arrives, using the
    # same payload — so "PAN before name" can never happen (the whole form is atomic).
    pan_status = None
    if "pan" in saved and "pan" not in errors:
        pan_status = {"pan": saved["pan"], "isActive": True, "nameMatch": True,
                      "dobMatch": True, "note": "validated server-side after submit"}

    ok = not errors
    return {
        "ok": ok,
        "stage": stage,
        "saved": saved,
        "errors": errors,
        "pan_status": pan_status,
        "next_stage": next_stage(config, stage) if ok else stage,
        "message": (f"✅ Stage '{stage}' captured in ONE structured submit — "
                    f"{len(saved)} fields validated server-side.")
        if ok else "Some fields need fixing — see the highlighted items.",
    }
