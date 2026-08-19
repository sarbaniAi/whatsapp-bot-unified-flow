"""Offline MLflow 3.x GenAI evaluation for the BrickFin WhatsApp agent.

Runs the agent's grounded-Q&A over the golden set, producing one MLflow trace per
row (with a RETRIEVER span), then scores every trace with built-in + custom judges
and logs an MLflow evaluation run.

Run:
  cd agent-dir already handled via sys.path
  DATABRICKS_HOST=... DATABRICKS_TOKEN=... VS_ENDPOINT=one-env-shared-endpoint-15 \
  MCP_SERVER_URL=in-process python eval/run_eval.py [--max N]

Env:
  MLFLOW_EXPERIMENT   experiment path (default /Users/<me>/whatsapp-agent-eval)
  JUDGE_MODEL         judge endpoint (default endpoints:/databricks-gpt-oss-120b)
  EVAL_MAX_ROWS       limit rows (for a quick pipeline check)
"""

import argparse
import os
import sys
import pathlib

import yaml
import mlflow
from mlflow.entities import SpanType

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "eval"))

import app  # builds the agent (LLM + KB + MCP)  # noqa: E402
from scorers import all_scorers, register_custom_judges  # noqa: E402

QA_PHONE = "+919999999999"  # dummy; Q&A path does not touch the LOS


@mlflow.trace(span_type=SpanType.RETRIEVER, name="kb_retrieve")
def kb_retrieve(question: str):
    """Vector Search retrieval, emitted as a RETRIEVER span for retrieval scorers."""
    docs = app.kb_search(question, 3)
    return [{"page_content": d, "metadata": {"doc_id": str(i)}} for i, d in enumerate(docs)]


@mlflow.trace(span_type=SpanType.AGENT, name="wa_agent_qa")
def predict_fn(question: str) -> str:
    """Grounded-answer generation for eval: retrieve -> draft. The MLflow scorers
    (built-in + custom judges) evaluate the draft; we deliberately do NOT run the
    app's internal two-gate judge here (that gates answer-vs-fallback at runtime;
    for eval we want to score the raw generated answer)."""
    passages = [d["page_content"] for d in kb_retrieve(question)]
    sysp = app.CONFIG.get("llm", {}).get("system_prompt", "")
    draft = app.call_llm(
        [{"role": "system", "content": sysp},
         {"role": "system", "content":
          "Answer using ONLY these passages, with the specific facts/numbers that answer it. "
          "2-3 short WhatsApp lines, India context. If the passages don't cover it, say you'll "
          "confirm and follow up (do not invent):\n" + "\n---\n".join(passages)},
         {"role": "user", "content": question}], tier="small")
    return draft or ""


def load_dataset(max_rows=None):
    raw = yaml.safe_load(open(ROOT / "eval" / "golden_dataset.yaml"))["queries"]
    if max_rows:
        raw = raw[:max_rows]
    data = []
    for q in raw:
        data.append({
            "inputs": {"question": q["question"]},
            "expectations": {
                "expected_facts": q.get("expected_facts", []),
                "guidelines": q.get("guidelines", []),
            },
            "tags": {"category": q.get("category", "")},
        })
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=int(os.environ.get("EVAL_MAX_ROWS", 0)) or None)
    args = ap.parse_args()

    mlflow.set_tracking_uri("databricks")
    exp = os.environ.get("MLFLOW_EXPERIMENT", "/Users/sarbani.maiti@databricks.com/whatsapp-agent-eval")
    mlflow.set_experiment(exp)
    print(f"Experiment: {exp}")

    print("Registering custom judges…")
    print("  registered:", register_custom_judges())

    data = load_dataset(args.max)
    print(f"Evaluating {len(data)} golden queries with {len(all_scorers())} scorers…")

    results = mlflow.genai.evaluate(
        data=data,
        predict_fn=predict_fn,
        scorers=all_scorers(),
    )
    print("\n=== Aggregate metrics (built-in scorers) ===")
    for k, v in sorted((results.metrics or {}).items()):
        print(f"  {k}: {v}")

    # Build a tidy per-row table from the assessments (custom judges live here).
    try:
        import json as _json
        import pandas as pd
        tbl = results.tables.get("eval_results")
        df = pd.DataFrame(tbl) if not isinstance(tbl, pd.DataFrame) else tbl

        def _norm(a):
            if isinstance(a, dict):
                name = a.get("assessment_name") or a.get("name")
                fb = a.get("feedback") or {}
                val = fb.get("value") if isinstance(fb, dict) else getattr(fb, "value", None)
            else:
                name = getattr(a, "name", None)
                fb = getattr(a, "feedback", None)
                val = getattr(fb, "value", None) if fb is not None else None
            return name, val

        rows, judge_names = [], set()
        for _, r in df.iterrows():
            req = r.get("request")
            try:
                q = _json.loads(req).get("question") if isinstance(req, str) else (req or {}).get("question")
            except Exception:
                q = str(req)[:80]
            row = {"question": q, "response": str(r.get("response", ""))[:200]}
            for a in (r.get("assessments") or []):
                n, v = _norm(a)
                if n:
                    row[n] = v
                    judge_names.add(n)
            rows.append(row)
        tidy = pd.DataFrame(rows)
        outdir = ROOT / "eval" / "results"; outdir.mkdir(exist_ok=True)
        csv = outdir / f"eval_{results.run_id}.csv"
        tidy.to_csv(csv, index=False)
        print(f"\nPer-row results saved: {csv}  ({len(tidy)} rows)")

        PASS = {"yes", "pass", "true", "1"}
        print("\n=== Per-judge pass-rates (all scorers) ===")
        for n in sorted(judge_names):
            vals = [str(x).lower() for x in tidy[n].dropna().tolist()]
            if vals:
                p = sum(v in PASS for v in vals)
                print(f"  {n}: {p}/{len(vals)} pass ({100*p//len(vals)}%)")
    except Exception as e:
        print(f"(per-row table dump skipped: {e})")

    print(f"\nRun ID: {results.run_id}")
    print(f"Open in MLflow UI → experiment '{exp}', run {results.run_id}")


if __name__ == "__main__":
    main()
