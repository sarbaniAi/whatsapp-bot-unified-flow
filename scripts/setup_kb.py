#!/usr/bin/env python
"""Provision the BrickFin knowledge base + Vector Search index on Databricks.

Creates:
  serverless_stable_v41mwb_catalog.whatsapp_agent.kb_docs   (Delta, CDF on)
  Vector Search endpoint  : whatsapp-agent-vs
  Delta Sync index        : serverless_stable_v41mwb_catalog.whatsapp_agent.kb_index

KB content is fictitious BrickFin personal-loan policy — it covers the facts the
golden probe set asks for AND fills the corpus gaps that previously fell back
(pincode reason, interest rate, processing fee, foreclosure, documents, timelines).

Run:  PROFILE=fevm-serverless-stable-v41mwb python scripts/setup_kb.py
"""

import os
import time

from databricks.sdk import WorkspaceClient

CATALOG = "serverless_stable_v41mwb_catalog"
SCHEMA = "whatsapp_agent"
TABLE = f"{CATALOG}.{SCHEMA}.kb_docs"
VS_ENDPOINT = "whatsapp-agent-vs"
INDEX = f"{CATALOG}.{SCHEMA}.kb_index"
EMBED_ENDPOINT = "databricks-gte-large-en"
WAREHOUSE_ID = os.environ.get("WAREHOUSE_ID", "87b872956b927e71")

# --- BrickFin KB (fictitious policy; covers probe facts + fills the gaps) ----
KB = [
    ("company", "BrickFin Financial Services is an RBI-registered NBFC in India offering unsecured personal loans. BrickFin is a genuine, regulated lender and never charges any upfront fee before disbursal."),
    ("rates", "The interest rate on a BrickFin personal loan ranges from 10.99% to 24% per annum on a reducing-balance basis. Your exact rate depends on your credit profile, income and CIBIL score, and is confirmed after bank-statement verification."),
    ("fees", "BrickFin charges a one-time processing fee of 2% of the sanctioned loan amount plus applicable GST. It is deducted from the disbursed amount. There are no hidden charges."),
    ("foreclosure", "BrickFin allows part-prepayment after 3 EMIs with no charge. Full foreclosure is free after 6 EMIs; if you foreclose earlier, a fee of 2% of the outstanding principal applies."),
    ("documents", "To apply for a BrickFin personal loan you need your PAN, an address/identity proof (Aadhaar), and your latest bank statement shared via Account Aggregator. KYC is fully digital — no physical documents or branch visit are required."),
    ("timelines", "BrickFin gives an in-principle eligibility decision instantly during the WhatsApp journey. After your bank statement is verified through Account Aggregator, final approval and disbursal typically happen within 24 to 48 hours."),
    ("pincode", "BrickFin asks for your 6-digit residence pincode to confirm the loan is serviceable in your area and to verify your address as part of KYC. Loans are offered only in serviceable pincodes."),
    ("eligibility", "To be eligible for a BrickFin personal loan you must be between 21 and 65 years of age and have a minimum household income of Rs 3,00,000 per year."),
    ("amount", "BrickFin personal loans go up to Rs 10,00,000, subject to eligibility. You cannot request more than the eligible limit shown to you during the application."),
    ("tenure", "BrickFin personal-loan repayment tenure ranges from 12 to 60 months. Your final EMI, rate and tenure are confirmed after income verification."),
    ("master_data", "For 'current residence type' the options are: Owned (self-owned), Owned by parents, Rented, Company-provided, or PG/Hostel."),
    ("banking", "An Account Aggregator (AA) is a secure, RBI-regulated framework that lets you share your bank statement digitally without ever sharing your net-banking username or password. It is the safest and fastest way to share financial data with BrickFin."),
    ("banking", "Sharing your bank statement is a mandatory step for every BrickFin loan. It is used to verify your income and repayment capacity, and is collected securely through the RBI Account Aggregator framework."),
    ("pan", "PAN (Permanent Account Number) is a 10-character ID issued by the Income Tax Department of India. BrickFin needs it to verify your identity and run the mandatory credit-bureau check, as required by RBI regulations."),
    ("pan", "For your security, only share your PAN through BrickFin's secure application flow — never as a plain message. BrickFin collects PAN over an encrypted channel and validates it against the official PAN record."),
    ("dob", "Your date of birth is used to verify your identity against the PAN database and to confirm you meet the 21-65 age eligibility for a BrickFin personal loan."),
    ("income", "BrickFin needs your net monthly income to assess your repayment capacity and meet RBI eligibility guidelines, so we can offer a loan that fits your financial profile."),
    ("consent", "The NDNC (National Do Not Call) consent lets BrickFin call you about your loan application even if your number is registered on the DND/NCPR registry. It is only for loan-related communication, never marketing."),
    ("consent", "The CKYC consent authorises BrickFin to fetch your KYC details from the CKYC registry; the credit-report consent authorises a detailed credit-bureau (hard) pull to assess eligibility."),
    ("general_finance", "EMI (Equated Monthly Installment) is the fixed amount you pay each month to repay a loan, covering both principal and interest. BrickFin calculates it from your loan amount, interest rate and tenure."),
    ("general_finance", "A CIBIL score (300-900) reflects your credit history and repayment behaviour. A higher score improves your chances of approval and a better rate. BrickFin considers your CIBIL score as part of the eligibility check."),
    ("general_finance", "KYC (Know Your Customer) is the RBI-mandated process of verifying your identity, address and PAN/Aadhaar to prevent fraud. BrickFin completes KYC digitally before approving any loan."),
    ("support", "For help with your BrickFin application you can call BrickFin support at 1800-200-4567 or email care@brickfin.com. A loan officer will guide you through the next steps."),
    ("safety", "BrickFin is an RBI-registered NBFC. BrickFin never asks for an upfront fee before disbursal and never asks for your net-banking password or OTP. Your data is encrypted and used only to process your loan."),
]


def sql(w, stmt):
    r = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=stmt, wait_timeout="50s")
    st = r.status.state.value if r.status and r.status.state else "?"
    if st not in ("SUCCEEDED",):
        raise RuntimeError(f"SQL {st}: {getattr(r.status, 'error', None)}\n{stmt[:200]}")
    return r


def main():
    prof = os.environ.get("PROFILE", "fevm-serverless-stable-v41mwb")
    os.environ["DATABRICKS_CONFIG_PROFILE"] = prof
    w = WorkspaceClient()

    print(f"1) schema + table {TABLE}")
    sql(w, f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    sql(w, f"DROP TABLE IF EXISTS {TABLE}")
    sql(w, f"CREATE TABLE {TABLE} (id STRING, content STRING, category STRING) "
           f"TBLPROPERTIES (delta.enableChangeDataFeed = true)")

    print(f"2) insert {len(KB)} docs")
    vals = []
    for i, (cat, content) in enumerate(KB, 1):
        c = content.replace("'", "''")
        vals.append(f"('doc{i:03d}', '{c}', '{cat}')")
    sql(w, f"INSERT INTO {TABLE} (id, content, category) VALUES " + ",\n".join(vals))
    cnt = sql(w, f"SELECT COUNT(*) FROM {TABLE}")
    print("   rows:", cnt.result.data_array[0][0] if cnt.result else "?")

    print(f"3) Vector Search endpoint {VS_ENDPOINT}")
    from databricks.vector_search.client import VectorSearchClient
    # VectorSearchClient doesn't read the CLI profile — pass workspace URL + a
    # bearer token derived from the same (OAuth) profile the SDK is using.
    host = w.config.host
    try:
        token = w.config.oauth_token().access_token
    except Exception:
        token = os.environ.get("DATABRICKS_TOKEN", "")
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=token, disable_notice=True)
    existing = [e.get("name") for e in (vsc.list_endpoints().get("endpoints") or [])]
    if VS_ENDPOINT not in existing:
        vsc.create_endpoint_and_wait(name=VS_ENDPOINT, endpoint_type="STANDARD", verbose=True)
    print("   endpoint ready")

    print(f"4) Delta Sync index {INDEX}")
    idxs = [i.get("name") for i in (vsc.list_indexes(VS_ENDPOINT).get("vector_indexes") or [])]
    if INDEX in idxs:
        print("   index exists — triggering sync")
        vsc.get_index(VS_ENDPOINT, INDEX).sync()
    else:
        vsc.create_delta_sync_index_and_wait(
            endpoint_name=VS_ENDPOINT, index_name=INDEX, source_table_name=TABLE,
            pipeline_type="TRIGGERED", primary_key="id",
            embedding_source_column="content",
            embedding_model_endpoint_name=EMBED_ENDPOINT, verbose=True)
    print("   index ready")

    print("5) test query")
    idx = vsc.get_index(VS_ENDPOINT, INDEX)
    res = idx.similarity_search(query_text="what is the interest rate?",
                                columns=["content", "category"], num_results=2)
    for row in res.get("result", {}).get("data_array", []):
        print("   →", row[0][:90])
    print("\nDONE.")


if __name__ == "__main__":
    main()
