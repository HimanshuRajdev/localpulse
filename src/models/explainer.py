"""
explainer.py

Adds plain-English LLM explanations to gap scores using Snowflake Cortex.
Runs as an optional enrichment step after gap_scorer.py.

Each gap row gets two new fields:
    explanation   — why this is an opportunity (2-3 sentences)
    business_plan — what to actually do about it (actionable stub)

Cost: ~$0.0007 per gap × ~10 gaps = negligible on free trial.
Run once after compute_gaps() — results cached in RESULTS.GAP_SCORES.

Usage:
    from src.models.explainer import enrich_with_llm
    gaps = enrich_with_llm(gaps, conn)
"""

import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()


# ── prompt builders ────────────────────────────────────────────────────────
def build_explanation_prompt(row: pd.Series) -> str:
    """
    Build a prompt that gives Cortex enough context to write
    a meaningful explanation — not just a generic sentence.
    """
    parts = [
        f"You are a local business analyst.",
        f"A market gap analysis found the following opportunity:",
        f"- Category: {row['category']}",
        f"- Specific type needed: {row.get('subcategory', 'unknown')}",
        f"- Supply gap score: {row['supply_gap']:.2f} (1.0 = completely missing)",
        f"- Nearest competitor: {row.get('nearest_competitor_km', 'unknown')} km away",
        f"- Missing price tier: {row.get('missing_price_tier', 'unknown')}",
        f"- Hours gap: {row.get('hours_gap', 'unknown')}",
        f"- Top customer complaint: {row.get('top_complaint', 'none')}",
        f"",
        f"Write 2 sentences explaining WHY this is a strong opportunity.",
        f"Be specific. Mention the distance, hours, and price tier.",
        f"Do not use bullet points. Plain text only.",
    ]
    return "\n".join(parts)


def build_business_plan_prompt(row: pd.Series) -> str:
    """
    Build a prompt for a concrete 3-point action plan.
    Deliberately brief — this is a stub, not a full plan.
    """
    parts = [
        f"You are a startup advisor.",
        f"Someone wants to start a {row.get('subcategory', row['category'])} business.",
        f"Key facts:",
        f"- Location: {row.get('avg_lat', ''):.4f}, {row.get('avg_lng', ''):.4f}",
        f"- Nearest competitor: {row.get('nearest_competitor_km', 'unknown')} km away",
        f"- Missing price tier: {row.get('missing_price_tier', 'not identified')}",
        f"- Hours opportunity: {row.get('hours_gap', 'not identified')}",
        f"- Customer pain point: {row.get('top_complaint', 'not identified')}",
        f"",
        f"Give exactly 3 bullet points:",
        f"1. What format to launch in (kiosk, studio, mobile service, etc.)",
        f"2. What hours and price point to target",
        f"3. One specific competitive advantage to lead with",
        f"",
        f"Keep each bullet under 15 words. Be concrete and actionable.",
    ]
    return "\n".join(parts)


# ── cortex caller ──────────────────────────────────────────────────────────
def call_cortex(prompt: str, conn, model: str = "mistral-7b") -> str:
    """
    Call Snowflake Cortex COMPLETE via SQL.
    Returns the generated text or empty string on failure.
    """
    # escape single quotes in prompt
    safe_prompt = prompt.replace("'", "''")
    sql = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}',
            '{safe_prompt}'
        ) AS response
    """
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchone()
        return result[0].strip() if result and result[0] else ""
    except Exception as e:
        print(f"    Cortex error: {e}")
        return ""


# ── main enrichment function ───────────────────────────────────────────────
def enrich_with_llm(
    gaps: pd.DataFrame,
    conn,
    model: str = "mistral-7b",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Add LLM-generated explanation and business_plan to each gap row.

    Only runs on top_n gaps to control cost and time.
    Gaps below threshold get empty strings.

    Args:
        gaps:   DataFrame from compute_gaps()
        conn:   active Snowflake connection
        model:  Cortex model to use
        top_n:  only enrich top N opportunities

    Returns:
        gaps DataFrame with two new columns added
    """
    print(f"\n[LLM] Enriching top {min(top_n, len(gaps))} gaps with Cortex...")

    gaps = gaps.copy()
    gaps["explanation"]   = ""
    gaps["business_plan"] = ""

    for idx, row in gaps.head(top_n).iterrows():
        print(f"  {row['category']:<25}", end=" ", flush=True)

        # explanation
        exp_prompt   = build_explanation_prompt(row)
        explanation  = call_cortex(exp_prompt, conn, model)
        gaps.at[idx, "explanation"] = explanation

        # business plan
        plan_prompt  = build_business_plan_prompt(row)
        business_plan = call_cortex(plan_prompt, conn, model)
        gaps.at[idx, "business_plan"] = business_plan

        print("done")

    print(f"\n  Sample explanation for top gap:")
    if len(gaps) > 0 and gaps.iloc[0]["explanation"]:
        print(f"  {gaps.iloc[0]['explanation'][:200]}...")

    return gaps


# ── push enriched results ──────────────────────────────────────────────────
def push_enriched_gaps(gaps: pd.DataFrame, conn) -> None:
    """
    Update GAP_SCORES with explanation and business_plan columns.
    Uses a temp table + MERGE to avoid overwriting existing scores.
    """
    from snowflake.connector.pandas_tools import write_pandas

    print("\n  Pushing enriched gaps to Snowflake...")

    # add columns to GAP_SCORES if they don't exist
    cursor = conn.cursor()
    for col, dtype in [("EXPLANATION", "VARCHAR"), ("BUSINESS_PLAN", "VARCHAR")]:
        try:
            cursor.execute(
                f"ALTER TABLE RESULTS.GAP_SCORES ADD COLUMN {col} {dtype}"
            )
        except Exception:
            pass   # column already exists

    # write enriched subset
    enriched = gaps[["cluster_id", "category",
                      "explanation", "business_plan"]].copy()
    enriched.columns = [c.upper() for c in enriched.columns]

    write_pandas(
        conn, enriched, "GAP_SCORES_ENRICHED",
        schema="RESULTS",
        auto_create_table=True,
        overwrite=True,
        quote_identifiers=False,
    )

    # merge back into main table
    cursor.execute("""
        UPDATE RESULTS.GAP_SCORES g
        SET g.EXPLANATION   = e.EXPLANATION,
            g.BUSINESS_PLAN = e.BUSINESS_PLAN
        FROM RESULTS.GAP_SCORES_ENRICHED e
        WHERE g.CLUSTER_ID = e.CLUSTER_ID
          AND g.CATEGORY    = e.CATEGORY
    """)

    # cleanup temp table
    cursor.execute("DROP TABLE IF EXISTS RESULTS.GAP_SCORES_ENRICHED")
    print("  Done.")