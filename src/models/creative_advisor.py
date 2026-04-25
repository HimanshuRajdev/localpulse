"""
creative_advisor.py

Takes all gap scores for a scanned area and sends them to GPT-4o
to generate 3 creative, specific business ideas.

GPT-4o sees the full picture — all gaps, complaint themes, supply/demand
signals — and thinks like an entrepreneur combining insights rather than
responding to one gap at a time.

Usage:
    from src.models.creative_advisor import generate_creative_ideas
    ideas = generate_creative_ideas(gaps_df, city_name, api_key)
"""

import json
import os
import requests
import pandas as pd


OPENAI_URL = "https://api.openai.com/v1/chat/completions"


# ── prompt builder ─────────────────────────────────────────────────────────
def build_prompt(gaps: pd.DataFrame, city: str) -> str:
    """
    Build a rich prompt that gives GPT-4o the full gap analysis context.
    The model reasons across all gaps holistically.
    """

    gap_lines = []
    for _, row in gaps.head(8).iterrows():
        line = (
            f"- {row['category'].replace('_',' ').title()}"
            f" (score {row['opportunity_score']:.2f})"
            f": supply_gap={row['supply_gap']:.2f}"
            f", demand={row['demand_proxy']:.2f}"
            f", complaint signal={row['complaint_signal']:.2f}"
        )

        if row.get("subcategory"):
            line += f" | specific gap: {row['subcategory']}"
        if row.get("nearest_competitor_km"):
            line += f" | nearest competitor: {row['nearest_competitor_km']}km"
        if row.get("missing_price_tier"):
            line += f" | missing: {row['missing_price_tier']}"

        hours = row.get("hours_gap") or ""
        if hours and hours not in ("hours data unavailable", "reasonable coverage"):
            line += f" | hours gap: {hours}"

        complaint = row.get("top_complaint") or ""
        clean = ", ".join([w for w in complaint.split(", ") if len(w) >= 4])
        if clean:
            line += f" | complaint themes: {clean}"

        gap_lines.append(line)

    gaps_text = "\n".join(gap_lines)

    from collections import Counter
    all_complaints = []
    for _, row in gaps.iterrows():
        c = row.get("top_complaint") or ""
        words = [w for w in c.split(", ") if len(w) >= 4]
        all_complaints.extend(words)

    top_themes = ", ".join([w for w, _ in Counter(all_complaints).most_common(8)])

    return f"""You are an entrepreneurial advisor analyzing real-world local service gaps in {city}.

MARKET DATA:

{gaps_text}

Recurring customer complaints in this area:
{top_themes or "not identified"}

YOUR TASK:
Generate exactly 3 highly specific, physically grounded business ideas for {city}.

CORE RULES:

1. Each idea must solve a REAL physical-world service gap (storefront, kiosk, mobile van, pop-up, hybrid logistics service).
   - Apps are allowed ONLY if they coordinate or enable a physical service.
   - Reject ideas that are purely digital marketplaces or already-saturated national services.

2. Only use opportunities where supply_gap >= 0.6.
   - Prioritize higher values first.

3. STRICTLY reject commodity markets that already have dominant national players:
   - grocery delivery, food delivery, ride-hailing, generic tutoring, basic e-commerce resellers.

4. Each idea must explicitly reference at least one data signal:
   - supply_gap score OR
   - competitor distance OR
   - complaint theme OR
   - service coverage gap

5. Each idea must specify:
   - exact physical format (storefront / van / kiosk / pop-up / hybrid)
   - exact location context (district, corridor, or landmark zone)
   - operating hours justified by local behavior patterns
   - realistic price point

6. Time-of-day logic must match US urban/suburban behavior:
   - Do NOT assume demand exists for late-night service unless supported by data
   - Valid only in contexts like hospital zones, campuses, airports, nightlife districts

7. Explain WHY competitors fail despite existing presence:
   - distance friction
   - poor coverage
   - long wait times
   - mismatch in service type

8. Think like a city operator, not a startup ideation tool:
   - focus on missing physical infrastructure or service delivery gaps seen in larger cities but absent here

OUTPUT FORMAT:
Return valid JSON only. No markdown, no explanation.

{{
  "city": "{city}",
  "ideas": [
    {{
      "title": "",
      "format": "",
      "gaps_addressed": [""],
      "description": "",
      "why_now": "",
      "startup_angle": "low | medium | high",
      "first_step": ""
    }}
  ]
}}"""

# ── GPT-4o caller ──────────────────────────────────────────────────────────
def call_gpt4o(prompt: str, api_key: str) -> dict | None:
    """
    Call GPT-4o with the gap analysis prompt.
    Returns parsed JSON dict or None on failure.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       "gpt-4o",
        "temperature": 0.85,   # creative but grounded
        "max_tokens":  1200,
        "messages": [
            {
                "role":    "system",
                "content": "You are a creative business advisor. Always respond with valid JSON only, no markdown, no preamble."
            },
            {
                "role":    "user",
                "content": prompt,
            }
        ],
    }
    try:
        r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        # strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except requests.exceptions.HTTPError as e:
        if r.status_code == 401:
            raise ValueError("Invalid OpenAI API key. Check OPENAI_API_KEY in your .env")
        elif r.status_code == 429:
            raise ValueError("OpenAI rate limit hit. Wait a moment and try again.")
        else:
            raise ValueError(f"OpenAI API error: {e}")
    except json.JSONDecodeError:
        raise ValueError(f"GPT-4o returned invalid JSON. Raw response: {raw[:200]}")
    except Exception as e:
        raise ValueError(f"GPT-4o call failed: {e}")


# ── main entry point ───────────────────────────────────────────────────────
def generate_creative_ideas(
    gaps: pd.DataFrame,
    city: str,
    api_key: str,
) -> list[dict]:
    """
    Generate 3 creative business ideas from the full gap analysis.

    Args:
        gaps:    DataFrame from RESULTS.GAP_SCORES (all rows)
        city:    scanned city name
        api_key: OpenAI API key

    Returns:
        list of 3 idea dicts, each with:
            title, format, gaps_addressed, description,
            why_now, startup_angle, first_step
    """
    if gaps.empty:
        return []

    prompt = build_prompt(gaps, city)
    result = call_gpt4o(prompt, api_key)

    if not result or "ideas" not in result:
        return []

    return result["ideas"][:3]