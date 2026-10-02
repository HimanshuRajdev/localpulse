"""
ideas.py

Sends the scored gaps for an area to an OpenAI model and asks for three
business ideas that hold up against common sense, not just the scores.
"""

import json
import os
import urllib.error
import urllib.request

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o"


class IdeasError(RuntimeError):
    pass


def _gap_line(g: dict) -> str:
    line = (f"- {g['label']}: {g['count']} here vs ~{g['expected']:.0f} expected "
            f"(location quotient {g['location_quotient']}, opportunity {round(g['score'] * 100)}/100)")
    if g.get("residents_per_business"):
        line += f", about 1 per {g['residents_per_business']:,.0f} residents"
    if g["nearest"]:
        line += "; nearest: " + ", ".join(f"{x['name']} ({x['km']} km)" for x in g["nearest"])
    if g.get("missing_subtype"):
        line += f"; no {g['missing_subtype']} at all"
    if g.get("hours_gap"):
        line += f"; hours: {g['hours_gap']}"
    if g.get("complaint_themes"):
        line += f"; common complaints about this kind of business nationally: {', '.join(g['complaint_themes'])}"
    return line


def build_prompt(gaps: list[dict], ctx: dict) -> str:
    flagged = [g for g in gaps if g["is_gap"]][:6] or gaps[:4]
    rest = [g for g in gaps if g not in flagged]
    people = (f"about {ctx['residents_estimate']:,.0f} residents in the circle "
              f"({ctx['residents_per_km2']:,.0f} per km2, 2020 Census)"
              if ctx.get("residents_estimate") else "resident count unknown")
    supply = ", ".join(f"{g['label']} {g['count']}" for g in sorted(gaps, key=lambda g: -g["count"]))

    return f"""You are advising someone with limited capital who wants to open a small business near {ctx['place']}.

AREA ({ctx['radius_km']:g} km radius, {ctx['density_tier']} density)
- {ctx['business_count']} businesses mapped in OpenStreetMap; {people}
- Current mix: {supply}
- "Expected" counts below come from this category's share of businesses in {ctx['benchmark']}. A location quotient under 1 means this area has fewer than its surroundings.

CANDIDATE GAPS
{chr(10).join(_gap_line(g) for g in flagged)}

OTHER CATEGORIES (for context, not gaps)
{chr(10).join(_gap_line(g) for g in rest)}

STEP 1. Use what you know about {ctx['place']} (who lives, works and visits there, rents, transit, car ownership, zoning) to judge each candidate gap. A gap is STRUCTURAL if there is an obvious reason it is empty, such as no car ownership for car washes, office districts with few residents for daycares, or rents too high for low-margin uses. Structural gaps are ruled out. Be strict. It is fine to rule out most of them.

STEP 2. For the gaps that survive, propose exactly 3 businesses. Each must
- serve a specific customer you can name (e.g. "residents of the new towers south of Chambers St who own dogs"), not "busy professionals"
- name where in or near the circle it should go, using streets, landmarks or the businesses listed above
- give a realistic price point and a rough monthly revenue needed to cover rent in this area
- say why the existing options listed above don't already cover this customer
- have a first step that costs under $500 and takes under two weeks and tests real demand (pre-sales, a pop-up, a waitlist), never "do a survey" or "research competitors"
If fewer than 3 gaps survive, use a gap combined with an underserved hour, subtype or customer within a category that is otherwise well supplied.

Write plainly, like a local operator talking to a friend. No hype words.

Return JSON only, exactly this shape:
{{
  "ruled_out": [{{"category": "", "reason": "one sentence"}}],
  "ideas": [
    {{
      "title": "",
      "format": "storefront | van | kiosk | pop-up | service | hybrid",
      "gaps_addressed": ["category label"],
      "customer": "",
      "where": "",
      "description": "",
      "price_and_math": "",
      "why_existing_options_fall_short": "",
      "honest_risks": "",
      "first_step": ""
    }}
  ]
}}"""


def generate_ideas(gaps: list[dict], ctx: dict) -> dict:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise IdeasError("Idea generation is not configured on this deployment.")
    payload = {
        "model": os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        "temperature": 0.6,
        "max_tokens": 2200,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a practical small-business advisor. Respond with valid JSON only."},
            {"role": "user", "content": build_prompt(gaps, ctx)},
        ],
    }
    req = urllib.request.Request(
        OPENAI_URL, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            content = json.loads(resp.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise IdeasError("The OpenAI key on this deployment is invalid.")
        if e.code == 429:
            raise IdeasError("The idea generator is rate limited or out of credit. Try again later.")
        raise IdeasError(f"OpenAI returned HTTP {e.code}.")
    except Exception as e:
        raise IdeasError(f"Could not reach OpenAI: {e}")
    try:
        out = json.loads(content)
        ideas = out["ideas"][:3]
    except (json.JSONDecodeError, KeyError, TypeError):
        raise IdeasError("The model returned something that wasn't valid JSON. Try again.")
    ruled_out = [r for r in out.get("ruled_out", []) if isinstance(r, dict)][:10]
    return {"ideas": ideas, "ruled_out": ruled_out,
            "model": payload["model"]}
