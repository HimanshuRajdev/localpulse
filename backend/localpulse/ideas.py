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

STEP 1. Judge every candidate gap, one by one. Use what you know about {ctx['place']} (who lives, works and visits there, rents, transit, car ownership, zoning).
- verdict "rule_out" only for a STRUCTURAL reason the counts can't see, such as no car ownership for car washes, an office district with few residents for daycares, or rent too high for a low-margin use.
- Do NOT rule a gap out by saying there are already enough of these businesses. The counts above show there are fewer than expected; that is the data, so don't contradict it.
- Otherwise the verdict is "keep".

STEP 2. Propose exactly 3 businesses. Each idea's category MUST be one you marked "keep", or a well-supplied category from the other list where you target a specific underserved customer, hour or subtype. Never use a category you ruled out. Each idea must
- serve a specific customer you can name (e.g. "residents of the new towers south of Chambers St who own dogs"), not "busy professionals"
- name where in or near the circle it should go, using streets, landmarks or the businesses listed above
- give a price per unit (per class, per visit, per item), how many units a month is realistic, and the monthly rent you'd expect for the space in this area. Don't do the multiplication, the app does it.
- say why the existing options listed above don't already cover this customer
- have a first step that costs under $500, takes under two weeks and tests real demand (pre-sales, a pop-up, a waitlist), never "do a survey" or "research competitors"

Write plainly, like a local operator talking to a friend. No hype words."""


def _schema(categories: list[str]) -> dict:
    s = lambda: {"type": "string"}  # noqa: E731
    cat = {"type": "string", "enum": categories}
    idea = {"type": "object", "additionalProperties": False, "properties": {
        "title": s(), "format": {"type": "string", "enum": ["storefront", "van", "kiosk", "pop-up", "service", "hybrid"]},
        "category": cat, "customer": s(), "where": s(), "description": s(),
        "price_usd": {"type": "number"}, "unit": s(), "units_per_month": {"type": "integer"},
        "monthly_rent_usd": {"type": "number"},
        "why_existing_options_fall_short": s(), "honest_risks": s(), "first_step": s()}}
    idea["required"] = list(idea["properties"])
    verdict = {"type": "object", "additionalProperties": False, "required": ["category", "verdict", "reason"],
               "properties": {"category": cat, "verdict": {"type": "string", "enum": ["keep", "rule_out"]},
                              "reason": s()}}
    return {"name": "localpulse_ideas", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": ["assessments", "ideas"],
        "properties": {"assessments": {"type": "array", "items": verdict},
                       "ideas": {"type": "array", "items": idea}}}}


def _call(messages: list, model: str, schema: dict, key: str) -> dict:
    payload = {"model": model, "temperature": 0.5, "max_tokens": 2600, "messages": messages,
               "response_format": {"type": "json_schema", "json_schema": schema}}
    req = urllib.request.Request(
        OPENAI_URL, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=50) as resp:
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
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        raise IdeasError("The model returned something that wasn't valid JSON. Try again.")


def check_ideas(out: dict) -> tuple[list, list, set]:
    """Split ideas into consistent ones and ones that use a ruled-out category."""
    ruled = {a["category"] for a in out.get("assessments", []) if a.get("verdict") == "rule_out"}
    good = [i for i in out.get("ideas", []) if i.get("category") not in ruled]
    bad = [i for i in out.get("ideas", []) if i.get("category") in ruled]
    return good, bad, ruled


def generate_ideas(gaps: list[dict], ctx: dict) -> dict:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise IdeasError("Idea generation is not configured on this deployment.")
    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    schema = _schema(sorted({g["label"] for g in gaps}))
    messages = [
        {"role": "system", "content": "You are a practical small-business advisor. Be consistent: never propose a business in a category you ruled out."},
        {"role": "user", "content": build_prompt(gaps, ctx)},
    ]
    out = _call(messages, model, schema, key)
    good, bad, ruled = check_ideas(out)

    if bad:  # one correction round, then drop anything still inconsistent
        messages += [
            {"role": "assistant", "content": json.dumps(out)},
            {"role": "user", "content": (
                f"You ruled out {', '.join(sorted(ruled))} but then proposed ideas in "
                f"{', '.join(sorted({i['category'] for i in bad}))}. Keep your assessments, replace those "
                "ideas with ones in categories you kept or in well-supplied categories with a specific "
                "underserved customer, and return the full JSON again.")},
        ]
        out = _call(messages, model, schema, key)
        good, bad, ruled = check_ideas(out)

    for i in good:   # the app does the arithmetic, not the model
        i["monthly_revenue_usd"] = round(max(i.get("price_usd") or 0, 0) * max(i.get("units_per_month") or 0, 0))
        rent = i.get("monthly_rent_usd") or 0
        i["rent_share"] = round(rent / i["monthly_revenue_usd"], 2) if i["monthly_revenue_usd"] and rent else None
    assessments = out.get("assessments", [])
    return {"ideas": good[:3],
            "ruled_out": [{"category": a["category"], "reason": a["reason"]} for a in assessments if a["verdict"] == "rule_out"],
            "kept": [{"category": a["category"], "reason": a["reason"]} for a in assessments if a["verdict"] == "keep"],
            "model": model}
