"""
handler.py

AWS Lambda entry point behind a Lambda Function URL. One function serves
both the web page and the JSON API, so the whole app lives at one link.

  GET  /                 the single-page app
  GET  /api/scan         ?lat=&lng=&radius=&name=  -> scored gaps + businesses
  POST /api/ideas        {"place": str, "gaps": [...]} -> 3 business ideas
  GET  /api/health       liveness check
"""

import base64
from concurrent.futures import ThreadPoolExecutor
import json
import time
import traceback
from pathlib import Path

from localpulse.categories import CATEGORIES, LABELS
from localpulse.ideas import IdeasError, generate_ideas
from localpulse.places import PlacesError, fetch_businesses
from localpulse.census import residents as census_residents
from localpulse.region import region_counts
from localpulse.scoring import score_area

INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
CACHE_TTL_S = 6 * 3600
_scan_cache: dict = {}   # lives as long as the warm Lambda container


def _json(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json", "Cache-Control": "no-store"},
        "body": json.dumps(body),
    }


def _html() -> dict:
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/html; charset=utf-8",
                    "Cache-Control": "public, max-age=300"},
        "body": INDEX_HTML,
    }


def scan(params: dict) -> dict:
    try:
        lat = float(params["lat"])
        lng = float(params["lng"])
        radius = float(params.get("radius", 1.5))
    except (KeyError, TypeError, ValueError):
        return _json(400, {"error": "lat and lng are required numbers."})
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return _json(400, {"error": "lat/lng out of range."})
    radius = min(max(radius, 0.5), 3.0)
    name = (params.get("name") or f"{lat:.4f}, {lng:.4f}")[:120]

    key = (round(lat, 3), round(lng, 3), radius)
    hit = _scan_cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL_S:
        return _json(200, {**hit[1], "place": name, "cached": True})

    t0 = time.perf_counter()
    # the three lookups are independent, so run them at the same time
    pool = ThreadPoolExecutor(max_workers=3)
    f_biz = pool.submit(fetch_businesses, lat, lng, radius)
    f_region = pool.submit(region_counts, lat, lng, radius)
    f_people = pool.submit(census_residents, lat, lng, radius)
    try:
        businesses, source = f_biz.result()
    except PlacesError as e:
        pool.shutdown(wait=False, cancel_futures=True)
        return _json(503, {"error": str(e)})
    region = f_region.result()
    people = f_people.result()
    pool.shutdown(wait=False)
    fetch_ms = round((time.perf_counter() - t0) * 1000)

    if len(businesses) < 5:
        return _json(200, {"place": name, "center": [lat, lng], "radius_km": radius,
                           "source": source, "business_count": len(businesses),
                           "gaps": [], "businesses": [], "model": {},
                           "warning": "Fewer than 5 businesses found. Try a bigger radius "
                                      "or a busier spot."})

    result = score_area(businesses, (lat, lng), radius, region=region, residents=people)
    result["model"]["fetch_ms"] = fetch_ms
    body = {
        "place": name, "center": [lat, lng], "radius_km": radius, "source": source,
        **result,
        "businesses": [{"name": b["name"], "lat": b["lat"], "lng": b["lng"],
                        "category": b["category"]} for b in businesses[:2000]],
        "scanned_at": int(time.time()),
    }
    if len(_scan_cache) > 200:
        _scan_cache.clear()
    _scan_cache[key] = (time.time(), body)
    return _json(200, body)


def _num(v, default=None):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else default


def _clean_gap(g: dict) -> dict | None:
    """Only known, typed fields reach the prompt; anything else is dropped."""
    if not isinstance(g, dict) or g.get("category") not in CATEGORIES:
        return None
    text = lambda k, n=160: str(g.get(k) or "")[:n]  # noqa: E731
    return {
        "label": LABELS[g["category"]], "is_gap": bool(g.get("is_gap")),
        "score": _num(g.get("score"), 0.0), "supply_gap": _num(g.get("supply_gap"), 0.0),
        "count": int(_num(g.get("count"), 0)), "expected": _num(g.get("expected"), 0.0),
        "location_quotient": _num(g.get("location_quotient")),
        "residents_per_business": _num(g.get("residents_per_business")),
        "nearest": [{"name": str(x.get("name", ""))[:60], "km": _num(x.get("km"), 0.0)}
                    for x in (g.get("nearest") or [])[:3] if isinstance(x, dict)],
        "missing_subtype": text("missing_subtype"), "hours_gap": text("hours_gap"),
        "complaint_themes": [str(t)[:40] for t in (g.get("complaint_themes") or [])][:6],
    }


def ideas(body: dict) -> dict:
    gaps = [c for c in (_clean_gap(g) for g in (body.get("gaps") or [])[:14]) if c]
    if not gaps:
        return _json(400, {"error": "No gaps to work from. Run a scan first."})
    ctx = body.get("context") if isinstance(body.get("context"), dict) else {}
    context = {
        "place": str(body.get("place") or "this area")[:120],
        "radius_km": _num(ctx.get("radius_km"), 1.5),
        "density_tier": str(ctx.get("density_tier") or "")[:20],
        "benchmark": str(ctx.get("benchmark") or "")[:60],
        "business_count": int(_num(ctx.get("business_count"), 0)),
        "residents_per_km2": _num(ctx.get("residents_per_km2")),
        "residents_estimate": _num(ctx.get("residents_estimate")),
    }
    try:
        return _json(200, generate_ideas(gaps, context))
    except IdeasError as e:
        return _json(502, {"error": str(e)})


def lambda_handler(event, context=None):
    if event.get("warmup"):  # scheduled ping that keeps one container warm
        return {"warm": True}
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    path = event.get("rawPath", "/")
    try:
        if path == "/" and method == "GET":
            return _html()
        if path == "/api/health":
            return _json(200, {"ok": True})
        if path == "/api/scan" and method == "GET":
            return scan(event.get("queryStringParameters") or {})
        if path == "/api/ideas" and method == "POST":
            raw = event.get("body") or "{}"
            if event.get("isBase64Encoded"):
                raw = base64.b64decode(raw).decode()
            return ideas(json.loads(raw))
        return _json(404, {"error": "Not found"})
    except json.JSONDecodeError:
        return _json(400, {"error": "Invalid JSON body."})
    except Exception as e:  # never leak a stack trace to the page
        traceback.print_exc()
        print(f"Unhandled error on {method} {path}: {e!r}")
        return _json(500, {"error": "Something went wrong on our side. Try again."})
