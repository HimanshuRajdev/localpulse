"""
overpass.py

Pulls named businesses around a point from OpenStreetMap's Overpass API.
Free, no key. Public Overpass servers are often slow or overloaded, so the
query goes to every mirror at once and the first good answer wins.
"""

import json
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .categories import OSM_AMENITY, OSM_LEISURE, osm_category

MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
USER_AGENT = "LocalPulse/2.0 (+https://github.com/HimanshuRajdev/localpulse)"


class OverpassError(RuntimeError):
    pass


def build_query(lat: float, lng: float, radius_km: float) -> str:
    r = int(radius_km * 1000)
    amenity = "|".join(sorted(OSM_AMENITY))
    leisure = "|".join(sorted(OSM_LEISURE))
    around = f"(around:{r},{lat:.6f},{lng:.6f})"
    # nwr = nodes, ways and relations; many shops are mapped as building
    # outlines (ways), so nodes alone miss a big share of businesses
    return (
        "[out:json][timeout:25];("
        f'nwr["amenity"~"^({amenity})$"]["name"]{around};'
        f'nwr["shop"]["name"]{around};'
        f'nwr["leisure"~"^({leisure})$"]["name"]{around};'
        ");out center tags;"
    )


def _query_one(url: str, body: bytes, timeout: float) -> list:
    req = urllib.request.Request(url, data=body, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    if "elements" not in data:
        raise OverpassError(data.get("remark", "no elements in response"))
    if data.get("remark") and "runtime error" in data["remark"]:
        raise OverpassError(data["remark"])
    return data["elements"]


def fetch(lat: float, lng: float, radius_km: float, timeout: float = 28) -> list:
    return run_query(build_query(lat, lng, radius_km), timeout)


def run_query(query: str, timeout: float = 28) -> list:
    """Send one Overpass query to every mirror at once; first good answer wins."""
    body = urllib.parse.urlencode({"data": query}).encode()
    pool = ThreadPoolExecutor(max_workers=len(MIRRORS))
    pending = {pool.submit(_query_one, u, body, timeout): u for u in MIRRORS}
    errors = []
    try:
        while pending:
            done, _ = wait(pending, timeout=timeout + 2, return_when=FIRST_COMPLETED)
            if not done:
                break
            for fut in done:
                url = pending.pop(fut)
                try:
                    return fut.result()
                except Exception as e:
                    errors.append(f"{url.split('/')[2]}: {e}")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    raise OverpassError("OpenStreetMap servers are busy right now. "
                        "Try again in a minute. (" + "; ".join(errors)[:300] + ")")


def parse(elements: list) -> list[dict]:
    """Elements -> deduplicated businesses with a unified category."""
    out, seen = [], set()
    for el in elements:
        tags = el.get("tags", {})
        name = (tags.get("name") or "").strip()
        lat = el.get("lat", el.get("center", {}).get("lat"))
        lng = el.get("lon", el.get("center", {}).get("lon"))
        if not name or lat is None or lng is None:
            continue
        category, raw = osm_category(tags)
        if not category:
            continue
        key = (name.lower(), round(lat, 3), round(lng, 3))
        if key in seen:  # same shop mapped as both a point and a building
            continue
        seen.add(key)
        out.append({
            "id": f"{el.get('type', 'n')[0]}{el['id']}",
            "name": name,
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "category": category,
            "raw": raw,
            "hours": tags.get("opening_hours", ""),
        })
    return out
