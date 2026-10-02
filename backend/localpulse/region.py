"""
region.py

The benchmark for "how many of these should there be". Instead of a
national average, LocalPulse compares the scanned circle with the wider
area around it (default 5 km), using the same OpenStreetMap data on both
sides. This is a location quotient: a category is a gap when its share of
local businesses is much smaller than its share in the surrounding area.

Same source on both sides means mapping quirks (OSM lists fewer bars than
Yelp, for example) cancel out, and a downtown is judged against its own
city rather than against car-dependent suburbs elsewhere.
"""

import time

from . import overpass
from .categories import CATEGORIES, OSM_AMENITY, OSM_LEISURE, OSM_SHOP

_cache: dict = {}
CACHE_TTL_S = 24 * 3600


def region_radius(radius_km: float) -> float:
    return min(max(4.0, radius_km * 3), 6.0)


def build_count_query(lat: float, lng: float, radius_km: float) -> str:
    around = f"(around:{int(radius_km * 1000)},{lat:.5f},{lng:.5f})"
    parts = ["[out:json][timeout:25];"]
    for cat in CATEGORIES:
        sets = []
        for key, table in (("amenity", OSM_AMENITY), ("shop", OSM_SHOP), ("leisure", OSM_LEISURE)):
            vals = sorted(v for v, c in table.items() if c == cat)
            if vals:
                sets.append(f'nwr["{key}"~"^({"|".join(vals)})$"]["name"]{around};')
        parts.append("(" + "".join(sets) + ");out count;")
    return "".join(parts)


def region_counts(lat: float, lng: float, radius_km: float) -> dict | None:
    """{category: count} for the surrounding area, or None if Overpass fails."""
    r = region_radius(radius_km)
    key = (round(lat, 2), round(lng, 2), r)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL_S:
        return hit[1]
    try:
        elements = overpass.run_query(build_count_query(lat, lng, r), timeout=25)
    except Exception as e:
        print(f"[region] count query failed: {e}", flush=True)
        return None
    counts = [int(el.get("tags", {}).get("total", 0)) for el in elements if el.get("type") == "count"]
    if len(counts) != len(CATEGORIES) or sum(counts) < 30:
        print(f"[region] unusable count result ({len(counts)} counts, total {sum(counts)})", flush=True)
        return None
    out = {"radius_km": r, "counts": dict(zip(CATEGORIES, counts))}
    _cache[key] = (time.time(), out)
    return out
