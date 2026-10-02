"""
census.py

Residents around a point, from the US Census geocoder (2020 Census
population and land area per census tract). Free and keyless. The tract
at the center plus four points around it are averaged, so one odd tract
(a park, an office block) doesn't decide the number. Returns None outside
the US or if the service is down.
"""

import json
import math
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"


def _tract(lat: float, lng: float) -> dict | None:
    q = urllib.parse.urlencode({
        "x": f"{lng:.6f}", "y": f"{lat:.6f}", "benchmark": "Public_AR_Current",
        "vintage": "Census2020_Current", "layers": "Census Tracts", "format": "json"})
    try:
        with urllib.request.urlopen(f"{URL}?{q}", timeout=8) as r:
            tracts = json.loads(r.read())["result"]["geographies"].get("Census Tracts", [])
    except Exception:
        return None
    if not tracts:
        return None
    t = tracts[0]
    return {"geoid": t["GEOID"], "pop": int(t["POP100"]), "land_m2": int(t["AREALAND"])}


def residents(lat: float, lng: float, radius_km: float) -> dict | None:
    d = radius_km * 0.6
    dlat = d / 111.0
    dlng = d / (111.0 * math.cos(math.radians(lat)))
    pts = [(lat, lng), (lat + dlat, lng), (lat - dlat, lng), (lat, lng + dlng), (lat, lng - dlng)]
    with ThreadPoolExecutor(max_workers=5) as pool:
        found = [t for t in pool.map(lambda p: _tract(*p), pts) if t]
    tracts = {t["geoid"]: t for t in found}.values()       # same tract hit twice counts once
    land = sum(t["land_m2"] for t in tracts) / 1e6
    if not tracts or land <= 0:
        return None
    density = sum(t["pop"] for t in tracts) / land
    return {"per_km2": round(density),
            "estimate": round(density * math.pi * radius_km ** 2, -2),
            "tracts": len(tracts)}
