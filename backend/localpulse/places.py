"""
places.py

Where business data comes from. Two OpenStreetMap-based sources:

  Geoapify Places API  fast (about 1 s), free tier of 3,000 requests/day,
                       needs GEOAPIFY_API_KEY. Used first when the key is set.
  Overpass API         free and keyless, but public servers can take
                       10-60 s or time out when busy. Used as the fallback.

Both return OSM data, so the same category mapping applies to either.
"""

import json
import os
import urllib.parse
import urllib.request

from . import overpass
from .categories import osm_category

GEOAPIFY_URL = "https://api.geoapify.com/v2/places"
GEOAPIFY_CATEGORIES = ("catering,commercial,healthcare,service,sport,childcare,"
                       "education,pet,entertainment")
PAGE = 500
MAX_PAGES = 3

# Geoapify category -> (osm key, osm value); used only when the raw OSM
# tags are missing from a result. Longest matching prefix wins.
GEOAPIFY_TO_OSM = {
    "catering.restaurant": ("amenity", "restaurant"),
    "catering.fast_food": ("amenity", "fast_food"),
    "catering.food_court": ("amenity", "food_court"),
    "catering.cafe": ("amenity", "cafe"),
    "catering.ice_cream": ("amenity", "ice_cream"),
    "catering.bar": ("amenity", "bar"),
    "catering.pub": ("amenity", "pub"),
    "commercial.supermarket": ("shop", "supermarket"),
    "commercial.convenience": ("shop", "convenience"),
    "commercial.food_and_drink.bakery": ("shop", "bakery"),
    "commercial.food_and_drink.butcher": ("shop", "butcher"),
    "commercial.food_and_drink.fruit_and_vegetable": ("shop", "greengrocer"),
    "commercial.clothing": ("shop", "clothes"),
    "commercial.books": ("shop", "books"),
    "commercial.elektronics": ("shop", "electronics"),
    "commercial.houseware_and_hardware": ("shop", "hardware"),
    "commercial.gift_and_souvenir": ("shop", "gift"),
    "commercial.florist": ("shop", "florist"),
    "commercial.department_store": ("shop", "department_store"),
    "commercial.pet": ("shop", "pet"),
    "commercial.health_and_beauty.pharmacy": ("amenity", "pharmacy"),
    "commercial.health_and_beauty.optician": ("shop", "optician"),
    "commercial.health_and_beauty.cosmetics": ("shop", "cosmetics"),
    "healthcare.pharmacy": ("amenity", "pharmacy"),
    "healthcare.clinic_or_praxis": ("amenity", "clinic"),
    "healthcare.dentist": ("amenity", "dentist"),
    "healthcare.hospital": ("amenity", "hospital"),
    "service.beauty.hairdresser": ("shop", "hairdresser"),
    "service.beauty.massage": ("shop", "massage"),
    "service.beauty": ("shop", "beauty"),
    "service.cleaning.laundry": ("shop", "laundry"),
    "service.cleaning.dry_cleaning": ("shop", "dry_cleaning"),
    "service.vehicle.fuel": ("amenity", "fuel"),
    "service.vehicle.car_wash": ("amenity", "car_wash"),
    "service.vehicle.repair": ("shop", "car_repair"),
    "sport.fitness": ("leisure", "fitness_centre"),
    "sport.sports_centre": ("leisure", "sports_centre"),
    "sport.swimming_pool": ("leisure", "swimming_pool"),
    "childcare": ("amenity", "kindergarten"),
    "education.driving_school": ("amenity", "driving_school"),
    "education.language_school": ("amenity", "language_school"),
    "education.music_school": ("amenity", "music_school"),
    "pet.veterinary": ("amenity", "veterinary"),
    "pet.shop": ("shop", "pet"),
    "entertainment.cinema": ("amenity", "cinema"),
    "entertainment.culture.theatre": ("amenity", "theatre"),
    "entertainment.bowling_alley": ("leisure", "bowling_alley"),
}


class PlacesError(RuntimeError):
    pass


def _tags_from_geoapify(props: dict) -> dict:
    raw = (props.get("datasource") or {}).get("raw") or {}
    tags = {k: raw[k] for k in ("amenity", "shop", "leisure") if raw.get(k)}
    if tags and osm_category(tags)[0]:
        return tags
    best = ""
    for cat in props.get("categories", []):
        for prefix in GEOAPIFY_TO_OSM:
            if (cat == prefix or cat.startswith(prefix + ".")) and len(prefix) > len(best):
                best = prefix
    if best:
        key, val = GEOAPIFY_TO_OSM[best]
        return {key: val}
    return tags


def geoapify_elements(lat: float, lng: float, radius_km: float, key: str) -> list:
    """Geoapify results reshaped to look like Overpass elements."""
    elements = []
    for page in range(MAX_PAGES):
        q = urllib.parse.urlencode({
            "categories": GEOAPIFY_CATEGORIES,
            "filter": f"circle:{lng:.6f},{lat:.6f},{int(radius_km * 1000)}",
            "limit": PAGE, "offset": page * PAGE, "apiKey": key,
        }, safe=",:")   # Geoapify expects literal commas in the category list
        req = urllib.request.Request(f"{GEOAPIFY_URL}?{q}",
                                     headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read())
        features = payload.get("features") or []
        if page == 0:
            sample = features[0].get("properties", {}) if features else payload
            print(f"[geoapify] {len(features)} features on first page; "
                  f"sample: {json.dumps(sample)[:600]}", flush=True)
        for f in features:
            p = f.get("properties", {})
            coords = (f.get("geometry") or {}).get("coordinates") or [None, None]
            raw = (p.get("datasource") or {}).get("raw") or {}
            tags = _tags_from_geoapify(p)
            tags["name"] = p.get("name") or raw.get("name") or ""
            hours = p.get("opening_hours") or raw.get("opening_hours") or ""
            tags["opening_hours"] = hours if isinstance(hours, str) else ""
            elements.append({
                "type": raw.get("osm_type", "g"),
                "id": raw.get("osm_id") or p.get("place_id", "")[:16],
                "lat": p.get("lat", coords[1]), "lon": p.get("lon", coords[0]),
                "tags": tags,
            })
        if len(features) < PAGE:
            break
    return elements


def fetch_businesses(lat: float, lng: float, radius_km: float) -> tuple[list[dict], str]:
    """Returns (businesses, source_name)."""
    key = os.environ.get("GEOAPIFY_API_KEY", "").strip()
    errors = []
    if key:
        try:
            businesses = overpass.parse(geoapify_elements(lat, lng, radius_km, key))
            print(f"[geoapify] {len(businesses)} businesses after category mapping", flush=True)
            if len(businesses) >= 5:
                return businesses, "geoapify"
            errors.append(f"Geoapify: only {len(businesses)} usable businesses")
        except Exception as e:
            errors.append(f"Geoapify: {e}")
        print(f"[places] falling back to Overpass ({errors[-1]})", flush=True)
    try:
        return overpass.parse(overpass.fetch(lat, lng, radius_km)), "overpass"
    except Exception as e:
        errors.append(str(e))
    raise PlacesError(" | ".join(errors))
