"""
overpass_client.py

Fetches business data from OpenStreetMap via the Overpass API
for any location on Earth. Completely free, no API key required.

No tiling needed — Overpass returns everything in one query,
unlike Google Places API which caps at 60 results per call.

Pipeline:
    1. Geocode city name → lat/lng (via Nominatim, free)
    2. Single Overpass query → all businesses in radius
    3. Parse + deduplicate by osm_id
    4. Push to RAW.OSM_BUSINESSES in Snowflake

Usage:
    python -m src.ingestion.overpass_client --city "Austin, TX" --no-upload
    python -m src.ingestion.overpass_client --city "Austin, TX" --radius 1.5
    python -m src.ingestion.overpass_client --lat 43.0731 --lng -89.4012
"""

import argparse
import os
import time
import uuid
from collections import Counter

import pandas as pd
import requests
import snowflake.connector
from dotenv import load_dotenv
from snowflake.connector.pandas_tools import write_pandas

load_dotenv()

OVERPASS_URL   = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL  = "https://nominatim.openstreetmap.org/search"
TIMEOUT_S      = 60
RETRY_ATTEMPTS = 3
RETRY_BACKOFF  = 5


def build_query(lat: float, lng: float, radius_km: float) -> str:
    r = int(radius_km * 1000)
    return f"""
[out:json][timeout:{TIMEOUT_S}];
(
  node["amenity"](around:{r},{lat},{lng});
  node["shop"](around:{r},{lat},{lng});
  node["leisure"~"fitness_centre|gym|sports_centre|swimming_pool|yoga"](around:{r},{lat},{lng});
  node["office"](around:{r},{lat},{lng});
  node["tourism"~"hotel|hostel|guest_house"](around:{r},{lat},{lng});
);
out body;
"""


def geocode(city_name: str) -> tuple | None:
    headers = {"User-Agent": "LocalPulse/1.0 (academic project)"}
    params  = {"q": city_name, "format": "json", "limit": 1}
    try:
        r       = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=10)
        results = r.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
        print(f"  City not found: {city_name}")
        return None
    except Exception as e:
        print(f"  Geocoding error: {e}")
        return None


def fetch_overpass(query: str, attempt: int = 1) -> dict | None:
    try:
        response = requests.post(OVERPASS_URL, data={"data": query}, timeout=TIMEOUT_S)
        if response.status_code == 200:
            return response.json()
        wait = RETRY_BACKOFF * (2 ** (attempt - 1))
        if response.status_code in (429, 504):
            print(f"  HTTP {response.status_code} — waiting {wait}s (attempt {attempt}/{RETRY_ATTEMPTS})")
            time.sleep(wait)
        else:
            print(f"  HTTP {response.status_code} — skipping")
            return None
    except requests.exceptions.Timeout:
        print(f"  Timeout (attempt {attempt}/{RETRY_ATTEMPTS})")
        time.sleep(RETRY_BACKOFF)
    except requests.exceptions.ConnectionError:
        print(f"  Connection error (attempt {attempt}/{RETRY_ATTEMPTS})")
        time.sleep(RETRY_BACKOFF)

    if attempt < RETRY_ATTEMPTS:
        return fetch_overpass(query, attempt + 1)
    print("  All retries exhausted")
    return None


def parse_element(el: dict, city: str, scan_id: str) -> dict | None:
    tags = el.get("tags", {})
    name = tags.get("name", "").strip()
    if not name:
        return None
    return {
        "osm_id":        str(el["id"]),
        "name":          name,
        "lat":           el.get("lat"),
        "lng":           el.get("lon"),
        "amenity":       tags.get("amenity", ""),
        "shop":          tags.get("shop", ""),
        "cuisine":       tags.get("cuisine", ""),
        "opening_hours": tags.get("opening_hours", ""),
        "city":          city,
        "scan_id":       scan_id,
    }


def get_category(record: dict) -> str:
    if record["amenity"]:
        return record["amenity"]
    if record["shop"]:
        return f"shop:{record['shop']}"
    return "other"


def scan_city(lat: float, lng: float, city_name: str, radius_km: float = 1.5) -> list:
    scan_id = str(uuid.uuid4())[:8]
    print(f"\nScanning : {city_name}")
    print(f"Center   : {lat:.4f}, {lng:.4f}")
    print(f"Radius   : {radius_km} km")
    print(f"Scan ID  : {scan_id}")

    print("\n[1/3] Querying Overpass API...")
    result = fetch_overpass(build_query(lat, lng, radius_km))
    if result is None:
        print("  Query failed")
        return []

    elements = result.get("elements", [])
    print(f"  Raw elements: {len(elements):,}")

    print("\n[2/3] Parsing and deduplicating...")
    seen, parsed = set(), []
    for el in elements:
        record = parse_element(el, city_name, scan_id)
        if record is None or record["osm_id"] in seen:
            continue
        seen.add(record["osm_id"])
        record["category"] = get_category(record)
        parsed.append(record)
    print(f"  Named + unique: {len(parsed):,}")

    print("\n[3/3] Category breakdown (top 10):")
    for cat, count in Counter(r["category"] for r in parsed).most_common(10):
        print(f"  {cat:<30} {count}")

    return parsed


def load_to_snowflake(businesses: list) -> None:
    if not businesses:
        print("Nothing to upload.")
        return
    print(f"\nUploading {len(businesses):,} rows to Snowflake...")
    conn = snowflake.connector.connect(
        account=os.getenv("SF_ACCOUNT"),
        user=os.getenv("SF_USER"),
        password=os.getenv("SF_PASSWORD"),
        database=os.getenv("SF_DATABASE"),
        warehouse=os.getenv("SF_WAREHOUSE"),
    )
    df = pd.DataFrame(businesses)
    df = df.drop(columns=["category"], errors="ignore")
    df.columns = [c.upper() for c in df.columns]
    # PERMANENT FIX: truncate before insert so table never grows
    # staging refresh then always works against a small fresh dataset
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE RAW.OSM_BUSINESSES")
    write_pandas(conn, df, "OSM_BUSINESSES", schema="RAW",
                 auto_create_table=False, overwrite=False,
                 quote_identifiers=False)
    conn.close()
    print(f"  Done — {len(businesses):,} rows loaded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan a city via Overpass API")
    parser.add_argument("--city",      type=str,   help="City name e.g. 'Austin, TX'")
    parser.add_argument("--lat",       type=float)
    parser.add_argument("--lng",       type=float)
    parser.add_argument("--radius",    type=float, default=1.5)
    parser.add_argument("--name",      type=str,   help="Human-readable city name (used as label in Snowflake)")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    if args.city:
        print(f"Geocoding '{args.city}'...")
        coords = geocode(args.city)
        if not coords:
            exit(1)
        lat, lng, city_name = coords[0], coords[1], args.city
    elif args.lat and args.lng:
        lat  = args.lat
        lng  = args.lng
        # use --name if provided, otherwise reverse-geocode, otherwise coords
        if args.name:
            city_name = args.name
        else:
            # reverse geocode to get a real name
            try:
                import requests as _req
                r = _req.get(
                    "https://nominatim.openstreetmap.org/reverse",
                    params={"lat": lat, "lon": lng, "format": "json"},
                    headers={"User-Agent": "LocalPulse/1.0"},
                    timeout=8,
                )
                data = r.json()
                addr = data.get("address", {})
                city_name = (
                    addr.get("city") or addr.get("town") or
                    addr.get("village") or addr.get("county") or
                    f"{lat:.4f},{lng:.4f}"
                )
            except Exception:
                city_name = f"{lat:.4f},{lng:.4f}"
    else:
        lat, lng, city_name = 43.0731, -89.4012, "Madison, WI"
        print("No location given — defaulting to Madison, WI")

    businesses = scan_city(lat, lng, city_name, radius_km=args.radius)

    if businesses and not args.no_upload:
        load_to_snowflake(businesses)
    elif args.no_upload:
        print(f"\nDry run — {len(businesses)} businesses found, not uploaded")
        print("\nSample (first 5):")
        for b in businesses[:5]:
            print(f"  {b['name']:<35} {b['category']:<20} {b['lat']:.4f},{b['lng']:.4f}")