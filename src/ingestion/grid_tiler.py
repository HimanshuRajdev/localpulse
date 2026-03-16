"""
grid_tiler.py

Generates a grid of (lat, lng) tile centers that covers a circular
search area. Tile radius is computed dynamically from a probe query
so the tiler self-adapts to city density anywhere on Earth.

Formula derived from:
    density = 4N / pi          (businesses per km²)
    tile_count = density * pi * r²
    solve for r where tile_count <= TARGET_PER_TILE:
        r = sqrt(TARGET_PER_TILE / (4 * N))
        r = sqrt(12.5 / N)
"""

import math


# ── constants ──────────────────────────────────────────────────────────────
KM_PER_DEG_LAT = 111.32          # roughly constant everywhere on Earth
TARGET_PER_TILE = 50              # stay comfortably below the 60-result cap
OVERLAP_FACTOR  = 1.5             # tile spacing = tile_radius * overlap_factor
                                  # <2.0 means tiles overlap (intentional)
MIN_TILE_RADIUS_KM = 0.2          # never go smaller — too many API calls
MAX_TILE_RADIUS_KM = 2.0          # never go larger — too many missed results


# ── core formula ───────────────────────────────────────────────────────────
def compute_tile_radius(probe_count: int) -> float:
    """
    Given the number of businesses returned by a 500m probe query,
    compute the ideal tile radius in km.

    Args:
        probe_count: number of businesses the probe query returned

    Returns:
        tile radius in km, clamped to [MIN, MAX]

    >>> compute_tile_radius(28)   # sparse city like Madison
    0.67
    >>> compute_tile_radius(58)   # dense city like Tokyo downtown
    0.46
    >>> compute_tile_radius(3)    # rural area
    2.0  (clamped to MAX)
    """
    if probe_count <= 0:
        # no businesses found — use maximum tile size
        return MAX_TILE_RADIUS_KM

    raw_radius = math.sqrt(12.5 / probe_count)
    return max(MIN_TILE_RADIUS_KM, min(MAX_TILE_RADIUS_KM, raw_radius))


# ── coordinate helpers ─────────────────────────────────────────────────────
def km_to_deg_lat(km: float) -> float:
    """Convert km to degrees of latitude (constant everywhere)."""
    return km / KM_PER_DEG_LAT


def km_to_deg_lng(km: float, lat: float) -> float:
    """
    Convert km to degrees of longitude.
    This varies with latitude — 1 degree of longitude is shorter
    near the poles than at the equator.
    """
    return km / (KM_PER_DEG_LAT * math.cos(math.radians(lat)))


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Great-circle distance between two points in km.
    Used to check whether a tile center is within the search radius.
    """
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


# ── main tiler ─────────────────────────────────────────────────────────────
def generate_grid(
    center_lat: float,
    center_lng: float,
    search_radius_km: float,
    probe_count: int,
) -> list[dict]:
    """
    Generate a list of tile centers covering the search area.

    Uses a rectangular grid (not hex) for simplicity — the haversine
    boundary check ensures we only query tiles that actually overlap
    the search circle.

    Args:
        center_lat:       latitude of the user's chosen location
        center_lng:       longitude of the user's chosen location
        search_radius_km: how far out to scan (from the UI slider)
        probe_count:      businesses returned by the 500m probe query

    Returns:
        list of dicts: [{"lat": float, "lng": float, "radius_km": float}, ...]

    Think about this:
        Why do we return radius_km in each tile dict instead of computing
        it once and storing it as a constant? What changes if we want
        variable-density tiling in the future?
    """
    tile_radius  = compute_tile_radius(probe_count)
    tile_spacing = tile_radius * OVERLAP_FACTOR

    # convert spacing from km to degrees for grid stepping
    step_lat = km_to_deg_lat(tile_spacing)
    step_lng = km_to_deg_lng(tile_spacing, center_lat)

    # how many steps do we need in each direction?
    steps = math.ceil(search_radius_km / tile_spacing) + 1

    tiles = []
    seen  = set()  # deduplicate tiles that round to the same center

    for row in range(-steps, steps + 1):
        for col in range(-steps, steps + 1):
            tile_lat = center_lat + row * step_lat
            tile_lng = center_lng + col * step_lng

            # only include tiles whose center is within search_radius + one
            # tile_radius (so edge businesses are still captured)
            dist = haversine_km(center_lat, center_lng, tile_lat, tile_lng)
            if dist > search_radius_km + tile_radius:
                continue

            # round to 5 decimal places (~1m precision) to deduplicate
            key = (round(tile_lat, 5), round(tile_lng, 5))
            if key in seen:
                continue
            seen.add(key)

            tiles.append({
                "lat":       tile_lat,
                "lng":       tile_lng,
                "radius_km": tile_radius,
            })

    return tiles


# ── summary helper ─────────────────────────────────────────────────────────
def grid_summary(tiles: list[dict]) -> dict:
    """
    Print a human-readable summary of the grid before committing
    to API calls. Always call this and review before running the scraper.
    """
    if not tiles:
        return {"tile_count": 0}

    r = tiles[0]["radius_km"]
    nearby_cost  = len(tiles) * 0.032   # Nearby Search price per call
    details_est  = len(tiles) * 20      # rough estimate: ~20 detail calls per tile
    details_cost = details_est * 0.017  # Place Details price per call

    return {
        "tile_count":        len(tiles),
        "tile_radius_km":    round(r, 3),
        "nearby_cost_usd":   round(nearby_cost, 2),
        "details_est_cost":  round(details_cost, 2),
        "total_est_cost":    round(nearby_cost + details_cost, 2),
    }


# ── quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Madison, WI — simulate a probe that returned 28 businesses
    tiles = generate_grid(
        center_lat=43.0731,
        center_lng=-89.4012,
        search_radius_km=2.0,
        probe_count=28,
    )

    summary = grid_summary(tiles)
    print(f"tile radius   : {summary['tile_radius_km']} km")
    print(f"tiles         : {summary['tile_count']}")
    print(f"nearby cost   : ${summary['nearby_cost_usd']}")
    print(f"details cost  : ~${summary['details_est_cost']}")
    print(f"total est.    : ~${summary['total_est_cost']}")
    print(f"\nfirst 3 tiles : {tiles[:3]}")

    print("\n--- Tokyo test (probe=58) ---")
    tokyo = generate_grid(35.6762, 139.6503, 2.0, probe_count=58)
    s2 = grid_summary(tokyo)
    print(f"tile radius   : {s2['tile_radius_km']} km")
    print(f"tiles         : {s2['tile_count']}")
    print(f"total est.    : ~${s2['total_est_cost']}")

    print("\n--- Rural Montana test (probe=3) ---")
    rural = generate_grid(46.8787, -110.3626, 2.0, probe_count=3)
    s3 = grid_summary(rural)
    print(f"tile radius   : {s3['tile_radius_km']} km")
    print(f"tiles         : {s3['tile_count']}")
    print(f"total est.    : ~${s3['total_est_cost']}")