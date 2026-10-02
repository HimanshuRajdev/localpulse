"""Small geo helpers, no dependencies."""

import math

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def geohash(lat: float, lng: float, precision: int = 6) -> str:
    """Standard geohash. Precision 6 is a cell of roughly 1.2 km x 0.6 km."""
    lat_rng, lng_rng = [-90.0, 90.0], [-180.0, 180.0]
    out, bits, ch, even = [], 0, 0, True
    while len(out) < precision:
        rng, val = (lng_rng, lng) if even else (lat_rng, lat)
        mid = (rng[0] + rng[1]) / 2
        if val >= mid:
            ch = (ch << 1) | 1
            rng[0] = mid
        else:
            ch <<= 1
            rng[1] = mid
        even = not even
        bits += 1
        if bits == 5:
            out.append(_BASE32[ch])
            bits, ch = 0, 0
    return "".join(out)
