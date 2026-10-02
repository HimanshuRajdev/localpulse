"""
Synthetic scan shaped like a real one: the category mix is what Overpass
returned for downtown Madison, WI (1.5 km around the Capitol) on
2026-10-02; positions are scattered around the center.
"""
import random

MIX = {
    ("amenity", "restaurant"): 101, ("amenity", "fast_food"): 38,
    ("amenity", "cafe"): 34, ("amenity", "bar"): 34, ("shop", "convenience"): 14,
    ("shop", "clothes"): 13, ("amenity", "theatre"): 11, ("amenity", "pub"): 9,
    ("shop", "jewelry"): 9, ("shop", "hairdresser"): 9, ("shop", "gift"): 8,
    ("amenity", "ice_cream"): 6, ("shop", "supermarket"): 5, ("shop", "bicycle"): 5,
    ("amenity", "nightclub"): 5, ("amenity", "clinic"): 5,
    ("leisure", "fitness_centre"): 5, ("leisure", "sports_centre"): 5,
    ("amenity", "dentist"): 5, ("shop", "beauty"): 4, ("amenity", "pharmacy"): 4,
    ("shop", "tattoo"): 4, ("shop", "florist"): 4, ("shop", "optician"): 3,
    ("shop", "bakery"): 2, ("amenity", "cinema"): 2, ("amenity", "hospital"): 2,
    ("shop", "books"): 1, ("amenity", "veterinary"): 1, ("amenity", "kindergarten"): 1,
    ("amenity", "childcare"): 1, ("amenity", "fuel"): 1, ("shop", "car"): 1,
    ("shop", "cannabis"): 9, ("shop", "alcohol"): 6,
}
HOURS = ["", "", "Mo-Fr 09:00-17:00", "Mo-Su 11:00-22:00",
         "Mo-Sa 08:00-20:00; Su 10:00-16:00", "Tu-Sa 07:00-17:00; Su 07:00-14:00",
         "Mo-Th 16:00-02:00; Fr,Sa 12:00-02:30"]
CENTER = (43.0731, -89.4012)


def elements(seed: int = 7) -> list:
    rnd = random.Random(seed)
    out, i = [], 0
    for (key, val), n in MIX.items():
        for _ in range(n):
            i += 1
            out.append({"type": "node", "id": i,
                        "lat": CENTER[0] + rnd.gauss(0, 0.004),
                        "lon": CENTER[1] + rnd.gauss(0, 0.005),
                        "tags": {"name": f"{val} {i}", key: val,
                                 "opening_hours": rnd.choice(HOURS)}})
    return out


# Made-up but plausible counts for the surrounding 4.5 km, and residents,
# so the location-quotient path can be tested offline.
REGION = {"radius_km": 4.5, "counts": {
    "pet_services": 40, "childcare": 30, "medical": 160, "gym": 45, "beauty": 140,
    "automotive": 60, "education": 15, "coffee": 120, "bar": 110, "grocery": 90,
    "food": 420, "entertainment": 35, "laundry": 25, "retail": 260}}
RESIDENTS = {"per_km2": 5200, "estimate": 36800, "tracts": 4}
