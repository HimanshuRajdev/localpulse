"""
categories.py

One place that defines the unified business categories and how raw
OpenStreetMap tags and Yelp category strings map onto them.

Used at build time (scripts/build_category_stats.py, Yelp side) and at
request time (OSM side), so both halves of the pipeline always agree.
"""

# Display order is also the tie-break priority when a Yelp business
# lists several categories ("Cafes, Restaurants" -> coffee).
CATEGORIES = [
    "pet_services", "childcare", "medical", "gym", "beauty", "automotive",
    "education", "coffee", "bar", "grocery", "food", "entertainment",
    "laundry", "retail",
]

LABELS = {
    "pet_services": "Pet services",
    "childcare": "Childcare",
    "medical": "Medical",
    "gym": "Fitness",
    "beauty": "Beauty & personal care",
    "automotive": "Automotive",
    "education": "Education",
    "coffee": "Cafes & bakeries",
    "bar": "Bars & nightlife",
    "grocery": "Grocery",
    "food": "Restaurants",
    "entertainment": "Entertainment",
    "laundry": "Laundry",
    "retail": "Retail",
}

# plural nouns for sentences: "No {noun} within 1.5 km"
NOUNS = {
    "pet_services": "vets or pet shops", "childcare": "daycares or preschools",
    "medical": "clinics, dentists or pharmacies", "gym": "gyms or fitness studios",
    "beauty": "salons or barbers", "automotive": "auto shops or gas stations",
    "education": "classes or schools", "coffee": "cafes or bakeries",
    "bar": "bars", "grocery": "grocery or convenience stores",
    "food": "restaurants", "entertainment": "cinemas, theatres or venues",
    "laundry": "laundromats or dry cleaners", "retail": "retail shops",
}

# ── OSM tag value -> unified category ─────────────────────────────────────
OSM_AMENITY = {
    "restaurant": "food", "fast_food": "food", "food_court": "food",
    "cafe": "coffee", "ice_cream": "coffee",
    "bar": "bar", "pub": "bar", "nightclub": "bar", "biergarten": "bar",
    "clinic": "medical", "doctors": "medical", "dentist": "medical",
    "pharmacy": "medical", "hospital": "medical",
    "veterinary": "pet_services",
    "kindergarten": "childcare", "childcare": "childcare",
    "driving_school": "education", "language_school": "education",
    "music_school": "education", "prep_school": "education",
    "fuel": "automotive", "car_wash": "automotive",
    "car_rental": "automotive", "charging_station": "automotive",
    "cinema": "entertainment", "theatre": "entertainment",
    "arts_centre": "entertainment",
}

OSM_SHOP = {
    "bakery": "coffee", "pastry": "coffee", "coffee": "coffee",
    "supermarket": "grocery", "convenience": "grocery",
    "greengrocer": "grocery", "butcher": "grocery", "deli": "grocery",
    "grocery": "grocery", "health_food": "grocery", "seafood": "grocery",
    "cheese": "grocery", "tea": "grocery", "confectionery": "grocery",
    "hairdresser": "beauty", "beauty": "beauty", "nails": "beauty",
    "massage": "beauty", "tattoo": "beauty", "cosmetics": "beauty",
    "barber": "beauty",
    "chemist": "medical", "optician": "medical", "medical_supply": "medical",
    "pet": "pet_services", "pet_grooming": "pet_services",
    "car_repair": "automotive", "car": "automotive", "tyres": "automotive",
    "car_parts": "automotive",
    "laundry": "laundry", "dry_cleaning": "laundry",
    "clothes": "retail", "shoes": "retail", "books": "retail",
    "electronics": "retail", "mobile_phone": "retail", "hardware": "retail",
    "doityourself": "retail", "gift": "retail", "florist": "retail",
    "department_store": "retail", "sports": "retail", "furniture": "retail",
    "jewelry": "retail", "variety_store": "retail", "toys": "retail",
    "bicycle": "retail", "outdoor": "retail", "second_hand": "retail",
    "games": "retail", "music": "retail", "hifi": "retail", "craft": "retail",
    "art": "retail", "stationery": "retail", "video_games": "retail",
}

OSM_LEISURE = {
    "fitness_centre": "gym", "sports_centre": "gym", "swimming_pool": "gym",
    "yoga": "gym", "bowling_alley": "entertainment",
    "amusement_arcade": "entertainment",
}

# raw OSM value -> specific label, used for "which subtype is missing"
SUBCATEGORIES = {
    "medical": {"clinic": "walk-in clinic", "doctors": "family doctor",
                "dentist": "dental clinic", "pharmacy": "pharmacy",
                "optician": "optician"},
    "gym": {"fitness_centre": "fitness centre", "yoga": "yoga studio",
            "sports_centre": "sports centre", "swimming_pool": "swimming pool"},
    "beauty": {"hairdresser": "hair salon", "barber": "barbershop",
               "nails": "nail salon", "beauty": "beauty salon",
               "massage": "massage studio"},
    "food": {"restaurant": "sit-down restaurant", "fast_food": "quick service",
             "food_court": "food hall"},
    "coffee": {"cafe": "cafe", "bakery": "bakery", "ice_cream": "dessert shop"},
    "bar": {"bar": "bar", "pub": "pub", "nightclub": "nightclub"},
    "grocery": {"supermarket": "full-size grocery store",
                "greengrocer": "fresh produce market",
                "convenience": "convenience store"},
    "pet_services": {"veterinary": "veterinary clinic", "pet": "pet store",
                     "pet_grooming": "pet grooming"},
    "childcare": {"childcare": "daycare", "kindergarten": "preschool"},
    "education": {"language_school": "language school",
                  "music_school": "music school",
                  "driving_school": "driving school"},
    "automotive": {"car_repair": "auto repair", "car_wash": "car wash",
                   "fuel": "fuel station", "tyres": "tire shop"},
    "entertainment": {"cinema": "cinema", "theatre": "theatre",
                      "bowling_alley": "bowling alley"},
    "laundry": {"laundry": "laundromat", "dry_cleaning": "dry cleaner"},
    "retail": {"books": "bookstore", "hardware": "hardware store",
               "clothes": "clothing store", "gift": "gift shop"},
}


def osm_category(tags: dict) -> tuple[str | None, str]:
    """Return (unified_category, raw_value) for an OSM element's tags."""
    for key, table in (("amenity", OSM_AMENITY), ("shop", OSM_SHOP),
                       ("leisure", OSM_LEISURE)):
        val = (tags.get(key) or "").strip().lower()
        if val in table:
            return table[val], val
    return None, ""


# ── Yelp category name -> unified category ────────────────────────────────
YELP = {
    "pet_services": ["pet services", "veterinarians", "pet stores",
                     "pet groomers", "pet sitting", "dog walkers", "pets"],
    "childcare": ["child care & day care", "preschools"],
    "medical": ["doctors", "dentists", "general dentistry", "pharmacy",
                "drugstores", "urgent care", "medical centers",
                "optometrists", "family practice", "walk-in clinics"],
    "gym": ["gyms", "fitness & instruction", "yoga", "pilates", "trainers",
            "cycling classes", "boxing", "barre classes"],
    "beauty": ["hair salons", "nail salons", "barbers", "massage",
               "day spas", "tattoo", "skin care", "waxing",
               "eyelash service", "beauty & spas"],
    "automotive": ["auto repair", "car wash", "gas stations", "tires",
                   "oil change stations", "automotive"],
    "education": ["tutoring centers", "driving schools", "language schools",
                  "specialty schools", "test preparation"],
    "coffee": ["coffee & tea", "cafes", "bakeries", "desserts",
               "ice cream & frozen yogurt", "juice bars & smoothies",
               "bubble tea", "donuts"],
    "bar": ["bars", "pubs", "cocktail bars", "wine bars", "sports bars",
            "breweries", "dance clubs", "nightlife"],
    "grocery": ["grocery", "convenience stores", "international grocery",
                "farmers market", "specialty food"],
    "food": ["restaurants", "fast food", "food trucks", "pizza",
             "sandwiches", "burgers"],
    "entertainment": ["cinema", "music venues", "bowling", "arcades",
                      "performing arts", "arts & entertainment"],
    "laundry": ["laundromat", "dry cleaning & laundry", "laundry services",
                "dry cleaning"],
    "retail": ["fashion", "women's clothing", "men's clothing",
               "department stores", "bookstores", "electronics",
               "hardware stores", "sporting goods", "furniture stores",
               "flowers & gifts", "shoe stores", "shopping"],
}

_YELP_LOOKUP = {}
for _cat in CATEGORIES:
    for _name in YELP[_cat]:
        _YELP_LOOKUP.setdefault(_name, _cat)


def yelp_category(categories_str: str | None) -> str | None:
    """Pick one unified category for a Yelp business, by CATEGORIES priority."""
    if not categories_str:
        return None
    found = {_YELP_LOOKUP[c.strip().lower()]
             for c in categories_str.split(",")
             if c.strip().lower() in _YELP_LOOKUP}
    for cat in CATEGORIES:
        if cat in found:
            return cat
    return None
