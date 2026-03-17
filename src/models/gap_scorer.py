"""
gap_scorer.py

Computes specific, actionable gap scores per cluster.

Fixes demand_proxy=0 bug (OSM has no reviews — uses tile_density instead).
Adds four specificity signals:
    1. Subcategory drill-down  — which specific type is missing
    2. Nearest competitor      — haversine distance to closest same-category biz
    3. Missing price tier      — which price band is underserved
    4. Hours gap               — which time slots have zero coverage

Output: one row per gap with a human-readable recommendation string.

Usage:
    from src.models.gap_scorer import compute_gaps
    gaps = compute_gaps(df, labels, category_topics, cfg)
"""

import math
import re
from collections import Counter

import numpy as np
import pandas as pd


# ── subcategory mapping ────────────────────────────────────────────────────
# maps unified_category → dict of OSM raw values → human subcategory label
SUBCATEGORY_MAP = {
    "medical": {
        "clinic":    "walk-in clinic",
        "doctors":   "GP / family doctor",
        "dentist":   "dental clinic",
        "pharmacy":  "pharmacy",
        "hospital":  "hospital",
        "therapist": "mental health / therapy",
        "optician":  "optician",
    },
    "gym": {
        "fitness_centre": "fitness centre",
        "gym":            "gym",
        "yoga":           "yoga studio",
        "pilates":        "pilates studio",
        "sports_centre":  "sports centre",
        "swimming_pool":  "swimming pool",
    },
    "beauty": {
        "hairdresser": "hair salon",
        "beauty":      "beauty salon",
        "nail_salon":  "nail salon",
        "massage":     "massage therapy",
        "spa":         "day spa",
        "tattoo":      "tattoo studio",
    },
    "food": {
        "restaurant": "restaurant",
        "cafe":       "cafe",
        "fast_food":  "fast food",
        "food_court": "food court",
        "bakery":     "bakery",
        "ice_cream":  "ice cream / dessert",
    },
    "education": {
        "school":       "school",
        "university":   "university",
        "tutoring":     "tutoring centre",
        "driving_school": "driving school",
        "language_school": "language school",
    },
    "pet_services": {
        "veterinary":  "veterinary clinic",
        "pet":         "pet store",
        "grooming":    "pet grooming",
    },
    "automotive": {
        "fuel":       "fuel station",
        "car_wash":   "car wash",
        "car_repair": "auto repair",
        "car_rental": "car rental",
    },
}

PRICE_LABELS = {
    1: "budget ($)",
    2: "moderate ($$)",
    3: "upscale ($$$)",
    4: "luxury ($$$$)",
}


# ── haversine distance ─────────────────────────────────────────────────────
def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Compute great-circle distance between two points in km."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


# ── signal 1: subcategory drill-down ──────────────────────────────────────
def get_missing_subcategory(
    gap_category: str,
    all_businesses: pd.DataFrame,
    gap_center_lat: float,
    gap_center_lng: float,
    radius_km: float = 2.0,
) -> str:
    """
    Within the gap category, find which specific subcategory is
    least represented in the scanned area.

    Strategy:
        - Get all businesses of this category within radius
        - Count by amenity/shop tag (the raw OSM subtype)
        - Map to human-readable subcategory labels
        - Return the subtype(s) with the lowest count
    """
    subcat_map = SUBCATEGORY_MAP.get(gap_category, {})
    if not subcat_map:
        return ""

    # filter to this category within radius
    nearby = all_businesses[
        all_businesses["unified_category"] == gap_category
    ].copy()

    if nearby.empty:
        # nothing present at all — return the most common expected subtype
        return list(subcat_map.values())[0]

    # compute distance from gap center
    nearby["dist_km"] = nearby.apply(
        lambda r: haversine_km(
            gap_center_lat, gap_center_lng, r["lat"], r["lng"]
        ), axis=1
    )
    nearby = nearby[nearby["dist_km"] <= radius_km]

    # count raw OSM amenity/shop values
    present_raw = set(
        nearby["amenity"].str.lower().tolist()
        + nearby["shop"].str.lower().tolist()
    )
    present_raw.discard("")

    # find which expected subtypes are missing
    missing = [
        label for raw, label in subcat_map.items()
        if raw not in present_raw
    ]

    if missing:
        return missing[0]   # most important missing subtype

    # all subtypes present — find the least common one
    counts = {}
    for raw, label in subcat_map.items():
        counts[label] = sum(
            1 for r in present_raw if r == raw
        )
    rarest = min(counts, key=counts.get)
    return f"{rarest} (underserved)"


# ── signal 2: nearest competitor distance ─────────────────────────────────
def get_nearest_competitor(
    gap_category: str,
    gap_center_lat: float,
    gap_center_lng: float,
    all_businesses: pd.DataFrame,
) -> float:
    """
    Find the closest existing business in the same category.
    Returns distance in km. Large distance = strong location gap.
    """
    same_cat = all_businesses[
        all_businesses["unified_category"] == gap_category
    ]

    if same_cat.empty:
        return 99.0   # nothing exists — extreme gap

    distances = same_cat.apply(
        lambda r: haversine_km(
            gap_center_lat, gap_center_lng, r["lat"], r["lng"]
        ), axis=1
    )
    return round(distances.min(), 2)


# ── signal 3: missing price tier ──────────────────────────────────────────
def get_missing_price_tier(
    gap_category: str,
    all_businesses: pd.DataFrame,
    category_medians: pd.DataFrame = None,
) -> str:
    """
    Find which price tier is missing from the scanned area
    for this category.

    Uses 'price_level' column if available (from Yelp data).
    OSM businesses mostly lack price data — falls back to
    comparing category median vs local median.
    """
    cat_biz = all_businesses[
        all_businesses["unified_category"] == gap_category
    ]

    if "price_level" not in cat_biz.columns or cat_biz["price_level"].isna().all():
        return ""

    present_tiers = set(
        cat_biz["price_level"].dropna().astype(int).tolist()
    )
    all_tiers = {1, 2, 3, 4}
    missing_tiers = all_tiers - present_tiers

    if not missing_tiers:
        return ""

    # return the most commonly demanded missing tier
    # (budget missing is almost always more impactful than luxury missing)
    priority = [1, 2, 3, 4]
    for tier in priority:
        if tier in missing_tiers:
            return PRICE_LABELS[tier]
    return ""


# ── signal 4: hours gap ────────────────────────────────────────────────────
def parse_opening_hours(hours_str: str) -> set:
    """
    Parse OSM opening_hours string into a set of (day, hour) tuples
    representing when the business is open.

    OSM opening hours format examples:
        "Mo-Fr 09:00-18:00"
        "Mo-Sa 08:00-22:00; Su 10:00-16:00"
        "24/7"

    Returns set of (weekday_abbr, hour_int) tuples.
    e.g. {("Mo", 9), ("Mo", 10), ..., ("Fr", 17)}
    """
    if not hours_str or hours_str.strip() == "":
        return set()

    open_slots = set()
    DAY_EXPAND = {
        "Mo": ["Mo"], "Tu": ["Tu"], "We": ["We"],
        "Th": ["Th"], "Fr": ["Fr"], "Sa": ["Sa"], "Su": ["Su"],
        "Mo-Fr": ["Mo", "Tu", "We", "Th", "Fr"],
        "Mo-Sa": ["Mo", "Tu", "We", "Th", "Fr", "Sa"],
        "Mo-Su": ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"],
        "Sa-Su": ["Sa", "Su"],
    }

    if "24/7" in hours_str:
        for day in ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]:
            for h in range(24):
                open_slots.add((day, h))
        return open_slots

    # split on semicolons for multiple rules
    for rule in hours_str.split(";"):
        rule = rule.strip()
        # match patterns like "Mo-Fr 09:00-18:00"
        match = re.match(
            r"([A-Za-z\-]+)\s+(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})",
            rule
        )
        if not match:
            continue

        day_part  = match.group(1)
        open_h    = int(match.group(2))
        close_h   = int(match.group(4))

        days = DAY_EXPAND.get(day_part, [day_part])
        for day in days:
            for h in range(open_h, close_h):
                open_slots.add((day, h))

    return open_slots


def get_hours_gap(
    gap_category: str,
    all_businesses: pd.DataFrame,
) -> str:
    """
    Find time slots where no business in this category is open.
    Returns a human-readable string describing the gap.

    Checks three key slots:
        - Evening (18:00-22:00)
        - Late night (22:00-02:00)
        - Weekend (Sa/Su all day)
    """
    cat_biz = all_businesses[
        (all_businesses["unified_category"] == gap_category)
        & (all_businesses["opening_hours"].notna())
        & (all_businesses["opening_hours"] != "")
    ]

    if cat_biz.empty:
        return "hours data unavailable"

    # build union of all open slots across all businesses in category
    all_open = set()
    for _, row in cat_biz.iterrows():
        all_open |= parse_opening_hours(row.get("opening_hours", ""))

    if not all_open:
        return "hours data unavailable"

    gaps = []

    # check evening coverage (18-22)
    evening_days = ["Mo", "Tu", "We", "Th", "Fr"]
    evening_covered = any(
        (day, h) in all_open
        for day in evening_days
        for h in range(18, 22)
    )
    if not evening_covered:
        gaps.append("no evening hours (6-10pm weekdays)")

    # check late night (22-02)
    late_covered = any(
        (day, h) in all_open
        for day in ["Fr", "Sa"]
        for h in range(22, 24)
    )
    if not late_covered:
        gaps.append("no late night coverage (10pm+)")

    # check weekend coverage
    weekend_covered = any(
        (day, h) in all_open
        for day in ["Sa", "Su"]
        for h in range(9, 18)
    )
    if not weekend_covered:
        gaps.append("no weekend hours")

    return " · ".join(gaps) if gaps else "reasonable coverage"


# ── demand proxy fix ───────────────────────────────────────────────────────
def compute_demand_proxy(group: pd.DataFrame) -> float:
    """
    Fix for demand_proxy=0 bug.

    OSM businesses have no reviews so review_log=0.
    Use tile_density as the demand signal instead:
        more businesses of this type = proven demand exists

    Yelp businesses use review_log × rating_norm as before.
    """
    osm_mask  = group["source"] == "osm"
    yelp_mask = group["source"] == "yelp"

    demand = 0.0
    count  = 0

    if osm_mask.any():
        # normalize tile_density to 0-1 using a soft cap of 20
        avg_density = group.loc[osm_mask, "tile_density"].mean()
        demand += min(avg_density / 20.0, 1.0)
        count  += 1

    if yelp_mask.any():
        yelp_demand = (
            group.loc[yelp_mask, "review_log"].mean()
            * group.loc[yelp_mask, "rating_norm"].mean()
        )
        # normalize: review_log max ~7, rating_norm max 1 → raw max ~7
        demand += min(yelp_demand / 7.0, 1.0)
        count  += 1

    return demand / max(count, 1)


# ── recommendation generator ───────────────────────────────────────────────
def build_recommendation(
    category: str,
    subcategory: str,
    nearest_km: float,
    price_gap: str,
    hours_gap: str,
    top_complaint: str,
) -> str:
    """
    Synthesize all signals into one actionable recommendation sentence.
    This is what appears in the dashboard as the opportunity description.
    """
    parts = []

    # what to open
    biz_type = subcategory if subcategory else category.replace("_", " ")
    parts.append(biz_type.capitalize())

    # location angle
    if nearest_km > 2.0:
        parts.append(f"— nearest competitor {nearest_km}km away")
    elif nearest_km > 1.0:
        parts.append(f"— limited competition ({nearest_km}km away)")

    # price angle
    if price_gap:
        parts.append(f"· {price_gap} option missing")

    # hours angle
    if hours_gap and hours_gap not in ("hours data unavailable",
                                       "reasonable coverage"):
        parts.append(f"· {hours_gap}")

    # complaint angle
    if top_complaint and len(top_complaint) > 5:
        # truncate long complaints
        short = top_complaint[:80].rstrip()
        parts.append(f"· customers say: \"{short}\"")

    return " ".join(parts)


# ── main gap scorer ────────────────────────────────────────────────────────
def compute_gaps(
    df: pd.DataFrame,
    labels: np.ndarray,
    category_topics: dict,
    cfg: dict,
) -> pd.DataFrame:
    """
    Main entry point. Computes fully-specified gap scores.

    Args:
        df:               feature DataFrame from Snowflake
        labels:           HDBSCAN cluster labels per row
        category_topics:  BERTopic output {category: top_words}
        cfg:              loaded config.yaml dict

    Returns:
        DataFrame with one row per gap, sorted by opportunity_score desc
    """
    print("\n[6/6] Computing gap scores with specificity signals...")

    df = df.copy()
    df["cluster_id"] = labels

    # remove noise points
    df_clean = df[df["cluster_id"] != -1].copy()

    # supply benchmark — p75 tile density
    p75_density = df_clean["tile_density"].quantile(0.75)
    p75_density = max(p75_density, 1)

    weights = cfg["gap_scoring"]["weights"]
    results = []

    for (cluster_id, category), group in df_clean.groupby(
        ["cluster_id", "unified_category"]
    ):
        if len(group) < 3:
            continue

        # gap center
        center_lat = group["lat"].mean()
        center_lng = group["lng"].mean()

        # ── supply gap ─────────────────────────────────────────────────────
        avg_density = group["tile_density"].mean()
        supply_gap  = max(0.0, 1.0 - (avg_density / p75_density))
        supply_gap  = min(supply_gap, 1.0)

        # ── demand proxy (fixed) ───────────────────────────────────────────
        demand_proxy = compute_demand_proxy(group)

        # ── complaint signal ───────────────────────────────────────────────
        complaint = min(group["negative_ratio"].mean(), 1.0)

        # ── opportunity score ──────────────────────────────────────────────
        score = (
            weights["supply_gap"]        * supply_gap
            + weights["demand_proxy"]    * demand_proxy
            + weights["complaint_signal"] * complaint
        )

        if score < cfg["gap_scoring"]["min_opportunity_score"]:
            continue

        # ── specificity signals ────────────────────────────────────────────
        subcategory  = get_missing_subcategory(
            category, df_clean, center_lat, center_lng
        )
        nearest_km   = get_nearest_competitor(
            category, center_lat, center_lng, df_clean
        )
        price_gap    = get_missing_price_tier(category, df_clean)
        hours_gap    = get_hours_gap(category, df_clean)
        top_complaint = category_topics.get(category, "")

        recommendation = build_recommendation(
            category, subcategory, nearest_km,
            price_gap, hours_gap, top_complaint
        )

        results.append({
            "cluster_id":        int(cluster_id),
            "category":          category,
            "subcategory":       subcategory,
            "opportunity_score": round(score, 4),
            "supply_gap":        round(supply_gap, 4),
            "demand_proxy":      round(demand_proxy, 4),
            "complaint_signal":  round(complaint, 4),
            "nearest_competitor_km": nearest_km,
            "missing_price_tier":    price_gap,
            "hours_gap":             hours_gap,
            "top_complaint":         top_complaint,
            "recommendation":        recommendation,
            "business_count":        len(group),
            "confidence":            group["nlp_confidence"].mode().iloc[0],
            "avg_lat":               round(center_lat, 5),
            "avg_lng":               round(center_lng, 5),
        })

    gaps = (
        pd.DataFrame(results)
        .sort_values("opportunity_score", ascending=False)
        .reset_index(drop=True)
    )

    # pretty print
    print(f"\n  {'Category':<20} {'Score':>6}  "
          f"{'Supply':>7}  {'Demand':>7}  {'Complaint':>9}")
    print("  " + "-" * 60)
    for _, row in gaps.head(7).iterrows():
        print(f"  {row['category']:<20} {row['opportunity_score']:>6.2f}  "
              f"{row['supply_gap']:>7.2f}  "
              f"{row['demand_proxy']:>7.2f}  "
              f"{row['complaint_signal']:>9.2f}")

    print(f"\n  Top recommendation:")
    if len(gaps) > 0:
        top = gaps.iloc[0]
        print(f"  {top['recommendation']}")

    print(f"\n  Total opportunities: {len(gaps)}")
    return gaps