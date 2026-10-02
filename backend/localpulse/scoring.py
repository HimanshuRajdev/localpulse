"""
scoring.py

Turns a list of OpenStreetMap businesses into ranked market gaps.

Steps:
  1. features   each business gets Yelp-derived category features (median
                rating, review sentiment, complaint share) plus its local
                density (same-category businesses in its geohash6 cell)
  2. clusters   HDBSCAN segments the local market into groups with similar
                quality / sentiment / density profiles
  3. gaps       every category is scored on supply, demand and complaints;
                within a category the least dense segment is the one used
  4. specifics  missing subtypes, nearest competitor, opening-hours holes
"""

import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .categories import CATEGORIES, LABELS, SUBCATEGORIES
from .geo import geohash, haversine_km

STATS = json.loads((Path(__file__).parent / "data/category_stats.json").read_text())
TIERS = STATS.pop("_tiers")

# Complaint rates are national per category (the same everywhere), so they
# are shown as context and passed to the LLM but not used for ranking.
WEIGHTS = {"supply_gap": 0.65, "demand_proxy": 0.35}
MIN_GAP_SCORE = 0.45
MIN_EXPECTED = 5          # missing 2 of an expected 3 shops is weak evidence
DENSE_RESIDENTS_KM2 = 8000
BUSY_AREA_PER_KM2 = 150  # businesses/km2 treated as "maximum foot traffic"

_MAX_REVIEW_LOG = max(s["avg_review_log"] for s in STATS.values())
_NEG = [s["negative_ratio"] for s in STATS.values()]
_NEG_MIN, _NEG_MAX = min(_NEG), max(_NEG)


def density_tier(businesses: list[dict]) -> str:
    """
    Same rule the Yelp benchmark was built with: each business sits in a
    geohash6 cell, and the area's tier comes from how crowded the cell of a
    typical (median) business is.
    """
    cells = Counter(geohash(b["lat"], b["lng"]) for b in businesses)
    per_biz = sorted(cells[geohash(b["lat"], b["lng"])] for b in businesses)
    median = per_biz[len(per_biz) // 2] if per_biz else 0
    tier = "suburban"
    for name, lo in sorted(TIERS["thresholds"].items(), key=lambda x: x[1]):
        if median >= lo:
            tier = name
    return tier


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ── 1. features ─────────────────────────────────────────────────────────────
def build_features(businesses: list[dict]) -> np.ndarray:
    cells = Counter((b["category"], geohash(b["lat"], b["lng"])) for b in businesses)
    rows = []
    for b in businesses:
        s = STATS[b["category"]]
        b["tile_density"] = cells[(b["category"], geohash(b["lat"], b["lng"]))]
        rows.append([
            (s["median_stars"] - 1) / 4,     # rating_norm
            s["avg_sentiment"],
            s["negative_ratio"],
            math.log1p(b["tile_density"]),
        ])
    return np.array(rows, dtype=float)


# ── 2. clustering ───────────────────────────────────────────────────────────
def cluster(X: np.ndarray) -> tuple[np.ndarray, dict]:
    n = len(X)
    if n < 15:
        return np.zeros(n, dtype=int), {"clusters": 1, "noise_ratio": 0.0,
                                        "silhouette": None}
    Xs = StandardScaler().fit_transform(X)
    labels = HDBSCAN(min_cluster_size=max(5, n // 40), min_samples=3,
                     copy=True).fit_predict(Xs)
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    sil = None
    if n_clusters >= 2 and mask.sum() > n_clusters:
        sil = round(float(silhouette_score(Xs[mask], labels[mask])), 3)
    return labels, {"clusters": n_clusters,
                    "noise_ratio": round(float((~mask).mean()), 3),
                    "silhouette": sil}


# ── 4. specificity signals ──────────────────────────────────────────────────
_DAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
_DAY_RE = r"(?:Mo|Tu|We|Th|Fr|Sa|Su)"
_DAYSPEC = re.compile(rf"^\s*({_DAY_RE}(?:-{_DAY_RE})?(?:\s*,\s*{_DAY_RE}(?:-{_DAY_RE})?)*)\b")
_TIME = re.compile(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})")


def _expand_days(spec: str) -> list[str]:
    days = []
    for part in spec.replace(" ", "").split(","):
        if "-" in part:
            a, b = (_DAYS.index(d) for d in part.split("-"))
            days += [_DAYS[i % 7] for i in range(a, a + (b - a) % 7 + 1)]
        else:
            days.append(part)
    return days


def parse_hours(text: str) -> set[tuple[str, int]]:
    """OSM opening_hours -> set of (day, hour) slots that are open."""
    text = (text or "").strip()
    if not text:
        return set()
    if "24/7" in text:
        return {(d, h) for d in _DAYS for h in range(24)}
    slots = set()
    for rule in text.split(";"):
        if "off" in rule or "closed" in rule:
            continue
        m = _DAYSPEC.match(rule)
        days = _expand_days(m.group(1)) if m else _DAYS
        for t in _TIME.finditer(rule):
            start, end = int(t.group(1)), int(t.group(3))
            if end <= start:            # 18:00-02:00 runs past midnight
                end += 24
            for h in range(start, end):
                for d in days:
                    nd = _DAYS[(_DAYS.index(d) + h // 24) % 7]
                    slots.add((nd, h % 24))
    return slots


EARLY = {"coffee", "gym", "grocery", "childcare", "medical"}
EVENING = {"medical", "beauty", "retail", "grocery", "gym", "pet_services",
           "automotive", "laundry", "childcare", "education"}
WEEKEND = EVENING | {"coffee", "food"}


def hours_gap(category: str, group: list[dict]) -> str:
    with_hours = [b for b in group if b.get("hours")]
    if len(with_hours) < 3:
        return ""
    open_slots = set().union(*(parse_hours(b["hours"]) for b in with_hours))
    if not open_slots:
        return ""
    gaps = []
    if category in EARLY and not any((d, h) in open_slots for d in _DAYS[:5] for h in range(6, 8)):
        gaps.append("nothing open before 8am on weekdays")
    if category in EVENING and not any((d, h) in open_slots for d in _DAYS[:5] for h in range(19, 22)):
        gaps.append("nothing open after 7pm on weekdays")
    if category in WEEKEND and not any((d, h) in open_slots for d in ("Sa", "Su") for h in range(9, 18)):
        gaps.append("closed on weekends")
    return "; ".join(gaps)


def missing_subtype(category: str, group: list[dict]) -> str:
    present = Counter(b["raw"] for b in group)
    expected = SUBCATEGORIES.get(category, {})
    missing = [label for raw, label in expected.items() if present[raw] == 0]
    return missing[0] if missing else ""


# ── 3. gap scoring ──────────────────────────────────────────────────────────
def score_area(businesses: list[dict], center: tuple[float, float], radius_km: float,
               region: dict | None = None, residents: dict | None = None) -> dict:
    """
    region     {"radius_km", "counts": {category: n}} for the surrounding area
               (see region.py). Without it, falls back to the Yelp mix for
               areas of the same density.
    residents  {"per_km2", "estimate"} from census.py, or None.
    """
    t0 = time.perf_counter()
    lat0, lng0 = center
    n_total = len(businesses)
    area_km2 = math.pi * radius_km ** 2
    tier = density_tier(businesses)

    if region:
        region_total = sum(region["counts"].values())
        mix = {c: region["counts"][c] / region_total for c in CATEGORIES}
        benchmark = f"the surrounding {region['radius_km']:g} km"
    else:
        mix = TIERS["mix"][tier]
        benchmark = f"{tier} areas in the Yelp dataset"

    # how many people the area serves: residents if known, business density otherwise
    if residents:
        people = clamp(residents["per_km2"] / DENSE_RESIDENTS_KM2)
    else:
        people = clamp(n_total / area_km2 / BUSY_AREA_PER_KM2)

    labels, model = cluster(build_features(businesses)) if businesses else (np.array([]), {})
    for b, lab in zip(businesses, labels):
        b["cluster"] = int(lab)
        b["dist_km"] = round(haversine_km(lat0, lng0, b["lat"], b["lng"]), 2)

    densities = [b["tile_density"] for b in businesses] or [1]
    p75 = max(float(np.percentile(densities, 75)), 1.0)

    by_cat = defaultdict(list)
    for b in businesses:
        by_cat[b["category"]].append(b)

    rows = []
    for cat in CATEGORIES:
        s = STATS[cat]
        group = sorted(by_cat.get(cat, []), key=lambda b: b["dist_km"])
        n = len(group)
        expected = n_total * mix[cat]
        if not group and expected < 1.5:
            continue  # rare everywhere nearby, so its absence means nothing

        # location quotient: local share of this category vs share around it
        lq = (n / n_total) / mix[cat] if mix[cat] > 0 and n_total else None
        short = clamp(1 - n / expected) if expected > 0 else 0.0
        short *= min(1.0, expected / MIN_EXPECTED)
        if n >= 3:   # the thinnest HDBSCAN segment sharpens a real shortfall
            segments = defaultdict(list)
            for b in group:
                segments[b["cluster"]].append(b)
            seg = min(segments.values(), key=lambda g: sum(b["tile_density"] for b in g) / len(g))
            dens_gap = clamp(1 - (sum(b["tile_density"] for b in seg) / len(seg)) / p75)
        else:
            dens_gap = 1.0
        supply = short * (0.7 + 0.3 * dens_gap)

        demand = 0.5 * (s["avg_review_log"] / _MAX_REVIEW_LOG) + 0.5 * people
        score = WEIGHTS["supply_gap"] * supply + WEIGHTS["demand_proxy"] * demand
        complaint = (s["negative_ratio"] - _NEG_MIN) / (_NEG_MAX - _NEG_MIN)

        per_resident = (round(residents["estimate"] / n, -2) if residents and n else None)
        sub = missing_subtype(cat, group)
        hrs = hours_gap(cat, group)
        rows.append({
            "category": cat,
            "label": LABELS[cat],
            "score": round(score, 3),
            "is_gap": score >= MIN_GAP_SCORE and supply >= 0.35 and expected >= 3,
            "supply_gap": round(supply, 3),
            "demand_proxy": round(demand, 3),
            "complaint_signal": round(complaint, 3),
            "count": n,
            "expected": round(expected, 1),
            "location_quotient": round(lq, 2) if lq is not None else None,
            "residents_per_business": per_resident,
            "nearest_km": group[0]["dist_km"] if group else None,
            "nearest": [{"name": b["name"], "km": b["dist_km"]} for b in group[:3]],
            "missing_subtype": sub,
            "hours_gap": hrs,
            "complaint_themes": s["complaint_themes"][:4],
            "summary": _summary(n, expected, benchmark, group, radius_km, sub, hrs, residents),
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    model["runtime_ms"] = round((time.perf_counter() - t0) * 1000)
    model["density_tier"] = tier
    model["benchmark"] = benchmark
    return {"gaps": rows, "model": model, "business_count": n_total,
            "residents": residents, "region": region}


def _summary(n, expected, benchmark, group, radius_km, sub, hrs, residents) -> str:
    # the category name is shown right above this line, so it isn't repeated
    exp = f"{expected:.0f}" if expected >= 1 else "fewer than 1"
    if n == 0:
        out = f"None within {radius_km:g} km. At the rate of {benchmark} you'd expect about {exp}."
    elif n < expected * 0.8:
        out = f"Only {n} here. At the rate of {benchmark} you'd expect about {exp}."
    else:
        out = f"{n} here, in line with or above {benchmark} (about {exp} expected)."
    if residents and n:
        out += f" Roughly 1 per {round(residents['estimate'] / n, -2):,.0f} residents."
    if group and group[0]["dist_km"] > 0.5:
        out += f" Nearest is {group[0]['name']}, {group[0]['dist_km']} km from the center."
    if sub:
        out += f" No {sub} at all."
    if hrs:
        out += f" {hrs[0].upper()}{hrs[1:]}."
    return out
