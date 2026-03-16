"""
filter_yelp.py

Filters the raw Yelp academic dataset down to what LocalPulse needs.

Reads both JSON files line-by-line (streaming) so we never load
5GB into memory. Outputs two filtered JSONL files ready for
Snowflake upload.

Run once. Takes ~10 minutes on a typical laptop.

Usage:
    python -m src.ingestion.filter_yelp
"""

import json
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────────
RAW_DIR   = Path("data/raw")
OUT_DIR   = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIZ_IN    = RAW_DIR / "yelp_academic_dataset_business.json"
REV_IN    = RAW_DIR / "yelp_academic_dataset_review.json"
BIZ_OUT   = OUT_DIR / "yelp_businesses_filtered.jsonl"
REV_OUT   = OUT_DIR / "yelp_reviews_filtered.jsonl"

MIN_REVIEW_LENGTH   = 40   # ignore very short reviews — useless for NLP
MAX_REVIEWS_PER_BIZ = 20   # cap per business to avoid class imbalance

# categories we care about for gap detection
RELEVANT_CATEGORIES = {
    "restaurants", "food", "coffee", "cafes", "bars", "nightlife",
    "gyms", "fitness", "yoga", "pilates", "crossfit",
    "beauty", "hair salons", "nail salons", "spas", "massage",
    "health", "medical", "dentists", "doctors", "pharmacy",
    "shopping", "retail", "clothing", "grocery",
    "auto", "automotive", "car wash", "gas stations",
    "education", "tutoring", "childcare",
    "home services", "contractors", "cleaning",
    "pet services", "veterinarians", "pet stores",
    "entertainment", "arts", "movies",
}


def is_relevant(categories_str: str) -> bool:
    """Check if a business belongs to a category we care about."""
    if not categories_str:
        return False
    cats = {c.strip().lower() for c in categories_str.split(",")}
    return bool(cats & RELEVANT_CATEGORIES)


# ── Step 1: filter businesses ───────────────────────────────────────────────
print("Filtering businesses...")

kept_biz_ids = set()
biz_count    = 0

with open(BIZ_IN, "r", encoding="utf-8") as fin, \
     open(BIZ_OUT, "w", encoding="utf-8") as fout:

    for line in fin:
        biz = json.loads(line)

        if not biz.get("latitude") or not biz.get("longitude"):
            continue

        if not is_relevant(biz.get("categories", "")):
            continue

        clean = {
            "business_id":  biz["business_id"],
            "name":         biz["name"],
            "city":         biz.get("city"),
            "state":        biz.get("state"),
            "lat":          biz["latitude"],
            "lng":          biz["longitude"],
            "stars":        biz.get("stars"),
            "review_count": biz.get("review_count"),
            "categories":   biz.get("categories"),
            "hours":        json.dumps(biz.get("hours", {})),
        }
        fout.write(json.dumps(clean) + "\n")
        kept_biz_ids.add(biz["business_id"])
        biz_count += 1

print(f"  Kept {biz_count:,} businesses")


# ── Step 2: filter reviews ──────────────────────────────────────────────────
print("Filtering reviews (this takes ~8 minutes)...")

rev_count    = 0
biz_rev_seen = {}   # track review count per business for capping

with open(REV_IN, "r", encoding="utf-8") as fin, \
     open(REV_OUT, "w", encoding="utf-8") as fout:

    for i, line in enumerate(fin):
        rev = json.loads(line)

        bid = rev.get("business_id")
        if bid not in kept_biz_ids:
            continue

        text = rev.get("text", "")
        if len(text) < MIN_REVIEW_LENGTH:
            continue

        biz_rev_seen[bid] = biz_rev_seen.get(bid, 0) + 1
        if biz_rev_seen[bid] > MAX_REVIEWS_PER_BIZ:
            continue

        clean = {
            "review_id":   rev["review_id"],
            "business_id": bid,
            "stars":       rev.get("stars"),
            "text":        text,
            "date":        rev.get("date"),
        }
        fout.write(json.dumps(clean) + "\n")
        rev_count += 1

        if i % 500_000 == 0 and i > 0:
            print(f"  processed {i:,} raw reviews, kept {rev_count:,}...")

print(f"  Kept {rev_count:,} reviews")
print(f"Done. Files written to {OUT_DIR}")