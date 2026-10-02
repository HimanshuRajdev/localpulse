"""
build_category_stats.py

Builds backend/localpulse/data/category_stats.json from the Yelp Open
Dataset. This replaces the Snowflake tables the first version needed
(CATEGORY_MEDIANS, NLP_SIGNALS, COMPLAINTS) with one small file that
ships inside the Lambda package. Run it once; the output is committed.

Two steps:

  1. aggregate   stream the filtered Yelp files (scripts/filter_yelp.py),
                 compute per-category rating/sentiment stats and sample
                 negative reviews  -> data/processed/category_raw.json
  2. topics      class-based TF-IDF over the sampled complaints, keep the
                 most distinctive complaint terms per category
                 -> backend/localpulse/data/category_stats.json

Usage:
    python scripts/build_category_stats.py aggregate
    python scripts/build_category_stats.py topics

Sentiment here is review-star based: (stars - 3) / 2 in [-1, 1], and a
review counts as a complaint when stars <= 2.
"""

import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from localpulse.categories import CATEGORIES, yelp_category  # noqa: E402
from localpulse.geo import geohash  # noqa: E402

# Yelp businesses per geohash6 cell (~0.7 km2) -> density tier. The category
# mix of a downtown is very different from a suburb (few auto shops, many
# bars), so the "expected" supply benchmark is taken per tier.
TIERS = [("suburban", 0), ("mixed", 10), ("urban", 40)]


def tier_for(cell_count: float) -> str:
    name = TIERS[0][0]
    for t, lo in TIERS:
        if cell_count >= lo:
            name = t
    return name

BIZ_FILE = ROOT / "data/processed/yelp_businesses_filtered.jsonl"
REV_FILE = ROOT / "data/processed/yelp_reviews_filtered.jsonl"
RAW_OUT = ROOT / "data/processed/category_raw.json"
FINAL_OUT = ROOT / "backend/localpulse/data/category_stats.json"

SAMPLE_PER_CATEGORY = 600
MAX_TEXT_CHARS = 700


def aggregate() -> None:
    random.seed(42)
    biz_cat, stars, review_logs = {}, defaultdict(list), defaultdict(list)
    biz_cell, cell_count = {}, Counter()
    with open(BIZ_FILE, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            cat = yelp_category(b.get("categories"))
            if not cat:
                continue
            biz_cat[b["business_id"]] = cat
            cell = geohash(b["lat"], b["lng"], 6)
            biz_cell[b["business_id"]] = cell
            cell_count[cell] += 1
            if b.get("stars") is not None:
                stars[cat].append(float(b["stars"]))
            review_logs[cat].append(math.log1p(b.get("review_count") or 0))
    print(f"businesses mapped: {len(biz_cat):,}")

    mix = {t: Counter() for t, _ in TIERS}
    for bid, cat in biz_cat.items():
        mix[tier_for(cell_count[biz_cell[bid]])][cat] += 1
    tier_mix = {t: {c: round(m[c] / max(sum(m.values()), 1), 5) for c in CATEGORIES}
                for t, m in mix.items()}
    for t, m in mix.items():
        print(f"  tier {t:<9} {sum(m.values()):>7,} businesses")

    n_rev, sent_sum, n_neg = Counter(), defaultdict(float), Counter()
    samples, seen_neg = defaultdict(list), Counter()
    with open(REV_FILE, encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            cat = biz_cat.get(r["business_id"])
            if not cat or r.get("stars") is None:
                continue
            s = float(r["stars"])
            n_rev[cat] += 1
            sent_sum[cat] += (s - 3.0) / 2.0
            if s <= 2:
                n_neg[cat] += 1
                seen_neg[cat] += 1
                text = r["text"][:MAX_TEXT_CHARS].replace("\n", " ")
                # reservoir sample so every complaint has equal odds
                if len(samples[cat]) < SAMPLE_PER_CATEGORY:
                    samples[cat].append(text)
                else:
                    j = random.randrange(seen_neg[cat])
                    if j < SAMPLE_PER_CATEGORY:
                        samples[cat][j] = text
            if i % 250_000 == 0:
                print(f"  {i:,} reviews", flush=True)

    out = {}
    for cat in CATEGORIES:
        out[cat] = {
            "yelp_businesses": len(stars[cat]),
            "median_stars": statistics.median(stars[cat]) if stars[cat] else 3.5,
            "avg_review_log": (sum(review_logs[cat]) / len(review_logs[cat])
                               if review_logs[cat] else 0.0),
            "reviews": n_rev[cat],
            "avg_sentiment": sent_sum[cat] / n_rev[cat] if n_rev[cat] else 0.0,
            "negative_ratio": n_neg[cat] / n_rev[cat] if n_rev[cat] else 0.0,
            "complaint_samples": samples[cat],
        }
        print(f"{cat:<14} biz={out[cat]['yelp_businesses']:>6} "
              f"reviews={n_rev[cat]:>7} neg={out[cat]['negative_ratio']:.2f}")
    out["_tiers"] = {"thresholds": dict(TIERS), "mix": tier_mix}
    RAW_OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {RAW_OUT}")


NOISE = {
    "place", "service", "just", "like", "time", "said", "told", "went",
    "didn", "don", "got", "came", "really", "did", "asked", "know", "going",
    "back", "minutes", "people", "order", "ordered", "way", "want", "make",
    "made", "say", "won", "ll", "ve", "isn", "wasn", "doesn", "star", "stars",
    "review", "reviews", "experience", "worst", "bad", "terrible", "horrible",
    "rude", "good", "great", "location", "business", "times", "today",
    "took", "called", "store", "customer", "money", "owner", "shop", "phone",
    "items", "food", "restaurant", "call", "left", "come", "away", "manager",
    "starbucks", "away", "thing", "right", "need", "needed", "pretty", "better",
}


# generic or brand/place bigrams that say nothing about the category
NOISE_PHRASES = {
    "days later", "hour later", "hours later", "hour half", "half hour",
    "used love", "looking forward", "moved area", "taco bell", "home depot",
    "santa barbara", "living social", "year olds", "second visit",
    "weeks weeks", "credit card", "bought groupon", "purchased groupon",
    "tracking number", "small claims", "real estate", "girl counter",
    "taste mouth", "couple days", "taken care", "super nice", "benefit doubt",
    "monday morning", "friend wanted", "worth price", "sign door",
    "staff friendly", "staff nice", "facebook page", "nearly years",
    "ready days", "feel comfortable", "year child", "weeks later",
    "extremely disappointed", "sent email", "treated criminal", "cars drive",
    "thousands dollars", "high school", "credit cards", "glass wine",
    "looking forward", "pair pants",
}


def topics() -> None:
    """
    Class-based TF-IDF (the representation step BERTopic uses), with each
    unified category as one class: term frequency inside that category's
    complaints, weighted by how rare the term is across all categories.
    The top terms are what customers complain about specifically in that
    category, not complaints in general.
    """
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer

    raw = json.loads(RAW_OUT.read_text(encoding="utf-8"))
    tiers = raw.pop("_tiers")
    cats = [c for c in raw if raw[c]["complaint_samples"]]
    class_docs = [" ".join(raw[c]["complaint_samples"]) for c in cats]
    print(f"c-TF-IDF over {sum(len(raw[c]['complaint_samples']) for c in cats):,} "
          f"complaints in {len(cats)} categories")

    vec = CountVectorizer(stop_words=list(
        set(CountVectorizer(stop_words="english").get_stop_words()) | NOISE),
        ngram_range=(2, 2), min_df=3, token_pattern=r"(?u)\b[a-zA-Z]{4,}\b")
    counts = vec.fit_transform(class_docs).toarray().astype(float)
    tf = counts / counts.sum(axis=1, keepdims=True)          # term share per class
    avg_words = counts.sum() / len(cats)                      # A in BERTopic paper
    idf = np.log(1 + avg_words / counts.sum(axis=0))          # rare across classes
    ctfidf = tf * idf
    vocab = np.array(vec.get_feature_names_out())

    stats = {}
    for i, cat in enumerate(raw):
        themes = []
        if cat in cats:
            row = ctfidf[cats.index(cat)]
            for term in vocab[np.argsort(row)[::-1][:40]]:
                # skip a unigram already covered by a chosen bigram, and vice versa
                if term in NOISE_PHRASES or any(term in t or t in term for t in themes):
                    continue
                themes.append(term)
                if len(themes) == 6:
                    break
        stats[cat] = {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in raw[cat].items() if k != "complaint_samples"}
        stats[cat]["complaint_themes"] = themes
        print(f"{cat:<14} {', '.join(themes)}")

    stats["_tiers"] = tiers
    FINAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_OUT.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"wrote {FINAL_OUT}")


if __name__ == "__main__":
    {"aggregate": aggregate, "topics": topics}[sys.argv[1]]()
