"""
cluster.py

Phase 4 — ML pipeline.

Chain:
    Snowflake → StandardScaler → UMAP → HDBSCAN → BERTopic
    → Gap Scorer → LLM Explainer → Snowflake

Usage:
    python -m src.models.cluster --city "Madison, WI"
    python -m src.models.cluster --city "Madison, WI" --evaluate
    python -m src.models.cluster --city "Madison, WI" --ablate
    python -m src.models.cluster --city "Madison, WI" --explain
"""

import argparse
import os
import warnings
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import snowflake.connector
import umap
import yaml
from bertopic import BERTopic
from dotenv import load_dotenv
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from snowflake.connector.pandas_tools import write_pandas

from src.models.gap_scorer import compute_gaps
from src.models.explainer import enrich_with_llm, push_enriched_gaps

warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

CONFIG_PATH = Path("config.yaml")
FEATURE_COLS = ["rating_norm", "review_log", "avg_sentiment", "negative_ratio"]


# ── config ─────────────────────────────────────────────────────────────────
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── snowflake ──────────────────────────────────────────────────────────────
def get_connection():
    return snowflake.connector.connect(
        account=os.getenv("SF_ACCOUNT"),
        user=os.getenv("SF_USER"),
        password=os.getenv("SF_PASSWORD"),
        database=os.getenv("SF_DATABASE"),
        warehouse=os.getenv("SF_WAREHOUSE"),
    )


def refresh_staging(conn) -> None:
    """
    Rebuild STAGING.BUSINESSES and STAGING.TILE_DENSITY
    from the latest RAW data. Must run after every Overpass scan
    so cluster.py sees the new businesses.
    """
    print("[0/6] Refreshing staging tables...")
    cur = conn.cursor()

    cur.execute("""
        CREATE OR REPLACE TABLE STAGING.BUSINESSES AS
        WITH osm_mapped AS (
            SELECT
                o.osm_id                        AS business_id,
                o.name, o.lat, o.lng, o.city,
                o.scan_id, 'osm'                AS source,
                COALESCE(cm_a.unified_category, cm_s.unified_category, 'other') AS unified_category,
                COALESCE(med.median_stars, 3.5) AS stars,
                0                               AS review_count,
                o.opening_hours
            FROM RAW.OSM_BUSINESSES o
            LEFT JOIN STAGING.CATEGORY_MAP cm_a
                ON LOWER(o.amenity) = cm_a.raw_value AND cm_a.source = 'osm'
            LEFT JOIN STAGING.CATEGORY_MAP cm_s
                ON LOWER(o.shop) = cm_s.raw_value AND cm_s.source = 'osm'
            LEFT JOIN STAGING.CATEGORY_MEDIANS med
                ON med.unified_category = COALESCE(cm_a.unified_category, cm_s.unified_category)
        ),
        yelp_mapped AS (
            SELECT
                b.business_id, b.name, b.lat, b.lng, b.city,
                NULL AS scan_id, 'yelp' AS source,
                MIN(cm.unified_category) AS unified_category,
                b.stars, b.review_count, NULL AS opening_hours
            FROM RAW.YELP_BUSINESSES b,
            LATERAL SPLIT_TO_TABLE(b.categories, ',') f
            JOIN STAGING.CATEGORY_MAP cm
                ON LOWER(TRIM(f.value::STRING)) ILIKE '%' || cm.raw_value || '%'
                AND cm.source = 'yelp'
            GROUP BY b.business_id, b.name, b.lat, b.lng, b.city, b.stars, b.review_count
        ),
        combined AS (
            SELECT * FROM osm_mapped
            UNION ALL
            SELECT * FROM yelp_mapped
        )
        SELECT
            business_id, name, lat, lng, city, scan_id, source, unified_category,
            stars, review_count, opening_hours,
            ROUND((stars - 1.0) / 4.0, 4) AS rating_norm,
            ROUND(LN(review_count + 1), 4) AS review_log,
            ST_GEOHASH(TO_GEOGRAPHY(OBJECT_CONSTRUCT('type','Point','coordinates',ARRAY_CONSTRUCT(lng,lat))),5) AS geohash5,
            ST_GEOHASH(TO_GEOGRAPHY(OBJECT_CONSTRUCT('type','Point','coordinates',ARRAY_CONSTRUCT(lng,lat))),6) AS geohash6
        FROM combined
        WHERE unified_category != 'other'
          AND lat IS NOT NULL AND lng IS NOT NULL AND name != ''
    """)
    print("  STAGING.BUSINESSES rebuilt")

    cur.execute("""
        CREATE OR REPLACE TABLE STAGING.TILE_DENSITY AS
        SELECT
            geohash6 AS tile_id, unified_category, source,
            COUNT(*) AS business_count,
            AVG(rating_norm) AS avg_rating_norm,
            SUM(review_log)  AS total_demand_signal
        FROM STAGING.BUSINESSES
        GROUP BY geohash6, unified_category, source
    """)
    print("  STAGING.TILE_DENSITY rebuilt")


def pull_features(city: str, conn) -> pd.DataFrame:
    print(f"\n[1/6] Pulling features for '{city}' from Snowflake...")

    # get the latest scan_id — avoids city name mismatch
    # (overpass stores coords as city name when --lat/--lng used)
    cur = conn.cursor()
    cur.execute("""
        SELECT scan_id FROM RAW.OSM_BUSINESSES
        ORDER BY scanned_at DESC LIMIT 1
    """)
    row = cur.fetchone()
    if not row:
        print("  No OSM scan found. Run the Overpass scan first.")
        return pd.DataFrame()
    scan_id = row[0]
    print(f"  Latest scan_id: {scan_id}")

    cur.execute(f"""
        SELECT
            b.business_id,
            b.name,
            b.lat,
            b.lng,
            b.unified_category,
            b.rating_norm,
            b.review_log,
            b.geohash6,
            b.source,
            b.opening_hours,
            COALESCE(n.avg_sentiment,      0.0)  AS avg_sentiment,
            COALESCE(n.negative_ratio,     0.0)  AS negative_ratio,
            COALESCE(n.unique_complaints,    0)   AS unique_complaints,
            COALESCE(n.confidence,        'low')  AS nlp_confidence,
            COALESCE(t.business_count,       1)   AS tile_density,
            COALESCE(o.amenity, '')               AS amenity,
            COALESCE(o.shop,    '')               AS shop
        FROM STAGING.BUSINESSES b
        LEFT JOIN FEATURES.NLP_SIGNALS n
            ON b.unified_category  = n.unified_category
        LEFT JOIN STAGING.TILE_DENSITY t
            ON b.geohash6          = t.tile_id
            AND b.unified_category = t.unified_category
        LEFT JOIN RAW.OSM_BUSINESSES o
            ON b.business_id = o.osm_id
        WHERE b.scan_id = '{scan_id}'
          AND b.source  = 'osm'
    """)
    rows = cur.fetchall()
    cols = [d[0].lower() for d in cur.description]
    df   = pd.DataFrame(rows, columns=cols)
    amenity_coverage = (df["amenity"] != "").sum()
    print(f"  Pulled {len(df):,} businesses "
          f"(amenity coverage: {amenity_coverage}/{len(df)})")
    return df


def pull_complaints(conn) -> pd.DataFrame:
    print("  Pulling complaint corpus...")
    query = """
        SELECT unified_category, complaint_summary
        FROM   FEATURES.COMPLAINTS
        WHERE  complaint_summary IS NOT NULL
          AND  confidence IN ('high', 'medium')
    """
    df = pd.read_sql(query, conn)
    df.columns = [c.lower() for c in df.columns]
    return df


# ── feature matrix ─────────────────────────────────────────────────────────
def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    print("\n[2/6] Building feature matrix...")
    X = df[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        X[col] = X[col].fillna(X[col].median())
    X_scaled = StandardScaler().fit_transform(X)
    print(f"  Shape: {X_scaled.shape}  (businesses × features)")
    return X_scaled


# ── UMAP ───────────────────────────────────────────────────────────────────
def run_umap(X: np.ndarray, cfg: dict) -> np.ndarray:
    print("\n[3/6] Running UMAP...")
    reducer = umap.UMAP(
        n_components=cfg["n_components"],
        n_neighbors=cfg["n_neighbors"],
        min_dist=cfg["min_dist"],
        metric=cfg["metric"],
        random_state=cfg["random_state"],
    )
    X_reduced = reducer.fit_transform(X)
    print(f"  Reduced to {X_reduced.shape[1]} dims")
    return X_reduced


def run_umap_2d(X: np.ndarray) -> np.ndarray:
    """2D projection for visualization only."""
    return umap.UMAP(
        n_components=2, n_neighbors=15,
        min_dist=0.1, random_state=42,
    ).fit_transform(X)


# ── HDBSCAN ────────────────────────────────────────────────────────────────
def run_hdbscan(X_reduced: np.ndarray, cfg: dict) -> tuple:
    print("\n[4/6] Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg["min_cluster_size"],
        min_samples=cfg["min_samples"],
        cluster_selection_method=cfg["cluster_selection_method"],
    )
    labels = clusterer.fit_predict(X_reduced)
    probs  = clusterer.probabilities_

    n_cl    = len(set(labels)) - (1 if -1 in labels else 0)
    noise   = (labels == -1).sum()
    n_ratio = noise / len(labels)

    print(f"  Clusters : {n_cl}")
    print(f"  Noise    : {noise} ({n_ratio:.1%})")
    if n_ratio > 0.4:
        print("  ⚠ High noise — lower min_cluster_size in config.yaml")

    return labels, probs, clusterer


# ── evaluation ─────────────────────────────────────────────────────────────
def evaluate_clusters(X_reduced: np.ndarray, labels: np.ndarray, clusterer) -> dict:
    print("\n  --- Cluster Evaluation ---")
    mask = labels != -1
    X_v, l_v = X_reduced[mask], labels[mask]

    if len(set(l_v)) < 2:
        print("  Not enough clusters to evaluate")
        return {}

    sil   = silhouette_score(X_v, l_v)
    db    = davies_bouldin_score(X_v, l_v)
    noise = (labels == -1).sum() / len(labels)

    print(f"  Silhouette     : {sil:.3f}  (target > 0.35)")
    print(f"  Davies-Bouldin : {db:.3f}  (lower is better)")
    print(f"  Noise ratio    : {noise:.1%}  (target < 20%)")

    grade = ("excellent" if sil > 0.5 else "good" if sil > 0.35
             else "fair — tune UMAP/HDBSCAN" if sil > 0.2
             else "poor — review feature engineering")
    print(f"  Grade          : {grade}")
    return {"silhouette": sil, "davies_bouldin": db, "noise_ratio": noise}


# ── BERTopic ───────────────────────────────────────────────────────────────
def run_bertopic(complaints_df: pd.DataFrame, cfg: dict) -> dict:
    print("\n[5/6] Running BERTopic...")
    docs = complaints_df["complaint_summary"].dropna().tolist()

    if len(docs) < 50:
        print("  Not enough complaints — skipping")
        return {}

    topic_model = BERTopic(
        language=cfg["language"],
        min_topic_size=cfg["min_topic_size"],
        nr_topics=cfg["nr_topics"],
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(docs)
    complaints_df = complaints_df.dropna(subset=["complaint_summary"]).copy()
    complaints_df["topic_id"] = topics

    info = topic_model.get_topic_info()
    print(f"  Topics found: {len(info) - 1}")
    print("\n  Top 5 (manual check — do these make sense?):")
    for _, row in info[info["Topic"] != -1].head(5).iterrows():
        words = [w for w, _ in topic_model.get_topic(row["Topic"])[:5]]
        print(f"  Topic {row['Topic']:>2} ({row['Count']:>4} docs): "
              f"{', '.join(words)}")

    category_topics = {}
    for cat, grp in complaints_df.groupby("unified_category"):
        top = grp["topic_id"].mode()
        if len(top) > 0 and top.iloc[0] != -1:
            words = [w for w, _ in topic_model.get_topic(top.iloc[0])[:5]]
            category_topics[cat] = ", ".join(words)

    return category_topics


# ── ablation study ─────────────────────────────────────────────────────────
def run_ablation(df: pd.DataFrame, cfg: dict):
    print("\n  --- Ablation Study ---")
    hcfg = cfg["ml"]["hdbscan"]
    ucfg = cfg["ml"]["umap"]

    variants = {
        "full":          FEATURE_COLS,
        "no NLP":        ["rating_norm", "review_log"],
        "no ratings":    ["review_log", "avg_sentiment", "negative_ratio"],
        "no review_log": ["rating_norm", "avg_sentiment", "negative_ratio"],
    }

    for name, cols in variants.items():
        X_v = StandardScaler().fit_transform(
            df[cols].fillna(df[cols].median())
        )
        n_comp = min(ucfg["n_components"], len(cols))
        X_r = umap.UMAP(
            n_components=n_comp, n_neighbors=ucfg["n_neighbors"],
            min_dist=ucfg["min_dist"], random_state=ucfg["random_state"],
        ).fit_transform(X_v)

        lbls = hdbscan.HDBSCAN(
            min_cluster_size=hcfg["min_cluster_size"],
            min_samples=hcfg["min_samples"],
        ).fit_predict(X_r)

        mask = lbls != -1
        sil  = (silhouette_score(X_r[mask], lbls[mask])
                if mask.sum() > 10 and len(set(lbls[mask])) >= 2 else 0.0)
        n_cl = len(set(lbls)) - (1 if -1 in lbls else 0)
        print(f"  {name:<20} sil={sil:.3f}  clusters={n_cl}"
              f"  noise={(lbls==-1).sum()/len(lbls):.1%}")


# ── push to snowflake ──────────────────────────────────────────────────────
def push_results(
    df: pd.DataFrame,
    gaps: pd.DataFrame,
    labels: np.ndarray,
    probs: np.ndarray,
    X_2d: np.ndarray,
    conn,
):
    print("\nPushing results to Snowflake...")

    # cluster assignments
    asgn = df[["business_id", "name", "unified_category"]].copy()
    asgn["cluster_id"]   = labels
    asgn["cluster_prob"] = probs.round(4)
    asgn["umap_x"]       = X_2d[:, 0].round(4)
    asgn["umap_y"]       = X_2d[:, 1].round(4)
    asgn.columns         = [c.upper() for c in asgn.columns]

    write_pandas(conn, asgn, "CLUSTER_ASSIGNMENTS",
                 schema="RESULTS", auto_create_table=False,
                 overwrite=True, quote_identifiers=False)
    print(f"  CLUSTER_ASSIGNMENTS : {len(asgn):,} rows")

    # gap scores
    g = gaps.copy()
    g.columns = [c.upper() for c in g.columns]
    write_pandas(conn, g, "GAP_SCORES",
                 schema="RESULTS", auto_create_table=False,
                 overwrite=True, quote_identifiers=False)
    print(f"  GAP_SCORES          : {len(g):,} rows")


# ── main ───────────────────────────────────────────────────────────────────
def main(city: str, evaluate: bool, ablate: bool, explain: bool):
    cfg  = load_config()
    conn = get_connection()

    print(f"\nLocalPulse — Phase 4 ML Pipeline")
    print(f"City    : {city}")
    print(f"Flags   : evaluate={evaluate}  ablate={ablate}  explain={explain}")
    print("=" * 55)

    refresh_staging(conn)
    df            = pull_features(city, conn)
    complaints_df = pull_complaints(conn)

    if len(df) < 20:
        print(f"Only {len(df)} businesses — run Overpass scan first.")
        return

    X_scaled  = build_feature_matrix(df)

    if ablate:
        run_ablation(df, cfg)

    X_reduced = run_umap(X_scaled, cfg["ml"]["umap"])
    X_2d      = run_umap_2d(X_scaled)

    labels, probs, clusterer = run_hdbscan(X_reduced, cfg["ml"]["hdbscan"])

    if evaluate:
        evaluate_clusters(X_reduced, labels, clusterer)

    category_topics = run_bertopic(complaints_df, cfg["ml"]["bertopic"])

    gaps = compute_gaps(df, labels, category_topics, cfg)

    if explain and len(gaps) > 0:
        gaps = enrich_with_llm(gaps, conn)
        push_enriched_gaps(gaps, conn)

    push_results(df, gaps, labels, probs, X_2d, conn)

    conn.close()
    print("\nPhase 4 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LocalPulse ML pipeline")
    parser.add_argument("--city",     type=str,  default="Madison, WI")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--ablate",   action="store_true")
    parser.add_argument("--explain",  action="store_true",
                        help="Enrich gaps with Cortex LLM explanations")
    args = parser.parse_args()
    main(args.city, args.evaluate, args.ablate, args.explain)