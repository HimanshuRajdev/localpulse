"""
app.py — LocalPulse dashboard.
Receives location from search.html via URL params, runs pipeline, shows results.
Based exactly on the working version — only addition is reading URL params on load.
"""

import os
import re
import subprocess
import sys
import time

import pandas as pd
import requests
import snowflake.connector
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from src.models.creative_advisor import generate_creative_ideas

load_dotenv()


def get_env(key: str) -> str:
    """Read from st.secrets (Streamlit Cloud) or .env (local dev)."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")


st.set_page_config(
    page_title="LocalPulse",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
  .block-container { padding-top: 2rem !important; max-width: 1100px !important; background: #F5F3EE; }
  .stApp { background: #F5F3EE; }
  h1,h2,h3 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; letter-spacing: -0.02em; color: #1A1A1A !important; }
  div[data-testid="metric-container"] {
    background: #fff !important;
    border: 1px solid #E2DFD8 !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }
  div[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; color: #1A1A1A !important; }
  div[data-testid="stMetricLabel"] { color: #7A7570 !important; font-size: 12px !important; }
  .stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
  }
  .stButton > button[kind="primary"] {
    background: #FF4F2B !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(255,79,43,0.3) !important;
    color: #fff !important;
  }
  .stButton > button[kind="primary"]:hover {
    background: #e63d1a !important;
    box-shadow: 0 6px 20px rgba(255,79,43,0.4) !important;
  }
  div[data-testid="stInfo"] {
    background: #FFF8F6 !important;
    border-left: 3px solid #FF4F2B !important;
    border-radius: 0 10px 10px 0 !important;
    color: #1A1A1A !important;
  }
  div[data-testid="stSuccess"] {
    background: #EDFAF2 !important;
    border-left: 3px solid #2D7D46 !important;
    border-radius: 0 10px 10px 0 !important;
  }
  div[data-testid="stContainer"] { border-radius: 14px !important; }
  .stExpander { border: 1px solid #E2DFD8 !important; border-radius: 12px !important; background: #fff !important; }
  .stProgress > div > div { background: #FF4F2B !important; }
  hr { border-color: #E2DFD8 !important; }
</style>
""", unsafe_allow_html=True)


# ── snowflake ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_conn():
    return snowflake.connector.connect(
        account=get_env("SF_ACCOUNT"),
        user=get_env("SF_USER"),
        password=get_env("SF_PASSWORD"),
        database=get_env("SF_DATABASE"),
        warehouse=get_env("SF_WAREHOUSE"),
    )


@st.cache_data(ttl=60)
def load_gaps(city: str) -> pd.DataFrame:
    """city arg = per-city cache key so switching cities always fetches fresh data."""
    cur = get_conn().cursor()
    cur.execute("""
        SELECT cluster_id, category, subcategory,
               opportunity_score, supply_gap, demand_proxy, complaint_signal,
               nearest_competitor_km, missing_price_tier, hours_gap,
               top_complaint, recommendation, explanation, business_plan,
               confidence, avg_lat, avg_lng, business_count
        FROM   RESULTS.GAP_SCORES
        ORDER  BY opportunity_score DESC
    """)
    rows = cur.fetchall()
    cols = [d[0].lower() for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


# ── pipeline ───────────────────────────────────────────────────────────────
def run_pipeline(city: str, lat: float, lng: float, radius: float) -> bool:
    try:
        r1 = subprocess.run(
            [sys.executable, "-m", "src.ingestion.overpass_client",
             "--lat", str(lat), "--lng", str(lng),
             "--radius", str(radius), "--name", city],
            capture_output=True, text=True, timeout=180,
        )
        if r1.returncode != 0:
            st.error(f"Scan failed:\n{r1.stderr[:400]}")
            return False
        r2 = subprocess.run(
            [sys.executable, "-m", "src.models.cluster",
             "--city", city, "--explain"],
            capture_output=True, text=True, timeout=600,
        )
        if r2.returncode != 0:
            st.error(f"ML pipeline failed:\n{r2.stderr[:400]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        st.error("Timed out — try a smaller radius.")
        return False
    except Exception as e:
        st.error(str(e))
        return False


def get_openai_key() -> str:
    key = get_env("OPENAI_API_KEY")
    if not key:
        st.error("Add OPENAI_API_KEY to your .env file. Get one at platform.openai.com")
    return key


# ── main ───────────────────────────────────────────────────────────────────
def fetch_category_detail(category: str) -> dict:
    """
    Fetch businesses present and competitor info for a specific category
    from the latest scan in Snowflake.
    """
    try:
        cur = get_conn().cursor()
        # get latest scan_id
        cur.execute("SELECT scan_id FROM RAW.OSM_BUSINESSES ORDER BY scanned_at DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return {}
        scan_id = row[0]

        # fetch businesses in this category from latest scan
        cur.execute(f"""
            SELECT b.name, b.lat, b.lng, b.opening_hours
            FROM STAGING.BUSINESSES b
            WHERE b.unified_category = %s
              AND b.scan_id = %s
              AND b.source  = 'osm'
            ORDER BY b.name
            LIMIT 20
        """, (category, scan_id))
        rows = cur.fetchall()
        businesses = [
            {"name": r[0], "lat": r[1], "lng": r[2], "hours": r[3]}
            for r in rows if r[0]
        ]

        # get gap detail for this category
        cur.execute("""
            SELECT nearest_competitor_km, subcategory,
                   hours_gap, missing_price_tier, top_complaint
            FROM RESULTS.GAP_SCORES
            WHERE LOWER(category) = LOWER(%s)
            ORDER BY scored_at DESC LIMIT 1
        """, (category,))
        gap = cur.fetchone()

        return {
            "businesses":      businesses,
            "nearest_km":      gap[0] if gap else None,
            "subcategory":     gap[1] if gap else None,
            "hours_gap":       gap[2] if gap else None,
            "price_tier":      gap[3] if gap else None,
            "top_complaint":   gap[4] if gap else None,
        }
    except Exception:
        return {}


def main():
    # ── header ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:1.5rem 0 1rem">
        <div style="font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;
                    letter-spacing:-0.03em;line-height:1;margin-bottom:6px">
            Local<span style="color:#FF4F2B">Pulse</span>
        </div>
        <div style="font-size:14px;color:#7A7570;font-family:DM Sans,sans-serif">
            Find untapped business opportunities in any US city
        </div>
    </div>
    <hr style="border:none;border-top:1.5px solid #E2DFD8;margin-bottom:1.5rem">
    """, unsafe_allow_html=True)

    # ── read URL params from search.html (auto-redirect) ───────────────────
    params = st.query_params
    url_name   = params.get("lp_name",   "")
    url_lat    = params.get("lp_lat",    "")
    url_lng    = params.get("lp_lng",    "")
    url_radius = params.get("lp_radius", "1.5")

    # if search.html redirected here with a new location, run the scan
    if url_name and url_name != st.session_state.get("scanned_city", ""):
        try:
            lat    = float(url_lat)
            lng    = float(url_lng)
            radius = float(url_radius)
        except (ValueError, TypeError):
            st.error("Invalid location data. Go back to search.html and try again.")
            st.stop()

        st.info(f"📍 **{url_name}** · {lat:.4f}, {lng:.4f} · {radius} km radius")

        # ── inline pipeline with real progress bar ────────────────────────
        prog = st.progress(0, "Starting scan...")

        try:
            import snowflake.connector as sf
            from src.ingestion.overpass_client import scan_city, load_to_snowflake
            from src.models.cluster import (
                refresh_staging, pull_features, pull_complaints,
                build_feature_matrix, run_umap, run_umap_2d,
                run_hdbscan, run_bertopic, push_results, load_config
            )
            from src.models.gap_scorer import compute_gaps

            cfg  = load_config()
            conn = get_conn()  # reuse cached connection

            prog.progress(8,  "Fetching businesses from OpenStreetMap...")
            businesses = scan_city(lat, lng, url_name, radius_km=radius)
            if not businesses:
                st.error("No businesses found. Try a larger radius.")
                prog.empty(); st.stop()

            prog.progress(22, "Uploading to Snowflake...")
            load_to_snowflake(businesses)

            prog.progress(34, "Rebuilding staging tables...")
            refresh_staging(conn)

            prog.progress(46, "Building feature matrix...")
            df            = pull_features(url_name, conn)
            complaints_df = pull_complaints(conn)
            if len(df) < 10:
                st.error("Too few businesses found. Try a larger radius.")
                prog.empty(); st.stop()

            X_scaled = build_feature_matrix(df)

            prog.progress(57, "Running UMAP...")
            X_reduced = run_umap(X_scaled, cfg["ml"]["umap"])
            X_2d      = run_umap_2d(X_scaled)

            prog.progress(68, "Clustering with HDBSCAN...")
            labels, probs, clusterer = run_hdbscan(X_reduced, cfg["ml"]["hdbscan"])

            prog.progress(78, "Extracting complaint themes...")
            category_topics = run_bertopic(complaints_df, cfg["ml"]["bertopic"])

            prog.progress(88, "Scoring market gaps...")
            gaps_df = compute_gaps(df, labels, category_topics, cfg)

            prog.progress(95, "Saving results...")
            push_results(df, gaps_df, labels, probs, X_2d, conn)
            # don't close — conn is cached and reused across scans

            prog.progress(100, "Scan complete!")
            time.sleep(0.5)
            prog.empty()
            load_gaps.clear()
            st.session_state["scanned_city"] = url_name
            st.session_state["scan_ts"]      = time.time()
            st.session_state["selected_idx"] = 0
            st.session_state.pop("ideas", None)
            st.query_params.clear()
            st.rerun()

        except Exception as e:
            prog.empty()
            st.error(f"Scan failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

            st.stop()

    # ── empty state ────────────────────────────────────────────────────────
    if "scanned_city" not in st.session_state:
        st.markdown("<br>", unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.info(
                "**How it works**\n\n"
                "1. Open `search.html` in your browser\n"
                "2. Type any city, neighborhood, or address\n"
                "3. Select from the Google dropdown\n"
                "4. Hit Scan — you'll be brought straight here\n\n"
                "LocalPulse uses unsupervised ML + Snowflake Cortex to find "
                "genuine market gaps and generate creative business ideas."
            )
        return

    # ── load results ───────────────────────────────────────────────────────
    try:
        city_key = st.session_state.get("scanned_city", "")
        gaps = load_gaps(city_key)
    except Exception as e:
        st.error(f"Could not load results: {e}")
        return

    if gaps.empty:
        st.warning("No opportunities found. Try a larger radius.")
        return

    # ── gap summary ────────────────────────────────────────────────────────
    # compute score from default weights (0.35 / 0.35 / 0.30)
    gaps["score"] = (
        0.35 * gaps["supply_gap"]
        + 0.35 * gaps["demand_proxy"]
        + 0.30 * gaps["complaint_signal"]
    ).round(3)
    # deduplicate — keep best score per category only
    gaps = (
        gaps.sort_values("score", ascending=False)
        .drop_duplicates(subset=["category"], keep="first")
        .reset_index(drop=True)
    )

    city_label = st.session_state.get("scanned_city", "Last scan")
    st.markdown(
        f'<div style="font-family:Syne,sans-serif;font-size:1.6rem;'
        f'font-weight:800;letter-spacing:-0.02em;margin-bottom:4px">'
        f'📍 {city_label}</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gaps found",     len(gaps))
    m2.metric("Top score",      f"{gaps['score'].max():.2f}")
    m3.metric("Avg supply gap", f"{gaps['supply_gap'].mean():.2f}")
    m4.metric("Businesses",     st.session_state.get("biz_count", "—"))
    st.markdown("")

    # ── gap pills + complaint summary ──────────────────────────────────────
    from collections import Counter

    all_complaints = []
    for _, row in gaps.iterrows():
        c = row.get("top_complaint") or ""
        words = [w for w in c.split(", ") if len(w) >= 4]
        all_complaints.extend(words)
    top_themes = [w for w, _ in Counter(all_complaints).most_common(8)]

    with st.container(border=True):
        st.markdown(
            '<div style="font-size:11px;font-weight:500;color:#7A7570;'
            'letter-spacing:.06em;margin-bottom:.75rem">'
            'GAPS DETECTED — click any category to explore</div>',
            unsafe_allow_html=True
        )

        # clickable pill buttons — 5 per row
        chunk = 5
        gap_list = list(gaps.iterrows())
        for i in range(0, len(gap_list), chunk):
            slice_ = gap_list[i:i+chunk]
            cols = st.columns(len(slice_))
            for col, (_, gap_row) in zip(cols, slice_):
                score = gap_row["score"]
                cat   = gap_row["category"].replace("_", " ").title()
                color = "#FF4F2B" if score >= 0.6 else "#BA7517"
                with col:
                    label = f"{cat}  {score:.2f}"
                    if st.button(label, key=f"pill_{gap_row['category']}",
                                 use_container_width=True):
                        cur = st.session_state.get("selected_category")
                        st.session_state["selected_category"] = (
                            None if cur == gap_row["category"]
                            else gap_row["category"]
                        )
                        st.rerun()

        # detail panel for selected category
        selected_cat = st.session_state.get("selected_category")
        if selected_cat:
            matched = gaps[gaps["category"] == selected_cat]
            if not matched.empty:
                gap_row   = matched.iloc[0]
                cat_label = selected_cat.replace("_", " ").title()
                st.markdown(
                    f'<div style="margin-top:1rem;padding:1rem 1.25rem;'
                    f'background:#fff;border-radius:12px;'
                    f'border:1.5px solid #FF4F2B">',
                    unsafe_allow_html=True
                )
                st.markdown(f"**{cat_label}** — gap score {gap_row['score']:.2f}")
                with st.spinner(f"Loading {cat_label} details..."):
                    detail = fetch_category_detail(selected_cat)

                d1, d2 = st.columns(2)
                with d1:
                    st.caption("BUSINESSES CURRENTLY IN AREA")
                    bizs = detail.get("businesses", [])
                    if bizs:
                        for b in bizs[:8]:
                            hours = (b.get("hours") or "hours unknown")[:40]
                            st.markdown(
                                f'<div style="font-size:13px;padding:5px 0;'
                                f'border-bottom:0.5px solid #F0EDE8">'
                                f'<b>{b["name"]}</b>'
                                f'<br><span style="color:#7A7570;font-size:11px">'
                                f'{hours}</span></div>',
                                unsafe_allow_html=True
                            )
                        if len(bizs) > 8:
                            st.caption(f"+ {len(bizs)-8} more not shown")
                    else:
                        st.info("No businesses found — supply gap confirmed.")

                with d2:
                    st.caption("GAP SIGNALS")
                    km = detail.get("nearest_km") or gap_row.get("nearest_competitor_km")
                    if km:
                        st.metric("Nearest competitor", f"{km} km away")
                    sub = detail.get("subcategory") or gap_row.get("subcategory") or ""
                    if sub:
                        st.markdown(f"**Specific gap:** {sub}")
                    hours_g = detail.get("hours_gap") or gap_row.get("hours_gap") or ""
                    if hours_g and hours_g not in ("hours data unavailable", "reasonable coverage"):
                        st.markdown(f"**Hours gap:** {hours_g}")
                    price = detail.get("price_tier") or gap_row.get("missing_price_tier") or ""
                    if price:
                        st.markdown(f"**Missing tier:** {price}")
                    complaint = detail.get("top_complaint") or gap_row.get("top_complaint") or ""
                    clean = ", ".join([w for w in complaint.split(", ") if len(w) >= 4])
                    if clean:
                        st.markdown(f"**Complaints:** {clean}")

                st.markdown('</div>', unsafe_allow_html=True)

        # complaint themes row
        if top_themes:
            st.markdown(
                '<div style="font-size:11px;font-weight:500;color:#7A7570;'
                'letter-spacing:.06em;margin:.75rem 0 .4rem">TOP COMPLAINT THEMES</div>',
                unsafe_allow_html=True
            )
            theme_html = "".join([
                f'<span style="display:inline-block;margin:2px 3px 2px 0;'
                f'padding:3px 10px;border-radius:20px;font-size:12px;'
                f'background:#FFF3E0;color:#BF360C;font-weight:500">{t}</span>'
                for t in top_themes
            ])
            st.markdown(theme_html, unsafe_allow_html=True)


    # ── GPT-4o ideas ───────────────────────────────────────────────────────
    if "gpt_ideas" not in st.session_state:
        if st.button(
            "Generate 3 creative business ideas with GPT-4o →",
            type="primary", use_container_width=True
        ):
            api_key = get_env("OPENAI_API_KEY")
            if not api_key:
                st.error("Add OPENAI_API_KEY to your .env file. Get one at platform.openai.com")
            else:
                with st.spinner("GPT-4o is analysing all gaps and thinking like an entrepreneur..."):
                    try:
                        ideas = generate_creative_ideas(
                            gaps,
                            city_label,
                            api_key,
                        )
                        st.session_state["gpt_ideas"] = ideas
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
    else:
        ideas = st.session_state["gpt_ideas"]

        st.markdown(f"**3 best opportunities for {city_label}**")
        st.caption("Generated by GPT-4o using your full gap analysis")
        st.markdown("")

        capital_bg = {"low": "#EAF3DE", "medium": "#FAEEDA", "high": "#FCEBEB"}
        capital_fg = {"low": "#27500A", "medium": "#633806", "high": "#A32D2D"}

        for i, idea in enumerate(ideas, 1):
            with st.container(border=True):
                # header row
                hc, fc = st.columns([3, 1])
                with hc:
                    st.caption(f"IDEA {i}  ·  addresses: {', '.join(idea.get('gaps_addressed', []))}")
                    st.markdown(f"**{idea.get('title', '')}**")
                with fc:
                    fmt = idea.get("format", "")
                    st.markdown(
                        f"<div style='text-align:right;margin-top:18px'>"
                        f"<span style='font-size:11px;padding:3px 9px;"
                        f"border-radius:10px;background:#E6F1FB;color:#0C447C;"
                        f"font-weight:500'>{fmt}</span></div>",
                        unsafe_allow_html=True,
                    )

                # description
                st.write(idea.get("description", ""))

                # why now + first step + capital
                cap = idea.get("startup_angle", "medium").lower()
                cbg = capital_bg.get(cap, "#F4F6F8")
                cfg = capital_fg.get(cap, "#555")

                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown(
                        f"<div style='background:#EAF3DE;border-radius:8px;"
                        f"padding:8px 10px;font-size:12px;color:#27500A'>"
                        f"<b>Why now</b><br>{idea.get('why_now','')}</div>",
                        unsafe_allow_html=True,
                    )
                with r2:
                    st.markdown(
                        f"<div style='background:#EEEDFE;border-radius:8px;"
                        f"padding:8px 10px;font-size:12px;color:#3C3489'>"
                        f"<b>First step</b><br>{idea.get('first_step','')}</div>",
                        unsafe_allow_html=True,
                    )
                with r3:
                    st.markdown(
                        f"<div style='background:{cbg};border-radius:8px;"
                        f"padding:8px 10px;font-size:12px;color:{cfg}'>"
                        f"<b>Capital needed</b><br>{cap}</div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("")
        if st.button("Regenerate ideas", use_container_width=True):
            del st.session_state["gpt_ideas"]
            st.rerun()

        # expandable raw data for reference
        with st.expander("View all gap scores"):
            show_cols = ["category", "score", "supply_gap",
                         "demand_proxy", "complaint_signal",
                         "nearest_competitor_km", "hours_gap"]
            show_cols = [c for c in show_cols if c in gaps.columns]
            st.dataframe(gaps[show_cols].round(3), use_container_width=True)


if __name__ == "__main__":
    main()