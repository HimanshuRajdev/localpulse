"""
app.py — LocalPulse dashboard.
Google Places autocomplete → Overpass scan → ML pipeline → results.
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

load_dotenv()

st.set_page_config(
    page_title="LocalPulse",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
    div[data-testid="metric-container"] {
        border: 1px solid rgba(49,51,63,0.1);
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── google places component ────────────────────────────────────────────────
def google_places_input(api_key: str, default_value: str = "") -> dict | None:
    """
    Renders a Google Places Autocomplete search box.
    
    The JS→Python bridge works by writing the selected place into
    a hidden Streamlit text_input via JavaScript, which Streamlit
    can then read on the next rerun.
    
    Returns the currently stored place from session_state, or None.
    """
    component_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0;
       font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
  body {{ background:transparent; padding:2px 0; }}
  #wrap {{ position:relative; width:100%; }}
  #loc-input {{
    width:100%; padding:11px 16px 11px 44px;
    font-size:15px; border:1.5px solid #d0d0d0;
    border-radius:24px; outline:none;
    background:white; color:#1a1a1a;
    transition:border-color .15s, box-shadow .15s;
  }}
  #loc-input:focus {{
    border-color:#185FA5;
    box-shadow:0 0 0 3px rgba(24,95,165,0.12);
  }}
  #pin {{
    position:absolute; left:14px; top:50%;
    transform:translateY(-50%);
    width:20px; height:20px; color:#888;
    pointer-events:none;
  }}
  /* fix broken icons and dropdown */
  .pac-icon, .pac-icon-marker {{ display:none !important; }}
  .pac-container {{
    border-radius:12px !important;
    margin-top:6px !important;
    box-shadow:0 4px 24px rgba(0,0,0,0.14) !important;
    border:1px solid rgba(0,0,0,0.08) !important;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important;
    font-size:13px !important;
    background:white !important;
    z-index:9999 !important;
  }}
  .pac-item {{
    padding:10px 16px !important;
    cursor:pointer; border-top:none !important;
    line-height:1.4 !important;
  }}
  .pac-item:hover {{ background:#f0f7ff !important; }}
  .pac-item-query {{
    font-size:14px !important;
    font-weight:500 !important;
    color:#1a1a1a !important;
  }}
  .pac-matched {{ color:#185FA5 !important; font-weight:700 !important; }}
  .pac-secondary-text {{ color:#888 !important; font-size:12px !important; }}
  #confirm {{
    display:none; margin-top:8px; padding:8px 14px;
    background:#e8f4fd; border-radius:8px;
    font-size:12px; color:#0c447c; line-height:1.5;
  }}
</style>
</head>
<body>
<div id="wrap">
  <svg id="pin" viewBox="0 0 24 24" fill="none"
       stroke="currentColor" stroke-width="2">
    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
    <circle cx="12" cy="10" r="3"/>
  </svg>
  <input id="loc-input" type="text"
    placeholder="Search any city, neighborhood, or address..."
    value="{default_value}" autocomplete="off"/>
</div>
<div id="confirm"></div>

<script>
var selectedPlace = null;

function initMap() {{
  var input = document.getElementById('loc-input');
  var auto = new google.maps.places.Autocomplete(input, {{
    types: ['geocode', 'establishment'],
    fields: ['formatted_address', 'name', 'geometry'],
  }});

  auto.addListener('place_changed', function() {{
    var place = auto.getPlace();
    if (!place || !place.geometry) return;

    selectedPlace = {{
      name: place.formatted_address || place.name,
      lat:  place.geometry.location.lat(),
      lng:  place.geometry.location.lng(),
    }};

    // show confirmation in the component
    var c = document.getElementById('confirm');
    c.style.display = 'block';
    c.innerHTML = '&#10003; <b>' + selectedPlace.name + '</b><br>'
      + selectedPlace.lat.toFixed(5) + ', ' + selectedPlace.lng.toFixed(5);

    // send to parent Streamlit frame
    window.parent.postMessage({{
      type:  'localpulse_place',
      name:  selectedPlace.name,
      lat:   selectedPlace.lat,
      lng:   selectedPlace.lng,
    }}, '*');
  }});
}}
</script>
<script
  src="https://maps.googleapis.com/maps/api/js?key={api_key}&libraries=places&callback=initMap"
  async defer>
</script>
</body>
</html>
"""
    # render the component — height 120 to show confirmation text too
    components.html(component_html, height=120, scrolling=False)


# ── message receiver — JS → Python ────────────────────────────────────────
def place_receiver():
    """
    Invisible component that listens for postMessage from the
    Google Places component and writes the result into Streamlit's
    session state via a hidden text input trick.
    """
    receiver_html = """
<script>
window.addEventListener('message', function(e) {
  if (!e.data || e.data.type !== 'localpulse_place') return;
  var p = e.data;
  // find the hidden streamlit inputs and update them
  function setInput(label, value) {
    var inputs = window.parent.document.querySelectorAll('input[type=text]');
    for (var i = 0; i < inputs.length; i++) {
      var lbl = inputs[i].closest('[data-testid="stTextInput"]');
      if (lbl && lbl.querySelector('label') &&
          lbl.querySelector('label').textContent.trim() === label) {
        var nativeInputValueSetter = Object.getOwnPropertyDescriptor(
          window.HTMLInputElement.prototype, 'value').set;
        nativeInputValueSetter.call(inputs[i], value);
        inputs[i].dispatchEvent(new Event('input', { bubbles: true }));
        return;
      }
    }
  }
  setInput('__place_name__', p.name);
  setInput('__place_lat__',  String(p.lat));
  setInput('__place_lng__',  String(p.lng));
}, false);
</script>
"""
    components.html(receiver_html, height=0)


# ── snowflake ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_conn():
    return snowflake.connector.connect(
        account=os.getenv("SF_ACCOUNT"),
        user=os.getenv("SF_USER"),
        password=os.getenv("SF_PASSWORD"),
        database=os.getenv("SF_DATABASE"),
        warehouse=os.getenv("SF_WAREHOUSE"),
    )


@st.cache_data(ttl=300)
def load_gaps() -> pd.DataFrame:
    df = pd.read_sql("""
        SELECT cluster_id, category, subcategory,
               opportunity_score, supply_gap, demand_proxy, complaint_signal,
               nearest_competitor_km, missing_price_tier, hours_gap,
               top_complaint, recommendation, explanation, business_plan,
               confidence, avg_lat, avg_lng, business_count
        FROM   RESULTS.GAP_SCORES
        ORDER  BY opportunity_score DESC
    """, get_conn())
    df.columns = [c.lower() for c in df.columns]
    return df


# ── pipeline ───────────────────────────────────────────────────────────────
def run_pipeline(city: str, lat: float, lng: float, radius: float) -> bool:
    try:
        r1 = subprocess.run(
            [sys.executable, "-m", "src.ingestion.overpass_client",
             "--lat", str(lat), "--lng", str(lng), "--radius", str(radius)],
            capture_output=True, text=True, timeout=120,
        )
        if r1.returncode != 0:
            st.error(f"Scan failed:\n{r1.stderr[:400]}")
            return False

        r2 = subprocess.run(
            [sys.executable, "-m", "src.models.cluster",
             "--city", city, "--explain"],
            capture_output=True, text=True, timeout=300,
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


# ── cortex ─────────────────────────────────────────────────────────────────
def generate_ideas(row: pd.Series) -> list[dict]:
    clean = ", ".join([
        w for w in (row.get("top_complaint") or "").split(", ")
        if len(w) >= 4
    ])
    prompt = f"""You are a creative business advisor.
Market gap:
- Category: {row['category']}
- Specific gap: {row.get('subcategory') or row['category']}
- Supply gap score: {row['supply_gap']:.2f}
- Nearest competitor: {row.get('nearest_competitor_km','unknown')} km
- Missing price tier: {row.get('missing_price_tier') or 'not identified'}
- Hours gap: {row.get('hours_gap') or 'not identified'}
- Customer complaints: {clean or 'not identified'}

Give exactly 3 creative business concepts:
CONCEPT 1: [title]
[2-sentence description]

CONCEPT 2: [title]
[2-sentence description]

CONCEPT 3: [title]
[2-sentence description]"""

    safe = prompt.replace("'", "''")
    try:
        cur = get_conn().cursor()
        cur.execute(
            f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-7b','{safe}') AS r"
        )
        res = cur.fetchone()
        return _parse(res[0]) if res and res[0] else []
    except Exception as e:
        st.error(f"Cortex: {e}")
        return []


def _parse(text: str) -> list[dict]:
    ideas = []
    for title, desc in re.findall(
        r"CONCEPT\s+\d+:\s*(.+?)\n(.*?)(?=CONCEPT\s+\d+:|$)",
        text, re.DOTALL
    ):
        ideas.append({"title": title.strip(), "description": desc.strip()})
    if not ideas:
        for i, chunk in enumerate(
            [c.strip() for c in text.split("\n\n") if c.strip()][:3]
        ):
            lines = chunk.split("\n")
            ideas.append({
                "title":       lines[0].replace(f"CONCEPT {i+1}:", "").strip(),
                "description": " ".join(lines[1:]).strip(),
            })
    return ideas[:3]


# ── main ───────────────────────────────────────────────────────────────────
def main():
    api_key = os.getenv("GOOGLE_MAPS_KEY", "")
    if not api_key:
        st.error(
            "Add `GOOGLE_MAPS_KEY=your_key` to your .env file.\n\n"
            "Free key: console.cloud.google.com → enable Maps JavaScript API + Places API"
        )
        st.stop()

    # ── header ─────────────────────────────────────────────────────────────
    st.markdown("## 📍 LocalPulse")
    st.caption("Discover untapped business opportunities in any US city")
    st.divider()

    # ── hidden state inputs (JS writes to these) ───────────────────────────
    # Invisible but present in DOM so the JS receiver can find them
    place_name = st.text_input("__place_name__", value=st.session_state.get("place_name", ""), label_visibility="hidden")
    place_lat  = st.text_input("__place_lat__",  value=st.session_state.get("place_lat",  ""), label_visibility="hidden")
    place_lng  = st.text_input("__place_lng__",  value=st.session_state.get("place_lng",  ""), label_visibility="hidden")

    # persist to session state when JS updates them
    if place_name: st.session_state["place_name"] = place_name
    if place_lat:  st.session_state["place_lat"]  = place_lat
    if place_lng:  st.session_state["place_lng"]  = place_lng

    # ── search + scan row ──────────────────────────────────────────────────
    search_col, radius_col, btn_col = st.columns([4, 1, 1])

    with search_col:
        default = st.session_state.get("place_name", "")
        google_places_input(api_key, default_value=default)
        place_receiver()   # invisible listener

    with radius_col:
        radius = st.select_slider(
            "Radius",
            options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            value=1.5,
            format_func=lambda x: f"{x} km",
            label_visibility="collapsed",
        )

    with btn_col:
        scan = st.button("Scan", type="primary", use_container_width=True)

    # ── scan logic ─────────────────────────────────────────────────────────
    if scan:
        city = st.session_state.get("place_name", "").strip()
        lat_s = st.session_state.get("place_lat",  "").strip()
        lng_s = st.session_state.get("place_lng",  "").strip()

        if not city:
            st.warning("Select a location from the search box first.")
            st.stop()

        # try lat/lng from autocomplete first
        try:
            lat = float(lat_s)
            lng = float(lng_s)
        except (ValueError, TypeError):
            # fallback: geocode via Nominatim
            st.info(f"Geocoding '{city}'...")
            try:
                r = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": city, "format": "json", "limit": 1},
                    headers={"User-Agent": "LocalPulse/1.0"},
                    timeout=10,
                )
                res = r.json()
                if not res:
                    st.error(f"Could not find '{city}'. "
                             "Try selecting from the dropdown.")
                    st.stop()
                lat = float(res[0]["lat"])
                lng = float(res[0]["lon"])
            except Exception as e:
                st.error(f"Geocoding failed: {e}")
                st.stop()

        st.info(
            f"📍 **{city}**  ·  {lat:.4f}, {lng:.4f}  ·  {radius}km radius"
        )

        with st.spinner("Scanning — this takes 2–4 minutes..."):
            prog = st.progress(0, "Fetching businesses from OpenStreetMap...")
            time.sleep(0.3)
            prog.progress(20, "Running ML pipeline...")
            ok = run_pipeline(city, lat, lng, radius)
            if ok:
                prog.progress(100, "Done!")
                time.sleep(0.4)
                prog.empty()
                load_gaps.clear()
                st.session_state["scanned_city"] = city
                st.session_state["selected_idx"] = 0
                st.session_state.pop("ideas", None)
                st.rerun()
            else:
                prog.empty()
                st.stop()

    # ── empty state ────────────────────────────────────────────────────────
    if "scanned_city" not in st.session_state:
        st.markdown("<br>", unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.info(
                "**How it works**\n\n"
                "1. Type any US city, neighborhood, or address above\n"
                "2. Select from the dropdown suggestions\n"
                "3. Choose a radius and hit **Scan**\n\n"
                "LocalPulse uses unsupervised ML + Snowflake Cortex to find "
                "genuine market gaps and generate creative business ideas."
            )
        return

    # ── load results ───────────────────────────────────────────────────────
    try:
        gaps = load_gaps()
    except Exception as e:
        st.error(f"Could not load results: {e}")
        return

    if gaps.empty:
        st.warning("No opportunities found. Try a larger radius.")
        return

    # weight sliders
    with st.expander("Adjust gap weights"):
        c1, c2, c3 = st.columns(3)
        ws = c1.slider("Supply gap",       0.0, 1.0, 0.35, 0.05)
        wd = c2.slider("Demand proxy",     0.0, 1.0, 0.35, 0.05)
        wc = c3.slider("Complaint signal", 0.0, 1.0, 0.30, 0.05)

    gaps["score"] = (
        ws * gaps["supply_gap"]
        + wd * gaps["demand_proxy"]
        + wc * gaps["complaint_signal"]
    ).round(3)
    gaps = gaps.sort_values("score", ascending=False).reset_index(drop=True)

    city_label = st.session_state.get("scanned_city", "Last scan")
    st.markdown(f"### Results — {city_label}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Opportunities",  len(gaps))
    m2.metric("Top score",      f"{gaps['score'].max():.2f}")
    m3.metric("Avg supply gap", f"{gaps['supply_gap'].mean():.2f}")
    m4.metric("Avg complaint",  f"{gaps['complaint_signal'].mean():.2f}")
    st.markdown("")

    left, right = st.columns([1, 1.3], gap="large")

    # ── ranked list ────────────────────────────────────────────────────────
    with left:
        st.markdown("**Ranked opportunities**")
        if "selected_idx" not in st.session_state:
            st.session_state.selected_idx = 0

        for i, row in gaps.iterrows():
            score  = row["score"]
            is_sel = (i == st.session_state.selected_idx)

            with st.container(border=True):
                hc, sc = st.columns([3, 1])
                with hc:
                    st.markdown(
                        f"**{row['category'].replace('_',' ').title()}**"
                    )
                    sub = row.get("subcategory") or ""
                    km  = row.get("nearest_competitor_km")
                    st.caption(
                        (sub or row["category"])
                        + (f" · {km}km away" if km else "")
                    )
                with sc:
                    color = "#185FA5" if score >= 0.6 else "#BA7517"
                    st.markdown(
                        f"<p style='text-align:right;font-size:20px;"
                        f"font-weight:700;color:{color};margin:4px 0'>"
                        f"{score:.2f}</p>",
                        unsafe_allow_html=True,
                    )

                st.caption("Supply gap")
                st.progress(float(row["supply_gap"]))
                st.caption("Demand proxy")
                st.progress(float(row["demand_proxy"]))
                st.caption("Complaint signal")
                st.progress(float(row["complaint_signal"]))

                # signal tags
                tags = []
                if row["supply_gap"]       > 0.7: tags.append("⬆ supply gap")
                if row["demand_proxy"]     > 0.6: tags.append("⬆ high demand")
                if row["complaint_signal"] > 0.6: tags.append("⚠ complaints")
                h = row.get("hours_gap") or ""
                if h and h not in ("hours data unavailable","reasonable coverage"):
                    tags.append("🕐 hours gap")
                if row.get("missing_price_tier"):
                    tags.append(f"💰 {row['missing_price_tier']}")
                if tags:
                    st.caption("  ".join(tags))

                if st.button(
                    "View ideas →",
                    key=f"sel_{i}",
                    use_container_width=True,
                    type="primary" if is_sel else "secondary",
                ):
                    st.session_state.selected_idx = i
                    st.session_state.pop("ideas", None)
                    st.rerun()

    # ── detail + ideas ─────────────────────────────────────────────────────
    with right:
        row = gaps.iloc[st.session_state.selected_idx]

        st.markdown(
            f"### {row['category'].replace('_',' ').title()}"
        )
        if row.get("subcategory"):
            st.caption(f"Specific gap: {row['subcategory']}")

        r1, r2, r3 = st.columns(3)
        r1.metric("Opportunity score", f"{row['score']:.2f}")
        r2.metric("Supply gap",        f"{row['supply_gap']:.2f}")
        r3.metric("Complaint signal",  f"{row['complaint_signal']:.2f}")

        exp = row.get("explanation") or row.get("recommendation") or ""
        if exp:
            st.info(exp)

        st.divider()
        st.markdown("**Creative business concepts**")
        st.caption("AI-generated ideas tailored to this specific gap")

        if row.get("business_plan") and "ideas" not in st.session_state:
            st.markdown(row["business_plan"])
            if st.button("Regenerate ideas"):
                st.session_state["ideas"] = None
                st.rerun()
        else:
            if st.button(
                "Generate ideas with Cortex AI",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Cortex is thinking like an entrepreneur..."):
                    st.session_state["ideas"] = generate_ideas(row)

            if st.session_state.get("ideas"):
                for i, idea in enumerate(st.session_state["ideas"], 1):
                    with st.container(border=True):
                        st.caption(f"CONCEPT {i}")
                        st.markdown(f"**{idea['title']}**")
                        st.write(idea["description"])

        with st.expander("Full gap details"):
            st.table(pd.DataFrame([
                {"Signal": k, "Value": str(v)}
                for k, v in {
                    "Nearest competitor":  f"{row.get('nearest_competitor_km','N/A')} km",
                    "Missing price tier":  row.get("missing_price_tier") or "not identified",
                    "Hours gap":           row.get("hours_gap") or "not identified",
                    "Customer complaint":  row.get("top_complaint") or "not identified",
                    "Businesses scanned":  row.get("business_count", "N/A"),
                    "Data confidence":     row.get("confidence", "N/A"),
                }.items()
            ]))


if __name__ == "__main__":
    main()