# 📍 LocalPulse — Market Gap Analyzer

> Scan any US city and discover untapped business opportunities using unsupervised ML, NLP, and GPT-4o.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://localpulse.streamlit.app)

---

## What it does

LocalPulse takes any US city or neighborhood, scans its local business landscape via OpenStreetMap, clusters businesses using unsupervised ML, identifies genuine market gaps by cross-referencing supply, demand, and customer complaint data — then hands the findings to GPT-4o to generate 3 creative, specific business ideas.

**Not a directory. Not a map. A gap detector.**

---

## Live Demo

| Interface | URL |
|---|---|
| Search page | `https://d3mh7f3cn8l730.cloudfront.net/` |
| Results dashboard | `https://localpulse.streamlit.app` |

**Flow:**
1. Open the search page
2. Type any US city, neighborhood, or address
3. Select from the Google Maps autocomplete dropdown
4. Choose a scan radius (0.5–3km)
5. Hit Scan — results appear in ~2 minutes

---

## Architecture

```
User → search.html (AWS S3 + CloudFront)
           ↓ Google Places autocomplete
           ↓ hit Scan → redirect with lat/lng params
       app.py (Streamlit Cloud)
           ↓
    ┌──────────────────────────────────────────────┐
    │              Pipeline                         │
    │                                              │
    │  Overpass API (OpenStreetMap)                │
    │      ↓ 400–700 businesses                    │
    │  Snowflake RAW.OSM_BUSINESSES                │
    │      ↓                                       │
    │  STAGING.BUSINESSES (category mapping)       │
    │      ↓                                       │
    │  StandardScaler → UMAP → HDBSCAN            │
    │      ↓ 8–12 clusters                        │
    │  BERTopic (complaint theme extraction)       │
    │      ↓                                       │
    │  Gap Scorer (supply + demand + NLP signals)  │
    │      ↓                                       │
    │  GPT-4o (creative business ideas)           │
    │      ↓                                       │
    │  RESULTS.GAP_SCORES → Streamlit Dashboard   │
    └──────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data ingestion | OpenStreetMap Overpass API (free, no key) |
| Review data | Yelp Open Dataset (6M reviews) |
| Data warehouse | Snowflake (RAW → STAGING → FEATURES → RESULTS) |
| Dimensionality reduction | UMAP |
| Clustering | HDBSCAN |
| Topic modelling | BERTopic |
| Gap scoring | Custom 4-signal scorer (supply, demand, NLP, spatial) |
| Creative ideas | GPT-4o via OpenAI API |
| Frontend search | Google Maps Places Autocomplete |
| Dashboard | Streamlit |
| Search hosting | AWS S3 + CloudFront |
| App hosting | Streamlit Community Cloud |

---

## ML Pipeline Detail

### 1. Feature Engineering
Each OSM business gets 4 features:
- `rating_norm` — normalised star rating (0–1)
- `review_log` — log-transformed review count (demand proxy)
- `avg_sentiment` — Yelp review sentiment from BERT (via Snowflake Cortex)
- `negative_ratio` — proportion of negative reviews per category

### 2. UMAP + HDBSCAN
Businesses are reduced to 5 dimensions then clustered. HDBSCAN finds variable-density clusters with no noise assignment forced — typical output: 8–12 clusters, 0% noise.

**Evaluation results (Madison, WI):**
```
Silhouette score   : 0.94  (excellent, target > 0.35)
Davies-Bouldin     : 0.13  (lower is better)
Noise ratio        : 0.0%
```

### 3. Gap Scoring
Each cluster is scored across 4 specificity signals:

```
opportunity_score = 0.35 × supply_gap
                  + 0.35 × demand_proxy
                  + 0.30 × complaint_signal
```

| Signal | Source | Meaning |
|---|---|---|
| `supply_gap` | OSM business count vs category median | How absent this category is |
| `demand_proxy` | Yelp review volume × rating | Proven customer interest |
| `complaint_signal` | BERTopic on negative reviews | Unmet need from existing customers |
| `subcategory` | OSM amenity/shop tags | Specific business type missing |
| `nearest_competitor_km` | Haversine distance | Physical gap size |
| `hours_gap` | OSM opening_hours parser | Temporal gap (e.g. no late night) |
| `missing_price_tier` | Yelp price attributes | Budget/luxury gap |

---

## Snowflake Schema

```
LOCALPULSE
├── RAW
│   ├── YELP_BUSINESSES     (6M businesses)
│   ├── YELP_REVIEWS        (filtered negative/neutral)
│   └── OSM_BUSINESSES      (latest scan only — truncated per scan)
├── STAGING
│   ├── CATEGORY_MAP        (OSM tag → unified category)
│   ├── CATEGORY_MEDIANS    (Yelp benchmarks per category)
│   ├── BUSINESSES          (OSM + Yelp unified, with scan_id)
│   ├── REVIEWS             (filtered for NLP)
│   └── TILE_DENSITY        (geohash6 demand signals)
├── FEATURES
│   ├── REVIEW_SENTIMENT    (BERT scores)
│   ├── COMPLAINTS          (negative review extraction)
│   └── NLP_SIGNALS         (per-category aggregates)
└── RESULTS
    ├── CLUSTER_ASSIGNMENTS (HDBSCAN output)
    └── GAP_SCORES          (final ranked opportunities)
```

---

## Local Setup

### Prerequisites
- Python 3.11+
- Snowflake account with LOCALPULSE database
- OpenAI API key (GPT-4o)
- Google Maps API key (Maps JS + Places APIs enabled)

### Installation

```bash
git clone https://github.com/yourusername/localpulse
cd localpulse

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### Configuration

```bash
# copy and fill in your credentials
cp .env.example .env
cp config.example.js config.js
```

`.env`:
```
SF_ACCOUNT=your_snowflake_account
SF_USER=your_username
SF_PASSWORD=your_password
SF_DATABASE=LOCALPULSE
SF_WAREHOUSE=COMPUTE_WH
OPENAI_API_KEY=sk-...
GOOGLE_MAPS_KEY=your_key
```

`config.js`:
```js
window.LOCALPULSE_CONFIG = {
    GOOGLE_MAPS_KEY: "your_key",
    STREAMLIT_URL:   "http://localhost:8501",
};
```

### Database Setup

Run the full schema in Snowflake:
```sql
-- paste and run snowflake_schema.sql in a Snowflake worksheet
```

Load Yelp data (one-time):
```bash
python -m src.ingestion.filter_yelp
python -m src.ingestion.load_yelp_to_snowflake
```

### Running Locally

```bash
# Terminal 1 — Streamlit results dashboard
streamlit run app.py

# Terminal 2 — static file server for search page
python -m http.server 8502

# Open search page
# http://localhost:8502/search.html
```

### Manual Pipeline Run

```bash
# scan a city directly
python -m src.ingestion.overpass_client --lat 43.0731 --lng -89.4012 --name "Madison, WI" --radius 1.5

# run ML pipeline
python -m src.models.cluster --city "Madison, WI" --evaluate

# run with LLM explanations
python -m src.models.cluster --city "Madison, WI" --evaluate --explain
```

---

## Project Structure

```
localpulse/
├── app.py                        # Streamlit dashboard
├── search.html                   # Google Places search page
├── config.js                     # local secrets (gitignored)
├── config.example.js             # template — commit this
├── config.yaml                   # non-secret settings
├── snowflake_schema.sql          # full schema — run once
├── requirements.txt
├── setup.py
├── .env.example
└── src/
    ├── ingestion/
    │   ├── overpass_client.py    # OSM business scanner
    │   ├── filter_yelp.py        # Yelp dataset filter
    │   └── load_yelp_to_snowflake.py
    └── models/
        ├── cluster.py            # UMAP → HDBSCAN → BERTopic pipeline
        ├── gap_scorer.py         # 4-signal opportunity scorer
        ├── explainer.py          # Snowflake Cortex enrichment
        └── creative_advisor.py   # GPT-4o business idea generator
```

---

## Deployment

| Component | Service |
|---|---|
| `search.html` + `config.js` | AWS S3 + CloudFront |
| `app.py` | Streamlit Community Cloud |
| Database | Snowflake (free trial or standard) |

See `DEPLOYMENT.md` for step-by-step instructions.

---

## Cost Estimate (production)

| Service | Cost |
|---|---|
| Streamlit Cloud | Free |
| AWS S3 + CloudFront | ~$0.10/month |
| OpenAI GPT-4o | ~$0.01 per scan |
| Snowflake | Free trial / ~$5–15/month standard |
| Google Maps API | Free (under 10k requests/month) |

---

## Data Sources

- **OpenStreetMap** via Overpass API — business locations, categories, opening hours
- **Yelp Open Dataset** — 6.9M reviews, 150k businesses across US/Canada (academic use)
- **Nominatim** — free geocoding for reverse location lookup

---

## Author

Built by [Himanshu Rajdev] · [linkedin.com/in/himanshu-rajdev-786043271](https://www.linkedin.com/in/himanshu-rajdev-786043271/) · [github.com/HimanshuRajdev](https://github.com/HimanshuRajdev)