# LocalPulse

Pick any neighborhood and LocalPulse tells you which kinds of local businesses are missing there, how strong the case is, and what someone could realistically open to fill the gap.

**Live app:** https://solu375zu4mfkt4yowl4ej62ma0zjwej.lambda-url.us-east-1.on.aws/

## What it does

1. **Maps the supply.** Every named business within 0.5 to 3 km of the point you pick, pulled live from OpenStreetMap and sorted into 14 categories (restaurants, cafes, medical, fitness, pet services and so on).
2. **Compares it to its surroundings.** The same OpenStreetMap categories are counted across the wider 4 to 6 km area. A category is a gap when its share of local businesses is much smaller than its share around it (a location quotient, standard in economic geography). Using the same source on both sides means mapping quirks cancel out, and a downtown is judged against its own city.
3. **Counts the people it serves.** 2020 Census population for the tracts around the point gives residents per business ("1 vet per 18,000 residents").
4. **Scores every category.** HDBSCAN segments the local market by rating, review sentiment, complaint rate and density, and the thinnest segment sharpens the supply signal.
5. **Pressure-tests ideas.** An LLM first rules out gaps with an obvious structural reason (no car washes where nobody owns a car), then proposes three businesses for what's left, each with a named customer, a location, price math and a first step under $500.

## How the score works

```
opportunity = 0.65 × short_supply + 0.35 × demand
```

| Signal | What it measures | Source |
|---|---|---|
| `short_supply` | How far below the expected count this category is, where expected = local business total × the category's share in the surrounding area. Zero when the area already has its share. | OSM scan vs OSM counts for the surrounding 4 to 6 km |
| `demand` | How much people use this kind of business (national review volume) and how many residents live here | Yelp review counts, 2020 Census |

Complaint rates from 1.8M Yelp reviews are the same everywhere for a category, so they aren't used for ranking. They're shown as context and passed to the LLM. If the surrounding-area count fails, the app falls back to the Yelp category mix for areas of the same density and says so on the page.

Each gap also comes with specifics: which subtype is missing entirely (e.g. no walk-in clinic), the nearest existing competitor, holes in opening hours (nothing open after 7pm, closed weekends) and the most distinctive complaint phrases for that category, found with class-based TF-IDF over sampled negative reviews.

## Architecture

```
Browser ──► Lambda Function URL (one link)
              │
              ├─ GET  /              single-page app (HTML, Leaflet map, Photon search)
              ├─ GET  /api/scan      three lookups in parallel ─► features ─► HDBSCAN ─► gap scores
              │                        ├─ businesses: Geoapify Places API, Overpass as fallback
              │                        ├─ surrounding-area counts: one Overpass count query
              │                        └─ residents: US Census geocoder (2020 tracts)
              └─ POST /api/ideas     OpenAI (gpt-4o by default)

Yelp benchmarks: built offline once ─► backend/localpulse/data/category_stats.json (6 KB)
```

There is no database and no always-on server. The Yelp-derived numbers never change between requests, so they are computed once and shipped inside the function. That is what took the first version from minutes per scan down to a couple of seconds.

Hosting is one AWS Lambda function (container image, 1 GB memory) with a Function URL. A scheduled ping every 5 minutes keeps one instance warm, so the app never "sleeps". Everything fits inside AWS's always-free Lambda allowance (1M requests and 400,000 GB-seconds a month).

## Honest limitations

- OpenStreetMap coverage varies. A neighborhood with poorly mapped shops will look emptier than it is.
- The location quotient says an area has less of something than its surroundings, not that the surroundings are well served. Residents are estimated from census tracts and ignore daytime workers and visitors.
- Category features come from Yelp averages per category, so businesses in the same category share most of their feature values. The silhouette score is high partly for that reason and shouldn't be read as proof of rich structure.
- The LLM ideas are a starting point for thinking, not business advice.

## Run it locally

```bash
pip install -r backend/requirements.txt pytest
pytest                                   # 8 tests, no network needed
python scripts/dev_server.py             # http://localhost:8000 with live data
python scripts/dev_server.py --offline   # canned Madison data, no keys needed
```

For live scans set `GEOAPIFY_API_KEY` (free at geoapify.com) or leave it unset to use Overpass. For ideas set `OPENAI_API_KEY`.

## Deploy

Every push to `main` runs the tests and deploys with AWS SAM through GitHub Actions (`.github/workflows/deploy.yml`). One-time setup:

1. In AWS IAM, create a user for deployments with programmatic access. Attach `AWSCloudFormationFullAccess`, `AWSLambda_FullAccess`, `AmazonEC2ContainerRegistryFullAccess`, `AmazonS3FullAccess`, `IAMFullAccess` and `AmazonEventBridgeSchedulerFullAccess`.
2. In the GitHub repo, go to Settings > Secrets and variables > Actions and add `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `OPENAI_API_KEY` and `GEOAPIFY_API_KEY`. Optionally add a variable `AWS_REGION` (defaults to `us-east-1`).
3. Push. The workflow summary prints the live link.

## Rebuilding the Yelp benchmarks

Only needed if you change the categories. Download the [Yelp Open Dataset](https://www.yelp.com/dataset) into `data/raw/`, then

```bash
python scripts/filter_yelp.py                 # ~10 min, streams the 5 GB review file
python scripts/build_category_stats.py aggregate
python scripts/build_category_stats.py topics
```

## Project layout

```
backend/
  handler.py              Lambda entry point and routes
  localpulse/
    categories.py         the 14 categories and OSM/Yelp mappings
    places.py, overpass.py  business data (Geoapify first, Overpass fallback)
    scoring.py            features, HDBSCAN, gap scores, hours parsing
    ideas.py              LLM prompt and call
    data/category_stats.json
  static/index.html       the whole frontend
  Dockerfile
scripts/                  offline data build + local dev server
tests/
template.yaml             AWS SAM (Lambda, Function URL, warm-up schedule, logs)
```

## History

Version 1 (March 2026) ran on Streamlit Community Cloud with a Snowflake warehouse, a separate search page on S3 and CloudFront, and Google Places for search. It needed two links, slept when idle and took a few minutes per scan. It's still in the git history.

## Author

Himanshu Rajdev · [LinkedIn](https://www.linkedin.com/in/himanshu-rajdev-786043271/) · [GitHub](https://github.com/HimanshuRajdev)
