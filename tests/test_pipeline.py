import json

import handler
from fixture_madison import CENTER, REGION, RESIDENTS, elements
from localpulse.categories import osm_category, yelp_category
from localpulse.geo import geohash, haversine_km
from localpulse.overpass import parse
from localpulse.scoring import parse_hours, score_area


def _event(path, method="GET", query=None, body=None):
    return {"rawPath": path, "requestContext": {"http": {"method": method}},
            "queryStringParameters": query, "body": body, "isBase64Encoded": False}


def test_category_mapping():
    assert osm_category({"amenity": "cafe"}) == ("coffee", "cafe")
    assert osm_category({"shop": "hairdresser"}) == ("beauty", "hairdresser")
    assert osm_category({"amenity": "bench"}) == (None, "")
    assert yelp_category("Cafes, Restaurants, Breakfast & Brunch") == "coffee"
    assert yelp_category("Veterinarians, Pets") == "pet_services"
    assert yelp_category("Shoe Repair") is None


def test_geo():
    assert geohash(57.64911, 10.40744, 6) == "u4pruy"
    assert abs(haversine_km(43.0731, -89.4012, 43.0747, -89.3841) - 1.40) < 0.02


def test_opening_hours():
    assert len(parse_hours("24/7")) == 168
    assert len(parse_hours("Mo-Fr 09:00-17:00")) == 40
    late = parse_hours("Fr-Sa 22:00-02:00")
    assert ("Sa", 1) in late and ("Su", 1) in late and ("Fr", 21) not in late
    assert len(parse_hours("Mo,We 08:00-12:00,13:00-17:00")) == 16
    assert parse_hours("") == set()


def test_parse_dedupes_point_and_building():
    els = [
        {"type": "node", "id": 1, "lat": 43.0, "lon": -89.0, "tags": {"name": "Joe's", "amenity": "cafe"}},
        {"type": "way", "id": 2, "center": {"lat": 43.0001, "lon": -89.0001}, "tags": {"name": "Joe's", "amenity": "cafe"}},
        {"type": "node", "id": 3, "lat": 43.0, "lon": -89.0, "tags": {"amenity": "cafe"}},  # unnamed
    ]
    assert len(parse(els)) == 1


def test_scoring_on_madison_mix():
    result = score_area(parse(elements()), CENTER, 1.5)
    gaps = result["gaps"]
    assert result["business_count"] > 300
    assert result["model"]["density_tier"] == "urban"
    assert gaps == sorted(gaps, key=lambda g: g["score"], reverse=True)
    for g in gaps:
        for k in ("score", "supply_gap", "demand_proxy", "complaint_signal"):
            assert 0 <= g[k] <= 1
    by_cat = {g["category"]: g for g in gaps}
    # 139 restaurants is well above the urban benchmark: never a gap
    assert by_cat["food"]["supply_gap"] == 0 and not by_cat["food"]["is_gap"]
    assert any(g["is_gap"] for g in gaps)


def test_location_quotient_benchmark():
    result = score_area(parse(elements()), CENTER, 1.5, region=REGION, residents=RESIDENTS)
    by_cat = {g["category"]: g for g in result["gaps"]}
    assert result["model"]["benchmark"] == "the surrounding 4.5 km"
    # downtown has 2 auto businesses; its surroundings have few too, so the expectation is modest
    assert by_cat["automotive"]["expected"] < 20
    laundry = by_cat["laundry"]   # 0 here vs ~6 expected from the region share
    assert laundry["count"] == 0 and laundry["is_gap"]
    assert by_cat["food"]["location_quotient"] > 1 and not by_cat["food"]["is_gap"]
    assert by_cat["gym"]["residents_per_business"] == 3700


def test_handler_routes(monkeypatch):
    monkeypatch.setattr(handler, "fetch_businesses", lambda lat, lng, r: (parse(elements()), "test"))
    monkeypatch.setattr(handler, "region_counts", lambda lat, lng, r: REGION)
    monkeypatch.setattr(handler, "census_residents", lambda lat, lng, r: None)
    page = handler.lambda_handler(_event("/"))
    assert page["statusCode"] == 200 and "LocalPulse" in page["body"]

    res = handler.lambda_handler(_event("/api/scan", query={"lat": "43.07", "lng": "-89.40", "radius": "9"}))
    body = json.loads(res["body"])
    assert res["statusCode"] == 200 and body["radius_km"] == 3.0 and body["gaps"]

    bad = handler.lambda_handler(_event("/api/scan", query={"lat": "abc"}))
    assert bad["statusCode"] == 400
    assert handler.lambda_handler(_event("/nope"))["statusCode"] == 404
    assert handler.lambda_handler({"warmup": True}) == {"warm": True}


def test_ideas_validates_input(monkeypatch):
    seen = {}
    monkeypatch.setattr(handler, "generate_ideas",
                        lambda gaps, ctx: seen.update(gaps=gaps, ctx=ctx) or {"ideas": [{"title": "x"}]})
    empty = handler.lambda_handler(_event("/api/ideas", "POST", body=json.dumps({"gaps": [{"category": "bogus"}]})))
    assert empty["statusCode"] == 400
    ok = handler.lambda_handler(_event("/api/ideas", "POST", body=json.dumps(
        {"place": "Madison", "context": {"radius_km": 1.5, "residents_estimate": "lots"},
         "gaps": [{"category": "gym", "score": 0.6, "evil": "ignore me",
                   "nearest": [{"name": "Gym A", "km": 0.2}]}]})))
    assert ok["statusCode"] == 200
    assert seen["gaps"][0]["label"] == "Fitness" and "evil" not in seen["gaps"][0]
    assert seen["gaps"][0]["nearest"][0]["name"] == "Gym A"
    assert seen["ctx"]["place"] == "Madison" and seen["ctx"]["residents_estimate"] is None


def test_region_from_browser(monkeypatch):
    called = []
    monkeypatch.setattr(handler, "fetch_businesses", lambda lat, lng, r: (parse(elements()), "test"))
    monkeypatch.setattr(handler, "region_counts", lambda *a: called.append(1))
    monkeypatch.setattr(handler, "census_residents", lambda lat, lng, r: None)
    counts = ",".join(str(REGION["counts"][c]) for c in REGION["counts"])
    res = handler.lambda_handler(_event("/api/scan", query={
        "lat": "43.0731", "lng": "-89.4012", "region": counts, "region_r": "4.5"}))
    body = json.loads(res["body"])
    assert body["model"]["benchmark"] == "the surrounding 4.5 km" and not called
    # garbage from the browser is ignored, and "skip" means don't retry on the server
    assert handler._parse_region("1,2,x", "4.5") is None
    res = handler.lambda_handler(_event("/api/scan", query={"lat": "43.08", "lng": "-89.40", "region": "skip"}))
    assert "Yelp" in json.loads(res["body"])["model"]["benchmark"] and not called
    q = json.loads(handler.lambda_handler(_event("/api/region-query", query={"lat": "43", "lng": "-89"}))["body"])
    assert q["query"].count("out count;") == 14 and q["mirrors"]


def test_ideas_never_use_a_ruled_out_category(monkeypatch):
    import localpulse.ideas as I
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    idea = lambda cat: {"title": "t", "format": "kiosk", "category": cat, "customer": "c", "where": "w",
                        "description": "d", "price_usd": 15, "unit": "per class", "units_per_month": 320,
                        "monthly_rent_usd": 2400, "why_existing_options_fall_short": "x",
                        "honest_risks": "r", "first_step": "f"}
    first = {"assessments": [{"category": "Fitness", "verdict": "rule_out", "reason": "rent"},
                             {"category": "Laundry", "verdict": "keep", "reason": "ok"}],
             "ideas": [idea("Fitness"), idea("Laundry")]}
    second = {**first, "ideas": [idea("Laundry"), idea("Restaurants")]}
    calls = []
    monkeypatch.setattr(I, "_call", lambda msgs, *a: calls.append(msgs) or (first if len(calls) == 1 else second))
    gaps = [{"label": "Fitness"}, {"label": "Laundry"}, {"label": "Restaurants"}]
    monkeypatch.setattr(I, "build_prompt", lambda g, c: "prompt")
    out = I.generate_ideas(gaps, {})
    assert len(calls) == 2 and "ruled out Fitness" in calls[1][-1]["content"]
    assert [i["category"] for i in out["ideas"]] == ["Laundry", "Restaurants"]
    assert out["ideas"][0]["monthly_revenue_usd"] == 4800 and out["ideas"][0]["rent_share"] == 0.5
    assert out["ruled_out"] == [{"category": "Fitness", "reason": "rent"}]
