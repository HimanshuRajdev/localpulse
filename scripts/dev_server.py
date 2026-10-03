"""
dev_server.py

Runs the Lambda handler locally at http://localhost:8000 so the page and
API can be tried without deploying.

    python scripts/dev_server.py              # live data (needs internet)
    python scripts/dev_server.py --offline    # canned Madison data, fake ideas
"""

import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import os

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "tests")]

# load keys from .env (OPENAI_API_KEY, GEOAPIFY_API_KEY) without extra packages
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import handler  # noqa: E402

if "--offline" in sys.argv:
    from fixture_madison import elements
    from localpulse.overpass import parse

    from fixture_madison import REGION, RESIDENTS

    handler.fetch_businesses = lambda lat, lng, r: (parse(elements()), "offline-fixture")
    handler.region_counts = lambda lat, lng, r: REGION
    handler.census_residents = lambda lat, lng, r: RESIDENTS
    handler.generate_ideas = lambda gaps, ctx: {"model": "offline", "kept": [], "ruled_out": [
        {"category": gaps[0]["label"], "reason": "Offline placeholder reason."}], "ideas": [{
        "title": f"Example idea {i} for {ctx['place']}", "format": "storefront",
        "category": gaps[1]["label"], "customer": "Placeholder.", "where": "Placeholder.",
        "description": "Offline placeholder.", "price_usd": 15, "unit": "per class",
        "units_per_month": 320, "monthly_revenue_usd": 4800, "monthly_rent_usd": 2400, "rent_share": 0.5,
        "why_existing_options_fall_short": "Placeholder.", "honest_risks": "Placeholder.",
        "first_step": "Placeholder."} for i in (1, 2, 3)]}


class Dev(BaseHTTPRequestHandler):
    def _run(self, method: str):
        url = urlsplit(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        event = {
            "rawPath": url.path,
            "queryStringParameters": dict(parse_qsl(url.query)) or None,
            "requestContext": {"http": {"method": method}},
            "body": self.rfile.read(length).decode() if length else None,
            "isBase64Encoded": False,
        }
        res = handler.lambda_handler(event)
        body = res["body"].encode()
        self.send_response(res["statusCode"])
        for k, v in res["headers"].items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._run("GET")

    def do_POST(self):
        self._run("POST")


if __name__ == "__main__":
    print("LocalPulse dev server on http://localhost:8000")
    print("  Geoapify key:", "found" if os.environ.get("GEOAPIFY_API_KEY") else "missing (using slower Overpass)")
    print("  OpenAI key:  ", "found" if os.environ.get("OPENAI_API_KEY") else "missing (ideas button won't work)")
    ThreadingHTTPServer(("127.0.0.1", 8000), Dev).serve_forever()
