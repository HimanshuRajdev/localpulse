import json
import os
from pathlib import Path
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from snowflake.connector.pandas_tools import write_pandas

load_dotenv()

PROCESSED_DIR = Path("data/processed")
CHUNK_SIZE    = 10_000 

def get_connection():
    return snowflake.connector.connect(
        account=os.getenv("SF_ACCOUNT"),
        user=os.getenv("SF_USER"),
        password=os.getenv("SF_PASSWORD"),
        database=os.getenv("SF_DATABASE"),
        warehouse=os.getenv("SF_WAREHOUSE"),
    )

def load_jsonl_in_chunks(filepath: Path, schema: str, table: str, conn) -> int:
    if not filepath.exists():
        print(f"Skipping {filepath.name} (not found).")
        return 0

    print(f"Loading {filepath.name} → {schema}.{table}")
    chunk, total, batches = [], 0, 0

    # Map JSON keys to Snowflake column names if they differ
    column_map = {
        "latitude": "LAT",
        "longitude": "LNG"
    }

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            chunk.append(json.loads(line))

            if len(chunk) >= CHUNK_SIZE:
                df = pd.DataFrame(chunk)
                
                # 1. Rename specific keys to match your .sql schema
                df = df.rename(columns=column_map)
                
                # 2. Filter to only include columns that exist in your SQL table
                # (Prevents error if JSON has extra fields like 'address' or 'is_open')
                if table == "YELP_BUSINESSES":
                    allowed = ["BUSINESS_ID", "NAME", "CITY", "STATE", "LAT", "LNG", "STARS", "REVIEW_COUNT", "CATEGORIES", "HOURS"]
                else:
                    allowed = ["REVIEW_ID", "BUSINESS_ID", "STARS", "TEXT", "DATE"]
                
                # 3. Force uppercase and keep only allowed columns
                df.columns = [col.upper() for col in df.columns]
                df = df[[c for c in allowed if c in df.columns]]

                write_pandas(conn, df, table, schema=schema, auto_create_table=False, 
                             overwrite=(batches == 0), quote_identifiers=False)
                
                total += len(chunk)
                batches += 1
                print(f"   batch {batches}: {total:,} rows")
                chunk = []

    if chunk:
        df = pd.DataFrame(chunk).rename(columns=column_map)
        df.columns = [col.upper() for col in df.columns]
        write_pandas(conn, df, table, schema=schema, auto_create_table=False, 
                     overwrite=(batches == 0), quote_identifiers=False)
        total += len(chunk)

    print(f"   Done. {total:,} rows loaded.\n")
    return total

def main():
    try:
        conn = get_connection()
        load_jsonl_in_chunks(PROCESSED_DIR / "yelp_businesses_filtered.jsonl", "RAW", "YELP_BUSINESSES", conn)
        load_jsonl_in_chunks(PROCESSED_DIR / "yelp_reviews_filtered.jsonl", "RAW", "YELP_REVIEWS", conn)
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    main()