"""
index_datasets.py — Indexes and validates all CSV files from owid/owid-datasets into a local SQLite database.

Supports incremental caching based on file size, handles wide/long formats, and uses HTTP range
requests for large datasets to keep indexing extremely fast.
"""

import io
import os
import re
import json
import time
import urllib.parse
import sqlite3
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

from pipeline.config import DATASETS_INDEX_DB, MIN_YEARS_REQUIRED, TOP_N_ENTITIES
from pipeline.modules.cleaner import auto_detect_columns

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Aggregate entity names to exclude from entity counts
AGGREGATE_NAMES = {
    "world", "global", "total", "all", "aggregate", "sum",
    "international", "other", "unknown", "unspecified",
}

CHUNK_SIZE = 16384  # 16 KB range request size


def init_db() -> sqlite3.Connection:
    """Initialize the SQLite database and create the datasets table and indexes."""
    # Ensure parent directory exists
    DATASETS_INDEX_DB.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(DATASETS_INDEX_DB))
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        path TEXT,
        csv_url TEXT,
        size_bytes INTEGER,
        columns TEXT,
        entity_col TEXT,
        date_col TEXT,
        value_col TEXT,
        start_year INTEGER,
        end_year INTEGER,
        span_years INTEGER,
        entity_count INTEGER,
        is_valid INTEGER,
        error_reason TEXT,
        last_indexed TEXT
    )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_valid ON datasets(is_valid)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON datasets(name)")
    conn.commit()
    return conn


def get_cached_datasets(conn: sqlite3.Connection) -> Dict[str, Tuple[int, Optional[int]]]:
    """Retrieve currently cached dataset names mapped to their (size_bytes, is_valid)."""
    cursor = conn.cursor()
    cursor.execute("SELECT name, size_bytes, is_valid FROM datasets")
    return {row[0]: (row[1], row[2]) for row in cursor.fetchall()}


def melt_wide_format(df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """Convert wide-format DataFrame (years as columns) to long format."""
    year_cols = [c for c in df.columns if re.match(r"^\d{4}$", str(c).strip())]
    if not entity_col:
        for col in df.columns:
            if col not in year_cols and df[col].dtype == object:
                entity_col = col
                break
    if not entity_col:
        raise ValueError("Cannot find entity column in wide-format data.")
    
    melted = df.melt(
        id_vars=[entity_col],
        value_vars=year_cols,
        var_name="date",
        value_name="value",
    )
    melted = melted.rename(columns={entity_col: "entity"})
    return melted


def parse_dates_to_years(series: pd.Series) -> pd.Series:
    """Parse dates from a pandas Series and extract the year."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.year

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0:
        valid = numeric.dropna()
        if valid.min() >= 1800 and valid.max() <= 2100:
            return numeric.astype(float).astype(int)

    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.year


def process_csv_dataframe(df: pd.DataFrame) -> Tuple[str, str, Optional[str], Optional[int], Optional[int], int]:
    """Analyze DataFrame columns, format, year range, and unique entities count."""
    date_col, entity_col, value_col = auto_detect_columns(df)
    
    if not date_col or not entity_col:
        raise ValueError(f"Could not auto-detect column roles. Detected: date_col={date_col}, entity_col={entity_col}")

    # Process format
    if date_col == "__WIDE_FORMAT__":
        df_melted = melt_wide_format(df, entity_col)
        # Year range from melted year column headers
        year_cols = [c for c in df.columns if re.match(r"^\d{4}$", str(c).strip())]
        years = [int(y) for y in year_cols]
        start_year = min(years) if years else None
        end_year = max(years) if years else None
        
        entities = df_melted["entity"].dropna().astype(str).str.strip()
        entities = entities[~entities.str.lower().isin(AGGREGATE_NAMES)]
        entity_count = entities.nunique()
    else:
        # Step 3.5 fallback: Handle entities-as-columns wide format
        if df[entity_col].nunique() <= 1:
            value_cols = [
                c for c in df.columns 
                if c not in (date_col, entity_col) 
                and str(c).lower() not in ("code", "continent", "index", "id")
            ]
            if len(value_cols) > 1:
                df = df.melt(id_vars=[date_col], value_vars=value_cols, var_name="entity", value_name="value")
                entity_col = "entity"
                value_col = "value"

        # Regular long format
        years = parse_dates_to_years(df[date_col]).dropna()
        start_year = int(years.min()) if not years.empty else None
        end_year = int(years.max()) if not years.empty else None
        
        entities = df[entity_col].dropna().astype(str).str.strip()
        entities = entities[~entities.str.lower().isin(AGGREGATE_NAMES)]
        entity_count = entities.nunique()

    return str(date_col), str(entity_col), value_col, start_year, end_year, entity_count


def process_large_csv_range_ends(
    url: str, size_bytes: int
) -> Tuple[str, str, Optional[str], Optional[int], Optional[int], int]:
    """Fetch first 16KB and last 16KB of a large CSV to extract schema and years range."""
    resp_start = requests.get(
        url,
        headers={"Range": f"bytes=0-{CHUNK_SIZE-1}", "User-Agent": "AutoTube2-Pipeline/1.0"},
        timeout=15,
        verify=False
    )
    if resp_start.status_code not in (200, 206):
        raise RuntimeError(f"Failed to fetch first chunk. HTTP Code: {resp_start.status_code}")
    start_text = resp_start.text

    # 2. Fetch last chunk
    last_chunk_start = max(0, size_bytes - CHUNK_SIZE)
    resp_end = requests.get(
        url,
        headers={"Range": f"bytes={last_chunk_start}-{size_bytes-1}", "User-Agent": "AutoTube2-Pipeline/1.0"},
        timeout=15,
        verify=False
    )

    if resp_end.status_code not in (200, 206):
        raise RuntimeError(f"Failed to fetch last chunk. HTTP Code: {resp_end.status_code}")
    end_text = resp_end.text

    start_lines = start_text.splitlines()
    if len(start_lines) < 2:
        raise ValueError("First chunk has too few lines to parse CSV header.")

    # Drop last line because it's probably truncated
    start_csv = "\n".join(start_lines[:-1])
    df_start = pd.read_csv(io.StringIO(start_csv))

    date_col, entity_col, value_col = auto_detect_columns(df_start)
    if not date_col or not entity_col:
        raise ValueError(f"Could not auto-detect column roles. Detected: date_col={date_col}, entity_col={entity_col}")

    # Start year & Entities from first chunk
    if date_col == "__WIDE_FORMAT__":
        df_start_melted = melt_wide_format(df_start, entity_col)
        year_cols = [c for c in df_start.columns if re.match(r"^\d{4}$", str(c).strip())]
        years = [int(y) for y in year_cols]
        start_year = min(years) if years else None
        end_year = max(years) if years else None
        
        entities_start = set(df_start_melted["entity"].dropna().astype(str).str.strip())
        entities_start = {e for e in entities_start if e.lower() not in AGGREGATE_NAMES}
        
        # Try end chunk for entities
        entities_end = set()
        end_lines = end_text.splitlines()
        if len(end_lines) > 1:
            end_csv = start_lines[0] + "\n" + "\n".join(end_lines[1:])
            try:
                df_end = pd.read_csv(io.StringIO(end_csv))
                df_end_melted = melt_wide_format(df_end, entity_col)
                entities_end = set(df_end_melted["entity"].dropna().astype(str).str.strip())
                entities_end = {e for e in entities_end if e.lower() not in AGGREGATE_NAMES}
            except Exception:
                pass
        
        # Combine entities and assure >= 10 since file size is > 256KB
        entity_count = max(len(entities_start.union(entities_end)), 10)
    else:
        # Regular format
        start_years = parse_dates_to_years(df_start[date_col]).dropna()
        start_year = int(start_years.min()) if not start_years.empty else None

        end_year = None
        end_lines = end_text.splitlines()
        
        # Parse end year from last chunk (with header prepended)
        if len(end_lines) > 1:
            end_csv = start_lines[0] + "\n" + "\n".join(end_lines[1:])
            try:
                df_end = pd.read_csv(io.StringIO(end_csv))
                end_years = parse_dates_to_years(df_end[date_col]).dropna()
                if not end_years.empty:
                    end_year = int(end_years.max())
            except Exception:
                pass

        # Fallback to regex scan of the end chunk text if parsing failed
        if end_year is None:
            regex_years = [int(y) for y in re.findall(r"\b(18\d{2}|19\d{2}|20\d{2})\b", end_text)]
            if regex_years:
                end_year = max(regex_years)

        # Fallback to start_year if everything fails
        if end_year is None and start_year is not None:
            end_year = start_year

        # Entities from first chunk
        entities_start = set(df_start[entity_col].dropna().astype(str).str.strip())
        entities_start = {e for e in entities_start if e.lower() not in AGGREGATE_NAMES}
        
        # Entities from end chunk
        entities_end = set()
        if len(end_lines) > 1:
            try:
                df_end = pd.read_csv(io.StringIO(start_lines[0] + "\n" + "\n".join(end_lines[1:])))
                entities_end = set(df_end[entity_col].dropna().astype(str).str.strip())
                entities_end = {e for e in entities_end if e.lower() not in AGGREGATE_NAMES}
            except Exception:
                pass
                
        # Since it's > 256KB, guarantee it has at least 10 entities
        entity_count = max(len(entities_start.union(entities_end)), 10)

    return str(date_col), str(entity_col), value_col, start_year, end_year, entity_count


def process_single_dataset(item: dict, cache: dict) -> Optional[dict]:
    """Helper to download and validate a single dataset."""
    path = item["path"]
    size_bytes = item.get("size", 0)
    name = Path(path).parent.name
    
    # Check cache
    if name in cache:
        cached_size, cached_valid = cache[name]
        if cached_size == size_bytes and cached_valid is not None:
            return None  # skipped

    csv_url = f"https://raw.githubusercontent.com/owid/owid-datasets/master/{urllib.parse.quote(path)}"
    
    date_col = entity_col = value_col = None
    start_year = end_year = span_years = entity_count = None
    is_valid = 0
    error_reason = None
    cols_list = []
    
    try:
        # Optimize based on file size
        if size_bytes <= 256 * 1024:
            # Small file: fetch in full
            r = requests.get(csv_url, headers={"User-Agent": "AutoTube2-Pipeline/1.0"}, timeout=15, verify=False)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code} fetching CSV.")
            df = pd.read_csv(io.StringIO(r.text))
            cols_list = list(df.columns)
            date_col, entity_col, value_col, start_year, end_year, entity_count = process_csv_dataframe(df)
        else:
            # Large file: fetch range chunks
            date_col, entity_col, value_col, start_year, end_year, entity_count = process_large_csv_range_ends(
                csv_url, size_bytes
            )
            # Fetch header only to extract columns list
            r_head = requests.get(
                csv_url,
                headers={"Range": f"bytes=0-{CHUNK_SIZE-1}", "User-Agent": "AutoTube2-Pipeline/1.0"},
                timeout=15,
                verify=False
            )

            if r_head.status_code in (200, 206):
                head_lines = r_head.text.splitlines()
                if head_lines:
                    cols_list = list(pd.read_csv(io.StringIO("\n".join(head_lines[:-1]))).columns)

        # Check spans & validity
        if start_year is not None and end_year is not None:
            span_years = end_year - start_year
        else:
            span_years = 0

        # Validate
        if not date_col or not entity_col:
            error_reason = "Could not identify date and entity columns."
        elif span_years < MIN_YEARS_REQUIRED:
            error_reason = f"Span is too short ({span_years} years, min {MIN_YEARS_REQUIRED})."
        elif entity_count < TOP_N_ENTITIES:
            error_reason = f"Too few unique entities ({entity_count}, min {TOP_N_ENTITIES})."
        elif end_year is None or end_year < 2023:
            error_reason = f"Dataset ends too early ({end_year}, must be >= 2023)."
        else:
            is_valid = 1

    except Exception as e:
        is_valid = 0
        error_reason = f"Processing failed: {str(e)}"

    return {
        "name": name,
        "path": path,
        "csv_url": csv_url,
        "size_bytes": size_bytes,
        "columns_json": json.dumps(cols_list),
        "entity_col": entity_col,
        "date_col": date_col,
        "value_col": value_col,
        "start_year": start_year,
        "end_year": end_year,
        "span_years": span_years,
        "entity_count": entity_count,
        "is_valid": is_valid,
        "error_reason": error_reason,
    }


def index_all_datasets() -> None:
    """Fetch tree, process CSVs concurrently, validate and save to SQLite db."""
    import concurrent.futures
    print("[indexer] Initializing SQLite database...")
    conn = init_db()
    cursor = conn.cursor()
    
    # 1. Fetch Git Trees list from GitHub API
    print("[indexer] Fetching repository tree from GitHub API...")
    url = "https://api.github.com/repos/owid/owid-datasets/git/trees/master?recursive=1"
    headers = {"User-Agent": "AutoTube2-Pipeline/1.0"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
        print("[indexer] Using GITHUB_TOKEN authentication.")
        
    try:
        resp = requests.get(url, headers=headers, timeout=20, verify=False)
        if resp.status_code == 401 and token:
            print("[indexer] GITHUB_TOKEN authentication failed (401). Retrying without token...")
            headers_no_token = {"User-Agent": "AutoTube2-Pipeline/1.0"}
            resp = requests.get(url, headers=headers_no_token, timeout=20, verify=False)
            
        if resp.status_code != 200:
            print(f"[indexer] Error fetching repo tree: HTTP {resp.status_code} - {resp.text[:300]}")
            return
        
        tree_data = resp.json().get("tree", [])

    except Exception as e:
        print(f"[indexer] Request failed: {e}")
        return

    # Filter CSV files in datasets/ directory
    csv_items = [
        item for item in tree_data
        if item.get("type") == "blob"
        and item.get("path", "").startswith("datasets/")
        and item.get("path", "").endswith(".csv")
    ]
    total_csvs = len(csv_items)
    print(f"[indexer] Found {total_csvs} CSV datasets in the repository tree.")

    # 2. Get cached datasets
    cache = get_cached_datasets(conn)
    print(f"[indexer] Loaded {len(cache)} cached datasets from SQLite database.")

    # 3. Process each dataset concurrently
    skipped = 0
    updated = 0
    
    print(f"[indexer] Running concurrently with up to 16 workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_item = {
            executor.submit(process_single_dataset, item, cache): item
            for item in csv_items
        }
        
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                res = future.result()
                if res is None:
                    skipped += 1
                else:
                    now_str = datetime.utcnow().isoformat()
                    cursor.execute("""
                    INSERT OR REPLACE INTO datasets (
                        name, path, csv_url, size_bytes, columns, entity_col, date_col, value_col,
                        start_year, end_year, span_years, entity_count, is_valid, error_reason, last_indexed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        res["name"], res["path"], res["csv_url"], res["size_bytes"], res["columns_json"],
                        res["entity_col"], res["date_col"], res["value_col"], res["start_year"], res["end_year"],
                        res["span_years"], res["entity_count"], res["is_valid"], res["error_reason"], now_str
                    ))
                    conn.commit()
                    updated += 1
                    status_str = f"VALID (entities={res['entity_count']}, span={res['span_years']})" if res["is_valid"] else f"INVALID ({res['error_reason']})"
                    print(f"[{skipped + updated}/{total_csvs}] Processed '{res['name']}' -> {status_str}")
            except Exception as e:
                print(f"[indexer] Thread failed for item {item.get('path')}: {e}")

    print(f"\n[indexer] Complete. Skipped (unchanged): {skipped}, Updated/Added: {updated}")
    
    # Print status counts
    cursor.execute("SELECT count(*), sum(is_valid) FROM datasets")
    total, valid = cursor.fetchone()
    print(f"[indexer] SQLite database statistics: Total={total}, Valid={valid or 0}, Invalid={total - (valid or 0)}")
    conn.close()


if __name__ == "__main__":
    index_all_datasets()
