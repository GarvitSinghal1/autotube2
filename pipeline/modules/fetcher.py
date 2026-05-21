"""
fetcher.py — Downloads and parses the dataset into a pandas DataFrame.

Handles CSV, JSON, and simple REST API responses.
"""

import io
import zipfile
from typing import Optional

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Timeout for HTTP requests (seconds)
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3


def fetch_dataset(url: str, fmt: str) -> pd.DataFrame:
    """Download a dataset from a URL and return it as a pandas DataFrame.

    Args:
        url: Direct download URL for the dataset.
        fmt: One of 'csv', 'json', or 'api'.

    Returns:
        Raw pandas DataFrame (uncleaned).

    Raises:
        RuntimeError: If the download fails, the data is empty, or parsing fails.
    """
    print(f"[fetcher] Downloading dataset from: {url}")
    print(f"[fetcher] Format: {fmt}")

    raw_data = _download(url)

    if fmt == "csv":
        df = _parse_csv(raw_data)
    elif fmt in ("json", "api"):
        df = _parse_json(raw_data, url)
    else:
        raise RuntimeError(f"Unsupported format: {fmt}")

    if df.empty:
        raise RuntimeError("Downloaded dataset is empty — no rows found.")

    print(f"[fetcher] Loaded DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[fetcher] Columns: {list(df.columns)}")

    return df


def _download(url: str) -> bytes:
    """Download raw bytes from a URL with retries.

    Args:
        url: URL to download from.

    Returns:
        Raw bytes of the response body.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "AutoTube2-Pipeline/1.0"},
                allow_redirects=True,
                verify=False,
            )
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:
            last_error = e
            print(f"[fetcher] Attempt {attempt}/{MAX_RETRIES} failed: {e}")

    raise RuntimeError(
        f"Failed to download dataset after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


def _parse_csv(raw_data: bytes) -> pd.DataFrame:
    """Parse raw bytes as CSV into a DataFrame.

    Handles ZIP files containing a single CSV, and tries multiple encodings.

    Args:
        raw_data: Raw bytes from the download.

    Returns:
        Parsed DataFrame.
    """
    # Check if it's a ZIP file
    if raw_data[:4] == b"PK\x03\x04":
        return _parse_csv_from_zip(raw_data)

    # Try UTF-8 first, then latin-1
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            text = raw_data.decode(encoding)
            # Try to detect the separator
            for sep in (",", "\t", ";"):
                try:
                    df = pd.read_csv(io.StringIO(text), sep=sep)
                    if df.shape[1] > 1:
                        return df
                except Exception:
                    continue
            # Fallback: just use comma
            return pd.read_csv(io.StringIO(text))
        except UnicodeDecodeError:
            continue

    raise RuntimeError("Failed to decode CSV file with any known encoding.")


def _parse_csv_from_zip(raw_data: bytes) -> pd.DataFrame:
    """Extract and parse the first CSV from a ZIP archive.

    Args:
        raw_data: Raw bytes of a ZIP file.

    Returns:
        Parsed DataFrame from the first CSV found in the archive.
    """
    with zipfile.ZipFile(io.BytesIO(raw_data)) as zf:
        csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_files:
            raise RuntimeError("ZIP archive contains no CSV files.")

        # Pick the largest CSV (usually the data file, not metadata)
        largest = max(csv_files, key=lambda n: zf.getinfo(n).file_size)
        print(f"[fetcher] Extracting '{largest}' from ZIP archive")

        with zf.open(largest) as f:
            return pd.read_csv(f)


def _parse_json(raw_data: bytes, url: str) -> pd.DataFrame:
    """Parse raw bytes as JSON into a DataFrame.

    Handles nested JSON structures common in APIs (e.g., World Bank API).

    Args:
        raw_data: Raw bytes from the download.
        url: Original URL (used to detect API-specific structures).

    Returns:
        Parsed DataFrame.
    """
    import json

    text = raw_data.decode("utf-8")
    data = json.loads(text)

    # World Bank API returns [metadata, data_array]
    if isinstance(data, list) and len(data) == 2 and isinstance(data[1], list):
        print("[fetcher] Detected World Bank API response format")
        return pd.json_normalize(data[1])

    # Simple JSON array
    if isinstance(data, list):
        return pd.json_normalize(data)

    # Object with a 'data' key
    if isinstance(data, dict):
        for key in ("data", "records", "results", "values", "rows"):
            if key in data and isinstance(data[key], list):
                return pd.json_normalize(data[key])
        # Try to normalize the entire dict
        return pd.json_normalize(data)

    raise RuntimeError("Could not parse JSON response into a DataFrame.")
