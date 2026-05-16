"""
cleaner.py — Cleans and normalizes a raw DataFrame into two standard forms.

Produces:
  - df_monthly: monthly granularity with interpolation
  - df_yearly: yearly granularity, clean and gap-filled

Both have columns: date (datetime), entity (str), value (float).
"""

import re
from typing import Optional

import numpy as np
import pandas as pd

import google.generativeai as genai
from pipeline.config import GEMINI_API_KEY, GEMINI_MODEL, MIN_YEARS_REQUIRED

# Column name patterns to auto-detect roles
_DATE_PATTERNS = re.compile(
    r"^(year|date|time|period|month|quarter|day|yr)$", re.IGNORECASE
)
_ENTITY_PATTERNS = re.compile(
    r"^(country|entity|name|region|state|city|team|company|brand|"
    r"country.name|country_name|countryname|location|area|nation)$",
    re.IGNORECASE,
)
_VALUE_PATTERNS = re.compile(
    r"^(value|amount|count|total|number|quantity|rate|gdp|score|"
    r"population|deaths|cases|emissions|revenue|sales|production)$",
    re.IGNORECASE,
)

# Columns to drop — metadata, codes, footnotes
_DROP_PATTERNS = re.compile(
    r"(iso|code|indicator|id$|_id$|footnote|source|unit$|series)",
    re.IGNORECASE,
)


def clean_dataframe(
    df: pd.DataFrame, topic_info: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean a raw DataFrame into standardized monthly and yearly forms.

    Uses heuristics and Gemini (as fallback) to identify the date, entity,
    and value columns from arbitrarily structured data.

    Args:
        df: Raw DataFrame from the fetcher.
        topic_info: Dict with 'topic', 'description', 'source', etc.

    Returns:
        Tuple of (df_monthly, df_yearly) both with columns [date, entity, value].

    Raises:
        RuntimeError: If the data can't be cleaned or has fewer than MIN_YEARS_REQUIRED
                      years of data.
    """
    print(f"[cleaner] Raw shape: {df.shape}")
    print(f"[cleaner] Columns: {list(df.columns)}")

    # --- Step 1: Try to detect column roles automatically ---
    date_col, entity_col, value_col = _auto_detect_columns(df)

    # --- Step 2: If auto-detection fails, ask Gemini ---
    if not all([date_col, entity_col, value_col]):
        print("[cleaner] Auto-detection incomplete, asking Gemini for column mapping...")
        date_col, entity_col, value_col = _gemini_detect_columns(df, topic_info)

    print(f"[cleaner] Column mapping: date={date_col}, entity={entity_col}, value={value_col}")

    # --- Step 3: Handle wide-format data (years as column headers) ---
    if date_col == "__WIDE_FORMAT__":
        df = _melt_wide_format(df, entity_col, value_col)
        date_col = "date"
        entity_col = "entity"
        value_col = "value"

    # --- Step 4: Build the standardized DataFrame ---
    df_clean = pd.DataFrame()

    # Parse dates
    df_clean["date"] = _parse_dates(df[date_col])

    # Entity names
    df_clean["entity"] = df[entity_col].astype(str).str.strip()

    # Numeric values
    df_clean["value"] = pd.to_numeric(df[value_col], errors="coerce")

    # --- Step 5: Drop nulls and duplicates ---
    df_clean = df_clean.dropna(subset=["date", "entity", "value"])
    df_clean = df_clean.drop_duplicates()

    # --- Step 6: Normalize entity names ---
    df_clean["entity"] = df_clean["entity"].apply(_normalize_entity)

    # Remove entities that are aggregates (e.g. "World", "Global")
    aggregate_names = {
        "world", "global", "total", "all", "aggregate", "sum",
        "international", "other", "unknown", "unspecified",
    }
    df_clean = df_clean[~df_clean["entity"].str.lower().isin(aggregate_names)]

    # --- Step 7: Validate data span ---
    year_min = df_clean["date"].dt.year.min()
    year_max = df_clean["date"].dt.year.max()
    span = year_max - year_min

    if span < MIN_YEARS_REQUIRED:
        raise RuntimeError(
            f"Dataset spans only {span} years ({year_min}–{year_max}). "
            f"Minimum required: {MIN_YEARS_REQUIRED} years."
        )

    print(f"[cleaner] Date range: {year_min} to {year_max} ({span} years)")
    print(f"[cleaner] Entities: {df_clean['entity'].nunique()}")
    print(f"[cleaner] Clean rows: {len(df_clean)}")

    # --- Step 8: Build yearly and monthly versions ---
    df_yearly = _build_yearly(df_clean)
    df_monthly = _build_monthly(df_clean, df_yearly)

    print(f"[cleaner] Yearly shape: {df_yearly.shape}")
    print(f"[cleaner] Monthly shape: {df_monthly.shape}")

    return df_monthly, df_yearly


def _auto_detect_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Try to auto-detect date, entity, and value columns from column names.

    Returns:
        (date_col, entity_col, value_col) — any may be None if not detected.
    """
    date_col = entity_col = value_col = None

    # Check if this is wide-format (years as column headers)
    numeric_cols = [c for c in df.columns if re.match(r"^\d{4}$", str(c).strip())]
    if len(numeric_cols) >= 5:
        # Wide format: columns are years
        # Find the entity column
        for col in df.columns:
            if col not in numeric_cols and _ENTITY_PATTERNS.match(str(col)):
                entity_col = col
                break
        if not entity_col:
            # Pick the first string column
            for col in df.columns:
                if col not in numeric_cols and df[col].dtype == object:
                    entity_col = col
                    break
        return "__WIDE_FORMAT__", entity_col, None

    for col in df.columns:
        col_str = str(col).strip()
        if not date_col and _DATE_PATTERNS.match(col_str):
            date_col = col
        elif not entity_col and _ENTITY_PATTERNS.match(col_str):
            entity_col = col
        elif not value_col and _VALUE_PATTERNS.match(col_str):
            value_col = col

    # If value column not found, try to find the first numeric column that isn't year
    if not value_col:
        for col in df.columns:
            if col == date_col or col == entity_col:
                continue
            if _DROP_PATTERNS.search(str(col)):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                value_col = col
                break

    return date_col, entity_col, value_col


def _gemini_detect_columns(
    df: pd.DataFrame, topic_info: dict
) -> tuple[str, str, str]:
    """Use Gemini to identify which columns are date, entity, and value.

    Args:
        df: Raw DataFrame.
        topic_info: Topic metadata for context.

    Returns:
        (date_col, entity_col, value_col)

    Raises:
        RuntimeError: If Gemini can't determine column roles.
    """
    import json

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    sample = df.head(5).to_string()
    columns = list(df.columns)
    dtypes = {str(c): str(df[c].dtype) for c in df.columns}

    prompt = f"""I have a dataset about: {topic_info.get('topic', 'unknown')}
Source: {topic_info.get('source', 'unknown')}

Columns: {columns}
Data types: {dtypes}

Sample rows:
{sample}

Identify which column is:
1. The DATE/TIME column (contains years, dates, or time periods)
2. The ENTITY column (contains country names, company names, team names, etc.)
3. The VALUE column (contains the numeric measurement being tracked)

If the data is in WIDE FORMAT (years are column headers), set date_col to "__WIDE_FORMAT__" and value_col to null.

Return ONLY a JSON object:
{{"date_col": "column_name", "entity_col": "column_name", "value_col": "column_name"}}
"""

    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            result = json.loads(raw)
            return result["date_col"], result["entity_col"], result.get("value_col")
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Gemini column detection failed after {max_retries} attempts: {e}") from e
            print(f"[cleaner] API error or rate limit: {e}. Waiting 30s (Attempt {attempt+1}/{max_retries})...")
            time.sleep(30)


def _melt_wide_format(
    df: pd.DataFrame, entity_col: Optional[str], value_col: Optional[str]
) -> pd.DataFrame:
    """Convert wide-format data (years as column headers) to long format.

    Args:
        df: Wide-format DataFrame.
        entity_col: Column containing entity names.
        value_col: Ignored for wide format.

    Returns:
        Long-format DataFrame with columns [entity, date, value].
    """
    year_cols = [c for c in df.columns if re.match(r"^\d{4}$", str(c).strip())]

    if not entity_col:
        # Try first string column
        for col in df.columns:
            if col not in year_cols and df[col].dtype == object:
                entity_col = col
                break

    if not entity_col:
        raise RuntimeError("Cannot find entity column in wide-format data.")

    melted = df.melt(
        id_vars=[entity_col],
        value_vars=year_cols,
        var_name="date",
        value_name="value",
    )
    melted = melted.rename(columns={entity_col: "entity"})
    melted["date"] = pd.to_datetime(melted["date"].astype(str), format="%Y")
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")

    return melted


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse a column into datetime, handling various formats.

    Args:
        series: Raw date column.

    Returns:
        Parsed datetime Series.
    """
    # If it's already datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    # If it looks like pure years (e.g. 1960, 2023)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0:
        valid = numeric.dropna()
        if valid.min() >= 1800 and valid.max() <= 2100:
            return pd.to_datetime(numeric.astype(int).astype(str), format="%Y", errors="coerce")

    # General date parsing
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)


def _normalize_entity(name: str) -> str:
    """Normalize entity names for consistency.

    Args:
        name: Raw entity name string.

    Returns:
        Cleaned entity name.
    """
    # Common country name fixes
    replacements = {
        "United States of America": "United States",
        "USA": "United States",
        "US": "United States",
        "UK": "United Kingdom",
        "Great Britain": "United Kingdom",
        "Russian Federation": "Russia",
        "Republic of Korea": "South Korea",
        "Korea, Rep.": "South Korea",
        "Korea, Dem. People's Rep.": "North Korea",
        "Iran, Islamic Rep.": "Iran",
        "Viet Nam": "Vietnam",
        "Türkiye": "Turkey",
        "Czechia": "Czech Republic",
        "Lao PDR": "Laos",
        "Congo, Dem. Rep.": "DR Congo",
        "Egypt, Arab Rep.": "Egypt",
        "Venezuela, RB": "Venezuela",
        "Syrian Arab Republic": "Syria",
        "Yemen, Rep.": "Yemen",
    }

    name = name.strip()
    return replacements.get(name, name)


def _build_yearly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate data to yearly granularity.

    Args:
        df: Cleaned DataFrame with date, entity, value columns.

    Returns:
        Yearly DataFrame with one row per entity per year.
    """
    df = df.copy()
    df["year"] = df["date"].dt.year
    yearly = df.groupby(["year", "entity"])["value"].mean().reset_index()
    yearly["date"] = pd.to_datetime(yearly["year"].astype(str), format="%Y")
    yearly = yearly.drop(columns=["year"])
    yearly = yearly.sort_values(["date", "entity"]).reset_index(drop=True)

    # Forward-fill gaps for each entity
    entities = yearly["entity"].unique()
    all_dates = pd.date_range(yearly["date"].min(), yearly["date"].max(), freq="YS")

    filled_parts = []
    for entity in entities:
        entity_data = yearly[yearly["entity"] == entity].set_index("date")
        entity_data = entity_data.reindex(all_dates)
        entity_data["entity"] = entity
        entity_data["value"] = entity_data["value"].interpolate(method="linear")
        entity_data = entity_data.dropna(subset=["value"])
        entity_data = entity_data.reset_index().rename(columns={"index": "date"})
        filled_parts.append(entity_data)

    if filled_parts:
        yearly = pd.concat(filled_parts, ignore_index=True)

    return yearly[["date", "entity", "value"]]


def _build_monthly(df: pd.DataFrame, df_yearly: pd.DataFrame) -> pd.DataFrame:
    """Build monthly data by interpolating from yearly if needed.

    Args:
        df: Cleaned DataFrame.
        df_yearly: Yearly DataFrame as fallback.

    Returns:
        Monthly DataFrame.
    """
    # Check if the raw data already has monthly or finer granularity
    date_diffs = df["date"].sort_values().diff().dropna()
    median_diff = date_diffs.median()

    if median_diff <= pd.Timedelta(days=45):
        # Data is already monthly or finer — aggregate to monthly
        df = df.copy()
        df["month"] = df["date"].dt.to_period("M")
        monthly = df.groupby(["month", "entity"])["value"].mean().reset_index()
        monthly["date"] = monthly["month"].dt.to_timestamp()
        monthly = monthly.drop(columns=["month"])
        return monthly[["date", "entity", "value"]].sort_values(["date", "entity"]).reset_index(drop=True)

    # Otherwise, interpolate from yearly to monthly
    print("[cleaner] No monthly data available — interpolating from yearly")

    entities = df_yearly["entity"].unique()
    all_months = pd.date_range(
        df_yearly["date"].min(),
        df_yearly["date"].max(),
        freq="MS",
    )

    parts = []
    for entity in entities:
        edata = df_yearly[df_yearly["entity"] == entity].set_index("date")
        edata = edata.reindex(all_months)
        edata["entity"] = entity
        edata["value"] = edata["value"].interpolate(method="linear")
        edata = edata.dropna(subset=["value"])
        edata = edata.reset_index().rename(columns={"index": "date"})
        parts.append(edata)

    if parts:
        monthly = pd.concat(parts, ignore_index=True)
    else:
        monthly = pd.DataFrame(columns=["date", "entity", "value"])

    return monthly[["date", "entity", "value"]]
