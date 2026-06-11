"""
cleaner.py — Cleans and normalizes a raw DataFrame into two standard forms.

Produces:
  - df_monthly: monthly granularity with interpolation
  - df_yearly: yearly granularity, clean and gap-filled

Both have columns: date (datetime), entity (str), value (float).
"""

import re
import json
import time
from typing import Optional

import numpy as np
import pandas as pd

from google import genai
from google.genai import types
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

    # --- Step 1: Try to detect column roles from topic_info or automatically ---
    date_col = topic_info.get("date_col")
    entity_col = topic_info.get("entity_col")
    value_col = topic_info.get("value_col")

    if not all([date_col, entity_col, value_col]):
        date_col_auto, entity_col_auto, value_col_auto = auto_detect_columns(df)
        if not date_col:
            date_col = date_col_auto
        if not entity_col:
            entity_col = entity_col_auto
        if not value_col:
            value_col = value_col_auto

    # --- Step 2: If detection still incomplete, ask Gemini ---
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

    # --- Step 3.5: Handle wide-format data (entities as column headers) ---
    if entity_col and date_col and date_col != "__WIDE_FORMAT__":
        if df[entity_col].nunique() <= 1:
            value_cols = [c for c in df.columns if c not in (date_col, entity_col) and str(c).lower() not in ("code", "continent", "index", "id")]
            if len(value_cols) > 1:
                print(f"[cleaner] Detected entities-as-columns wide format. Melting {len(value_cols)} columns.")
                if entity_col in df.columns:
                    df = df.drop(columns=[entity_col])
                df = df.melt(id_vars=[date_col], value_vars=value_cols, var_name="entity", value_name="value")
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

    # Clean cryptic entity names using Gemini mapping
    unique_entities = list(df_clean["entity"].unique())
    entity_mapping = _clean_all_entity_names_with_gemini(unique_entities, topic_info)
    df_clean["entity"] = df_clean["entity"].map(entity_mapping)

    # Detect dataset units using suggested units or fall back to Gemini
    suggested_full = topic_info.get("suggested_full_unit", "").strip()
    suggested_short = topic_info.get("suggested_short_unit", "").strip()

    if suggested_full and suggested_short:
        topic_info["full_unit"] = suggested_full
        topic_info["short_unit"] = suggested_short
        print(f"[cleaner] Using suggested units: full='{suggested_full}', short='{suggested_short}'")
    else:
        unit_info = _gemini_detect_unit(df, topic_info)
        topic_info["full_unit"] = unit_info.get("full_unit", "")
        topic_info["short_unit"] = unit_info.get("short_unit", "")
    print(f"[cleaner] Final units: full='{topic_info['full_unit']}', short='{topic_info['short_unit']}'")

    # Remove entities that are aggregates (e.g. "World", "Global", continents, regions)
    aggregate_names = {
        "world", "global", "total", "all", "aggregate", "sum",
        "international", "other", "unknown", "unspecified",
        # continents
        "africa", "asia", "europe", "north america", "south america", "oceania", "antarctica",
        # regional / global groups
        "european union", "americas", "european region", "western pacific", "south-east asia",
        "eastern mediterranean", "pan america", "latin america & caribbean", "sub-saharan africa",
        "east asia & pacific", "europe & central asia", "middle east & north africa", "south asia",
        "latin america", "caribbean", "middle east", "north africa", "central america", "western europe",
        "eastern europe", "southern europe", "northern europe", "northern america", "high income",
        "low income", "lower middle income", "upper middle income", "high-income", "low-income",
        "lower-middle-income", "upper-middle-income", "g20", "oecd", "asean", "eu", "brics",
        "east asia and pacific", "europe and central asia", "latin america and caribbean",
        "middle east and north africa", "sub-saharan africa", "high-income countries",
        "low-income countries", "lower-middle-income countries", "upper-middle-income countries",
        "world bank", "imf", "un", "unesco", "who", "wto", "nato", "unicef", "undp", "unhcr",
    }
    df_clean = df_clean[~df_clean["entity"].str.lower().str.strip().isin(aggregate_names)]

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


def auto_detect_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str], Optional[str]]:
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
    from pipeline.modules.gemini_helper import build_gemini_client
    client = build_gemini_client()

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

    from pipeline.modules.gemini_helper import generate_content_with_retry
    max_json_retries = 3
    for attempt in range(max_json_retries):
        try:
            response = generate_content_with_retry(
                client=client,
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                ),
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            result = json.loads(raw)
            return result.get("date_col"), result.get("entity_col"), result.get("value_col")
        except json.JSONDecodeError as e:
            if attempt == max_json_retries - 1:
                raise RuntimeError(f"Gemini column detection invalid JSON after {max_json_retries} attempts: {e}") from e
            print(f"[cleaner] Invalid JSON: {e}. Retrying JSON generation (Attempt {attempt+1}/{max_json_retries})...")
            time.sleep(5)
        except Exception as e:
            raise RuntimeError(f"Gemini column detection failed: {e}") from e



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
        "Democratic Republic of Congo": "DR Congo",
        "Egypt, Arab Rep.": "Egypt",
        "Venezuela, RB": "Venezuela",
        "Syrian Arab Republic": "Syria",
        "Yemen, Rep.": "Yemen",
        "Côte d'Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "Myanmar (Burma)": "Myanmar",
        "Timor-Leste": "East Timor",
        "Eswatini": "Swaziland",
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


def _clean_all_entity_names_with_gemini(entities: list[str], topic_info: dict) -> dict[str, str]:
    """Uses Gemini to clean up a list of raw entity/metric names into natural, human-readable titles."""
    if not entities:
        return {}

    needs_cleaning = False
    for name in entities:
        # Check if the name looks like a cryptic/machine-readable identifier.
        # Clean country/entity names can contain spaces, hyphens, periods, parentheses, apostrophes, and commas.
        clean_name = name.strip()
        if "_" in clean_name or clean_name.startswith("number_") or (clean_name.islower() and len(clean_name) > 3):
            needs_cleaning = True
            break

    if not needs_cleaning:
        return {e: e for e in entities}

    import json
    from pipeline.modules.gemini_helper import build_gemini_client
    client = build_gemini_client()

    prompt = f"""You are an expert data visualization editor.
Your task is to take a list of raw entity or metric names from a dataset and translate them into clean, polished, human-readable, natural English labels suitable for display on a professional bar/line chart race.

Topic: {topic_info.get('topic', 'Data Visualization')}
Description: {topic_info.get('description', '')}

Rules:
1. Translate cryptic column/variable names (e.g., "number_nuclweap_possession" -> "Possession", or "number_nuclweap_pursuit" -> "Pursuit", or "gdp_per_capita" -> "GDP per Capita") into elegant, concise labels.
2. If the entity names are already clean names (like country names), leave them exactly as they are.
3. Keep the output labels short and concise so they don't get cut off on the chart.
4. Return ONLY a JSON dictionary where the keys are the raw strings and the values are the cleaned, human-readable names. Do not return markdown, codeblocks, or any other text.

List of raw names to clean:
{json.dumps(entities, indent=2)}
"""

    from pipeline.modules.gemini_helper import generate_content_with_retry
    try:
        response = generate_content_with_retry(
            client=client,
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            )
        )
        mapping = json.loads(response.text.strip())
        if isinstance(mapping, dict):
            return {e: mapping.get(e, e) for e in entities}
    except Exception as e:
        print(f"[cleaner] Failed to clean entity names with Gemini: {e}")
        # Fallback to a basic string replacement
        fallback_mapping = {}
        for e in entities:
            clean = e.replace("_", " ").strip()
            if clean.lower().startswith("number "):
                clean = clean[7:]
            clean = re.sub(r"\bnuclweap\b", "nuclear weapons", clean, flags=re.IGNORECASE)
            fallback_mapping[e] = clean.title()
            print(f"[cleaner] Fallback mapping: {e} -> {fallback_mapping[e]}")
        return fallback_mapping


    return {e: e for e in entities}


def _gemini_detect_unit(df: pd.DataFrame, topic_info: dict) -> dict:
    """Ask Gemini to determine the unit of measurement for this dataset.

    Returns a dict with 'full_unit' and 'short_unit'.
    """
    import json
    from pipeline.modules.gemini_helper import build_gemini_client
    client = build_gemini_client()

    # Get a sample of the data and columns
    sample = df.head(5).to_string()
    columns = list(df.columns)

    prompt = f"""You are a data analyst.
We have a dataset that we are preparing for a visualization.
Your task is to identify the unit of measurement for the numeric values in this dataset.

Topic: {topic_info.get('topic', '')}
Description: {topic_info.get('description', '')}
Dataset Name: {topic_info.get('dataset_name', '')}

Columns: {columns}
Sample Data:
{sample}

Rules:
1. Identify the unit of measurement of the primary value/metric in the dataset (e.g., 'liters', 'liters per capita', 'USD', 'percentage', 'tons', 'deaths', 'people', 'kW', etc.).
2. Return a JSON object with two fields:
   - "full_unit": the complete, formal unit name (e.g., "liters of pure alcohol per capita", "number of nuclear weapons", "metric tons per capita")
   - "short_unit": a very short version (1 word or abbreviation/symbol, e.g., "liters", "weapons", "tons", "%", "$") to display next to numbers.
3. Do not explain anything. Return ONLY a valid JSON object.
"""

    from pipeline.modules.gemini_helper import generate_content_with_retry
    try:
        response = generate_content_with_retry(
            client=client,
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            )
        )
        res = json.loads(response.text.strip())
        if isinstance(res, dict):
            full_unit = res.get("full_unit", "").strip()
            short_unit = res.get("short_unit", "").strip()
            return {"full_unit": full_unit, "short_unit": short_unit}
    except Exception as e:
        print(f"[cleaner] Failed to detect unit with Gemini: {e}")
        # Fallback to empty units
        return {"full_unit": "", "short_unit": ""}

    return {"full_unit": "", "short_unit": ""}

