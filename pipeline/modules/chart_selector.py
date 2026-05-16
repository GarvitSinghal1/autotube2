"""

chart_selector.py — Picks the best chart type based on data characteristics.

Uses rules first, then Gemini as a tiebreaker.
"""

import json
import re

import pandas as pd
import google.generativeai as genai

from pipeline.config import GEMINI_API_KEY, GEMINI_MODEL, CHART_TYPES


# Known geographic entity keywords
_GEO_KEYWORDS = {
    "country", "countries", "nation", "nations", "region", "regions",
    "state", "states", "province", "continent",
}


def select_chart_type(df_yearly: pd.DataFrame, topic_info: dict) -> str:
    """Select the best chart type for the data.

    Rules:
    - More than 5 entities competing over time → bar_chart_race
    - Geographic data with country/region entities → map_animation
    - 2–5 entities with continuous values → line_chart_race
    - Single metric changing dramatically over time → area_chart
    - Two numeric variables per entity over time → bubble_chart

    Falls back to Gemini if the rules are ambiguous.

    Args:
        df_yearly: Yearly DataFrame with [date, entity, value].
        topic_info: Topic metadata.

    Returns:
        One of the CHART_TYPES strings.
    """
    n_entities = df_yearly["entity"].nunique()
    topic_lower = topic_info.get("topic", "").lower()
    desc_lower = topic_info.get("description", "").lower()

    # Check for geographic data
    is_geo = any(kw in topic_lower or kw in desc_lower for kw in _GEO_KEYWORDS)

    # Sample entity names to check if they look like countries
    sample_entities = df_yearly["entity"].unique()[:20]
    _COUNTRY_SAMPLES = {
        "united states", "china", "india", "germany", "japan", "brazil",
        "france", "united kingdom", "russia", "canada", "australia",
        "south korea", "mexico", "indonesia", "italy", "spain",
    }
    country_matches = sum(
        1 for e in sample_entities if e.lower() in _COUNTRY_SAMPLES
    )
    if country_matches >= 3:
        is_geo = True

    print(f"[chart_selector] Entities: {n_entities}, Geographic: {is_geo}")

    # Apply rules
    if n_entities == 1:
        chart_type = "area_chart"
    elif is_geo and n_entities > 10:
        chart_type = "map_animation"
    elif n_entities > 5:
        chart_type = "bar_chart_race"
    elif 2 <= n_entities <= 5:
        chart_type = "line_chart_race"
    else:
        chart_type = "bar_chart_race"  # safe default

    # Verify with Gemini if we have a tiebreaker situation
    if n_entities > 5 and is_geo:
        chart_type = _gemini_tiebreak(df_yearly, topic_info, chart_type)

    print(f"[chart_selector] Selected: {chart_type}")
    return chart_type


def _gemini_tiebreak(
    df_yearly: pd.DataFrame, topic_info: dict, default: str
) -> str:
    """Ask Gemini to break a tie between chart types.

    Args:
        df_yearly: Yearly DataFrame.
        topic_info: Topic metadata.
        default: Default chart type to fall back to.

    Returns:
        Selected chart type string.
    """
    if not GEMINI_API_KEY:
        return default

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    entities = list(df_yearly["entity"].unique()[:20])
    n_entities = df_yearly["entity"].nunique()
    years = sorted(df_yearly["date"].dt.year.unique())

    prompt = f"""I have a dataset about: {topic_info.get('topic', 'unknown')}
Number of entities: {n_entities}
Sample entities: {entities}
Year range: {years[0]} to {years[-1]}

Which chart type would create the most compelling YouTube video?
Options: {CHART_TYPES}

Rules:
- bar_chart_race: best when many entities compete for rank over time
- map_animation: best when showing geographic spread or country comparisons on a map
- line_chart_race: best for 2-5 entities with continuous values to compare trends
- area_chart: best for a single metric over time
- bubble_chart: best when there are two numeric dimensions per entity

Return ONLY the chart type name, nothing else.
"""

    try:
        response = model.generate_content(prompt)
        result = response.text.strip().lower().replace('"', "").replace("'", "")
        if result in CHART_TYPES:
            return result
    except Exception as e:
        print(f"[chart_selector] Gemini tiebreak failed: {e}")

    return default
