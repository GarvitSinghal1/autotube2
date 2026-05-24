"""
analyzer.py — Finds the most dramatic/extreme segment in the yearly data.

Used to determine what the YouTube Short should highlight.
"""

import numpy as np
import pandas as pd


def find_extreme_segment(df_yearly: pd.DataFrame, topic_info: dict) -> dict:
    """Analyze yearly data to find the most dramatic time window.

    "Most dramatic" is determined by:
    1. Rank changes — periods where entities swap positions the most
    2. Steepest growth/decline — absolute rate of change
    3. Overtake moments — when a lower-ranked entity surpasses a higher one

    The returned segment is constrained to 10–20 years (ideal for a 50-59s Short).

    Args:
        df_yearly: DataFrame with columns [date, entity, value].
        topic_info: Topic metadata context.

    Returns:
        Dict with keys: start_year, end_year, reason, hook

    Raises:
        RuntimeError: If analysis fails.
    """
    print("[analyzer] Finding the most dramatic segment...")

    years = sorted(df_yearly["date"].dt.year.unique())
    if len(years) < 10:
        # Use the entire range if data is short
        return {
            "start_year": int(years[0]),
            "end_year": int(years[-1]),
            "reason": "Full dataset span used (limited data)",
            "hook": f"What changed from {years[0]} to {years[-1]}?",
        }

    # --- Build rank matrix ---
    rank_data = {}
    for year in years:
        year_slice = df_yearly[df_yearly["date"].dt.year == year]
        ranked = year_slice.sort_values("value", ascending=False)
        rank_data[year] = {
            row["entity"]: rank + 1
            for rank, (_, row) in enumerate(ranked.iterrows())
        }

    # --- Strategy 1: Find the window with max total rank changes ---
    best_score = -1
    best_window = (years[0], years[-1])
    best_reason = ""

    min_window = min(10, len(years) - 1)
    max_window = min(20, len(years) - 1)

    for window_size in range(min_window, max_window + 1):
        # Force the window to end at the latest available year of the dataset
        i = len(years) - 1 - window_size
        start_yr = years[i]
        end_yr = years[-1]

        score, reason = _score_window(
            rank_data, df_yearly, start_yr, end_yr, years
        )

        if score > best_score:
            best_score = score
            best_window = (start_yr, end_yr)
            best_reason = reason

    start_year, end_year = best_window

    # --- Generate hook and metadata ---
    reasons_list = [r.strip() for r in best_reason.split(";") if r.strip()]
    hook_and_meta = _generate_hook_and_metadata(df_yearly, rank_data, start_year, end_year, topic_info, reasons_list)

    result = {
        "start_year": int(start_year),
        "end_year": int(end_year),
        "reason": best_reason,
        "hook": hook_and_meta.get("hook", ""),
        "metadata": {
            "long_form": {
                "title": hook_and_meta.get("long_title", ""),
                "description": hook_and_meta.get("long_description", ""),
                "tags": hook_and_meta.get("long_tags", [])
            },
            "short": {
                "title": hook_and_meta.get("short_title", ""),
                "description": hook_and_meta.get("short_description", ""),
                "tags": hook_and_meta.get("short_tags", [])
            }
        }
    }

    print(f"[analyzer] Extreme segment: {start_year}–{end_year}")
    print(f"[analyzer] Reason: {best_reason}")
    print(f"[analyzer] Hook: {result['hook']}")

    return result


def _score_window(
    rank_data: dict,
    df_yearly: pd.DataFrame,
    start_yr: int,
    end_yr: int,
    years: list,
) -> tuple[float, str]:
    """Score a time window for drama/interest.

    Args:
        rank_data: {year: {entity: rank}} mapping.
        df_yearly: Full yearly DataFrame.
        start_yr: Start year of window.
        end_yr: End year of window.
        years: Sorted list of all years.

    Returns:
        (score, reason) tuple.
    """
    window_years = [y for y in years if start_yr <= y <= end_yr]
    if len(window_years) < 2:
        return 0.0, ""

    # 1. Total rank changes in top 10
    rank_change_score = 0
    overtake_details = []

    start_ranks = rank_data.get(start_yr, {})
    end_ranks = rank_data.get(end_yr, {})

    # Get top 10 entities at start and end
    top_start = sorted(start_ranks.items(), key=lambda x: x[1])[:10]
    top_end = sorted(end_ranks.items(), key=lambda x: x[1])[:10]

    for entity, start_rank in top_start:
        end_rank = end_ranks.get(entity, 99)
        change = abs(end_rank - start_rank)
        rank_change_score += change

        # Detect overtakes
        if end_rank < start_rank and start_rank > 3:
            overtake_details.append(
                f"{entity} rose from #{start_rank} to #{end_rank}"
            )

    # Check for dramatic new entrants
    start_entities = set(e for e, _ in top_start)
    end_entities = set(e for e, _ in top_end)
    new_in_top = end_entities - start_entities
    if new_in_top:
        for entity in new_in_top:
            rank_change_score += 5
            overtake_details.append(f"{entity} entered top 10")

    # 2. Steepest growth among top entities
    growth_score = 0
    growth_details = []

    for entity in end_entities:
        start_val = df_yearly[
            (df_yearly["date"].dt.year == start_yr) &
            (df_yearly["entity"] == entity)
        ]["value"]
        end_val = df_yearly[
            (df_yearly["date"].dt.year == end_yr) &
            (df_yearly["entity"] == entity)
        ]["value"]

        if not start_val.empty and not end_val.empty:
            sv = start_val.iloc[0]
            ev = end_val.iloc[0]
            if sv > 0:
                pct_change = (ev - sv) / sv
                if abs(pct_change) > 1.0:  # More than 100% change
                    growth_score += abs(pct_change)
                    direction = "grew" if pct_change > 0 else "declined"
                    growth_details.append(
                        f"{entity} {direction} by {abs(pct_change)*100:.0f}%"
                    )

    total_score = rank_change_score + growth_score * 2

    # Build reason
    reasons = overtake_details[:3] + growth_details[:2]
    reason = "; ".join(reasons) if reasons else f"Significant changes from {start_yr} to {end_yr}"
    return total_score, reason


def _generate_hook_and_metadata(
    df_yearly: pd.DataFrame,
    rank_data: dict,
    start_year: int,
    end_year: int,
    topic_info: dict,
    reasons_list: list[str]
) -> dict:
    """Generate both the video hook and the YouTube upload metadata using a single Gemini call."""
    from pipeline.modules.gemini_helper import build_gemini_client, generate_content_with_retry
    from pipeline.config import GEMINI_MODEL
    from google.genai import types
    from pydantic import BaseModel
    import json
    import re

    # 1. Setup fallback dict structure
    topic_title = topic_info.get("topic") or "Data Visualization"
    source = topic_info.get("source") or "Our World in Data"
    clean_title = re.sub(r'[^\w\s]', '', topic_title)
    tags = [t.lower() for t in clean_title.split() if len(t) > 3][:12]
    if "data" not in tags:
        tags.append("data")
    if "visualization" not in tags:
        tags.append("visualization")

    span = end_year - start_year
    start_ranks = rank_data.get(start_year, {})
    end_ranks = rank_data.get(end_year, {})

    # Find the entity with the biggest rank improvement
    best_climb = None
    best_climb_delta = 0
    for entity, end_rank in end_ranks.items():
        start_rank = start_ranks.get(entity, 99)
        delta = start_rank - end_rank
        if delta > best_climb_delta and end_rank <= 5:
            best_climb_delta = delta
            best_climb = entity

    if best_climb and best_climb_delta >= 3:
        fallback_hook = f"In just {span} years, {best_climb} rose from #{start_ranks.get(best_climb, '?')} to #{end_ranks[best_climb]}."
    elif reasons_list:
        fallback_hook = reasons_list[0] + "."
    else:
        fallback_hook = f"How the world changed from {start_year} to {end_year}."

    fallback_result = {
        "hook": fallback_hook,
        "long_title": f"How {topic_title} Changed Over Time ({topic_info.get('start_year', start_year)}-{topic_info.get('end_year', end_year)})",
        "long_description": f"A comprehensive data visualization tracking {topic_title} from {topic_info.get('start_year', start_year)} to {topic_info.get('end_year', end_year)}.\n\nData source: {source}",
        "long_tags": tags + ["chart race", "bar chart race", "statistics"],
        "short_title": f"The Dramatic Shift in {topic_title} ({start_year}-{end_year}) #Shorts",
        "short_description": f"Highlighting the most extreme changes in {topic_title} from {start_year} to {end_year}.",
        "short_tags": tags + ["shorts", "trending", "history"]
    }

    # 2. Define schema for structured outputs
    class CombinedOutputs(BaseModel):
        hook: str
        long_title: str
        long_description: str
        long_tags: list[str]
        short_title: str
        short_description: str
        short_tags: list[str]

    # 3. Call Gemini
    try:
        client = build_gemini_client()
    except Exception as e:
        print(f"[analyzer] Failed to build Gemini client: {e}. Using fallback.")
        return fallback_result

    topic_desc = topic_info.get("description", "")
    short_unit = topic_info.get("short_unit", "")
    full_start = topic_info.get("start_year", start_year)
    full_end = topic_info.get("end_year", end_year)

    prompt = f"""You are a viral YouTube video producer. We are creating two versions of a video about a dataset.
    
Dataset details:
- Topic Title: {topic_title}
- Description: {topic_desc}
- Data source: {source}
- Unit: {short_unit}
- Full dataset date range: {full_start} to {full_end}

We have detected the most dramatic/extreme segment of the data:
- Time window: {start_year} to {end_year}
- Significant data highlights in this window:
{chr(10).join('- ' + r for r in reasons_list if r)}

Your task is to generate the following elements:

1. A video HOOK for the opening intro card:
   - Must be a single short sentence (max 85 characters, ~10-14 words).
   - Instant attention-grabber, highlighting a key shift or theme.
   - Respect the tone of the topic (serious/sober for tragedies/death/conflict, exciting/dramatic for space/tech/sports).
   - Do NOT mention chart terms (e.g. "bar chart", "data", "visualization").

2. YouTube Metadata for TWO videos:

   VIDEO 1 — LONG FORM (5-10 min, full dataset visualization):
   - Title: Informative, specific, includes full date range. (e.g. "How World GDP Changed From 1960 to 2023")
   - Description: 3-4 sentences explaining what the data shows, why it matters, and crediting the source.
   - Tags: 10-15 relevant tags for SEO.

   VIDEO 2 — SHORT (60 sec, highlights the extreme segment):
   - Title: Punchy, hook-driven, highlights the most dramatic moment, ends with " #Shorts". (e.g. "China Overtook Japan's Economy In Just 15 Years #Shorts")
   - Description: 2 sentences max.
   - Tags: 10 relevant tags.

Return ONLY a valid JSON object matching the requested schema.
"""

    try:
        response = generate_content_with_retry(
            client=client,
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CombinedOutputs,
                temperature=0.4,
            ),
        )
        res = json.loads(response.text.strip())
        
        # Validate and clean hook
        hook = res.get("hook")
        if not hook or not isinstance(hook, str) or not hook.strip():
            hook = fallback_result["hook"]
        else:
            hook = hook.strip()

        # Validate and clean titles
        long_title = res.get("long_title")
        if not long_title or not isinstance(long_title, str) or not long_title.strip():
            long_title = fallback_result["long_title"]
        else:
            long_title = long_title.strip()

        short_title = res.get("short_title")
        if not short_title or not isinstance(short_title, str) or not short_title.replace("#Shorts", "").replace("#shorts", "").strip():
            short_title = fallback_result["short_title"]
        else:
            short_title = short_title.strip()
            if not short_title.endswith("#Shorts"):
                short_title = short_title + " #Shorts"

        # Validate and clean descriptions
        long_desc = res.get("long_description")
        if not long_desc or not isinstance(long_desc, str) or not long_desc.strip():
            long_desc = fallback_result["long_description"]
        else:
            long_desc = long_desc.strip()

        short_desc = res.get("short_description")
        if not short_desc or not isinstance(short_desc, str) or not short_desc.strip():
            short_desc = fallback_result["short_description"]
        else:
            short_desc = short_desc.strip()

        # Validate and clean tags
        long_tags = res.get("long_tags")
        if not long_tags or not isinstance(long_tags, list):
            long_tags = fallback_result["long_tags"]
        else:
            long_tags = [str(t).strip() for t in long_tags if str(t).strip()]
            if not long_tags:
                long_tags = fallback_result["long_tags"]

        short_tags = res.get("short_tags")
        if not short_tags or not isinstance(short_tags, list):
            short_tags = fallback_result["short_tags"]
        else:
            short_tags = [str(t).strip() for t in short_tags if str(t).strip()]
            if not short_tags:
                short_tags = fallback_result["short_tags"]

        # Enforce YouTube title limit of 100 characters
        if len(long_title) > 100:
            long_title = long_title[:97] + "..."
        if len(short_title) > 100:
            short_title_clean = short_title.replace(" #Shorts", "").replace("#Shorts", "").strip()
            short_title = short_title_clean[:88] + " #Shorts"

        return {
            "hook": hook,
            "long_title": long_title,
            "long_description": long_desc,
            "long_tags": long_tags,
            "short_title": short_title,
            "short_description": short_desc,
            "short_tags": short_tags,
        }
    except Exception as e:
        print(f"[analyzer] Gemini combined hook & metadata generation failed: {e}. Using fallback.")
        return fallback_result
