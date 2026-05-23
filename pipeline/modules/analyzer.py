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

    # --- Generate hook ---
    reasons_list = [r.strip() for r in best_reason.split(";") if r.strip()]
    hook = _generate_hook(df_yearly, rank_data, start_year, end_year, topic_info, reasons_list)

    result = {
        "start_year": int(start_year),
        "end_year": int(end_year),
        "reason": best_reason,
        "hook": hook,
    }

    print(f"[analyzer] Extreme segment: {start_year}–{end_year}")
    print(f"[analyzer] Reason: {best_reason}")
    print(f"[analyzer] Hook: {hook}")

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


def _generate_hook(
    df_yearly: pd.DataFrame,
    rank_data: dict,
    start_year: int,
    end_year: int,
    topic_info: dict,
    reasons_list: list[str]
) -> str:
    """Generate a context-aware hook using Gemini (with rule-based fallback)."""
    # 1. Try to generate with Gemini
    hook = _generate_hook_with_gemini(df_yearly, start_year, end_year, topic_info, reasons_list)
    if hook:
        return hook

    # 2. Fallback to generic but safe rule-based hook
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
        return (
            f"In just {span} years, {best_climb} rose from "
            f"#{start_ranks.get(best_climb, '?')} to "
            f"#{end_ranks[best_climb]}."
        )

    return f"How the world changed from {start_year} to {end_year}."


def _generate_hook_with_gemini(
    df_yearly: pd.DataFrame,
    start_year: int,
    end_year: int,
    topic_info: dict,
    reasons_list: list[str]
) -> str:
    """Use Gemini to generate a highly compelling, dramatic, and context-aware hook for the video intro."""
    from pipeline.modules.gemini_helper import build_gemini_client, generate_content_with_retry
    from pipeline.config import GEMINI_MODEL
    from google.genai import types
    import json
    
    try:
        client = build_gemini_client()
    except Exception as e:
        print(f"[analyzer] Failed to build Gemini client: {e}")
        return ""
    
    topic_title = topic_info.get("topic", "Data Visualization")
    topic_desc = topic_info.get("description", "")
    short_unit = topic_info.get("short_unit", "")
    
    prompt = f"""You are a viral YouTube Shorts producer. We are making a vertical bar chart race video.
Topic Title: {topic_title}
Description: {topic_desc}
Unit: {short_unit}
Time window: {start_year} to {end_year}

Significant data highlights in this window:
{chr(10).join('- ' + r for r in reasons_list if r)}

Your task is to create a compelling, punchy, context-aware HOOK (maximum 85 characters, ~10-14 words) to display on the video's opening intro card.
The hook must capture the viewer's attention instantly, highlight a key shift or theme, and respect the tone of the topic (e.g., use a serious/sober tone for tragedy/suicide/death/conflict data, and an exciting/dramatic tone for space/technology/sports data).

Rules:
1. Must be a single short sentence.
2. Keep it under 85 characters so it fits on a single mobile screen overlay without wrapping too much.
3. Do not mention chart terms (e.g., "bar chart", "data", "visualization", "axis"). Speak about the real-world topic.
4. Return ONLY a JSON object:
{{"hook": "your compelling hook here"}}
"""
    try:
        response = generate_content_with_retry(
            client=client,
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.4,
            ),
        )
        res = json.loads(response.text.strip())
        return res.get("hook", "").strip()
    except Exception as e:
        print(f"[analyzer] Gemini hook generation failed: {e}. Using fallback.")
        return ""
