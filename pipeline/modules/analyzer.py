"""
analyzer.py — Finds the most dramatic/extreme segment in the yearly data.

Used to determine what the YouTube Short should highlight.
"""

import numpy as np
import pandas as pd


def find_extreme_segment(df_yearly: pd.DataFrame) -> dict:
    """Analyze yearly data to find the most dramatic time window.

    "Most dramatic" is determined by:
    1. Rank changes — periods where entities swap positions the most
    2. Steepest growth/decline — absolute rate of change
    3. Overtake moments — when a lower-ranked entity surpasses a higher one

    The returned segment is constrained to 10–20 years (ideal for a 50-59s Short).

    Args:
        df_yearly: DataFrame with columns [date, entity, value].

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
            "hook": f"What changed from {years[0]} to {years[-1]}? The answer will surprise you.",
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
    hook = _generate_hook(df_yearly, rank_data, start_year, end_year)

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
) -> str:
    """Generate a compelling hook line for the Short.

    Args:
        df_yearly: Full yearly DataFrame.
        rank_data: Rank data per year.
        start_year: Start of extreme segment.
        end_year: End of extreme segment.

    Returns:
        Hook string for display at the start of the Short.
    """
    span = end_year - start_year
    start_ranks = rank_data.get(start_year, {})
    end_ranks = rank_data.get(end_year, {})

    # Find the entity with the biggest rank improvement
    best_climb = None
    best_climb_delta = 0

    for entity, end_rank in end_ranks.items():
        start_rank = start_ranks.get(entity, 99)
        delta = start_rank - end_rank  # positive means improvement
        if delta > best_climb_delta and end_rank <= 5:
            best_climb_delta = delta
            best_climb = entity

    if best_climb and best_climb_delta >= 3:
        return (
            f"In just {span} years, {best_climb} went from "
            f"#{start_ranks.get(best_climb, '?')} to "
            f"#{end_ranks[best_climb]}. Nobody saw it coming."
        )

    # Find the entity with biggest percentage growth
    biggest_growth_entity = None
    biggest_growth_pct = 0

    top_end = sorted(end_ranks.items(), key=lambda x: x[1])[:5]
    for entity, _ in top_end:
        start_val = df_yearly[
            (df_yearly["date"].dt.year == start_year) &
            (df_yearly["entity"] == entity)
        ]["value"]
        end_val = df_yearly[
            (df_yearly["date"].dt.year == end_year) &
            (df_yearly["entity"] == entity)
        ]["value"]

        if not start_val.empty and not end_val.empty:
            sv = start_val.iloc[0]
            ev = end_val.iloc[0]
            if sv > 0:
                pct = (ev - sv) / sv * 100
                if pct > biggest_growth_pct:
                    biggest_growth_pct = pct
                    biggest_growth_entity = entity

    if biggest_growth_entity and biggest_growth_pct > 100:
        return (
            f"{biggest_growth_entity} grew {biggest_growth_pct:.0f}% in {span} years. "
            f"Watch what happened."
        )

    return f"From {start_year} to {end_year} — everything changed. Watch how."
