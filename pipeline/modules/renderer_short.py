"""
renderer_short.py — Renders the YouTube Short from the extreme segment.

Vertical format (1080x1920), 20–35 seconds. Supports all chart types:
  - bar_chart_race: horizontal bar race (fixed-slot system)
  - line_chart_race: animated lines for 2–5 entities
  - area_chart: single-metric filled area with value counter
  - bubble_chart: animated bubble sizes representing values
  - map_animation: choropleth world map animation
"""

import random
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Optional
import os
import requests
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import pandas as pd

from pipeline.config import (
    ACCENT_COLORS, FPS, FRAMES_SHORT_DIR, SHORT_FINAL,
    SHORT_MIN_DURATION, SHORT_MAX_DURATION, MUSIC_DIR,
    DEFAULT_VOLUME, TOP_N_ENTITIES, TMP_DIR, DRAW_MAP_LABELS,
)
from pipeline.modules.renderer_long import (
    SLOTS, OFF_SCREEN_Y, BAR_HEIGHT,
    assign_entity_colors, format_value, _rank_entities, _ease,
)
from pipeline.modules.font_loader import FONT_BOLD, FONT_REGULAR, overlay_watermark

# Short-form constants
SHORT_FRAMES_PER_STEP = 15   # fewer frames for faster pacing
SHORT_INTRO_FRAMES = 36      # 1.2s intro: 18 static + 18 animated

# Path effects for thick black outline (legibility on busy backgrounds)
OUTLINE = [path_effects.Stroke(linewidth=3.5, foreground="black"), path_effects.Normal()]


def render_short(
    df_yearly: pd.DataFrame,
    chart_type: str,
    topic_info: dict,
    extreme_segment: dict,
    entity_colors: Optional[dict] = None,
) -> tuple[Path, dict[str, str]]:
    """Render the YouTube Short video from the extreme segment.

    Routes to the appropriate chart-type-specific renderer.

    Args:
        df_yearly: Yearly DataFrame with columns [date, entity, value].
        chart_type: One of 'bar_chart_race', 'line_chart_race', 'area_chart',
                    'bubble_chart', 'map_animation'.
        topic_info: Dict with keys: topic, description, source, hook.
        extreme_segment: Dict with start_year, end_year, reason, hook.
        entity_colors: Pre-assigned color mapping from the long-form render.

    Returns:
        (output_path, entity_colors_used)
    """
    # Common setup
    start_yr = extreme_segment["start_year"]
    end_yr   = extreme_segment["end_year"]
    print(f"[renderer_short] Rendering Short ({chart_type}): {start_yr}–{end_yr}")

    # Use unique directories per chart type to support safe concurrent execution
    frames_dir = TMP_DIR / f"frames_short_{chart_type}"
    output_path = TMP_DIR / f"short_{chart_type}.mp4"

    import shutil
    if frames_dir.exists():
        print(f"[renderer_short] Cleaning up existing frames in {frames_dir}...")
        shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Filter to the extreme segment window
    df_seg = df_yearly[
        (df_yearly["date"].dt.year >= start_yr) &
        (df_yearly["date"].dt.year <= end_yr)
    ].copy()

    if df_seg.empty:
        raise RuntimeError(f"No data found for segment {start_yr}–{end_yr}")

    all_entities = sorted(df_yearly["entity"].unique())
    if entity_colors is None:
        entity_colors = assign_entity_colors(all_entities)
    else:
        for e in all_entities:
            if e not in entity_colors:
                entity_colors[e] = ACCENT_COLORS[len(entity_colors) % len(ACCENT_COLORS)]

    # Route to the appropriate renderer
    if chart_type == "line_chart_race":
        path = _render_line_chart(df_seg, topic_info, extreme_segment, entity_colors, frames_dir, output_path)
    elif chart_type == "area_chart":
        path = _render_area_chart(df_seg, topic_info, extreme_segment, entity_colors, frames_dir, output_path)
    elif chart_type == "bubble_chart":
        path = _render_bubble_chart(df_seg, topic_info, extreme_segment, entity_colors, frames_dir, output_path)
    elif chart_type == "map_animation":
        path = _render_map_chart(df_seg, topic_info, extreme_segment, entity_colors, frames_dir, output_path)
    else:
        # Default: bar_chart_race
        path = _render_bar_chart_race(df_seg, topic_info, extreme_segment, entity_colors, frames_dir, output_path)

    print(f"[renderer_short] Output: {path}")
    return path, entity_colors


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _compute_frames_per_step(n_steps: int) -> int:
    """Calculate frames per step to hit target Short duration."""
    usable_frames = SHORT_MIN_DURATION * FPS - SHORT_INTRO_FRAMES
    frames_per_step = max(SHORT_FRAMES_PER_STEP, usable_frames // max(n_steps - 1, 1))
    max_frames = SHORT_MAX_DURATION * FPS - SHORT_INTRO_FRAMES
    frames_per_step = min(frames_per_step, max_frames // max(n_steps - 1, 1))
    return max(frames_per_step, 4)


def _build_step_values(df_seg: pd.DataFrame) -> tuple[list, list[dict[str, float]]]:
    """Build per-timestep value dicts from a filtered DataFrame."""
    time_steps = sorted(df_seg["date"].unique())
    step_values = []
    for ts in time_steps:
        row = df_seg[df_seg["date"] == ts]
        step_values.append({r.entity: float(r.value) for r in row.itertuples()})
    return time_steps, step_values


def _render_intro_frames(
    fig: plt.Figure,
    wrapped_hook: str,
    wrapped_title: str,
    background_draw_fn,
    frame_number: int,
    frames_dir: Optional[Path] = None,
) -> int:
    """Render the intro card with hook → title transition.

    Args:
        fig: matplotlib Figure.
        wrapped_hook: Pre-wrapped hook text.
        wrapped_title: Pre-wrapped title text.
        background_draw_fn: Callable(fig, ax) to draw the background chart state.
        frame_number: Starting frame number.
        frames_dir: Output frames directory.

    Returns:
        Next frame_number.
    """
    if frames_dir is None:
        frames_dir = FRAMES_SHORT_DIR

    for f in range(SHORT_INTRO_FRAMES):
        fig.clf()
        fig.patch.set_facecolor("#000000")

        # Draw background chart
        ax = fig.add_axes([0.08, 0.10, 0.84, 0.70])
        background_draw_fn(fig, ax)

        # Translucent overlay to dim chart
        overlay = plt.Rectangle(
            (0, 0), 1, 1, facecolor="#000000", alpha=0.70,
            transform=fig.transFigure, zorder=5,
        )
        fig.patches.append(overlay)

        # Hook + title animation
        half = SHORT_INTRO_FRAMES // 2
        if f < half:
            hook_alpha, title_alpha, hook_y = 1.0, 0.0, 0.5
        else:
            t = (f - half) / half
            eased_t = _ease(t)
            hook_y = 0.5 + eased_t * 0.45
            hook_alpha = 1.0 - eased_t
            title_alpha = eased_t

        if hook_alpha > 0.01:
            fig.text(
                0.5, hook_y, wrapped_hook,
                ha="center", va="center",
                color=(1, 1, 1, hook_alpha),
                fontsize=28, fontproperties=FONT_BOLD,
                wrap=True, transform=fig.transFigure,
                bbox=dict(
                    boxstyle="round,pad=0.8",
                    facecolor="#111111",
                    edgecolor=(1, 1, 1, hook_alpha * 0.4),
                    linewidth=1.5,
                ),
                zorder=10,
            )

        if title_alpha > 0.01:
            fig.text(
                0.5, 0.97, wrapped_title,
                ha="center", va="top",
                color=(1, 1, 1, title_alpha),
                fontsize=26, fontproperties=FONT_BOLD,
                transform=fig.transFigure, zorder=10, wrap=True,
            )

        fig.savefig(
            frames_dir / f"frame_{frame_number:05d}.png",
            dpi=100, facecolor="#000000", pad_inches=0,
            pil_kwargs={"compress_level": 1},
        )
        frame_number += 1

    return frame_number


def _draw_frame_chrome(
    ax: plt.Axes,
    fig: plt.Figure,
    wrapped_title: str,
    source: str,
    date_label: str,
    topic_info: dict,
) -> None:
    """Draw title, source, year, and watermark on a chart frame."""
    full_unit = topic_info.get("full_unit", "") if topic_info else ""

    # Year — centered above the chart
    ax.text(
        0.5, 0.84, date_label,
        ha="center", va="center",
        color="white", fontsize=48, fontproperties=FONT_BOLD,
        transform=fig.transFigure,
        path_effects=OUTLINE, zorder=10,
    )

    # Title — figure-level, top
    ax.text(
        0.5, 0.97, wrapped_title,
        ha="center", va="top",
        color="white", fontsize=24, fontproperties=FONT_BOLD,
        transform=fig.transFigure, wrap=True,
    )

    # Source / Unit text
    source_text = f"Source: {source}" if source else ""
    if full_unit:
        if source_text:
            source_text += f" | Unit: {full_unit}"
        else:
            source_text = f"Unit: {full_unit}"

    if source_text:
        ax.text(
            0.5, 0.91, source_text,
            ha="center", va="top",
            color="#bbbbbb", fontsize=14, fontproperties=FONT_REGULAR, style="italic",
            transform=fig.transFigure,
        )

    # Watermark
    overlay_watermark(fig, x=0.04, y=0.04, size=55, alpha=0.22)


def _save_frame(fig: plt.Figure, frame_number: int, frames_dir: Optional[Path] = None) -> None:
    """Save a figure as a PNG frame with retry logic for cloud-synced folders."""
    if frames_dir is None:
        frames_dir = FRAMES_SHORT_DIR
    
    import time
    file_path = frames_dir / f"frame_{frame_number:05d}.png"
    
    for attempt in range(5):
        try:
            frames_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                file_path,
                dpi=100, facecolor="#000000", pad_inches=0,
                pil_kwargs={"compress_level": 1},
            )
            return
        except FileNotFoundError as e:
            if attempt == 4:
                raise e
            time.sleep(0.1)
        except Exception as e:
            if attempt == 4:
                raise e
            time.sleep(0.1)


# ══════════════════════════════════════════════════════════════════════════════
# CHART TYPE 1: BAR CHART RACE (existing, refactored)
# ══════════════════════════════════════════════════════════════════════════════

def _render_bar_chart_race(
    df_seg: pd.DataFrame,
    topic_info: dict,
    extreme_segment: dict,
    entity_colors: dict,
    frames_dir: Path,
    output_path: Path,
) -> Path:
    """Render bar chart race Short."""
    start_yr = extreme_segment["start_year"]

    time_steps, step_values = _build_step_values(df_seg)
    n_steps = len(time_steps)
    frames_per_step = _compute_frames_per_step(n_steps)

    est_duration = (SHORT_INTRO_FRAMES + (n_steps - 1) * frames_per_step) / FPS
    print(f"[renderer_short] Bar chart race: {n_steps} steps, {frames_per_step} f/step, ~{est_duration:.0f}s")

    title = topic_info.get("topic", "Data Visualization")
    source = topic_info.get("source", "")
    hook = extreme_segment.get("hook", title)
    wrapped_title = "\n".join(textwrap.wrap(title, width=28))
    wrapped_hook = "\n".join(textwrap.wrap(hook, width=24))

    fig = plt.figure(figsize=(10.8, 19.2), dpi=100)
    fig.patch.set_facecolor("#000000")

    # Compute initial state for intro background
    first_vals = step_values[0]
    sorted_ents_first = sorted(first_vals.keys(), key=lambda e: first_vals[e], reverse=True)
    top10_first = sorted_ents_first[:TOP_N_ENTITIES]

    entities_data_first = []
    for rank, entity in enumerate(top10_first):
        entities_data_first.append({
            "entity": entity,
            "value":  float(first_vals[entity]),
            "y_pos":  SLOTS[rank],
            "color":  entity_colors.get(entity, ACCENT_COLORS[0]),
            "left_offset": 0.0,
        })

    def draw_bg(fig, ax):
        _draw_bar_chart_frame(ax, fig, entities_data_first, "", "", str(start_yr),
                               topic_info, save=False)

    frame_number = _render_intro_frames(fig, wrapped_hook, wrapped_title, draw_bg, 0, frames_dir)

    # Recreate axes
    fig.clf()
    fig.patch.set_facecolor("#000000")
    ax = fig.add_axes([0.08, 0.10, 0.84, 0.70])

    # Chart animation
    prev_ranks: dict[str, int] = {e: r for r, e in enumerate(sorted_ents_first)}
    entities_data = []
    date_label = str(start_yr)

    for step_idx in range(len(time_steps) - 1):
        prev_vals = step_values[step_idx]
        next_vals = step_values[step_idx + 1]
        all_shown = set(prev_vals.keys()) | set(next_vals.keys())
        ts_start = pd.Timestamp(time_steps[step_idx])
        ts_end   = pd.Timestamp(time_steps[step_idx + 1])

        for interp_frame in range(frames_per_step):
            alpha = _ease(interp_frame / frames_per_step)

            interp_vals = {}
            for entity in all_shown:
                v0 = prev_vals.get(entity, 0.0)
                v1 = next_vals.get(entity, 0.0)
                interp_vals[entity] = v0 + (v1 - v0) * alpha

            sorted_ents = sorted(interp_vals.keys(), key=lambda e: interp_vals[e], reverse=True)
            current_top10 = sorted_ents[:TOP_N_ENTITIES]

            entities_data = []
            for rank, entity in enumerate(current_top10):
                prev_slot_rank = prev_ranks.get(entity, TOP_N_ENTITIES)
                prev_y = SLOTS.get(prev_slot_rank, OFF_SCREEN_Y)
                cur_y  = SLOTS[rank]

                if prev_slot_rank == TOP_N_ENTITIES:
                    y_pos = cur_y
                    left_offset = 1.0 - alpha
                else:
                    y_pos = prev_y + (cur_y - prev_y) * alpha
                    left_offset = 0.0

                entities_data.append({
                    "entity": entity,
                    "value":  interp_vals[entity],
                    "y_pos":  y_pos,
                    "color":  entity_colors.get(entity, ACCENT_COLORS[0]),
                    "left_offset": left_offset,
                })

            interp_ts = ts_start + (ts_end - ts_start) * alpha
            date_label = str(interp_ts.year)

            _draw_bar_chart_frame(ax, fig, entities_data, wrapped_title, source,
                                  date_label, topic_info)
            _save_frame(fig, frame_number, frames_dir)
            frame_number += 1

        prev_ranks = {e: r for r, e in enumerate(
            sorted(next_vals.keys(), key=lambda k: next_vals[k], reverse=True)
        )}

    # Hold last frame
    for _ in range(FPS * 1):
        _draw_bar_chart_frame(ax, fig, entities_data, wrapped_title, source,
                              date_label, topic_info)
        _save_frame(fig, frame_number, frames_dir)
        frame_number += 1

    plt.close(fig)
    print(f"[renderer_short] Total frames: {frame_number}")
    _encode_short(frames_dir, output_path)
    return output_path


def _draw_bar_chart_frame(
    ax: plt.Axes,
    fig: plt.Figure,
    entities_data: list[dict],
    title: str,
    source: str,
    date_label: str,
    topic_info: Optional[dict] = None,
    save: bool = False,
) -> None:
    """Draw a single vertical bar chart race frame."""
    ax.cla()
    ax.set_facecolor("#000000")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 9.6)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors="white", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#444444")

    if not entities_data:
        return

    max_value = max(d["value"] for d in entities_data) * 1.1
    if max_value <= 0:
        max_value = 1.0

    short_unit = topic_info.get("short_unit", "") if topic_info else ""
    full_unit = topic_info.get("full_unit", "") if topic_info else ""

    for d in entities_data:
        norm_val = d["value"] / max_value
        y = d["y_pos"]
        color = d["color"]
        left_offset = d.get("left_offset", 0.0)

        # Glassmorphic neon style
        rgba_face = to_rgba(color, alpha=0.45)
        rgba_edge = to_rgba(color, alpha=0.95)
        ax.barh(y, norm_val, height=BAR_HEIGHT, facecolor=rgba_face,
                edgecolor=rgba_edge, linewidth=2.5, left=left_offset)

        clean_name = re.sub(r"\s*\([^)]*\)\s*$", "", d["entity"]).strip()
        val_str = format_value(d["value"], short_unit)
        actual_end = left_offset + norm_val

        if norm_val > 0.50:
            max_chars = max(8, int((norm_val - 0.18) / 0.022))
            display_name = clean_name[:max_chars - 3] + "..." if len(clean_name) > max_chars else clean_name

            flag_w = _add_flag_to_axes(ax, clean_name, left_offset + 0.04, y, box_alignment=(0.0, 0.5))
            text_x = left_offset + 0.04 + ((flag_w + 14.0) / 622.08 if flag_w > 0 else 0.0)
            ax.text(text_x, y, display_name,
                    ha="left", va="center", color="white",
                    fontsize=18, fontproperties=FONT_BOLD,
                    path_effects=OUTLINE, clip_on=True)
            ax.text(actual_end - 0.04, y, val_str,
                    ha="right", va="center", color="white",
                    fontsize=18, fontproperties=FONT_BOLD,
                    path_effects=OUTLINE, clip_on=True)
        else:
            label_text = f"{clean_name} ({val_str})"
            flag_w = _add_flag_to_axes(ax, clean_name, actual_end + 0.03, y, box_alignment=(0.0, 0.5))
            text_x = actual_end + 0.03 + ((flag_w + 14.0) / 622.08 if flag_w > 0 else 0.0)
            ax.text(text_x, y, label_text,
                    ha="left", va="center", color="white",
                    fontsize=18, fontproperties=FONT_BOLD,
                    path_effects=OUTLINE, clip_on=True)

    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([format_value(t * max_value) for t in ticks],
                       color="white", fontsize=12, fontweight="bold")

    if not save:
        _draw_frame_chrome(ax, fig, title, source, date_label, topic_info)


# ══════════════════════════════════════════════════════════════════════════════
# CHART TYPE 2: LINE CHART RACE
# ══════════════════════════════════════════════════════════════════════════════

def _render_line_chart(
    df_seg: pd.DataFrame,
    topic_info: dict,
    extreme_segment: dict,
    entity_colors: dict,
    frames_dir: Path,
    output_path: Path,
) -> Path:
    """Render animated line chart Short for 2–5 entities."""
    start_yr = extreme_segment["start_year"]
    time_steps, step_values = _build_step_values(df_seg)
    n_steps = len(time_steps)
    frames_per_step = _compute_frames_per_step(n_steps)

    est_duration = (SHORT_INTRO_FRAMES + (n_steps - 1) * frames_per_step) / FPS
    print(f"[renderer_short] Line chart: {n_steps} steps, ~{est_duration:.0f}s")

    title = topic_info.get("topic", "Data Visualization")
    source = topic_info.get("source", "")
    hook = extreme_segment.get("hook", title)
    wrapped_title = "\n".join(textwrap.wrap(title, width=28))
    wrapped_hook = "\n".join(textwrap.wrap(hook, width=24))

    # Collect all entities and all values for axis scaling
    all_entities = sorted(df_seg["entity"].unique())
    all_values = df_seg["value"].dropna()
    y_min = float(all_values.min()) * 0.9
    y_max = float(all_values.max()) * 1.1
    if y_min == y_max:
        y_min, y_max = y_min - 1, y_max + 1

    years = [pd.Timestamp(ts).year for ts in time_steps]
    x_min, x_max = years[0], years[-1]

    fig = plt.figure(figsize=(10.8, 19.2), dpi=100)
    fig.patch.set_facecolor("#000000")

    # Build full entity timeseries for line drawing
    entity_series = {}
    for entity in all_entities:
        entity_series[entity] = []
        for ts in time_steps:
            vals = step_values[time_steps.index(ts)]
            entity_series[entity].append(vals.get(entity, np.nan))

    # Intro: draw static first-frame lines as background
    def draw_bg(fig, ax):
        ax.set_facecolor("#000000")
        ax.set_xlim(x_min - 0.5, x_max + 1.2)
        ax.set_ylim(y_min, y_max)
        for entity in all_entities:
            color = entity_colors.get(entity, ACCENT_COLORS[0])
            ax.plot([years[0]], [entity_series[entity][0]], "o",
                    color=color, markersize=8, alpha=0.6)

    frame_number = _render_intro_frames(fig, wrapped_hook, wrapped_title, draw_bg, 0, frames_dir)

    fig.clf()
    fig.patch.set_facecolor("#000000")
    ax = fig.add_axes([0.12, 0.10, 0.80, 0.70])

    short_unit = topic_info.get("short_unit", "") if topic_info else ""

    # Animate line growth
    for step_idx in range(len(time_steps) - 1):
        ts_start = pd.Timestamp(time_steps[step_idx])
        ts_end   = pd.Timestamp(time_steps[step_idx + 1])

        for interp_frame in range(frames_per_step):
            alpha = _ease(interp_frame / frames_per_step)
            ax.cla()
            ax.set_facecolor("#000000")
            ax.set_xlim(x_min - 0.5, x_max + 1.2)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(years)
            ax.set_xticklabels([str(y) for y in years], color="white", fontsize=11)
            ax.tick_params(axis="both", colors="white", labelsize=11)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#444444")
            ax.spines["bottom"].set_color("#444444")

            # Format y-axis
            ax.set_yticks(np.linspace(y_min, y_max, 5))
            ax.set_yticklabels([format_value(v, short_unit) for v in np.linspace(y_min, y_max, 5)],
                               color="white", fontsize=10)

            interp_year = ts_start.year + (ts_end.year - ts_start.year) * alpha
            date_label = str(int(interp_year))

            for entity in all_entities:
                color = entity_colors.get(entity, ACCENT_COLORS[0])
                series = entity_series[entity]

                # Draw completed segments
                x_complete = years[:step_idx + 1]
                y_complete = series[:step_idx + 1]

                # Interpolate current segment
                v0 = series[step_idx] if not np.isnan(series[step_idx]) else 0
                v1 = series[step_idx + 1] if not np.isnan(series[step_idx + 1]) else 0
                interp_val = v0 + (v1 - v0) * alpha

                x_line = x_complete + [interp_year]
                y_line = y_complete + [interp_val]

                # Filter NaN
                valid = [(x, y) for x, y in zip(x_line, y_line) if not np.isnan(y)]
                if not valid:
                    continue

                xv, yv = zip(*valid)

                # Glow effect (wider transparent line behind)
                ax.plot(xv, yv, color=color, linewidth=6, alpha=0.15, solid_capstyle="round")
                ax.plot(xv, yv, color=color, linewidth=3, alpha=0.9, solid_capstyle="round")

                # Dot at current position
                ax.plot(xv[-1], yv[-1], "o", color=color, markersize=10,
                        markeredgecolor="white", markeredgewidth=1.5, zorder=5)

                # Entity label at endpoint
                clean_name = re.sub(r"\s*\([^)]*\)\s*$", "", entity).strip()
                if len(clean_name) > 16:
                    clean_name = clean_name[:13] + "..."

                x_flag = xv[-1] + 0.3
                flag_w = _add_flag_to_axes(ax, clean_name, x_flag, yv[-1], box_alignment=(0.0, 0.5))
                if flag_w > 0:
                    x_span = (x_max + 0.5) - (x_min - 0.5)
                    shift_years = (flag_w + 14.0) * x_span / 864.0
                    text_x = x_flag + shift_years
                else:
                    text_x = x_flag

                ax.text(
                    text_x, yv[-1], clean_name,
                    ha="left", va="center", color="white",
                    fontsize=14, fontproperties=FONT_BOLD,
                    path_effects=OUTLINE, clip_on=False,
                )

            _draw_frame_chrome(ax, fig, wrapped_title, source, date_label, topic_info)
            _save_frame(fig, frame_number, frames_dir)
            frame_number += 1

    # Hold last frame
    for _ in range(FPS * 1):
        _save_frame(fig, frame_number, frames_dir)
        frame_number += 1

    plt.close(fig)
    print(f"[renderer_short] Total frames: {frame_number}")
    _encode_short(frames_dir, output_path)
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# CHART TYPE 3: AREA CHART
# ══════════════════════════════════════════════════════════════════════════════

def _render_area_chart(
    df_seg: pd.DataFrame,
    topic_info: dict,
    extreme_segment: dict,
    entity_colors: dict,
    frames_dir: Path,
    output_path: Path,
) -> Path:
    """Render filled area chart Short for a single metric."""
    start_yr = extreme_segment["start_year"]
    time_steps, step_values = _build_step_values(df_seg)
    n_steps = len(time_steps)
    frames_per_step = _compute_frames_per_step(n_steps)

    est_duration = (SHORT_INTRO_FRAMES + (n_steps - 1) * frames_per_step) / FPS
    print(f"[renderer_short] Area chart: {n_steps} steps, ~{est_duration:.0f}s")

    title = topic_info.get("topic", "Data Visualization")
    source = topic_info.get("source", "")
    hook = extreme_segment.get("hook", title)
    wrapped_title = "\n".join(textwrap.wrap(title, width=28))
    wrapped_hook = "\n".join(textwrap.wrap(hook, width=24))
    short_unit = topic_info.get("short_unit", "") if topic_info else ""

    # For area chart, pick the primary entity (largest average value)
    entity_avg = df_seg.groupby("entity")["value"].mean()
    primary_entity = entity_avg.idxmax()
    primary_color = entity_colors.get(primary_entity, ACCENT_COLORS[0])

    # Build series for primary entity
    years = [pd.Timestamp(ts).year for ts in time_steps]
    series_vals = []
    for sv in step_values:
        series_vals.append(sv.get(primary_entity, 0.0))

    y_min = 0
    y_max = max(series_vals) * 1.15 if max(series_vals) > 0 else 1
    x_min, x_max = years[0], years[-1]

    fig = plt.figure(figsize=(10.8, 19.2), dpi=100)
    fig.patch.set_facecolor("#000000")

    # Intro
    def draw_bg(fig, ax):
        ax.set_facecolor("#000000")
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(y_min, y_max)

    frame_number = _render_intro_frames(fig, wrapped_hook, wrapped_title, draw_bg, 0, frames_dir)

    fig.clf()
    fig.patch.set_facecolor("#000000")
    ax = fig.add_axes([0.12, 0.10, 0.80, 0.70])

    # Animate area fill growth
    for step_idx in range(len(time_steps) - 1):
        ts_start = pd.Timestamp(time_steps[step_idx])
        ts_end   = pd.Timestamp(time_steps[step_idx + 1])

        for interp_frame in range(frames_per_step):
            alpha = _ease(interp_frame / frames_per_step)
            ax.cla()
            ax.set_facecolor("#000000")
            ax.set_xlim(x_min - 0.5, x_max + 0.5)
            ax.set_ylim(y_min, y_max)
            ax.tick_params(axis="both", colors="white", labelsize=11)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#444444")
            ax.spines["bottom"].set_color("#444444")

            ax.set_yticks(np.linspace(y_min, y_max, 5))
            ax.set_yticklabels([format_value(v, short_unit) for v in np.linspace(y_min, y_max, 5)],
                               color="white", fontsize=10)

            interp_year = ts_start.year + (ts_end.year - ts_start.year) * alpha
            v0 = series_vals[step_idx]
            v1 = series_vals[step_idx + 1]
            interp_val = v0 + (v1 - v0) * alpha
            date_label = str(int(interp_year))

            x_line = list(years[:step_idx + 1]) + [interp_year]
            y_line = list(series_vals[:step_idx + 1]) + [interp_val]

            # Gradient fill — solid color with alpha gradient
            rgba_fill = to_rgba(primary_color, alpha=0.3)
            rgba_line = to_rgba(primary_color, alpha=0.95)

            ax.fill_between(x_line, y_line, y2=0, color=rgba_fill, zorder=2)
            ax.plot(x_line, y_line, color=rgba_line, linewidth=3, zorder=3)

            # Glow at current point
            ax.plot(interp_year, interp_val, "o", color=primary_color,
                    markersize=12, markeredgecolor="white", markeredgewidth=2, zorder=5)

            # Big value counter in center of chart
            val_str = format_value(interp_val, short_unit)
            ax.text(
                0.5, 0.45, val_str,
                ha="center", va="center",
                color="white", fontsize=72, fontproperties=FONT_BOLD,
                transform=ax.transAxes,
                alpha=0.25, zorder=1,
            )

            _draw_frame_chrome(ax, fig, wrapped_title, source, date_label, topic_info)
            _save_frame(fig, frame_number, frames_dir)
            frame_number += 1

    # Hold last frame
    for _ in range(FPS * 1):
        _save_frame(fig, frame_number, frames_dir)
        frame_number += 1

    plt.close(fig)
    print(f"[renderer_short] Total frames: {frame_number}")
    _encode_short(frames_dir, output_path)
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# CHART TYPE 4: BUBBLE CHART
# ══════════════════════════════════════════════════════════════════════════════

def _run_bubble_physics_step(positions, velocities, radii, center_x, center_y, axes_height, dt=0.5):
    """Run a single sub-step of the force-directed bubble collision physics."""
    forces = {ent: np.array([0.0, 0.0]) for ent in positions}
    
    # 1. Gravity attraction to center of screen
    gravity = 0.03
    for ent in positions:
        if radii[ent] <= 0.01:
            continue
        forces[ent] += (np.array([center_x, center_y]) - positions[ent]) * gravity
        
    # Pre-filter active entities to optimize N^2 collision check
    active_ents = [ent for ent in positions if radii[ent] > 0.01]
        
    # 2. Pairwise bubble-bubble repulsion force to prevent overlap (soft force)
    for i in range(len(active_ents)):
        entA = active_ents[i]
        rA = radii[entA]
        for j in range(i + 1, len(active_ents)):
            entB = active_ents[j]
            rB = radii[entB]
                
            delta = positions[entA] - positions[entB]
            dist = np.hypot(delta[0], delta[1])
            target_dist = (rA + rB) * 1.05
            
            if dist < target_dist:
                if dist < 0.001:
                    angle = np.random.uniform(0, 2 * np.pi)
                    delta = np.array([np.cos(angle), np.sin(angle)]) * 0.01
                    dist = 0.01
                
                overlap = target_dist - dist
                dir_vector = delta / dist
                f = dir_vector * overlap * 1.5
                
                forces[entA] += f
                forces[entB] -= f
                
    # 3. Update velocity and position with friction/damping
    damping = 0.6
    for ent in positions:
        if radii[ent] <= 0.01:
            # Keep inactive/invisible bubbles centered with small noise to prevent drift
            positions[ent] = np.array([
                center_x + np.random.uniform(-1.0, 1.0),
                center_y + np.random.uniform(-1.0, 1.0)
            ])
            velocities[ent] = np.array([0.0, 0.0])
            continue
            
        velocities[ent] = (velocities[ent] + forces[ent] * dt) * damping
        positions[ent] = positions[ent] + velocities[ent] * dt
        
    # 4. Hard geometric overlap resolution (multiple iterations for stability)
    for _ in range(8):
        # Resolve bubble-bubble overlaps
        for i in range(len(active_ents)):
            entA = active_ents[i]
            rA = radii[entA]
            for j in range(i + 1, len(active_ents)):
                entB = active_ents[j]
                rB = radii[entB]
                
                delta = positions[entA] - positions[entB]
                dist = np.hypot(delta[0], delta[1])
                target_dist = (rA + rB) * 1.02  # keep 2% safety gap
                if dist < target_dist:
                    overlap = target_dist - dist
                    if dist > 0.001:
                        dir_vector = delta / dist
                    else:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dir_vector = np.array([np.cos(angle), np.sin(angle)])
                    
                    # Push apart geometrically (50% each)
                    positions[entA] += dir_vector * (overlap * 0.5)
                    positions[entB] -= dir_vector * (overlap * 0.5)
                    
                    # Prevent them from moving deeper into each other by projecting velocities
                    v_rel = velocities[entA] - velocities[entB]
                    vn = np.dot(v_rel, dir_vector)
                    if vn < 0:
                        # Cancel relative normal velocity
                        impulse = vn * 0.5
                        velocities[entA] -= impulse * dir_vector
                        velocities[entB] += impulse * dir_vector
                        
        # Enforce boundary constraints
        for ent in active_ents:
            r = radii[ent]
            positions[ent][0] = np.clip(positions[ent][0], r + 3.0, 97.0 - r)
            positions[ent][1] = np.clip(positions[ent][1], r + 3.0, axes_height - r - 3.0)


def _draw_bubble_scene(
    ax: plt.Axes,
    bubbles: list,
    entity_colors: dict,
    short_unit: str,
    axes_height: float,
    interp_frame: int,
) -> None:
    """Helper to draw all active bubbles, their halos, and text/flag labels."""
    for ent, cx, cy, radius, val in bubbles:
        color = entity_colors.get(ent, ACCENT_COLORS[0])

        # 1. Glow (zorder=1)
        glow = plt.Circle(
            (cx, cy), radius + 2.5,
            facecolor="none",
            edgecolor=to_rgba(color, alpha=0.15),
            linewidth=5.0, zorder=1,
        )
        ax.add_patch(glow)

        # 2. Main bubble (zorder=2)
        bubble = plt.Circle(
            (cx, cy), radius,
            facecolor=to_rgba(color, alpha=0.50),
            edgecolor=to_rgba(color, alpha=0.90),
            linewidth=2.5, zorder=2,
        )
        ax.add_patch(bubble)

        # 3. Inner highlight (zorder=3)
        highlight = plt.Circle(
            (cx - radius * 0.15, cy + radius * 0.15),
            radius * 0.4,
            facecolor=to_rgba(color, alpha=0.12),
            edgecolor="none", zorder=3,
        )
        ax.add_patch(highlight)

        # 4. Text Labels (zorder=5)
        clean_name = re.sub(r"\s*\([^)]*\)\s*$", "", ent).strip()
        val_str = format_value(val, short_unit)

        if radius >= 7.5:
            # Inside bubble labels
            flag_drawn = _add_flag_to_axes(ax, clean_name, cx, cy + radius * 0.35, box_alignment=(0.5, 0.5), size=20)
            if interp_frame == 0:
                print(f"[debug_bubble] Large Entity: {clean_name}, flag_drawn: {flag_drawn}, radius: {radius:.2f}")
            name_lines = textwrap.wrap(clean_name, width=max(8, int(radius * 0.75)))
            display_text = "\n".join(name_lines) + f"\n{val_str}"
            font_size = max(11, min(22, int(radius * 0.80)))
            text_y = cy - radius * 0.15 if flag_drawn else cy
            ax.text(
                cx, text_y, display_text,
                ha="center", va="center", multialignment="center",
                color="white", fontsize=font_size, fontproperties=FONT_BOLD,
                path_effects=OUTLINE, zorder=5,
            )
        elif radius >= 4.0:
            # Value inside, name outside
            ax.text(
                cx, cy, val_str,
                ha="center", va="center",
                color="white", fontsize=12, fontproperties=FONT_BOLD,
                path_effects=OUTLINE, zorder=5,
            )
            is_above = True
            label_y = cy + radius + 7
            if label_y > axes_height - 10:
                label_y = cy - radius - 7
                is_above = False

            ax.plot(
                [cx, cx], [cy + radius if is_above else cy - radius, label_y - 2 if is_above else label_y + 2],
                color=(*to_rgba(color)[:3], 0.4),
                linewidth=1.0, zorder=4,
            )
            trunc_name = clean_name if len(clean_name) <= 14 else clean_name[:11] + "..."

            # Compute flag position: to the left of text.
            flag_y = label_y + 0.8 if is_above else label_y - 0.8
            flag_w = _add_flag_to_axes(ax, clean_name, cx - 2.5, flag_y, box_alignment=(0.0, 0.5), size=18)
            if interp_frame == 0:
                print(f"[debug_bubble] Medium Entity: {clean_name}, flag_w: {flag_w}, radius: {radius:.2f}")

            if flag_w > 0:
                text_x = cx - 2.5 + (flag_w + 14.0) / 9.72
                ax.text(
                    text_x, label_y, trunc_name,
                    ha="left", va="bottom" if is_above else "top",
                    color=(*to_rgba("white")[:3], 0.85),
                    fontsize=12, fontproperties=FONT_REGULAR,
                    path_effects=OUTLINE, zorder=5,
                )
            else:
                ax.text(
                    cx, label_y, trunc_name,
                    ha="center", va="bottom" if is_above else "top",
                    color=(*to_rgba("white")[:3], 0.85),
                    fontsize=12, fontproperties=FONT_REGULAR,
                    path_effects=OUTLINE, zorder=5,
                )
        else:
            # Small labels completely outside bubble
            label_y = cy + radius + 6
            if label_y > axes_height - 10:
                label_y = cy - radius - 6
            ax.plot(
                [cx, cx], [cy + radius, label_y - 2],
                color=(*to_rgba(color)[:3], 0.3),
                linewidth=0.8, zorder=4,
            )
            trunc_name = clean_name if len(clean_name) <= 10 else clean_name[:8] + ".."
            ax.text(
                cx, label_y, f"{trunc_name}\n{val_str}",
                ha="center", va="bottom", multialignment="center",
                color=(*to_rgba("white")[:3], 0.7),
                fontsize=10, fontproperties=FONT_REGULAR,
                path_effects=OUTLINE, zorder=5,
            )


def _render_bubble_chart(
    df_seg: pd.DataFrame,
    topic_info: dict,
    extreme_segment: dict,
    entity_colors: dict,
    frames_dir: Path,
    output_path: Path,
) -> Path:
    """Render animated bubble chart Short — bubble size = value.

    Uses a force-directed layout simulation updated frame-by-frame to keep
    bubble positions fluid, organic, and overlapping naturally without fly-throughs.
    """
    start_yr = extreme_segment["start_year"]
    time_steps, step_values = _build_step_values(df_seg)
    n_steps = len(time_steps)
    frames_per_step = _compute_frames_per_step(n_steps)

    est_duration = (SHORT_INTRO_FRAMES + (n_steps - 1) * frames_per_step) / FPS
    print(f"[renderer_short] Bubble chart: {n_steps} steps, ~{est_duration:.0f}s")

    title = topic_info.get("topic", "Data Visualization")
    source = topic_info.get("source", "")
    hook = extreme_segment.get("hook", title)
    wrapped_title = "\n".join(textwrap.wrap(title, width=28))
    wrapped_hook = "\n".join(textwrap.wrap(hook, width=24))
    short_unit = topic_info.get("short_unit", "") if topic_info else ""

    fig = plt.figure(figsize=(10.8, 19.2), dpi=100)
    fig.patch.set_facecolor("#000000")

    # Isotropic coordinate system matching the axes physical dimensions
    axes_aspect = 13.44 / 9.72
    axes_height = 100.0 * axes_aspect  # ~138.27

    center_x = 50.0
    center_y = axes_height / 2.0

    # Global value range for consistent radius scaling
    all_step_vals = [v for sv in step_values for v in sv.values() if v > 0]
    global_max_val = max(all_step_vals) if all_step_vals else 1.0

    all_entities = sorted(df_seg["entity"].unique())

    # 1. Identify all entities that ever enter the top 10
    all_top_entities = set()
    for sv in step_values:
        top_ents = sorted(sv.keys(), key=lambda e: sv[e], reverse=True)[:TOP_N_ENTITIES]
        all_top_entities.update(top_ents)
    all_top_entities = sorted(list(all_top_entities))

    # 2. Get peak values for these entities in the segment
    peak_vals = {}
    for ent in all_top_entities:
        peak_vals[ent] = max(sv.get(ent, 0.0) for sv in step_values)

    # 3. Calculate packed positions at test scale
    R_test_max = 20.0
    pack_input = []
    for ent in all_top_entities:
        val = peak_vals[ent]
        r_test = R_test_max * np.sqrt(max(val, 0) / global_max_val)
        pack_input.append((ent, r_test))
    pack_input.sort(key=lambda x: x[1], reverse=True)

    packed_positions = _pack_circles(pack_input, 0.0, 0.0)

    # 4. Find bounding box of the packed layout
    X_min = min(packed_positions[ent][0] - r_test for ent, r_test in pack_input)
    X_max = max(packed_positions[ent][0] + r_test for ent, r_test in pack_input)
    Y_min = min(packed_positions[ent][1] - r_test for ent, r_test in pack_input)
    Y_max = max(packed_positions[ent][1] + r_test for ent, r_test in pack_input)

    W_packed = X_max - X_min
    H_packed = Y_max - Y_min

    # 5. Fit packed layout to screen dimensions with margins
    margin_x = 6.0
    margin_y = 12.0
    W_target = 100.0 - 2 * margin_x
    H_target = axes_height - 2 * margin_y
    S = min(W_target / W_packed, H_target / H_packed)

    X_center_packed = (X_min + X_max) / 2.0
    Y_center_packed = (Y_min + Y_max) / 2.0
    X_center_screen = 50.0
    Y_center_screen = axes_height / 2.0

    # Map entities to final screen positions
    final_positions = {}
    for ent in all_top_entities:
        px, py = packed_positions[ent]
        final_positions[ent] = (
            X_center_screen + S * (px - X_center_packed),
            Y_center_screen + S * (py - Y_center_packed)
        )

    # Final maximum radius scale factor on screen
    scale_factor = S * R_test_max

    # 6. Initialize positions and velocities for all entities
    np.random.seed(42)
    positions = {}
    velocities = {}
    for ent in all_entities:
        positions[ent] = np.array([
            center_x + np.random.uniform(-1.0, 1.0),
            center_y + np.random.uniform(-1.0, 1.0)
        ])
        velocities[ent] = np.array([0.0, 0.0])

    # 7. Pre-calculate starting step radii and run warm-up steps
    first_vals = step_values[0]
    sorted_ents_first = sorted(first_vals.keys(), key=lambda e: first_vals[e], reverse=True)
    top_ents_first = set(sorted_ents_first[:TOP_N_ENTITIES])

    initial_radii = {}
    for ent in all_entities:
        if ent in top_ents_first:
            val = first_vals.get(ent, 0.0)
            initial_radii[ent] = scale_factor * np.sqrt(max(val, 0) / global_max_val)
        else:
            initial_radii[ent] = 0.0

    # Run physics engine warm-up so initial frames start nicely clustered
    for _ in range(150):
        _run_bubble_physics_step(positions, velocities, initial_radii, center_x, center_y, axes_height, dt=0.25)

    # Capture state for intro frames
    intro_positions = {ent: pos.copy() for ent, pos in positions.items()}
    intro_radii = {ent: r for ent, r in initial_radii.items()}
    intro_vals = {ent: first_vals.get(ent, 0.0) for ent in all_entities}

    # Intro background drawing function
    def draw_bg(fig, ax):
        ax.set_facecolor("#000000")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, axes_height)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        sorted_ents = sorted(all_entities, key=lambda e: intro_radii.get(e, 0.0), reverse=True)
        for ent in sorted_ents:
            r = intro_radii.get(ent, 0.0)
            if r <= 2.0:
                continue
            cx, cy = intro_positions[ent]
            val = intro_vals[ent]
            color = entity_colors.get(ent, ACCENT_COLORS[0])

            # Glow
            glow = plt.Circle(
                (cx, cy), r + 2.5,
                facecolor="none",
                edgecolor=to_rgba(color, alpha=0.15),
                linewidth=5.0, zorder=1,
            )
            ax.add_patch(glow)
            # Main bubble
            bubble = plt.Circle(
                (cx, cy), r,
                facecolor=to_rgba(color, alpha=0.50),
                edgecolor=to_rgba(color, alpha=0.90),
                linewidth=2.5, zorder=2,
            )
            ax.add_patch(bubble)
            # Highlight
            highlight = plt.Circle(
                (cx - r * 0.15, cy + r * 0.15),
                r * 0.4,
                facecolor=to_rgba(color, alpha=0.12),
                edgecolor="none", zorder=3,
            )
            ax.add_patch(highlight)

    frame_number = _render_intro_frames(fig, wrapped_hook, wrapped_title, draw_bg, 0, frames_dir)

    # Re-create axes for the main video
    fig.clf()
    fig.patch.set_facecolor("#000000")
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.70])

    # Animate frame-by-frame using the physics engine
    for step_idx in range(len(time_steps) - 1):
        prev_vals = step_values[step_idx]
        next_vals = step_values[step_idx + 1]
        ts_start = pd.Timestamp(time_steps[step_idx])
        ts_end   = pd.Timestamp(time_steps[step_idx + 1])

        prev_sorted = sorted(prev_vals.keys(), key=lambda e: prev_vals[e], reverse=True)
        next_sorted = sorted(next_vals.keys(), key=lambda e: next_vals[e], reverse=True)

        prev_top = set(prev_sorted[:TOP_N_ENTITIES])
        next_top = set(next_sorted[:TOP_N_ENTITIES])

        for interp_frame in range(frames_per_step):
            t = _ease(interp_frame / frames_per_step)

            # Interpolate value and radius for all entities
            current_radii = {}
            current_vals = {}
            for ent in all_entities:
                v0 = prev_vals.get(ent, 0.0)
                v1 = next_vals.get(ent, 0.0)
                val = v0 + (v1 - v0) * t
                current_vals[ent] = val

                r0 = scale_factor * np.sqrt(max(v0, 0) / global_max_val) if ent in prev_top else 0.0
                r1 = scale_factor * np.sqrt(max(v1, 0) / global_max_val) if ent in next_top else 0.0
                current_radii[ent] = r0 + (r1 - r0) * t

            # Advance physics engine (4 sub-steps for stability and smoothness)
            for _ in range(4):
                _run_bubble_physics_step(positions, velocities, current_radii, center_x, center_y, axes_height, dt=0.25)

            ax.cla()
            ax.set_facecolor("#000000")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, axes_height)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            interp_year = ts_start.year + (ts_end.year - ts_start.year) * t
            date_label = str(int(interp_year))

            # Draw all active bubbles
            bubbles = []
            for ent in all_entities:
                r = current_radii[ent]
                if r > 2.0:
                    cx, cy = positions[ent]
                    val = current_vals[ent]
                    bubbles.append((ent, cx, cy, r, val))

            # Draw largest bubbles first (in the background)
            bubbles.sort(key=lambda b: b[3], reverse=True)

            _draw_bubble_scene(ax, bubbles, entity_colors, short_unit, axes_height, interp_frame)

            _draw_frame_chrome(ax, fig, wrapped_title, source, date_label, topic_info)
            _save_frame(fig, frame_number, frames_dir)
            frame_number += 1

    # Hold last frame
    final_radii = {ent: current_radii[ent] for ent in all_entities}
    final_vals = {ent: current_vals[ent] for ent in all_entities}
    for _ in range(FPS * 1):
        for _ in range(4):
            _run_bubble_physics_step(positions, velocities, final_radii, center_x, center_y, axes_height, dt=0.25)

        ax.cla()
        ax.set_facecolor("#000000")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, axes_height)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        bubbles = []
        for ent in all_entities:
            r = final_radii[ent]
            if r > 2.0:
                cx, cy = positions[ent]
                val = final_vals[ent]
                bubbles.append((ent, cx, cy, r, val))
        bubbles.sort(key=lambda b: b[3], reverse=True)

        _draw_bubble_scene(ax, bubbles, entity_colors, short_unit, axes_height, interp_frame=1)

        _draw_frame_chrome(ax, fig, wrapped_title, source, date_label, topic_info)
        _save_frame(fig, frame_number, frames_dir)
        frame_number += 1

    plt.close(fig)
    print(f"[renderer_short] Total frames: {frame_number}")
    _encode_short(frames_dir, output_path)
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# CHART TYPE 5: MAP ANIMATION
# ══════════════════════════════════════════════════════════════════════════════

# Country name → ISO A3 mapping for joining with natural earth geometry
_COUNTRY_TO_ISO = {
    "united states": "USA", "china": "CHN", "india": "IND", "germany": "DEU",
    "japan": "JPN", "brazil": "BRA", "france": "FRA", "united kingdom": "GBR",
    "russia": "RUS", "canada": "CAN", "australia": "AUS", "south korea": "KOR",
    "mexico": "MEX", "indonesia": "IDN", "italy": "ITA", "spain": "ESP",
    "turkey": "TUR", "saudi arabia": "SAU", "argentina": "ARG", "poland": "POL",
    "netherlands": "NLD", "switzerland": "CHE", "sweden": "SWE", "norway": "NOR",
    "belgium": "BEL", "austria": "AUT", "iran": "IRN", "thailand": "THA",
    "south africa": "ZAF", "egypt": "EGY", "nigeria": "NGA", "pakistan": "PAK",
    "bangladesh": "BGD", "vietnam": "VNM", "philippines": "PHL", "colombia": "COL",
    "chile": "CHL", "peru": "PER", "czech republic": "CZE", "romania": "ROU",
    "portugal": "PRT", "greece": "GRC", "hungary": "HUN", "israel": "ISR",
    "ireland": "IRL", "denmark": "DNK", "finland": "FIN", "new zealand": "NZL",
    "singapore": "SGP", "malaysia": "MYS", "ukraine": "UKR", "morocco": "MAR",
    "kenya": "KEN", "ethiopia": "ETH", "tanzania": "TZA", "ghana": "GHA",
    "dr congo": "COD", "algeria": "DZA", "iraq": "IRQ", "venezuela": "VEN",
    "cuba": "CUB", "north korea": "PRK", "myanmar": "MMR", "sri lanka": "LKA",
    "nepal": "NPL", "cambodia": "KHM", "laos": "LAO", "syria": "SYR",
    "jordan": "JOR", "lebanon": "LBN", "kuwait": "KWT", "qatar": "QAT",
    "bahrain": "BHR", "oman": "OMN", "yemen": "YEM", "afghanistan": "AFG",
    "uzbekistan": "UZB", "kazakhstan": "KAZ", "mongolia": "MNG",
    "papua new guinea": "PNG", "zimbabwe": "ZWE", "mozambique": "MOZ",
    "angola": "AGO", "cameroon": "CMR", "ivory coast": "CIV", "senegal": "SEN",
    "mali": "MLI", "burkina faso": "BFA", "niger": "NER", "chad": "TCD",
    "uganda": "UGA", "rwanda": "RWA", "madagascar": "MDG", "sudan": "SDN",
    "tunisia": "TUN", "libya": "LBY", "somalia": "SOM",
    "dominican republic": "DOM", "haiti": "HTI", "jamaica": "JAM",
    "trinidad and tobago": "TTO", "panama": "PAN", "costa rica": "CRI",
    "uruguay": "URY", "paraguay": "PRY", "bolivia": "BOL", "ecuador": "ECU",
    "el salvador": "SLV", "honduras": "HND", "guatemala": "GTM",
    "nicaragua": "NIC", "iceland": "ISL", "luxembourg": "LUX",
    "estonia": "EST", "latvia": "LVA", "lithuania": "LTU",
    "croatia": "HRV", "slovenia": "SVN", "slovakia": "SVK",
    "serbia": "SRB", "bosnia and herzegovina": "BIH", "albania": "ALB",
    "north macedonia": "MKD", "montenegro": "MNE", "bulgaria": "BGR",
    "belarus": "BLR", "moldova": "MDA", "georgia": "GEO",
    "armenia": "ARM", "azerbaijan": "AZE",
}


_COUNTRY_TO_ISO2 = {
    "united states": "us", "china": "cn", "india": "in", "germany": "de",
    "japan": "jp", "brazil": "br", "france": "fr", "united kingdom": "gb",
    "russia": "ru", "canada": "ca", "australia": "au", "south korea": "kr",
    "mexico": "mx", "indonesia": "id", "italy": "it", "spain": "es",
    "turkey": "tr", "saudi arabia": "sa", "argentina": "ar", "poland": "pl",
    "netherlands": "nl", "switzerland": "ch", "sweden": "se", "norway": "no",
    "belgium": "be", "austria": "at", "iran": "ir", "thailand": "th",
    "south africa": "za", "egypt": "eg", "nigeria": "ng", "pakistan": "pk",
    "bangladesh": "bd", "vietnam": "vn", "philippines": "ph", "colombia": "co",
    "chile": "cl", "peru": "pe", "czech republic": "cz", "romania": "ro",
    "portugal": "pt", "greece": "gr", "hungary": "hu", "israel": "il",
    "ireland": "ie", "denmark": "dk", "finland": "fi", "new zealand": "nz",
    "singapore": "sg", "malaysia": "my", "ukraine": "ua", "morocco": "ma",
    "kenya": "ke", "ethiopia": "et", "tanzania": "tz", "ghana": "gh",
    "dr congo": "cd", "algeria": "dz", "iraq": "iq", "venezuela": "ve",
    "cuba": "cu", "north korea": "kp", "myanmar": "mm", "sri lanka": "lk",
    "nepal": "np", "cambodia": "kh", "laos": "la", "syria": "sy",
    "jordan": "jo", "lebanon": "lb", "kuwait": "kw", "qatar": "qa",
    "bahrain": "bh", "oman": "om", "yemen": "ye", "afghanistan": "af",
    "uzbekistan": "uz", "kazakhstan": "kz", "mongolia": "mn",
    "papua new guinea": "pg", "zimbabwe": "zw", "mozambique": "mz",
    "angola": "ao", "cameroon": "cm", "ivory coast": "ci", "senegal": "sn",
    "mali": "ml", "burkina faso": "bf", "niger": "ne", "chad": "td",
    "uganda": "ug", "rwanda": "rw", "madagascar": "mg", "sudan": "sd",
    "tunisia": "tn", "libya": "ly", "somalia": "so",
    "dominican republic": "do", "haiti": "ht", "jamaica": "jm",
    "trinidad and tobago": "tt", "panama": "pa", "costa rica": "cr",
    "uruguay": "uy", "paraguay": "py", "bolivia": "bo", "ecuador": "ec",
    "el salvador": "sv", "honduras": "hn", "guatemala": "gt",
    "nicaragua": "ni", "iceland": "is", "luxembourg": "lu",
    "estonia": "ee", "latvia": "lv", "lithuania": "lt",
    "croatia": "hr", "slovenia": "si", "slovakia": "sk",
    "serbia": "rs", "bosnia and herzegovina": "ba", "albania": "al",
    "north macedonia": "mk", "montenegro": "me", "bulgaria": "bg",
    "belarus": "by", "moldova": "md", "georgia": "ge",
    "armenia": "am", "azerbaijan": "az",
    "eswatini": "sz", "timor-leste": "tl", "cote d'ivoire": "ci",
}


_ISO3_TO_ISO2 = {
    "USA": "us", "CHN": "cn", "IND": "in", "DEU": "de", "JPN": "jp", "BRA": "br", "FRA": "fr", "GBR": "gb",
    "RUS": "ru", "CAN": "ca", "AUS": "au", "KOR": "kr", "MEX": "mx", "IDN": "id", "ITA": "it", "ESP": "es",
    "TUR": "tr", "SAU": "sa", "ARG": "ar", "POL": "pl", "NLD": "nl", "CHE": "ch", "SWE": "se", "NOR": "no",
    "BEL": "be", "AUT": "at", "IRN": "ir", "THA": "th", "ZAF": "za", "EGY": "eg", "NGA": "ng", "PAK": "pk",
    "BGD": "bd", "VNM": "vn", "PHL": "ph", "COL": "co", "CHL": "cl", "PER": "pe", "CZE": "cz", "ROU": "ro",
    "PRT": "pt", "GRC": "gr", "HUN": "hu", "ISR": "il", "IRL": "ie", "DNK": "dk", "FIN": "fi", "NZL": "nz",
    "SGP": "sg", "MYS": "my", "UKR": "ua", "MAR": "ma", "KEN": "ke", "ETH": "et", "TZA": "tz", "GHA": "gh",
    "COD": "cd", "DZA": "dz", "IRQ": "iq", "VEN": "ve", "CUB": "cu", "PRK": "kp", "MMR": "mm", "LKA": "lk",
    "NPL": "np", "KHM": "kh", "LAO": "la", "SYR": "sy", "JOR": "jo", "LBN": "lb", "KWT": "kw", "QAT": "qa",
    "BHR": "bh", "OMN": "om", "YEM": "ye", "AFG": "af", "UZB": "uz", "KAZ": "kz", "MNG": "mn", "PNG": "pg",
    "ZWE": "zw", "MOZ": "mz", "AGO": "ao", "CMR": "cm", "CIV": "ci", "SEN": "sn", "MLI": "ml", "BFA": "bf",
    "NER": "ne", "TCD": "td", "UGA": "ug", "RWA": "rw", "MDG": "mg", "SDN": "sd", "TUN": "tn", "LBY": "ly",
    "SOM": "so", "DOM": "do", "HTI": "ht", "JAM": "jm", "TTO": "tt", "PAN": "pa", "CRI": "cr", "URY": "uy",
    "PRY": "py", "BOL": "bo", "ECU": "ec", "SLV": "sv", "HND": "hn", "GTM": "gt", "NIC": "ni", "ISL": "is",
    "LUX": "lu", "EST": "ee", "LVA": "lv", "LTU": "lt", "HRV": "hr", "SVN": "si", "SVK": "sk", "SRB": "rs",
    "BIH": "ba", "ALB": "al", "MKD": "mk", "MNE": "me", "BGR": "bg", "BLR": "by", "MDA": "md", "GEO": "ge",
    "ARM": "am", "AZE": "az", "SWZ": "sz", "TLS": "tl",
}


def _get_flag_path(entity: str) -> Optional[str]:
    """Get path to the cached flag PNG for an entity, downloading if necessary."""
    name_lower = entity.lower().strip()
    
    # 1. Resolve to ISO-2 lowercase code
    iso2 = None
    if name_lower in _COUNTRY_TO_ISO2:
        iso2 = _COUNTRY_TO_ISO2[name_lower]
    else:
        # Fallback to world geometry / ISO-3 mapping
        if name_lower in _COUNTRY_TO_ISO:
            iso3 = _COUNTRY_TO_ISO[name_lower]
            iso2 = _ISO3_TO_ISO2.get(iso3, None)

    if not iso2:
        return None

    # 2. Local cache directory setup
    flags_dir = Path("assets/flags")
    flags_dir.mkdir(parents=True, exist_ok=True)
    flag_path = flags_dir / f"{iso2}.png"

    # 3. Download if not exists
    if not flag_path.exists():
        url = f"https://flagcdn.com/w80/{iso2}.png"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                flag_path.write_bytes(r.content)
            else:
                return None
        except Exception as e:
            print(f"[renderer_short] Failed to download flag for {entity} from {url}: {e}")
            return None

    return str(flag_path)


def _add_flag_to_axes(ax: plt.Axes, entity: str, x: float, y: float, box_alignment=(0.0, 0.5), size=16) -> int:
    """Fetch/load flag for entity and add to axes at (x, y) with AnnotationBbox.
    Returns the width of the resized flag image in points (pixels), or 0 if failed.
    """
    flag_path = _get_flag_path(entity)
    if not flag_path:
        return 0
    try:
        flag_img = Image.open(flag_path).convert("RGBA")
        # Resize standard height, keeping aspect ratio
        w, h = flag_img.size
        new_h = size
        new_w = int(w * (new_h / h))
        flag_img = flag_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        im = OffsetImage(flag_img, zoom=1.0)
        ab = AnnotationBbox(
            im, (x, y),
            xycoords='data',
            frameon=False,
            box_alignment=box_alignment,
            pad=0.0,
        )
        ab.set_zorder(5)
        ax.add_artist(ab)
        return new_w
    except Exception as e:
        print(f"[renderer_short] Failed to draw flag for {entity}: {e}")
        return 0


def _load_world_geometry():
    """Load world country geometry from geopandas datasets."""
    try:
        import geopandas as gpd
        import ssl
        from shapely.geometry import Polygon, MultiPolygon
        # Disable SSL verification for naturalearth download on macOS/proxies
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except Exception:
            pass

        # Try built-in dataset first
        try:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        except Exception:
            # Fallback: download from Natural Earth
            url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
            world = gpd.read_file(url)

        # Normalize column names to lowercase to support uppercase schemas in Natural Earth
        world.columns = [c.lower() for c in world.columns]

        # Normalize names for matching
        world["name_lower"] = world["name"].str.lower().str.strip()

        # ── Official J&K / Ladakh / Aksai Chin correction ──────────────
        # Uses exact naturalearth vertex coordinates on the LoC (India-Pakistan)
        # and LAC (India-China) shared borders to produce seamless edges.
        # Two separate capture polygons: one for AJK+GB, one for Aksai Chin.

        # AJK + Gilgit-Baltistan capture polygon:
        # East side follows the LoC (PAK-IND shared vertices),
        # north/west follows PAK-CHN and PAK-AFG border vertices,
        # cutting line from LoC bottom to Durand Line excludes KP.
        ajk_gb_capture = Polygon([
            (74.42, 30.98),   # LoC bottom (international border junction)
            (70.88, 33.99),   # Diagonal cut to Afghan border (excludes KP)
            (71.16, 34.35),   # PAK-AFG border northward
            (71.12, 34.73),
            (71.61, 35.15),
            (71.50, 35.65),
            (71.26, 36.07),
            (71.85, 36.51),
            (72.92, 36.72),
            (74.07, 36.84),
            (74.58, 37.02),
            (75.16, 37.13),   # PAK-CHN border
            (75.90, 36.67),
            (76.19, 35.90),
            (77.84, 35.49),   # Triple junction (PAK-IND-CHN)
            (76.87, 34.65),   # LoC southward (shared vertices)
            (75.76, 34.50),
            (74.24, 34.75),
            (73.75, 34.32),
            (74.10, 33.44),
            (74.45, 32.76),
            (75.26, 32.27),
            (74.41, 31.69),
            (74.42, 30.98),   # Close
        ])

        # Aksai Chin capture polygon:
        # West side follows the LAC (CHN-IND shared vertices),
        # east/south sweeps through India (no CHN territory there).
        aksai_chin_capture = Polygon([
            (77.84, 35.49),   # Triple junction
            (74.0, 35.49),    # West into India (no CHN here)
            (74.0, 29.0),     # South (below the region)
            (82.0, 29.0),     # East
            (82.0, 30.18),    # Up to LAC bottom
            (81.11, 30.18),   # LAC northward (shared vertices)
            (79.72, 30.88),
            (78.74, 31.52),
            (78.46, 32.62),
            (79.18, 32.48),
            (79.21, 32.99),
            (78.81, 33.51),
            (78.91, 34.32),
            (77.84, 35.49),   # Close
        ])

        pak_kashmir = None
        chn_kashmir = None

        def remove_holes(geom):
            if geom.is_empty:
                return geom
            if geom.geom_type == "Polygon":
                return Polygon(geom.exterior)
            elif geom.geom_type == "MultiPolygon":
                return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
            return geom

        def remove_slivers(geom, min_area=0.05):
            if geom.is_empty:
                return geom
            if geom.geom_type == "Polygon":
                return geom if geom.area >= min_area else Polygon()
            elif geom.geom_type == "MultiPolygon":
                valid_polys = [p for p in geom.geoms if p.area >= min_area]
                if not valid_polys:
                    return Polygon()
                elif len(valid_polys) == 1:
                    return valid_polys[0]
                else:
                    return MultiPolygon(valid_polys)
            return geom

        # 1. Extract and subtract AJK+GB from Pakistan
        pak_idx = world[world["iso_a3"] == "PAK"].index
        if not pak_idx.empty:
            pak_geom = world.loc[pak_idx[0], "geometry"]
            pak_kashmir = pak_geom.intersection(ajk_gb_capture)
            world.loc[pak_idx[0], "geometry"] = remove_slivers(pak_geom.difference(pak_kashmir))

        # 2. Extract and subtract Aksai Chin from China
        chn_idx = world[world["iso_a3"] == "CHN"].index
        if not chn_idx.empty:
            chn_geom = world.loc[chn_idx[0], "geometry"]
            chn_kashmir = chn_geom.intersection(aksai_chin_capture)
            world.loc[chn_idx[0], "geometry"] = remove_slivers(chn_geom.difference(chn_kashmir))

        # 3. Union both parts with India
        ind_idx = world[world["iso_a3"] == "IND"].index
        if not ind_idx.empty:
            ind_geom = world.loc[ind_idx[0], "geometry"]
            if pak_kashmir is not None and not pak_kashmir.is_empty:
                ind_geom = ind_geom.union(pak_kashmir)
            if chn_kashmir is not None and not chn_kashmir.is_empty:
                ind_geom = ind_geom.union(chn_kashmir)
            # Apply buffer closing and remove holes/slivers
            world.loc[ind_idx[0], "geometry"] = remove_slivers(
                remove_holes(ind_geom.buffer(0.005).buffer(-0.005))
            )

        return world
    except Exception as e:
        print(f"[renderer_short] Failed to load world geometry: {e}")
        return None


def _match_entity_to_iso(entity: str, world_df) -> Optional[str]:
    """Match an entity name to an ISO A3 code."""
    name_lower = entity.lower().strip()

    # Direct lookup
    if name_lower in _COUNTRY_TO_ISO:
        return _COUNTRY_TO_ISO[name_lower]

    # Try matching against world dataframe names
    if world_df is not None:
        match = world_df[world_df["name_lower"] == name_lower]
        if not match.empty:
            return match.iloc[0].get("iso_a3", None)

    return None


def _render_map_chart(
    df_seg: pd.DataFrame,
    topic_info: dict,
    extreme_segment: dict,
    entity_colors: dict,
    frames_dir: Path,
    output_path: Path,
) -> Path:
    """Render choropleth map animation Short.

    Shows a world map with countries colored by value using a sequential
    colormap (plasma). Annotates the top 5 countries with value labels.
    Includes a horizontal colorbar legend at the bottom.

    Falls back to bar_chart_race if geopandas/geometry unavailable or
    too few countries match.
    """
    world = _load_world_geometry()
    if world is None:
        print("[renderer_short] Map animation unavailable, falling back to bar chart race.")
        return _render_bar_chart_race(df_seg, topic_info, extreme_segment, entity_colors, frames_dir, output_path)

    start_yr = extreme_segment["start_year"]
    time_steps, step_values = _build_step_values(df_seg)
    n_steps = len(time_steps)
    frames_per_step = _compute_frames_per_step(n_steps)

    # Build entity→ISO mapping
    all_entities = sorted(df_seg["entity"].unique())
    entity_iso = {}
    for entity in all_entities:
        iso = _match_entity_to_iso(entity, world)
        if iso:
            entity_iso[entity] = iso

    matched_count = len(entity_iso)
    print(f"[renderer_short] Map: matched {matched_count}/{len(all_entities)} entities to countries")

    if matched_count < 3:
        print("[renderer_short] Too few country matches for map. Falling back to bar chart race.")
        return _render_bar_chart_race(df_seg, topic_info, extreme_segment, entity_colors, frames_dir, output_path)

    est_duration = (SHORT_INTRO_FRAMES + (n_steps - 1) * frames_per_step) / FPS
    print(f"[renderer_short] Map chart: {n_steps} steps, ~{est_duration:.0f}s")

    title = topic_info.get("topic", "Data Visualization")
    source = topic_info.get("source", "")
    hook = extreme_segment.get("hook", title)
    wrapped_title = "\n".join(textwrap.wrap(title, width=28))
    wrapped_hook = "\n".join(textwrap.wrap(hook, width=24))
    short_unit = topic_info.get("short_unit", "") if topic_info else ""

    # Global value range for consistent color scaling
    all_vals = [v for sv in step_values for v in sv.values() if v > 0]
    val_min = min(all_vals) if all_vals else 0
    val_max = max(all_vals) if all_vals else 1

    # Colormap for choropleth
    cmap = plt.cm.plasma

    fig = plt.figure(figsize=(10.8, 19.2), dpi=100)
    fig.patch.set_facecolor("#000000")

    # Pre-compute country centroids for labeling (using representative_point)
    country_centroids = {}
    for entity, iso in entity_iso.items():
        row = world[world["iso_a3"] == iso]
        if not row.empty:
            try:
                pt = row.geometry.iloc[0].representative_point()
                country_centroids[entity] = (pt.x, pt.y)
            except Exception:
                pass

    # Intro
    def draw_bg(fig, ax):
        ax.set_facecolor("#000000")
        ax.set_axis_off()
        try:
            world.plot(ax=ax, color="#1a1a2e", edgecolor="#333333", linewidth=0.3)
        except Exception:
            pass

    frame_number = _render_intro_frames(fig, wrapped_hook, wrapped_title, draw_bg, 0, frames_dir)

    fig.clf()
    fig.patch.set_facecolor("#000000")
    # Map axes — use full width and most of the height for maximum map size
    ax = fig.add_axes([0.0, 0.12, 1.0, 0.76])

    # Animate map
    for step_idx in range(len(time_steps) - 1):
        prev_vals = step_values[step_idx]
        next_vals = step_values[step_idx + 1]
        ts_start = pd.Timestamp(time_steps[step_idx])
        ts_end   = pd.Timestamp(time_steps[step_idx + 1])

        for interp_frame in range(frames_per_step):
            alpha_t = _ease(interp_frame / frames_per_step)
            ax.cla()
            ax.set_facecolor("#000000")
            ax.set_axis_off()

            interp_year = ts_start.year + (ts_end.year - ts_start.year) * alpha_t
            date_label = str(int(interp_year))

            # Interpolate values
            interp_vals = {}
            for entity in entity_iso:
                v0 = prev_vals.get(entity, 0.0)
                v1 = next_vals.get(entity, 0.0)
                interp_vals[entity] = v0 + (v1 - v0) * alpha_t

            # Build ISO → value map
            iso_values = {}
            for entity, iso in entity_iso.items():
                if entity in interp_vals:
                    iso_values[iso] = interp_vals[entity]

            # Create color column
            world_copy = world.copy()
            world_copy["_value"] = world_copy["iso_a3"].map(iso_values)

            # Base map (unmatched countries)
            world_copy[world_copy["_value"].isna()].plot(
                ax=ax, color="#1a1a2e", edgecolor="#2a2a3e", linewidth=0.3,
            )

            # Choropleth (matched countries)
            matched = world_copy[world_copy["_value"].notna()]
            if not matched.empty:
                matched.plot(
                    ax=ax, column="_value", cmap="plasma",
                    vmin=val_min, vmax=val_max,
                    edgecolor="#888888", linewidth=0.7,
                    legend=False,
                )

            # Crop out Antarctica and empty ocean — zoom into populated areas
            ax.set_xlim(-180, 180)
            ax.set_ylim(-58, 85)

            # Label top 5 countries with values on the map
            if DRAW_MAP_LABELS:
                top5 = sorted(interp_vals.items(), key=lambda kv: kv[1], reverse=True)[:5]
                for rank, (entity, val) in enumerate(top5):
                    if entity in country_centroids:
                        cx, cy = country_centroids[entity]
                        clean_name = re.sub(r"\s*\([^)]*\)\s*$", "", entity).strip()
                        if len(clean_name) > 12:
                            clean_name = clean_name[:10] + ".."
                        val_str = format_value(val, short_unit)

                        # Background pill for readability
                        label_text = f"{clean_name}\n{val_str}"
                        ax.text(
                            cx, cy, label_text,
                            ha="center", va="center", multialignment="center",
                            color="white", fontsize=13 if rank == 0 else 10,
                            fontproperties=FONT_BOLD,
                            path_effects=OUTLINE, zorder=10,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="#000000",
                                edgecolor="none",
                                alpha=0.55,
                            ),
                        )

            # Draw a horizontal colorbar below the map
            cbar_ax = fig.add_axes([0.15, 0.09, 0.70, 0.012])
            norm = plt.Normalize(vmin=val_min, vmax=val_max)
            sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
            cbar.ax.tick_params(labelsize=9, colors="white", length=3)
            cbar.outline.set_edgecolor("#444444")
            cbar.outline.set_linewidth(0.5)

            # Colorbar label
            unit_label = short_unit if short_unit else ""
            if unit_label:
                cbar.set_label(unit_label, color="white", fontsize=10, labelpad=4)

            _draw_frame_chrome(ax, fig, wrapped_title, source, date_label, topic_info)
            _save_frame(fig, frame_number, frames_dir)
            frame_number += 1

            # Remove the colorbar axes so it doesn't accumulate
            fig.delaxes(cbar_ax)

    # Hold last frame (re-add colorbar for final hold)
    cbar_ax = fig.add_axes([0.15, 0.10, 0.70, 0.015])
    norm = plt.Normalize(vmin=val_min, vmax=val_max)
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=9, colors="white", length=3)
    cbar.outline.set_edgecolor("#444444")
    cbar.outline.set_linewidth(0.5)

    for _ in range(FPS * 1):
        _save_frame(fig, frame_number, frames_dir)
        frame_number += 1

    plt.close(fig)
    print(f"[renderer_short] Total frames: {frame_number}")
    _encode_short(frames_dir, output_path)
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# FFmpeg ENCODER
# ══════════════════════════════════════════════════════════════════════════════

def _encode_short(frames_dir: Path, output_path: Path) -> None:
    """Encode all Short PNG frames into an MP4 with optional background music."""
    print("[renderer_short] Encoding Short with FFmpeg...")

    music_files = list(MUSIC_DIR.glob("*.mp3")) + list(MUSIC_DIR.glob("*.wav"))

    if music_files:
        bg_music = random.choice(music_files)
        print(f"[renderer_short] Adding background music: {bg_music.name}")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", str(frames_dir / "frame_%05d.png"),
            "-stream_loop", "-1",
            "-i", str(bg_music),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "slow",
            "-c:a", "aac",
            "-b:a", "192k",
            "-filter:a", f"volume={DEFAULT_VOLUME}",
            "-shortest",
            "-ac", "2",
            "-ar", "44100",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", str(frames_dir / "frame_%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "slow",
            str(output_path),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")
    print(f"[renderer_short] Encoded: {output_path}")


def _pack_circles(entities_radii: list[tuple[str, float]], center_x: float, center_y: float):
    """Deterministic spiral-based circle packing.

    Places the largest circle at center, then places subsequent circles
    in a tight spiral around already-placed circles.

    Args:
        entities_radii: List of (entity_name, radius) sorted by radius descending.
        center_x: X center of the layout area.
        center_y: Y center of the layout area.

    Returns:
        Dict mapping entity name to (x, y) position.
    """
    positions = {}
    if not entities_radii:
        return positions

    # Place largest at center
    positions[entities_radii[0][0]] = (center_x, center_y)

    for idx in range(1, len(entities_radii)):
        ent, r = entities_radii[idx]
        best_pos = None
        best_dist = float('inf')

        # Try angles around each already-placed circle
        for placed_ent, (px, py) in list(positions.items()):
            placed_r = dict(entities_radii)[placed_ent]
            target_dist = placed_r + r + 4.0  # 4 units gap

            for angle_deg in range(0, 360, 15):
                angle = np.radians(angle_deg)
                cx = px + target_dist * np.cos(angle)
                cy = py + target_dist * np.sin(angle)

                # Check overlap with all placed circles
                overlap = False
                for other_ent, (ox, oy) in positions.items():
                    other_r = dict(entities_radii)[other_ent]
                    if np.hypot(cx - ox, cy - oy) < r + other_r + 3.0:
                        overlap = True
                        break

                if not overlap:
                    dist_to_center = np.hypot(cx - center_x, cy - center_y)
                    if dist_to_center < best_dist:
                        best_dist = dist_to_center
                        best_pos = (cx, cy)

        if best_pos is None:
            # Fallback: spiral outward from center
            for spiral_r in np.arange(r + 5, 80, 3):
                for angle_deg in range(0, 360, 10):
                    angle = np.radians(angle_deg)
                    cx = center_x + spiral_r * np.cos(angle)
                    cy = center_y + spiral_r * np.sin(angle)
                    overlap = False
                    for other_ent, (ox, oy) in positions.items():
                        other_r_val = dict(entities_radii)[other_ent]
                        if np.hypot(cx - ox, cy - oy) < r + other_r_val + 3.0:
                            overlap = True
                            break
                    if not overlap:
                        best_pos = (cx, cy)
                        break
                if best_pos:
                    break

        if best_pos is None:
            best_pos = (center_x + idx * 8, center_y)

        positions[ent] = best_pos

    return positions

