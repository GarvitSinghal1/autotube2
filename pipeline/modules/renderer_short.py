"""
renderer_short.py — Renders the YouTube Short from the extreme segment.

Vertical format (1080x1920), 50–59 seconds. Uses the same fixed-slot bar chart
race system as renderer_long. Shares color assignments and utility functions.
"""

import random
import subprocess
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.config import (
    ACCENT_COLORS, FPS, FRAMES_SHORT_DIR, SHORT_FINAL,
    SHORT_MIN_DURATION, SHORT_MAX_DURATION, MUSIC_DIR,
    DEFAULT_VOLUME, TOP_N_ENTITIES, TMP_DIR,
)
from pipeline.modules.renderer_long import (
    SLOTS, OFF_SCREEN_Y, BAR_HEIGHT,
    assign_entity_colors, format_value, _rank_entities, _ease,
    _draw_intro_frame,
)

# Short-form constants
SHORT_FRAMES_PER_STEP = 15   # fewer frames for faster pacing
SHORT_INTRO_FRAMES = 90      # same intro: 45 static + 45 animated


def render_short(
    df_yearly: pd.DataFrame,
    chart_type: str,
    topic_info: dict,
    extreme_segment: dict,
    entity_colors: Optional[dict] = None,
) -> tuple[Path, dict[str, str]]:
    """Render the YouTube Short video from the extreme segment.

    Args:
        df_yearly: Yearly DataFrame with columns [date, entity, value].
        chart_type: Ignored — always renders bar chart race.
        topic_info: Dict with keys: topic, description, source, hook.
        extreme_segment: Dict with start_year, end_year, reason, hook.
        entity_colors: Pre-assigned color mapping from the long-form render.

    Returns:
        (output_path, entity_colors_used)
    """
    start_yr = extreme_segment["start_year"]
    end_yr   = extreme_segment["end_year"]
    print(f"[renderer_short] Rendering Short: {start_yr}–{end_yr}")

    FRAMES_SHORT_DIR.mkdir(parents=True, exist_ok=True)
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

    time_steps = sorted(df_seg["date"].unique())
    n_steps = len(time_steps)
    print(f"[renderer_short] Time steps: {n_steps}")

    # Build per-timestep value dicts
    step_values: list[dict[str, float]] = []
    for ts in time_steps:
        row = df_seg[df_seg["date"] == ts]
        step_values.append({r.entity: float(r.value) for r in row.itertuples()})

    # Calculate frames per step to hit ~50s target (minus intro)
    usable_frames = SHORT_MIN_DURATION * FPS - SHORT_INTRO_FRAMES
    frames_per_step = max(SHORT_FRAMES_PER_STEP, usable_frames // max(n_steps - 1, 1))
    # Cap to SHORT_MAX_DURATION
    max_frames = SHORT_MAX_DURATION * FPS - SHORT_INTRO_FRAMES
    frames_per_step = min(frames_per_step, max_frames // max(n_steps - 1, 1))
    frames_per_step = max(frames_per_step, 4)

    est_duration = (SHORT_INTRO_FRAMES + (n_steps - 1) * frames_per_step) / FPS
    print(f"[renderer_short] Frames/step: {frames_per_step}, Duration: {est_duration:.0f}s")

    title = topic_info.get("topic", "Data Visualization")
    source = topic_info.get("source", "")
    hook = extreme_segment.get("hook", title)

    # Create figure once
    fig = plt.figure(figsize=(10.8, 19.2), dpi=100)
    fig.patch.set_facecolor("#000000")
    ax = fig.add_axes([0.20, 0.08, 0.75, 0.78])

    frame_number = 0

    # ── Intro ────────────────────────────────────────────────────────────
    for f in range(SHORT_INTRO_FRAMES):
        _draw_intro_frame(fig, hook, title, f, SHORT_INTRO_FRAMES, FRAMES_SHORT_DIR, frame_number)
        frame_number += 1

    # ── Chart animation ──────────────────────────────────────────────────
    prev_ranks: dict[str, int] = {}
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

            interp_vals: dict[str, float] = {}
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
                y_pos  = prev_y + (cur_y - prev_y) * alpha

                entities_data.append({
                    "entity": entity,
                    "value":  interp_vals[entity],
                    "y_pos":  y_pos,
                    "color":  entity_colors.get(entity, ACCENT_COLORS[0]),
                })

            interp_ts = ts_start + (ts_end - ts_start) * alpha
            date_label = str(interp_ts.year)

            _draw_short_chart_frame(ax, fig, entities_data, title, source, date_label,
                                    FRAMES_SHORT_DIR, frame_number)
            frame_number += 1

        # Update prev_ranks to end state of this step
        prev_ranks = {e: r for r, e in enumerate(
            sorted(next_vals.keys(), key=lambda k: next_vals[k], reverse=True)
        )}

    # Hold last frame for 2 seconds
    for _ in range(FPS * 2):
        _draw_short_chart_frame(ax, fig, entities_data, title, source, date_label,
                                FRAMES_SHORT_DIR, frame_number)
        frame_number += 1

    plt.close(fig)
    print(f"[renderer_short] Total frames: {frame_number}")

    _encode_short(FRAMES_SHORT_DIR, SHORT_FINAL)
    print(f"[renderer_short] Output: {SHORT_FINAL}")
    return SHORT_FINAL, entity_colors


# ── Private helpers ───────────────────────────────────────────────────────────

def _draw_short_chart_frame(
    ax: plt.Axes,
    fig: plt.Figure,
    entities_data: list[dict],
    title: str,
    source: str,
    date_label: str,
    frames_dir: Path,
    frame_number: int,
) -> None:
    """Draw a single vertical-format chart frame and save to disk."""
    ax.cla()

    ax.set_facecolor("#000000")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 9.6)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors="white", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#444444")

    if not entities_data:
        plt.savefig(frames_dir / f"frame_{frame_number:05d}.png",
                    dpi=100, facecolor="#000000", pad_inches=0)
        return

    max_value = max(d["value"] for d in entities_data) * 1.1
    if max_value <= 0:
        max_value = 1.0

    for d in entities_data:
        norm_val = d["value"] / max_value
        y = d["y_pos"]
        color = d["color"]

        ax.barh(y, norm_val, height=BAR_HEIGHT, color=color, alpha=0.9, left=0)

        # Entity name — left margin
        ax.text(
            -0.01, y, d["entity"],
            ha="right", va="center",
            color="white", fontsize=9, fontweight="bold",
        )

        # Value label — right end of bar
        ax.text(
            norm_val + 0.01, y, format_value(d["value"]),
            ha="left", va="center",
            color="white", fontsize=8,
        )

    # X-axis tick labels
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([format_value(t * max_value) for t in ticks], color="white", fontsize=7)

    # Ghost year — figure-level, bottom right, large and transparent
    fig.text(
        0.95, 0.05, date_label,
        ha="right", va="bottom",
        color="white", alpha=0.15,
        fontsize=80, fontweight="bold",
        transform=fig.transFigure,
    )

    # Title — figure-level, top
    fig.text(
        0.5, 0.97, title,
        ha="center", va="top",
        color="white", fontsize=12, fontweight="bold",
        transform=fig.transFigure,
    )

    if source:
        fig.text(
            0.5, 0.93, f"Source: {source}",
            ha="center", va="top",
            color="#888888", fontsize=8, style="italic",
            transform=fig.transFigure,
        )

    plt.savefig(
        frames_dir / f"frame_{frame_number:05d}.png",
        dpi=100,
        facecolor="#000000",
        pad_inches=0,
    )


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
