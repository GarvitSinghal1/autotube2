"""
renderer_long.py — Renders the full long-form animated data visualization video.

Uses a fixed-slot bar chart race system. 10 bars, 10 fixed y-positions. Entities
animate between slots smoothly. X-axis is normalized 0-1 each frame — never rescaled.
Resolution: 1920x1080 (16:9). Approximate duration: 5–10 min at 30fps.
"""

import random
import re
import subprocess
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.config import (
    ACCENT_COLORS, BG_COLOR, FPS, FRAMES_LONG_DIR, LONG_FORM_FINAL,
    LONG_FORM_MAX_DURATION, LONG_FORM_MIN_DURATION, MUSIC_DIR,
    DEFAULT_VOLUME, TOP_N_ENTITIES, TMP_DIR,
)
from pipeline.modules.font_loader import FONT_BOLD, FONT_REGULAR, overlay_watermark

# ── Constants ────────────────────────────────────────────────────────────────

# Fixed y-positions for each rank slot (slot 0 = rank 1 = top)
SLOTS = {
    0: 9.0,
    1: 8.0,
    2: 7.0,
    3: 6.0,
    4: 5.0,
    5: 4.0,
    6: 3.0,
    7: 2.0,
    8: 1.0,
    9: 0.0,
}
OFF_SCREEN_Y = -2.0   # y position for entities outside top-10

LONG_FRAMES_PER_STEP = 30   # interpolation frames between consecutive time steps
INTRO_FRAMES = 90           # 45 static + 45 animated fade
BAR_HEIGHT = 0.65


# ── Public API ───────────────────────────────────────────────────────────────

def assign_entity_colors(entities: list[str]) -> dict[str, str]:
    """Return a stable color mapping for a list of entity names, maximizing distinct colors."""
    # Permuted color palette designed to maximize hue contrast between adjacent items
    high_contrast_colors = [
        "#FF2A6D",  # Neon Rose / Hot Pink
        "#05D9E8",  # Neon Cyan / Electric Blue
        "#39FF14",  # Neon Green / Lime
        "#FFCC00",  # Neon Gold / Yellow
        "#B026FF",  # Neon Purple / Violet
        "#FF5E36",  # Neon Orange / Coral
        "#0072FF",  # Neon Bright Blue
        "#FF00FF",  # Neon Magenta
        "#00F5D4",  # Neon Bright Teal / Mint
        "#FF9F1C",  # Neon Tangerine
        "#9B5DE5",  # Neon Lavender
        "#00BFFF",  # Neon Deep Sky Blue
        "#F15BB5",  # Neon Pink
        "#ADFF2F",  # Neon Lime Green
        "#EE5253",  # Neon Bright Red
        "#54A0FF",  # Neon Ice Blue
        "#A55EEA",  # Neon Violet / Purple
        "#10AC84",  # Neon Dark Teal
        "#FF6B6B",  # Neon Coral
        "#70A1FF",  # Neon Periwinkle Blue
    ]
    sorted_entities = sorted(entities)
    mapping = {}
    for i, e in enumerate(sorted_entities):
        mapping[e] = high_contrast_colors[i % len(high_contrast_colors)]
    return mapping


def render_long_form(
    df: pd.DataFrame,
    chart_type: str,
    topic_info: dict,
    entity_colors: Optional[dict] = None,
) -> tuple[Path, dict[str, str]]:
    """Render the long-form video.

    Args:
        df: DataFrame with columns [date, entity, value].
        chart_type: Ignored — always renders bar chart race per spec.
        topic_info: Dict with keys: topic, description, source.
        entity_colors: Optional pre-assigned color mapping.

    Returns:
        (output_path, entity_colors_used)
    """
    print(f"[renderer_long] Chart type: {chart_type}")
    print(f"[renderer_long] Data shape: {df.shape}")

    import shutil
    if FRAMES_LONG_DIR.exists():
        print(f"[renderer_long] Cleaning up existing frames in {FRAMES_LONG_DIR}...")
        shutil.rmtree(FRAMES_LONG_DIR, ignore_errors=True)
    FRAMES_LONG_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    all_entities = sorted(df["entity"].unique())
    if entity_colors is None:
        entity_colors = assign_entity_colors(all_entities)
    else:
        for e in all_entities:
            if e not in entity_colors:
                entity_colors[e] = ACCENT_COLORS[len(entity_colors) % len(ACCENT_COLORS)]

    time_steps = sorted(df["date"].unique())
    n_steps = len(time_steps)
    print(f"[renderer_long] Time steps: {n_steps}")

    # Build per-timestep value dicts
    step_values: list[dict[str, float]] = []
    for ts in time_steps:
        row = df[df["date"] == ts]
        step_values.append({r.entity: float(r.value) for r in row.itertuples()})

    title = topic_info.get("topic", "Data Visualization")
    source = topic_info.get("source", "")
    hook = topic_info.get("hook", title)

    # Calculate frames per step to hit target duration (5-10 min)
    hold_frames = FPS * 2
    usable_frames = LONG_FORM_MIN_DURATION * FPS - INTRO_FRAMES - hold_frames
    frames_per_step = usable_frames // max(n_steps - 1, 1)
    frames_per_step = max(20, frames_per_step)  # at least 20 frames for smooth pacing
    
    # Cap to LONG_FORM_MAX_DURATION
    max_usable_frames = LONG_FORM_MAX_DURATION * FPS - INTRO_FRAMES - hold_frames
    frames_per_step = min(frames_per_step, max_usable_frames // max(n_steps - 1, 1))
    frames_per_step = max(frames_per_step, 4)

    est_duration = (INTRO_FRAMES + (n_steps - 1) * frames_per_step + hold_frames) / FPS
    print(f"[renderer_long] Frames/step: {frames_per_step}, Duration: {est_duration:.0f}s")

    # Create figure once
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("#000000")
    ax = fig.add_axes([0.20, 0.12, 0.75, 0.74])

    frame_number = 0

    # ── Intro ────────────────────────────────────────────────────────────
    for f in range(INTRO_FRAMES):
        _draw_intro_frame(fig, hook, title, f, INTRO_FRAMES, FRAMES_LONG_DIR, frame_number)
        frame_number += 1

    # Recreate axes after fig.clf() destroyed it in intro
    fig.clf()
    fig.patch.set_facecolor("#000000")
    ax = fig.add_axes([0.20, 0.12, 0.75, 0.74])

    # ── Chart animation ──────────────────────────────────────────────────
    # Track each entity's previous slot rank (for smooth y interpolation)
    prev_ranks: dict[str, int] = {}

    for step_idx in range(len(time_steps) - 1):
        prev_vals = step_values[step_idx]
        next_vals = step_values[step_idx + 1]

        # Compute rankings at step boundaries
        prev_ranking = _rank_entities(prev_vals)
        next_ranking = _rank_entities(next_vals)

        # All entities visible in either frame
        all_shown = set(prev_ranking.keys()) | set(next_ranking.keys())

        # Get the date label for this step
        ts_start = pd.Timestamp(time_steps[step_idx])
        ts_end   = pd.Timestamp(time_steps[step_idx + 1])
        date_gap_days = (ts_end - ts_start).days

        for interp_frame in range(frames_per_step):
            alpha = interp_frame / frames_per_step
            alpha = _ease(alpha)

            # Interpolate all values
            interp_vals: dict[str, float] = {}
            for entity in all_shown:
                v0 = prev_vals.get(entity, 0.0)
                v1 = next_vals.get(entity, 0.0)
                interp_vals[entity] = v0 + (v1 - v0) * alpha

            # Rank by interpolated value → slot assignment
            sorted_ents = sorted(interp_vals.keys(), key=lambda e: interp_vals[e], reverse=True)
            current_top10 = sorted_ents[:TOP_N_ENTITIES]
            current_slot: dict[str, int] = {e: rank for rank, e in enumerate(current_top10)}

            # Y positions — interpolate between prev slot and current slot
            entities_data = []
            for rank, entity in enumerate(current_top10):
                prev_slot_rank = prev_ranks.get(entity, TOP_N_ENTITIES)  # off-screen if new
                prev_y = SLOTS.get(prev_slot_rank, OFF_SCREEN_Y)
                cur_y  = SLOTS[rank]
                y_pos  = prev_y + (cur_y - prev_y) * alpha

                entities_data.append({
                    "entity": entity,
                    "value":  interp_vals[entity],
                    "y_pos":  y_pos,
                    "color":  entity_colors.get(entity, ACCENT_COLORS[0]),
                })

            # Date label
            interp_ts = ts_start + (ts_end - ts_start) * alpha
            date_label = interp_ts.strftime("%b %Y") if date_gap_days < 400 else str(interp_ts.year)

            _draw_chart_frame(ax, fig, entities_data, title, source, date_label,
                              FRAMES_LONG_DIR, frame_number, topic_info)
            frame_number += 1

        # After this step is done, update prev_ranks to the END state of the step
        prev_ranks = {e: rank for rank, e in enumerate(sorted(next_vals.keys(),
                                                              key=lambda k: next_vals[k], reverse=True))}

        if step_idx % 5 == 0:
            print(f"[renderer_long] Progress: step {step_idx + 1}/{n_steps - 1}")

    # Hold last frame for 2 seconds
    hold_frames = FPS * 2
    for _ in range(hold_frames):
        _draw_chart_frame(ax, fig, entities_data, title, source, date_label,
                          FRAMES_LONG_DIR, frame_number, topic_info)
        frame_number += 1

    plt.close(fig)
    print(f"[renderer_long] Total frames rendered: {frame_number}")

    _encode_video(FRAMES_LONG_DIR, LONG_FORM_FINAL)
    print(f"[renderer_long] Output: {LONG_FORM_FINAL}")
    return LONG_FORM_FINAL, entity_colors


# ── Private helpers ───────────────────────────────────────────────────────────

def _rank_entities(values: dict[str, float]) -> dict[str, int]:
    """Return {entity: rank_index} sorted by descending value. Rank 0 = highest."""
    sorted_ents = sorted(values.keys(), key=lambda e: values[e], reverse=True)
    return {e: i for i, e in enumerate(sorted_ents)}


def _ease(t: float) -> float:
    """Smooth ease in-out (cubic)."""
    return t * t * (3 - 2 * t)


def format_value(value: float, short_unit: str = "") -> str:
    """Format large numbers with K/M/B suffixes and optional unit."""
    if not isinstance(short_unit, str):
        short_unit = ""
    if value >= 1_000_000_000:
        formatted = f"{value / 1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        formatted = f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        formatted = f"{value / 1_000:.1f}K"
    else:
        if 0 < abs(value) < 10:
            formatted = f"{value:.1f}"
        else:
            formatted = f"{value:.0f}"

    if not short_unit:
        return formatted

    short_unit = short_unit.strip()
    if short_unit == "%":
        return f"{formatted}%"
    elif short_unit in ("$", "£", "€"):
        return f"{short_unit}{formatted}"
    else:
        return f"{formatted} {short_unit}"


def _draw_intro_frame(
    fig: plt.Figure,
    hook_text: str,
    title: str,
    frame_idx: int,
    total_intro: int,
    frames_dir: Path,
    frame_number: int,
) -> None:
    """Draw a single intro frame (hook card + animated title fade-in)."""
    fig.clf()
    fig.patch.set_facecolor("#000000")

    half = total_intro // 2

    if frame_idx < half:
        # Phase 1 — static hook text
        hook_alpha = 1.0
        title_alpha = 0.0
        hook_y = 0.5
    else:
        # Phase 2 — hook slides up and fades; title fades in
        t = (frame_idx - half) / half
        eased_t = _ease(t)
        hook_y = 0.5 + eased_t * 0.45
        hook_alpha = 1.0 - eased_t
        title_alpha = eased_t

    if hook_alpha > 0.01:
        fig.text(
            0.5, hook_y, hook_text,
            ha="center", va="center",
            color=(1, 1, 1, hook_alpha),
            fontsize=18, fontproperties=FONT_BOLD,
            wrap=True,
            transform=fig.transFigure,
            bbox=dict(
                boxstyle="round,pad=0.8",
                facecolor="#111111",
                edgecolor=(1, 1, 1, hook_alpha * 0.4),
                linewidth=1,
            ),
        )

    if title_alpha > 0.01:
        fig.text(
            0.02, 0.97, title,
            ha="left", va="top",
            color=(1, 1, 1, title_alpha),
            fontsize=13, fontproperties=FONT_BOLD,
            transform=fig.transFigure,
        )

    fig.savefig(
        frames_dir / f"frame_{frame_number:05d}.png",
        dpi=100,
        facecolor="#000000",
        pad_inches=0,
        pil_kwargs={"compress_level": 1},
    )


def _draw_chart_frame(
    ax: plt.Axes,
    fig: plt.Figure,
    entities_data: list[dict],
    title: str,
    source: str,
    date_label: str,
    frames_dir: Path,
    frame_number: int,
    topic_info: Optional[dict] = None,
) -> None:
    """Draw a single chart frame and save to disk."""
    ax.cla()

    # Re-apply fixed axis settings after cla()
    ax.set_facecolor("#000000")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 9.6)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors="white", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#444444")

    if not entities_data:
        fig.savefig(
            frames_dir / f"frame_{frame_number:05d}.png",
            dpi=100,
            facecolor="#000000",
            pad_inches=0,
            pil_kwargs={"compress_level": 1},
        )
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

        # Bar
        ax.barh(y, norm_val, height=BAR_HEIGHT, color=color, alpha=0.9, left=0)

        # Clean/truncate entity name to prevent cutting off
        clean_name = re.sub(r"\s*\([^)]*\)\s*$", "", d["entity"]).strip()
        if len(clean_name) > 26:
            clean_name = clean_name[:23] + "..."

        # Entity name — LEFT of plot area (negative x in data coords)
        ax.text(
            -0.01, y, clean_name,
            ha="right", va="center",
            color="white", fontsize=10, fontproperties=FONT_BOLD,
        )

        # Value label — right end of bar
        ax.text(
            norm_val + 0.01, y, format_value(d["value"], short_unit),
            ha="left", va="center",
            color="white", fontsize=9, fontproperties=FONT_REGULAR,
        )

    # X-axis tick labels showing real values
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([format_value(t * max_value) for t in ticks], color="white", fontsize=8)

    # Ghost year counter — inside axes, bottom right, behind bars
    ax.text(
        0.98, 0.05, date_label,
        ha="right", va="bottom",
        color="white",
        alpha=0.20,
        fontsize=140, fontproperties=FONT_BOLD,
        transform=ax.transAxes,
        zorder=0,
    )

    # Title — figure-level, top left
    ax.text(
        0.02, 0.97, title,
        ha="left", va="top",
        color="white", fontsize=13, fontproperties=FONT_BOLD,
        transform=fig.transFigure,
    )

    # Source / Unit text — below title
    source_text = f"Source: {source}" if source else ""
    if full_unit:
        if source_text:
            source_text += f" | Unit: {full_unit}"
        else:
            source_text = f"Unit: {full_unit}"

    if source_text:
        ax.text(
            0.02, 0.93, source_text,
            ha="left", va="top",
            color="#888888", fontsize=9, fontproperties=FONT_REGULAR, style="italic",
            transform=fig.transFigure,
        )

    # Watermark
    overlay_watermark(fig, x=0.92, y=0.02, size=50, alpha=0.20)

    fig.savefig(
        frames_dir / f"frame_{frame_number:05d}.png",
        dpi=100,
        facecolor="#000000",
        pad_inches=0,
        pil_kwargs={"compress_level": 1},
    )


def _encode_video(frames_dir: Path, output_path: Path) -> None:
    """Encode all PNG frames into an MP4 with optional background music."""
    print("[renderer_long] Encoding video with FFmpeg...")

    music_files = list(MUSIC_DIR.glob("*.mp3")) + list(MUSIC_DIR.glob("*.wav"))

    if music_files:
        bg_music = random.choice(music_files)
        print(f"[renderer_long] Adding background music: {bg_music.name}")
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
    print(f"[renderer_long] Encoded: {output_path}")
