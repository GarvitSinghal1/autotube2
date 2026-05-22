"""
renderer_short.py — Renders the YouTube Short from the extreme segment.

Vertical format (1080x1920). Renders fast, high-quality, and modern bar chart races.
Includes font downloading, dynamic overlays, rank badges, gridlines, and overtake markers.
"""

import random
import subprocess
import urllib.request
import urllib.error
import re
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

from pipeline.config import (
    ACCENT_COLORS, FPS, FRAMES_SHORT_DIR, SHORT_FINAL,
    SHORT_MIN_DURATION, SHORT_MAX_DURATION, MUSIC_DIR,
    DEFAULT_VOLUME, TOP_N_ENTITIES, TMP_DIR, FONTS_DIR,
    SHORT_FRAMES_PER_STEP,
)
from pipeline.modules.renderer_long import (
    SLOTS, OFF_SCREEN_Y, BAR_HEIGHT,
    assign_entity_colors, format_value, _rank_entities, _ease,
)


def _download_fonts_if_needed() -> None:
    """Download Google Font 'Outfit' (Regular & Bold) if not present locally."""
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    
    font_files = {
        "Outfit-Regular.ttf": "https://raw.githubusercontent.com/Outfitio/Outfit-Fonts/main/fonts/ttf/Outfit-Regular.ttf",
        "Outfit-Bold.ttf": "https://raw.githubusercontent.com/Outfitio/Outfit-Fonts/main/fonts/ttf/Outfit-Bold.ttf",
    }
    
    for filename, url in font_files.items():
        dest_path = FONTS_DIR / filename
        if not dest_path.exists():
            print(f"[renderer_short] Downloading {filename} from Google Fonts...")
            try:
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
                )
                with urllib.request.urlopen(req, timeout=15) as response, open(dest_path, "wb") as out_file:
                    out_file.write(response.read())
                print(f"[renderer_short] Saved {filename} to {dest_path}")
            except Exception as e:
                print(f"[renderer_short] Warning: Failed to download {filename}: {e}. Falling back to system fonts.")


def _get_font_properties(bold: bool = False) -> FontProperties:
    """Get FontProperties for the Outfit font, falling back if not found."""
    filename = "Outfit-Bold.ttf" if bold else "Outfit-Regular.ttf"
    font_path = FONTS_DIR / filename
    if font_path.exists():
        return FontProperties(fname=str(font_path))
    else:
        return FontProperties(family="sans-serif", weight="bold" if bold else "normal")


def render_short(
    df_data: pd.DataFrame,
    chart_type: str,
    topic_info: dict,
    extreme_segment: dict,
    entity_colors: Optional[dict] = None,
) -> tuple[Path, dict[str, str]]:
    """Render the YouTube Short video from the extreme segment.

    Args:
        df_data: Monthly or Yearly DataFrame with columns [date, entity, value].
        chart_type: Ignored — always renders bar chart race.
        topic_info: Dict with keys: topic, description, source, hook.
        extreme_segment: Dict with start_year, end_year, reason, hook.
        entity_colors: Pre-assigned color mapping from the long-form render.

    Returns:
        (output_path, entity_colors_used)
    """
    _download_fonts_if_needed()

    start_yr = extreme_segment["start_year"]
    end_yr   = extreme_segment["end_year"]
    print(f"[renderer_short] Rendering Short: {start_yr}–{end_yr}")

    FRAMES_SHORT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Filter to the extreme segment window
    df_seg = df_data[
        (df_data["date"].dt.year >= start_yr) &
        (df_data["date"].dt.year <= end_yr)
    ].copy()

    if df_seg.empty:
        raise RuntimeError(f"No data found for segment {start_yr}–{end_yr}")

    all_entities = sorted(df_data["entity"].unique())
    if entity_colors is None:
        entity_colors = assign_entity_colors(all_entities)
    else:
        for e in all_entities:
            if e not in entity_colors:
                entity_colors[e] = ACCENT_COLORS[len(entity_colors) % len(ACCENT_COLORS)]

    time_steps = sorted(df_seg["date"].unique())
    n_steps = len(time_steps)
    print(f"[renderer_short] Time steps in Short: {n_steps}")

    # Build per-timestep value dicts
    step_values: list[dict[str, float]] = []
    for ts in time_steps:
        row = df_seg[df_seg["date"] == ts]
        step_values.append({r.entity: float(r.value) for r in row.itertuples()})

    # Set up intro and hold frame specs
    short_intro_frames = 60  # 2 seconds intro
    hold_frames = FPS * 2    # 2 seconds hold at the end

    # Calculate frames per step to hit target duration (min 20s, max 35s)
    usable_frames = SHORT_MIN_DURATION * FPS - short_intro_frames
    frames_per_step = max(SHORT_FRAMES_PER_STEP, usable_frames // max(n_steps - 1, 1))
    
    max_frames = SHORT_MAX_DURATION * FPS - short_intro_frames
    frames_per_step = min(frames_per_step, max_frames // max(n_steps - 1, 1))
    frames_per_step = max(frames_per_step, 4)

    total_frames = short_intro_frames + (n_steps - 1) * frames_per_step + hold_frames
    est_duration = total_frames / FPS
    print(f"[renderer_short] Frames/step: {frames_per_step}, Total Frames: {total_frames}, Duration: {est_duration:.1f}s")

    title = topic_info.get("topic", "Data Visualization")
    source = topic_info.get("source", "")
    hook = extreme_segment.get("hook", title)

    # Create figure once
    fig = plt.figure(figsize=(10.8, 19.2), dpi=100)
    fig.patch.set_facecolor("#000000")
    ax = fig.add_axes([0.30, 0.10, 0.65, 0.76])

    frame_number = 0

    # Calculate starting entities data for the intro background
    first_step_vals = step_values[0]
    first_ranking = sorted(first_step_vals.keys(), key=lambda e: first_step_vals[e], reverse=True)
    first_top10 = first_ranking[:TOP_N_ENTITIES]
    
    initial_entities_data = []
    for rank, entity in enumerate(first_top10):
        initial_entities_data.append({
            "entity": entity,
            "value":  first_step_vals[entity],
            "y_pos":  SLOTS[rank],
            "color":  entity_colors.get(entity, ACCENT_COLORS[0]),
            "rank":   rank,
            "is_climbing": False,
        })

    # ── Intro ────────────────────────────────────────────────────────────
    for f in range(short_intro_frames):
        _draw_short_intro_frame(
            fig, hook, title, f, short_intro_frames, FRAMES_SHORT_DIR, frame_number,
            initial_entities_data, topic_info
        )
        frame_number += 1

    # Recreate axes after fig.clf() in intro
    fig.clf()
    fig.patch.set_facecolor("#000000")
    ax = fig.add_axes([0.30, 0.10, 0.65, 0.76])

    # Calculate total span in days to determine date formatting (month/year vs year-only)
    date_gap_days = (df_seg["date"].max() - df_seg["date"].min()).days

    # ── Chart animation ──────────────────────────────────────────────────
    prev_ranks: dict[str, int] = {}
    entities_data = []
    date_label = str(start_yr)

    # Initialize prev_ranks for step 0
    prev_ranks = {e: r for r, e in enumerate(
        sorted(first_step_vals.keys(), key=lambda k: first_step_vals[k], reverse=True)
    )}

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

                is_climbing = False
                if entity in prev_ranks and rank < prev_ranks[entity]:
                    is_climbing = True

                entities_data.append({
                    "entity": entity,
                    "value":  interp_vals[entity],
                    "y_pos":  y_pos,
                    "color":  entity_colors.get(entity, ACCENT_COLORS[0]),
                    "rank":   rank,
                    "is_climbing": is_climbing,
                })

            interp_ts = ts_start + (ts_end - ts_start) * alpha
            date_label = interp_ts.strftime("%b %Y") if date_gap_days < 4000 else str(interp_ts.year)

            _draw_short_chart_frame(
                ax, fig, entities_data, title, source, date_label,
                FRAMES_SHORT_DIR, frame_number, topic_info, total_frames
            )
            frame_number += 1

        # Update prev_ranks to end state of this step
        prev_ranks = {e: r for r, e in enumerate(
            sorted(next_vals.keys(), key=lambda k: next_vals[k], reverse=True)
        )}

    # Hold last frame for 2 seconds
    for _ in range(hold_frames):
        _draw_short_chart_frame(
            ax, fig, entities_data, title, source, date_label,
            FRAMES_SHORT_DIR, frame_number, topic_info, total_frames
        )
        frame_number += 1

    plt.close(fig)
    print(f"[renderer_short] Total frames generated: {frame_number}")

    _encode_short(FRAMES_SHORT_DIR, SHORT_FINAL)
    print(f"[renderer_short] Output: {SHORT_FINAL}")
    return SHORT_FINAL, entity_colors


# ── Private helpers ───────────────────────────────────────────────────────────

def _draw_short_chart_frame_contents(
    ax: plt.Axes,
    fig: plt.Figure,
    entities_data: list[dict],
    title: str,
    source: str,
    date_label: str,
    topic_info: Optional[dict],
    frame_number: int,
    draw_progress: bool = True,
    total_frames: int = 1,
) -> None:
    """Helper to draw all details of the horizontal bar chart race for Shorts."""
    ax.cla()
    
    # 1. Background grid & styling
    ax.set_facecolor("#0a0a0a")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 9.6)
    ax.set_yticks([])
    
    # Custom fonts
    font_bold = _get_font_properties(bold=True)
    font_regular = _get_font_properties(bold=False)
    
    # Spines and ticks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(axis="x", colors="white", labelsize=8)
    
    # Add horizontal grid lines with low opacity for high-tech premium feel
    ax.grid(axis='x', color='#ffffff', linestyle='--', alpha=0.1, zorder=0)
    ax.set_axisbelow(True)
    
    if not entities_data:
        return
        
    max_value = max(d["value"] for d in entities_data) * 1.1
    if max_value <= 0:
        max_value = 1.0
        
    short_unit = topic_info.get("short_unit", "") if topic_info else ""
    full_unit = topic_info.get("full_unit", "") if topic_info else ""
    
    # Draw bars & labels
    for d in entities_data:
        norm_val = d["value"] / max_value
        y = d["y_pos"]
        color = d["color"]
        rank = d.get("rank", 9)
        entity = d["entity"]
        
        # Style borders for Top 3 slots to make them stand out
        if rank == 0:
            edge_color = "#FFD700"  # Gold
            line_width = 1.8
        elif rank == 1:
            edge_color = "#C0C0C0"  # Silver
            line_width = 1.4
        elif rank == 2:
            edge_color = "#CD7F32"  # Bronze
            line_width = 1.4
        else:
            edge_color = "#ffffff"  # Standard White
            line_width = 0.6
            
        # Draw standard bar
        ax.barh(
            y, norm_val, height=BAR_HEIGHT, color=color, 
            edgecolor=edge_color, linewidth=line_width, alpha=0.9, left=0, zorder=3
        )
        
        # Clean/truncate entity name to prevent cutting off
        import re
        clean_name = re.sub(r"\s*\([^)]*\)\s*$", "", entity).strip()
        if len(clean_name) > 20:
            clean_name = clean_name[:17] + "..."
            
        # Draw Rank Number
        rank_str = f"#{rank+1}"
        rank_color = "#FFD700" if rank == 0 else "#C0C0C0" if rank == 1 else "#CD7F32" if rank == 2 else "#888888"
        ax.text(
            -0.015, y, rank_str,
            ha="right", va="center",
            color=rank_color, fontproperties=font_bold, fontsize=10,
        )
        
        # Draw Entity Name (shifted left to make room for rank)
        ax.text(
            -0.055, y, clean_name,
            ha="right", va="center",
            color="white", fontproperties=font_regular, fontsize=9,
        )
        
        # Overtake highlight / climbing check
        is_climbing = d.get("is_climbing", False)
        val_str = format_value(d["value"], short_unit)
        
        if is_climbing:
            # Draw green value label to signal climbing rank
            ax.text(
                norm_val + 0.01, y, val_str,
                ha="left", va="center",
                color="#2ecc71", fontproperties=font_bold, fontsize=8.5,
            )
        else:
            ax.text(
                norm_val + 0.01, y, val_str,
                ha="left", va="center",
                color="white", fontproperties=font_regular, fontsize=8,
            )
            
    # X-axis tick labels
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [format_value(t * max_value) for t in ticks], 
        color="white", fontproperties=font_regular, fontsize=8
    )
    
    # Ghost year in background
    ax.text(
        0.98, 0.05, date_label,
        ha="right", va="bottom",
        color="white",
        alpha=0.15,
        fontproperties=font_bold,
        fontsize=110,
        transform=ax.transAxes,
        zorder=1,
    )
    
    # Title — figure-level, top
    fig.text(
        0.5, 0.96, title,
        ha="center", va="top",
        color="white", fontproperties=font_bold, fontsize=13,
        wrap=True,
    )
    
    # Source / Unit text — below title
    source_text = f"Source: {source}" if source else ""
    if full_unit:
        if source_text:
            source_text += f" | Unit: {full_unit}"
        else:
            source_text = f"Unit: {full_unit}"
            
    if source_text:
        fig.text(
            0.5, 0.92, source_text,
            ha="center", va="top",
            color="#888888", fontproperties=font_regular, fontsize=8, style="italic",
            wrap=True,
        )
        
    # Draw dynamic progress line at the very top of the frame
    if draw_progress and total_frames > 1:
        progress = min(1.0, max(0.0, frame_number / (total_frames - 1)))
        progress_rect = plt.Rectangle(
            (0, 0.992), progress, 0.008, 
            facecolor='#FF6B6B', transform=fig.transFigure, zorder=100
        )
        fig.patches.append(progress_rect)


def _draw_short_chart_frame(
    ax: plt.Axes,
    fig: plt.Figure,
    entities_data: list[dict],
    title: str,
    source: str,
    date_label: str,
    frames_dir: Path,
    frame_number: int,
    topic_info: Optional[dict] = None,
    total_frames: int = 1,
) -> None:
    """Draw a single vertical-format chart frame and save to disk."""
    _draw_short_chart_frame_contents(
        ax, fig, entities_data, title, source, date_label,
        topic_info, frame_number, draw_progress=True, total_frames=total_frames
    )
    
    fig.savefig(
        frames_dir / f"frame_{frame_number:05d}.png",
        dpi=100,
        facecolor="#000000",
        pad_inches=0,
        pil_kwargs={"compress_level": 1},
    )
    # Clear the progress bar patch so it doesn't build up
    fig.patches.clear()


def _draw_short_intro_frame(
    fig: plt.Figure,
    hook_text: str,
    title: str,
    frame_idx: int,
    total_intro: int,
    frames_dir: Path,
    frame_number: int,
    initial_entities_data: list[dict],
    topic_info: dict,
) -> None:
    """Draw a single intro frame: the starting chart with a dimmed mask and hook card overlay."""
    fig.clf()
    fig.patch.set_facecolor("#000000")
    
    # 1. Recreate axes and draw the initial frame of the chart (so the user sees the chart in background)
    ax = fig.add_axes([0.30, 0.10, 0.65, 0.76])
    
    # Extract source
    source = topic_info.get("source", "")
    
    # Draw standard background chart on ax
    _draw_short_chart_frame_contents(
        ax, fig, initial_entities_data, title, source, 
        str(initial_entities_data[0]["year"] if "year" in initial_entities_data[0] else ""),
        topic_info, frame_number, draw_progress=False
    )
    
    # 2. Add full-screen semi-transparent dark mask and overlay card
    half = total_intro // 2
    if frame_idx < half:
        alpha_mask = 0.8
        alpha_card = 1.0
        card_y = 0.5
    else:
        # Interpolate fade out
        t = (frame_idx - half) / half
        eased_t = t * t * (3 - 2 * t)  # ease-in-out
        alpha_mask = 0.8 * (1.0 - eased_t)
        alpha_card = 1.0 - eased_t
        card_y = 0.5 + eased_t * 0.15 # slide up slightly
        
    if alpha_mask > 0.01:
        rect = plt.Rectangle((0, 0), 1, 1, facecolor='#000000', alpha=alpha_mask, transform=fig.transFigure, zorder=5)
        fig.patches.append(rect)
        
    if alpha_card > 0.01:
        font_bold = _get_font_properties(bold=True)
        font_regular = _get_font_properties(bold=False)
        
        # Border outline colors alternate slightly to feel dynamic
        border_color = "#FFD93D" if frame_idx % 4 < 2 else "#4ECDC4"
        
        # Background box for hook text
        card_box = dict(
            boxstyle="round,pad=0.8",
            facecolor="#181818",
            edgecolor=border_color,
            alpha=alpha_card,
            linewidth=1.8,
        )
        
        # Display the hook text in a prominent centered card
        fig.text(
            0.5, card_y, hook_text,
            ha="center", va="center",
            color=(1, 1, 1, alpha_card),
            fontproperties=font_bold,
            fontsize=15,
            wrap=True,
            transform=fig.transFigure,
            bbox=card_box,
            zorder=10,
        )
        
        # Small call to action at the bottom of the card
        fig.text(
            0.5, card_y - 0.12, "Watch to see who wins!",
            ha="center", va="center",
            color=(1.0, 1.0, 1.0, alpha_card * 0.9),
            fontproperties=font_bold,
            fontsize=10,
            transform=fig.transFigure,
            zorder=10,
        )
        
    fig.savefig(
        frames_dir / f"frame_{frame_number:05d}.png",
        dpi=100,
        facecolor="#000000",
        pad_inches=0,
        pil_kwargs={"compress_level": 1},
    )
    
    # Clean up patches to prevent leaking into next frames
    fig.patches.clear()


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
