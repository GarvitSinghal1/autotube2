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
from matplotlib.patches import FancyBboxPatch
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


def _detect_overtakes(v0: dict[str, float], v1: dict[str, float], top_n: int = 5) -> float:
    """Calculate an activity score based on swaps in the top_n and new entries in the top 3."""
    sorted_0 = sorted(v0.keys(), key=lambda e: v0[e], reverse=True)
    sorted_1 = sorted(v1.keys(), key=lambda e: v1[e], reverse=True)
    
    rank_0 = {e: r for r, e in enumerate(sorted_0)}
    rank_1 = {e: r for r, e in enumerate(sorted_1)}
    
    weight = 1.0
    
    # Check top 5 swaps
    top_entities = set(sorted_0[:top_n]) | set(sorted_1[:top_n])
    top_list = list(top_entities)
    for i in range(len(top_list)):
        for j in range(i + 1, len(top_list)):
            e1, e2 = top_list[i], top_list[j]
            r0_1 = rank_0.get(e1, 999)
            r0_2 = rank_0.get(e2, 999)
            r1_1 = rank_1.get(e1, 999)
            r1_2 = rank_1.get(e2, 999)
            
            if (r0_1 < r0_2) != (r1_1 < r1_2):
                weight += 0.8
                
    # Check new entries in top 3
    top3_0 = set(sorted_0[:3])
    top3_1 = set(sorted_1[:3])
    new_top3 = top3_1 - top3_0
    weight += len(new_top3) * 1.5
    
    return weight


def render_short(
    df_data: pd.DataFrame,
    chart_type: str,
    topic_info: dict,
    extreme_segment: dict,
    entity_colors: Optional[dict] = None,
) -> tuple[Path, dict[str, str]]:
    """Render the YouTube Short video from the extreme segment with dynamic pacing.

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

    import shutil
    if FRAMES_SHORT_DIR.exists():
        print(f"[renderer_short] Cleaning up existing frames in {FRAMES_SHORT_DIR}...")
        shutil.rmtree(FRAMES_SHORT_DIR, ignore_errors=True)
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

    # Calculate dynamic variable pacing based on activity weights
    weights = []
    for step_idx in range(n_steps - 1):
        w = _detect_overtakes(step_values[step_idx], step_values[step_idx + 1])
        weights.append(w)

    sum_w = sum(weights) if weights else 1.0

    # Define targets for animation frames
    min_anim_frames = SHORT_MIN_DURATION * FPS - short_intro_frames - hold_frames  # 20s - 2s - 2s = 480 frames
    max_anim_frames = SHORT_MAX_DURATION * FPS - short_intro_frames - hold_frames  # 35s - 2s - 2s = 930 frames
    target_anim_frames = 600  # Default target around 20 seconds of animation

    # Distribute target frames proportionally
    step_frames = []
    for w in weights:
        f = int(round((w / sum_w) * target_anim_frames))
        f = max(3, min(24, f))  # Clamp step allocations to preserve fluidity
        step_frames.append(f)

    # Scale step frames if the total sum is outside allowed duration bounds
    total_anim = sum(step_frames)
    if total_anim < min_anim_frames:
        scale = min_anim_frames / total_anim
        step_frames = [max(3, min(24, int(round(f * scale)))) for f in step_frames]
    elif total_anim > max_anim_frames:
        scale = max_anim_frames / total_anim
        step_frames = [max(3, min(24, int(round(f * scale)))) for f in step_frames]

    total_frames = short_intro_frames + sum(step_frames) + hold_frames
    est_duration = total_frames / FPS
    print(f"[renderer_short] Dynamic Pacing -> Total Animation Frames: {sum(step_frames)}, Total Video Frames: {total_frames}, Duration: {est_duration:.1f}s")

    title = topic_info.get("topic", "Data Visualization")
    source = topic_info.get("source", "")
    hook = extreme_segment.get("hook", title)

    # Create figure once with obsidian background color
    fig = plt.figure(figsize=(10.8, 19.2), dpi=100)
    fig.patch.set_facecolor("#06050a")
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

    # Calculate date labels
    date_gap_days = (df_seg["date"].max() - df_seg["date"].min()).days
    first_ts = pd.Timestamp(time_steps[0])
    first_date_label = first_ts.strftime("%b %Y") if date_gap_days < 4000 else str(first_ts.year)

    # ── Intro ────────────────────────────────────────────────────────────
    for f in range(short_intro_frames):
        _draw_short_intro_frame(
            fig, hook, title, f, short_intro_frames, FRAMES_SHORT_DIR, frame_number,
            initial_entities_data, topic_info, first_date_label
        )
        frame_number += 1

    # Recreate axes after fig.clf() in intro
    fig.clf()
    fig.patch.set_facecolor("#06050a")
    ax = fig.add_axes([0.30, 0.10, 0.65, 0.76])

    # ── Chart animation ──────────────────────────────────────────────────
    prev_ranks: dict[str, int] = {}
    entities_data = []
    date_label = first_date_label

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
        f_step = step_frames[step_idx]

        for interp_frame in range(f_step):
            alpha = _ease(interp_frame / f_step)

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
    ax.set_facecolor("#0e0d16")
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
    ax.spines["bottom"].set_color("#1a1a2e")
    ax.tick_params(axis="x", colors="white", labelsize=8)
    
    # Add horizontal grid lines with low opacity for high-tech premium feel
    ax.grid(axis='x', color='#00f0ff', linestyle='-', alpha=0.08, zorder=0)
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
            
        # Draw Neon Glow Capsule Bars:
        # Glow Layer 1: very soft, wide
        glow_width_1 = norm_val
        glow_height_1 = BAR_HEIGHT * 1.3
        glow_y_1 = y - glow_height_1 / 2.0
        if glow_width_1 > 0:
            glow_patch_1 = FancyBboxPatch(
                (0, glow_y_1), glow_width_1, glow_height_1,
                boxstyle=f"round,pad=0,rounding_size={min(glow_width_1 * 0.4, glow_height_1 * 0.4)}",
                facecolor=color, edgecolor="none", alpha=0.08, zorder=2
            )
            ax.add_patch(glow_patch_1)
            
            # Glow Layer 2: slightly tighter, a bit more opacity
            glow_height_2 = BAR_HEIGHT * 1.15
            glow_y_2 = y - glow_height_2 / 2.0
            glow_patch_2 = FancyBboxPatch(
                (0, glow_y_2), glow_width_1, glow_height_2,
                boxstyle=f"round,pad=0,rounding_size={min(glow_width_1 * 0.4, glow_height_2 * 0.4)}",
                facecolor=color, edgecolor="none", alpha=0.18, zorder=2
            )
            ax.add_patch(glow_patch_2)

        # Draw the main capsule bar
        if norm_val > 0:
            r_size = min(norm_val * 0.4, BAR_HEIGHT * 0.4)
            main_patch = FancyBboxPatch(
                (0, y - BAR_HEIGHT/2.0), norm_val, BAR_HEIGHT,
                boxstyle=f"round,pad=0,rounding_size={r_size}",
                facecolor=color, edgecolor=edge_color, linewidth=line_width,
                alpha=0.95, zorder=3
            )
            ax.add_patch(main_patch)
        
        # Clean/truncate entity name to prevent cutting off
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
        
        # Draw circular knob at bar tip
        ax.plot(
            norm_val, y, marker='o', markersize=22, 
            color=color, markeredgecolor=edge_color, markeredgewidth=1.2, zorder=4
        )
        # Initials inside knob
        initial = clean_name[0].upper() if clean_name else ""
        ax.text(
            norm_val, y, initial,
            ha="center", va="center",
            color="#0e0d16", fontproperties=font_bold, fontsize=8, zorder=5
        )
        
        # Climbing arrow indicator + value text positioning
        is_climbing = d.get("is_climbing", False)
        val_str = format_value(d["value"], short_unit)
        
        if is_climbing:
            # Draw green up-arrow marker at norm_val + 0.045
            ax.plot(
                norm_val + 0.045, y, marker='^', color='#2ecc71', markersize=8, zorder=4
            )
            # Draw green value label at norm_val + 0.07
            ax.text(
                norm_val + 0.07, y, val_str,
                ha="left", va="center",
                color="#2ecc71", fontproperties=font_bold, fontsize=8.5,
            )
        else:
            # Draw standard value label at norm_val + 0.045
            ax.text(
                norm_val + 0.045, y, val_str,
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
    
    # Ghost year in background (glowing cyan subtle style)
    ax.text(
        0.98, 0.05, date_label,
        ha="right", va="bottom",
        color="#00f0ff",
        alpha=0.06,
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
        
    # Draw dynamic progress line at the very top of the frame (neon pink '#ff007f')
    if draw_progress and total_frames > 1:
        progress = min(1.0, max(0.0, frame_number / (total_frames - 1)))
        progress_rect = plt.Rectangle(
            (0, 0.992), progress, 0.008, 
            facecolor='#ff007f', transform=fig.transFigure, zorder=100
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
        facecolor="#06050a",
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
    first_date_label: str,
) -> None:
    """Draw a single intro frame: the starting chart with a dimmed mask and hook card overlay."""
    fig.clf()
    fig.patch.set_facecolor("#06050a")
    
    # 1. Recreate axes and draw the initial frame of the chart (so the user sees the chart in background)
    ax = fig.add_axes([0.30, 0.10, 0.65, 0.76])
    
    # Extract source
    source = topic_info.get("source", "")
    
    # Draw standard background chart on ax
    _draw_short_chart_frame_contents(
        ax, fig, initial_entities_data, title, source, 
        first_date_label,
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
        border_color = "#00f0ff" if frame_idx % 4 < 2 else "#ff007f"
        
        # Background box for hook text
        card_box = dict(
            boxstyle="round,pad=1.0",
            facecolor="#0e0d16",
            edgecolor=border_color,
            alpha=alpha_card,
            linewidth=2.0,
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
            color=(0.0, 0.94, 1.0, alpha_card * 0.9), # neon cyan text
            fontproperties=font_bold,
            fontsize=10,
            transform=fig.transFigure,
            zorder=10,
        )
        
    fig.savefig(
        frames_dir / f"frame_{frame_number:05d}.png",
        dpi=100,
        facecolor="#06050a",
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
