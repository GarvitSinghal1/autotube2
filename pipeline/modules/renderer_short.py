"""
renderer_short.py — Renders the YouTube Short from the extreme segment.

Vertical format (1080x1920), 50–59 seconds, year-wise granularity.
Uses the same color assignment logic as renderer_long for consistency.
"""

import subprocess
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from pipeline.config import (
    SHORT_WIDTH, SHORT_HEIGHT, FPS,
    SHORT_FRAMES_PER_STEP, SHORT_MIN_DURATION, SHORT_MAX_DURATION,
    BG_COLOR, TEXT_COLOR, SUBTITLE_COLOR, ACCENT_COLORS,
    FRAMES_SHORT_DIR, SHORT_FINAL, TOP_N_ENTITIES, TMP_DIR,
    MUSIC_DIR, DEFAULT_VOLUME,
)
from pipeline.modules.renderer_long import assign_entity_colors, _ease_in_out, _format_value


def render_short(
    df_yearly: pd.DataFrame,
    chart_type: str,
    topic_info: dict,
    extreme_segment: dict,
    entity_colors: Optional[dict] = None,
) -> Path:
    """Render the YouTube Short video from the extreme segment.

    Args:
        df_yearly: Yearly DataFrame with [date, entity, value].
        chart_type: Chart type string.
        topic_info: Topic metadata dict.
        extreme_segment: Dict with start_year, end_year, reason, hook.
        entity_colors: Pre-assigned colors from long-form render.

    Returns:
        Path to the rendered short video.

    Raises:
        RuntimeError: If rendering or encoding fails.
    """
    print(f"[renderer_short] Rendering Short: {extreme_segment['start_year']}–{extreme_segment['end_year']}")

    FRAMES_SHORT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Filter to extreme segment
    start_yr = extreme_segment["start_year"]
    end_yr = extreme_segment["end_year"]
    df_segment = df_yearly[
        (df_yearly["date"].dt.year >= start_yr) &
        (df_yearly["date"].dt.year <= end_yr)
    ].copy()

    if df_segment.empty:
        raise RuntimeError(f"No data in segment {start_yr}–{end_yr}")

    # Re-derive colors using same logic for consistency
    all_entities = sorted(df_yearly["entity"].unique())
    if entity_colors is None:
        entity_colors = assign_entity_colors(all_entities)

    time_steps = sorted(df_segment["date"].unique())
    n_steps = len(time_steps)

    # Calculate frames per step for 50–59 second target
    hook_frames = FPS * 3  # 3 seconds for hook
    data_frames_target = SHORT_MIN_DURATION * FPS - hook_frames
    frames_per_step = max(SHORT_FRAMES_PER_STEP, data_frames_target // max(n_steps, 1))

    # Cap total to SHORT_MAX_DURATION
    total_frames = hook_frames + n_steps * frames_per_step
    if total_frames > SHORT_MAX_DURATION * FPS:
        frames_per_step = (SHORT_MAX_DURATION * FPS - hook_frames) // max(n_steps, 1)
        frames_per_step = max(4, frames_per_step)

    total_frames = hook_frames + n_steps * frames_per_step
    print(f"[renderer_short] Steps: {n_steps}, FPS: {frames_per_step}/step")
    print(f"[renderer_short] Duration: {total_frames / FPS:.0f}s")

    # Render hook card
    frame_num = _render_hook(extreme_segment["hook"], hook_frames)

    # Render data frames
    if chart_type == "bar_chart_race":
        frame_num = _render_bar_race_short(
            df_segment, time_steps, entity_colors, topic_info,
            frames_per_step, frame_num,
        )
    elif chart_type == "line_chart_race":
        frame_num = _render_line_short(
            df_segment, time_steps, entity_colors, topic_info,
            frames_per_step, frame_num,
        )
    else:
        frame_num = _render_bar_race_short(
            df_segment, time_steps, entity_colors, topic_info,
            frames_per_step, frame_num,
        )

    # Encode
    _encode_short(FRAMES_SHORT_DIR, SHORT_FINAL)
    print(f"[renderer_short] Output: {SHORT_FINAL}")
    return SHORT_FINAL


def _render_hook(hook: str, n_frames: int) -> int:
    """Render hook text card for the first 3 seconds."""
    fig, ax = plt.subplots(figsize=(10.80, 19.20), dpi=100)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    ax.text(0.5, 0.5, hook, transform=ax.transAxes, fontsize=32,
            color=TEXT_COLOR, ha="center", va="center", fontweight="bold",
            fontfamily="sans-serif", wrap=True,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=BG_COLOR,
                     edgecolor=ACCENT_COLORS[0], linewidth=2))

    for i in range(n_frames):
        frame_path = FRAMES_SHORT_DIR / f"frame_{i:05d}.png"
        fig.savefig(frame_path, facecolor=BG_COLOR, dpi=100)
    plt.close(fig)
    return n_frames


def _render_bar_race_short(
    df: pd.DataFrame, time_steps: list, colors: dict, topic_info: dict,
    frames_per_step: int, start_frame: int,
) -> int:
    """Render bar chart race in vertical format."""
    frame_num = start_frame
    dpi = 100
    figsize = (10.80, 19.20)
    topic_title = topic_info.get("topic", "")

    for step_idx in range(len(time_steps)):
        current_ts = pd.Timestamp(time_steps[step_idx])
        current_data = df[df["date"] == time_steps[step_idx]]

        if step_idx < len(time_steps) - 1:
            next_ts = pd.Timestamp(time_steps[step_idx + 1])
            next_data = df[df["date"] == time_steps[step_idx + 1]]
        else:
            next_data = current_data
            next_ts = current_ts

        current_data_sorted = current_data.sort_values("value", ascending=True).reset_index(drop=True)
        next_data_sorted = next_data.sort_values("value", ascending=True).reset_index(drop=True)
        
        current_top = current_data_sorted.tail(TOP_N_ENTITIES)
        next_top = next_data_sorted.tail(TOP_N_ENTITIES)
        
        current_ranks = {row.entity: i for i, row in enumerate(current_top.itertuples())}
        next_ranks = {row.entity: i for i, row in enumerate(next_top.itertuples())}
        
        all_shown = set(current_ranks.keys()).union(set(next_ranks.keys()))

        max_val_current = current_top["value"].max() if not current_top.empty else 1
        max_val_next = next_top["value"].max() if not next_top.empty else 1

        for interp_idx in range(frames_per_step):
            t = _ease_in_out(interp_idx / max(frames_per_step - 1, 1))
            interp_max_val = max_val_current + (max_val_next - max_val_current) * t

            entities_to_draw = []
            values_to_draw = []
            y_positions = []
            bar_colors = []

            for entity in all_shown:
                cv = current_data[current_data["entity"] == entity]["value"]
                nv = next_data[next_data["entity"] == entity]["value"]
                cv = cv.iloc[0] if len(cv) > 0 else 0
                nv = nv.iloc[0] if len(nv) > 0 else 0
                val = cv + (nv - cv) * t
                
                r0 = current_ranks.get(entity, -1)
                r1 = next_ranks.get(entity, -1)
                y_pos = r0 + (r1 - r0) * t
                
                if y_pos > -1.5:
                    entities_to_draw.append(entity)
                    values_to_draw.append(val)
                    y_positions.append(y_pos)
                    bar_colors.append(colors.get(entity, ACCENT_COLORS[0]))

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            fig.patch.set_facecolor(BG_COLOR)
            ax.set_facecolor(BG_COLOR)

            bars = ax.barh(y_positions, values_to_draw, color=bar_colors,
                          height=0.7, edgecolor="none")

            ax.set_yticks(y_positions)
            ax.set_yticklabels(entities_to_draw, fontsize=16, color=TEXT_COLOR,
                              fontweight="bold", fontfamily="sans-serif")

            ax.set_ylim(-0.5, TOP_N_ENTITIES - 0.5)
            if interp_max_val > 0:
                ax.set_xlim(0, interp_max_val * 1.1)

            for i, (val, bar, y_pos) in enumerate(zip(values_to_draw, bars, y_positions)):
                ax.text(val + interp_max_val * 0.01, y_pos, _format_value(val),
                       fontsize=14, color=TEXT_COLOR, va="center")

            ax.text(0.5, 0.08, str(current_ts.year), transform=ax.transAxes,
                   fontsize=72, color=SUBTITLE_COLOR, ha="center", va="bottom",
                   fontweight="bold", alpha=0.3)

            ax.set_title(topic_title, fontsize=24, color=TEXT_COLOR,
                        fontweight="bold", fontfamily="sans-serif", pad=25)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color(SUBTITLE_COLOR)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="x", colors=SUBTITLE_COLOR)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: _format_value(x)))

            plt.tight_layout(pad=2)
            frame_path = FRAMES_SHORT_DIR / f"frame_{frame_num:05d}.png"
            fig.savefig(frame_path, facecolor=BG_COLOR, dpi=dpi)
            plt.close(fig)
            frame_num += 1

    return frame_num


def _render_line_short(
    df: pd.DataFrame, time_steps: list, colors: dict, topic_info: dict,
    frames_per_step: int, start_frame: int,
) -> int:
    """Render line chart in vertical format."""
    frame_num = start_frame
    dpi = 100
    figsize = (10.80, 19.20)
    entities = df["entity"].unique()
    topic_title = topic_info.get("topic", "")

    for step_idx in range(len(time_steps)):
        for interp_idx in range(frames_per_step):
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            fig.patch.set_facecolor(BG_COLOR)
            ax.set_facecolor(BG_COLOR)

            for entity in entities:
                edata = df[df["entity"] == entity].sort_values("date")
                mask = edata["date"] <= time_steps[step_idx]
                visible = edata[mask]
                if visible.empty:
                    continue
                color = colors.get(entity, ACCENT_COLORS[0])
                ax.plot(visible["date"], visible["value"],
                       color=color, linewidth=3, label=entity)
                ax.scatter([visible["date"].iloc[-1]], [visible["value"].iloc[-1]],
                          color=color, s=80, zorder=5)

            ax.set_title(topic_title, fontsize=22, color=TEXT_COLOR,
                        fontweight="bold", fontfamily="sans-serif")
            ax.legend(loc="upper left", fontsize=12, facecolor=BG_COLOR,
                     edgecolor=SUBTITLE_COLOR, labelcolor=TEXT_COLOR)

            current_ts = pd.Timestamp(time_steps[step_idx])
            ax.text(0.5, 0.08, str(current_ts.year), transform=ax.transAxes,
                   fontsize=72, color=SUBTITLE_COLOR, ha="center", va="bottom",
                   fontweight="bold", alpha=0.3)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color(SUBTITLE_COLOR)
            ax.spines["left"].set_color(SUBTITLE_COLOR)
            ax.tick_params(colors=SUBTITLE_COLOR)

            plt.tight_layout()
            frame_path = FRAMES_SHORT_DIR / f"frame_{frame_num:05d}.png"
            fig.savefig(frame_path, facecolor=BG_COLOR, dpi=dpi)
            plt.close(fig)
            frame_num += 1

    return frame_num


def _encode_short(frames_dir: Path, output_path: Path) -> None:
    """Encode Short frames to MP4 via FFmpeg."""
    print("[renderer_short] Encoding Short with FFmpeg...")
    import random
    music_files = list(MUSIC_DIR.glob("*.mp3")) + list(MUSIC_DIR.glob("*.wav"))
    
    if music_files:
        bg_music = random.choice(music_files)
        print(f"[renderer_short] Adding background music: {bg_music.name}")
        cmd = [
            "ffmpeg", "-y",
            "-r", str(FPS),
            "-i", str(frames_dir / "frame_%05d.png"),
            "-stream_loop", "-1",
            "-i", str(bg_music),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-filter_complex", f"[1:a]volume={DEFAULT_VOLUME}[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-crf", "18",
            "-preset", "slow",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-r", str(FPS),
            "-i", str(frames_dir / "frame_%05d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
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
