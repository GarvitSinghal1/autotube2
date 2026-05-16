"""
renderer_long.py — Renders the full long-form animated data visualization video.

Uses matplotlib to generate individual frames, then FFmpeg to encode.
Resolution: 1920x1080 (16:9 landscape). Duration: 5–10 minutes at 30fps.
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
    LONG_FORM_WIDTH, LONG_FORM_HEIGHT, FPS,
    LONG_FRAMES_PER_STEP, LONG_FORM_MIN_DURATION, LONG_FORM_MAX_DURATION,
    BG_COLOR, TEXT_COLOR, SUBTITLE_COLOR, ACCENT_COLORS,
    FRAMES_LONG_DIR, LONG_FORM_FINAL, TOP_N_ENTITIES, TMP_DIR,
)


def assign_entity_colors(entities: list[str]) -> dict[str, str]:
    """Assign a consistent color to each entity. Used across both renderers.

    Args:
        entities: List of unique entity names.

    Returns:
        Dict mapping entity name to hex color string.
    """
    colors = {}
    for i, entity in enumerate(entities):
        colors[entity] = ACCENT_COLORS[i % len(ACCENT_COLORS)]
    return colors


def render_long_form(
    df: pd.DataFrame,
    chart_type: str,
    topic_info: dict,
    entity_colors: Optional[dict] = None,
) -> tuple[Path, dict[str, str]]:
    """Render the full long-form video.

    Args:
        df: DataFrame with [date, entity, value] — monthly or yearly.
        chart_type: One of the supported chart types.
        topic_info: Dict with topic, description, source, etc.
        entity_colors: Pre-assigned entity colors. If None, assigns them.

    Returns:
        Tuple of (output_path, entity_colors_used).

    Raises:
        RuntimeError: If rendering or FFmpeg encoding fails.
    """
    print(f"[renderer_long] Chart type: {chart_type}")
    print(f"[renderer_long] Data shape: {df.shape}")

    # Prepare output directories
    FRAMES_LONG_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Get all unique entities and assign colors
    all_entities = sorted(df["entity"].unique())
    if entity_colors is None:
        entity_colors = assign_entity_colors(all_entities)
    else:
        # Ensure all entities have colors
        for e in all_entities:
            if e not in entity_colors:
                entity_colors[e] = ACCENT_COLORS[len(entity_colors) % len(ACCENT_COLORS)]

    # Get sorted unique time steps
    time_steps = sorted(df["date"].unique())
    n_steps = len(time_steps)
    print(f"[renderer_long] Time steps: {n_steps}")

    # Calculate frames per step to hit target duration
    total_frames_min = LONG_FORM_MIN_DURATION * FPS
    total_frames_max = LONG_FORM_MAX_DURATION * FPS
    frames_per_step = max(LONG_FRAMES_PER_STEP, total_frames_min // max(n_steps, 1))
    frames_per_step = min(frames_per_step, total_frames_max // max(n_steps, 1))
    frames_per_step = max(frames_per_step, 10)

    # Add title card frames (3 seconds)
    title_frames = FPS * 3
    total_frames = title_frames + n_steps * frames_per_step

    print(f"[renderer_long] Frames per step: {frames_per_step}")
    print(f"[renderer_long] Total frames: {total_frames}")
    print(f"[renderer_long] Estimated duration: {total_frames / FPS:.0f}s")

    # Render title card
    frame_num = 0
    frame_num = _render_title_card(
        topic_info, time_steps, frame_num, title_frames
    )

    # Render data frames
    if chart_type == "bar_chart_race":
        frame_num = _render_bar_chart_race(
            df, time_steps, entity_colors, topic_info,
            frames_per_step, frame_num, is_short=False,
        )
    elif chart_type == "line_chart_race":
        frame_num = _render_line_chart_race(
            df, time_steps, entity_colors, topic_info,
            frames_per_step, frame_num, is_short=False,
        )
    elif chart_type == "area_chart":
        frame_num = _render_area_chart(
            df, time_steps, entity_colors, topic_info,
            frames_per_step, frame_num, is_short=False,
        )
    else:
        # Default to bar chart race for unsupported types
        print(f"[renderer_long] Falling back to bar_chart_race for '{chart_type}'")
        frame_num = _render_bar_chart_race(
            df, time_steps, entity_colors, topic_info,
            frames_per_step, frame_num, is_short=False,
        )

    # Encode with FFmpeg
    _encode_video(FRAMES_LONG_DIR, LONG_FORM_FINAL)

    print(f"[renderer_long] Output: {LONG_FORM_FINAL}")
    return LONG_FORM_FINAL, entity_colors


def _render_title_card(
    topic_info: dict, time_steps: list, start_frame: int, n_frames: int
) -> int:
    """Render title card frames."""
    fig, ax = plt.subplots(figsize=(19.20, 10.80), dpi=100)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    title = topic_info.get("topic", "Data Visualization")
    ts_start = pd.Timestamp(time_steps[0])
    ts_end = pd.Timestamp(time_steps[-1])
    subtitle = f"{ts_start.year} — {ts_end.year}"
    source = f"Source: {topic_info.get('source', 'Public Data')}"

    ax.text(0.5, 0.55, title, transform=ax.transAxes, fontsize=42,
            color=TEXT_COLOR, ha="center", va="center", fontweight="bold",
            fontfamily="sans-serif", wrap=True)
    ax.text(0.5, 0.40, subtitle, transform=ax.transAxes, fontsize=28,
            color=ACCENT_COLORS[0], ha="center", va="center", fontfamily="sans-serif")
    ax.text(0.5, 0.15, source, transform=ax.transAxes, fontsize=16,
            color=SUBTITLE_COLOR, ha="center", va="center", fontfamily="sans-serif")

    for i in range(n_frames):
        frame_path = FRAMES_LONG_DIR / f"frame_{start_frame + i:05d}.png"
        fig.savefig(frame_path, facecolor=BG_COLOR, bbox_inches="tight",
                    pad_inches=0.5, dpi=100)
    plt.close(fig)

    return start_frame + n_frames


def _render_bar_chart_race(
    df: pd.DataFrame, time_steps: list, colors: dict, topic_info: dict,
    frames_per_step: int, start_frame: int, is_short: bool = False,
) -> int:
    """Render horizontal bar chart race frames with interpolation."""
    frame_num = start_frame
    dpi = 100
    if is_short:
        figsize = (10.80, 19.20)
        title_size, label_size, value_size, date_size = 28, 16, 14, 36
    else:
        figsize = (19.20, 10.80)
        title_size, label_size, value_size, date_size = 24, 14, 12, 32

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

        # Get top N for current and next
        current_top = current_data.nlargest(TOP_N_ENTITIES, "value")
        next_top = next_data.nlargest(TOP_N_ENTITIES, "value")

        # All entities in either top list
        all_shown = set(current_top["entity"]).union(set(next_top["entity"]))

        for interp_idx in range(frames_per_step):
            t = interp_idx / max(frames_per_step - 1, 1)
            t = _ease_in_out(t)

            # Interpolate values
            interp_values = {}
            for entity in all_shown:
                cv = current_data[current_data["entity"] == entity]["value"]
                nv = next_data[next_data["entity"] == entity]["value"]
                cv = cv.iloc[0] if len(cv) > 0 else 0
                nv = nv.iloc[0] if len(nv) > 0 else 0
                interp_values[entity] = cv + (nv - cv) * t

            # Sort by interpolated value
            sorted_entities = sorted(interp_values.items(), key=lambda x: x[1])
            sorted_entities = sorted_entities[-TOP_N_ENTITIES:]  # bottom = smallest

            entities = [e for e, _ in sorted_entities]
            values = [v for _, v in sorted_entities]
            bar_colors = [colors.get(e, ACCENT_COLORS[0]) for e in entities]

            # Draw frame
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            fig.patch.set_facecolor(BG_COLOR)
            ax.set_facecolor(BG_COLOR)

            bars = ax.barh(range(len(entities)), values, color=bar_colors,
                          height=0.7, edgecolor="none")

            # Entity labels
            ax.set_yticks(range(len(entities)))
            ax.set_yticklabels(entities, fontsize=label_size, color=TEXT_COLOR,
                              fontweight="bold", fontfamily="sans-serif")

            # Value labels on bars
            max_val = max(values) if values else 1
            for i, (val, bar) in enumerate(zip(values, bars)):
                ax.text(val + max_val * 0.01, i, _format_value(val),
                       fontsize=value_size, color=TEXT_COLOR, va="center",
                       fontfamily="sans-serif")

            # Date display
            interp_date = current_ts + (next_ts - current_ts) * t
            if is_short:
                date_label = str(interp_date.year)
            else:
                date_label = interp_date.strftime("%b %Y") if (next_ts - current_ts).days < 400 else str(interp_date.year)

            ax.text(0.95, 0.05, date_label, transform=ax.transAxes,
                   fontsize=date_size, color=SUBTITLE_COLOR, ha="right",
                   va="bottom", fontweight="bold", fontfamily="sans-serif",
                   alpha=0.6)

            # Title
            ax.set_title(topic_title, fontsize=title_size, color=TEXT_COLOR,
                        fontweight="bold", fontfamily="sans-serif", pad=20)

            # Style axes
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color(SUBTITLE_COLOR)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="x", colors=SUBTITLE_COLOR, labelsize=10)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: _format_value(x)))

            plt.tight_layout(pad=1.5)

            frame_path = FRAMES_LONG_DIR / f"frame_{frame_num:05d}.png"
            fig.savefig(frame_path, facecolor=BG_COLOR, dpi=dpi)
            plt.close(fig)
            frame_num += 1

        if step_idx % 10 == 0:
            print(f"[renderer_long] Progress: step {step_idx+1}/{len(time_steps)}")

    return frame_num


def _render_line_chart_race(
    df: pd.DataFrame, time_steps: list, colors: dict, topic_info: dict,
    frames_per_step: int, start_frame: int, is_short: bool = False,
) -> int:
    """Render animated line chart frames."""
    frame_num = start_frame
    dpi = 100
    if is_short:
        figsize = (10.80, 19.20)
    else:
        figsize = (19.20, 10.80)

    entities = df["entity"].unique()
    topic_title = topic_info.get("topic", "")

    # Build full time series per entity
    entity_series = {}
    for entity in entities:
        edata = df[df["entity"] == entity].sort_values("date")
        entity_series[entity] = edata

    for step_idx in range(len(time_steps)):
        current_ts = pd.Timestamp(time_steps[step_idx])

        for interp_idx in range(frames_per_step):
            if step_idx < len(time_steps) - 1:
                next_ts = pd.Timestamp(time_steps[step_idx + 1])
                t = interp_idx / max(frames_per_step - 1, 1)
                interp_date = current_ts + (next_ts - current_ts) * t
            else:
                interp_date = current_ts

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            fig.patch.set_facecolor(BG_COLOR)
            ax.set_facecolor(BG_COLOR)

            for entity in entities:
                edata = entity_series[entity]
                mask = edata["date"] <= time_steps[step_idx]
                visible = edata[mask]
                if visible.empty:
                    continue
                color = colors.get(entity, ACCENT_COLORS[0])
                ax.plot(visible["date"], visible["value"],
                       color=color, linewidth=2.5, label=entity)
                # Dot at current position
                ax.scatter([visible["date"].iloc[-1]], [visible["value"].iloc[-1]],
                          color=color, s=60, zorder=5)

            ax.set_title(topic_title, fontsize=22, color=TEXT_COLOR,
                        fontweight="bold", fontfamily="sans-serif", pad=15)
            ax.legend(loc="upper left", fontsize=11, facecolor=BG_COLOR,
                     edgecolor=SUBTITLE_COLOR, labelcolor=TEXT_COLOR)

            # Date label
            date_label = interp_date.strftime("%Y") if is_short else interp_date.strftime("%b %Y")
            ax.text(0.95, 0.05, date_label, transform=ax.transAxes,
                   fontsize=28, color=SUBTITLE_COLOR, ha="right", va="bottom",
                   fontweight="bold", alpha=0.6)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color(SUBTITLE_COLOR)
            ax.spines["left"].set_color(SUBTITLE_COLOR)
            ax.tick_params(colors=SUBTITLE_COLOR)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: _format_value(x)))

            plt.tight_layout()
            frame_path = FRAMES_LONG_DIR / f"frame_{frame_num:05d}.png"
            fig.savefig(frame_path, facecolor=BG_COLOR, dpi=dpi)
            plt.close(fig)
            frame_num += 1

        if step_idx % 10 == 0:
            print(f"[renderer_long] Progress: step {step_idx+1}/{len(time_steps)}")

    return frame_num


def _render_area_chart(
    df: pd.DataFrame, time_steps: list, colors: dict, topic_info: dict,
    frames_per_step: int, start_frame: int, is_short: bool = False,
) -> int:
    """Render animated area chart frames."""
    frame_num = start_frame
    dpi = 100
    figsize = (10.80, 19.20) if is_short else (19.20, 10.80)

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
                ax.fill_between(visible["date"], visible["value"],
                               alpha=0.4, color=color)
                ax.plot(visible["date"], visible["value"],
                       color=color, linewidth=2, label=entity)

            ax.set_title(topic_title, fontsize=22, color=TEXT_COLOR,
                        fontweight="bold", fontfamily="sans-serif")
            if len(entities) > 1:
                ax.legend(loc="upper left", fontsize=10, facecolor=BG_COLOR,
                         edgecolor=SUBTITLE_COLOR, labelcolor=TEXT_COLOR)

            current_ts = pd.Timestamp(time_steps[step_idx])
            ax.text(0.95, 0.05, str(current_ts.year), transform=ax.transAxes,
                   fontsize=28, color=SUBTITLE_COLOR, ha="right", va="bottom",
                   fontweight="bold", alpha=0.6)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color(SUBTITLE_COLOR)
            ax.spines["left"].set_color(SUBTITLE_COLOR)
            ax.tick_params(colors=SUBTITLE_COLOR)

            plt.tight_layout()
            frame_path = FRAMES_LONG_DIR / f"frame_{frame_num:05d}.png"
            fig.savefig(frame_path, facecolor=BG_COLOR, dpi=dpi)
            plt.close(fig)
            frame_num += 1

    return frame_num


def _encode_video(frames_dir: Path, output_path: Path) -> None:
    """Encode frames into MP4 using FFmpeg."""
    print(f"[renderer_long] Encoding video with FFmpeg...")
    cmd = [
        "ffmpeg", "-y",
        "-r", str(FPS),
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


def _ease_in_out(t: float) -> float:
    """Smooth easing function for interpolation."""
    return t * t * (3 - 2 * t)


def _format_value(val: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    val = float(val)
    if abs(val) >= 1e12:
        return f"{val/1e12:.1f}T"
    elif abs(val) >= 1e9:
        return f"{val/1e9:.1f}B"
    elif abs(val) >= 1e6:
        return f"{val/1e6:.1f}M"
    elif abs(val) >= 1e3:
        return f"{val/1e3:.1f}K"
    else:
        return f"{val:.1f}"
