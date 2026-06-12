"""
thumbnail.py — Generates a YouTube thumbnail image for the Short.

Creates a 1080x1920 vertical thumbnail with:
- Clean chart geometry rendering as background (without title/source chrome)
- Semi-transparent dark overlay (85% black) for a dimmed background texture
- High-tech pink/purple gradient overlay (15% alpha)
- DataDrift logo watermark
- Hook text (large, bold)
- Key stat from extreme segment (oversized accent color)
- Year range badge
"""

import re
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd

from pipeline.config import ACCENT_COLORS, OUTPUT_DIR, TOP_N_ENTITIES
from pipeline.modules.font_loader import FONT_BOLD, FONT_REGULAR, overlay_watermark
from pipeline.modules.renderer_long import format_value


OUTLINE_THICK = [
    path_effects.Stroke(linewidth=5, foreground="black"),
    path_effects.Normal(),
]


# ── Clean Chart Draw Helpers ──────────────────────────────────────────────────

def _draw_bars_clean(
    ax: plt.Axes,
    df_seg: pd.DataFrame,
    entity_colors: dict,
    end_year: int,
    short_unit: str,
) -> None:
    """Draw clean horizontal bar chart for the end year."""
    year_data = df_seg[df_seg["date"].dt.year == end_year]
    sorted_year = year_data.sort_values("value", ascending=False)
    top_10 = sorted_year.head(TOP_N_ENTITIES)
    top_10 = top_10.iloc[::-1]  # reverse to draw top values at the top

    y_positions = np.arange(len(top_10))
    values = top_10["value"].values
    entities = top_10["entity"].values

    max_val = max(values) * 1.1 if len(values) > 0 else 1.0
    if max_val <= 0:
        max_val = 1.0

    ax.set_xlim(0, max_val)
    ax.set_ylim(-0.6, 9.6)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, (ent, val) in enumerate(zip(entities, values)):
        color = entity_colors.get(ent, ACCENT_COLORS[0])
        rgba_face = to_rgba(color, alpha=0.35)
        rgba_edge = to_rgba(color, alpha=0.85)
        ax.barh(i, val, height=0.6, facecolor=rgba_face, edgecolor=rgba_edge, linewidth=2)

        # Entity label
        clean_name = re.sub(r"\s*\([^)]*\)\s*$", "", ent).strip()
        val_str = format_value(val, short_unit)
        ax.text(
            val + (max_val * 0.02), i, f"{clean_name} ({val_str})",
            ha="left", va="center", color="white", fontsize=11, fontproperties=FONT_BOLD,
            path_effects=[path_effects.Stroke(linewidth=2.5, foreground="black"), path_effects.Normal()]
        )


def _draw_lines_clean(
    ax: plt.Axes,
    df_seg: pd.DataFrame,
    entity_colors: dict,
    start_year: int,
    end_year: int,
    short_unit: str,
) -> None:
    """Draw clean line plot over the segment years for 2-5 entities."""
    entities = sorted(df_seg["entity"].unique())
    years = sorted(df_seg["date"].dt.year.unique())

    y_min = float(df_seg["value"].min()) * 0.9
    y_max = float(df_seg["value"].max()) * 1.1
    if y_min == y_max:
        y_min, y_max = y_min - 1, y_max + 1

    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for entity in entities:
        color = entity_colors.get(entity, ACCENT_COLORS[0])
        ent_data = df_seg[df_seg["entity"] == entity].sort_values("date")
        x_vals = ent_data["date"].dt.year.values
        y_vals = ent_data["value"].values

        # Plot glow line + main line
        ax.plot(x_vals, y_vals, color=color, linewidth=5, alpha=0.15, solid_capstyle="round")
        ax.plot(x_vals, y_vals, color=color, linewidth=2.5, alpha=0.85, solid_capstyle="round")

        # End dot + label
        if len(x_vals) > 0:
            ax.plot(
                x_vals[-1], y_vals[-1], "o", color=color, markersize=8,
                markeredgecolor="white", markeredgewidth=1
            )
            clean_name = re.sub(r"\s*\([^)]*\)\s*$", "", entity).strip()
            if len(clean_name) > 12:
                clean_name = clean_name[:10] + "..."
            ax.text(
                x_vals[-1] + 0.2, y_vals[-1], clean_name, ha="left", va="center",
                color="white", fontsize=11, fontproperties=FONT_BOLD,
                path_effects=[path_effects.Stroke(linewidth=2.5, foreground="black"), path_effects.Normal()]
            )


def _draw_area_clean(
    ax: plt.Axes,
    df_seg: pd.DataFrame,
    entity_colors: dict,
    start_year: int,
    end_year: int,
    short_unit: str,
) -> None:
    """Draw clean filled area plot for the primary entity."""
    entity_avg = df_seg.groupby("entity")["value"].mean()
    primary_entity = entity_avg.idxmax()
    primary_color = entity_colors.get(primary_entity, ACCENT_COLORS[0])

    ent_data = df_seg[df_seg["entity"] == primary_entity].sort_values("date")
    x_vals = ent_data["date"].dt.year.values
    y_vals = ent_data["value"].values

    y_max = max(y_vals) * 1.15 if len(y_vals) > 0 else 1.0

    ax.set_xlim(x_vals[0] - 0.5, x_vals[-1] + 0.5)
    ax.set_ylim(0, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    rgba_fill = to_rgba(primary_color, alpha=0.25)
    rgba_line = to_rgba(primary_color, alpha=0.85)

    ax.fill_between(x_vals, y_vals, y2=0, color=rgba_fill)
    ax.plot(x_vals, y_vals, color=rgba_line, linewidth=3)

    if len(x_vals) > 0:
        ax.plot(
            x_vals[-1], y_vals[-1], "o", color=primary_color, markersize=10,
            markeredgecolor="white", markeredgewidth=1.5
        )


def _draw_bubbles_clean(
    ax: plt.Axes,
    df_seg: pd.DataFrame,
    entity_colors: dict,
    end_year: int,
    short_unit: str,
) -> None:
    """Draw clean packed bubbles for the end year."""
    from pipeline.modules.renderer_short import _pack_circles

    year_data = df_seg[df_seg["date"].dt.year == end_year]
    all_vals = df_seg["value"].values
    global_max_val = max(all_vals) if len(all_vals) > 0 else 1.0

    entities_max = df_seg.groupby("entity")["value"].max()
    sorted_entities = entities_max.sort_values(ascending=False).index.tolist()
    top_entities = sorted_entities[:TOP_N_ENTITIES]

    # Packing max radii setup
    R_test_max = 20.0
    pack_input = []
    for ent in top_entities:
        val = entities_max.get(ent, 0.0)
        r_test = R_test_max * np.sqrt(max(val, 0) / global_max_val)
        pack_input.append((ent, r_test))
    pack_input.sort(key=lambda x: x[1], reverse=True)

    axes_aspect = 19.2 / 10.8
    axes_height = 100.0 * axes_aspect

    packed_positions = _pack_circles(pack_input, 0.0, 0.0)

    # Calculate bounding box
    X_min = min(packed_positions[ent][0] - r_test for ent, r_test in pack_input)
    X_max = max(packed_positions[ent][0] + r_test for ent, r_test in pack_input)
    Y_min = min(packed_positions[ent][1] - r_test for ent, r_test in pack_input)
    Y_max = max(packed_positions[ent][1] + r_test for ent, r_test in pack_input)

    W_packed = X_max - X_min
    H_packed = Y_max - Y_min

    margin_x = 6.0
    margin_y = 12.0
    W_target = 100.0 - 2 * margin_x
    H_target = axes_height - 2 * margin_y
    S = min(W_target / W_packed, H_target / H_packed)

    X_center_packed = (X_min + X_max) / 2.0
    Y_center_packed = (Y_min + Y_max) / 2.0
    X_center_screen = 50.0
    Y_center_screen = axes_height / 2.0

    scale_factor = S * R_test_max

    ax.set_xlim(0, 100)
    ax.set_ylim(0, axes_height)
    ax.set_aspect('equal')
    ax.set_axis_off()

    for ent in top_entities:
        if ent not in packed_positions:
            continue
        val_arr = year_data[year_data["entity"] == ent]["value"].values
        val = val_arr[0] if len(val_arr) > 0 else 0.0
        radius = scale_factor * np.sqrt(max(val, 0) / global_max_val)
        if radius <= 0.01:
            continue

        px, py = packed_positions[ent]
        cx = X_center_screen + S * (px - X_center_packed)
        cy = Y_center_screen + S * (py - Y_center_packed)

        color = entity_colors.get(ent, ACCENT_COLORS[0])

        glow = plt.Circle(
            (cx, cy), radius + 2,
            facecolor="none",
            edgecolor=to_rgba(color, alpha=0.15),
            linewidth=4,
        )
        ax.add_patch(glow)

        bubble = plt.Circle(
            (cx, cy), radius,
            facecolor=to_rgba(color, alpha=0.45),
            edgecolor=to_rgba(color, alpha=0.85),
            linewidth=2,
        )
        ax.add_patch(bubble)

        clean_name = re.sub(r"\s*\([^)]*\)\s*$", "", ent).strip()
        val_str = format_value(val, short_unit)

        if radius >= 7.5:
            name_lines = textwrap.wrap(clean_name, width=max(10, int(radius * 0.9)))
            display_text = "\n".join(name_lines) + f"\n{val_str}"
            fontsize = max(8, min(14, int(radius * 0.55)))
            ax.text(
                cx, cy, display_text, ha="center", va="center", color="white",
                fontsize=fontsize, fontproperties=FONT_BOLD,
                path_effects=[path_effects.Stroke(linewidth=2.5, foreground="black"), path_effects.Normal()]
            )
        elif radius >= 4.0:
            ax.text(
                cx, cy, val_str, ha="center", va="center", color="white",
                fontsize=8, fontproperties=FONT_BOLD,
                path_effects=[path_effects.Stroke(linewidth=2, foreground="black"), path_effects.Normal()]
            )


def _draw_map_clean(
    ax: plt.Axes,
    df_seg: pd.DataFrame,
    entity_colors: dict,
    end_year: int,
    short_unit: str,
) -> None:
    """Draw clean world map choropleth for the end year."""
    from pipeline.modules.renderer_short import _load_world_geometry, _match_entity_to_iso

    world = _load_world_geometry()
    if world is None:
        return

    year_data = df_seg[df_seg["date"].dt.year == end_year]
    entities = year_data["entity"].unique()

    iso_values = {}
    for ent in entities:
        iso = _match_entity_to_iso(ent, world)
        if iso:
            val_arr = year_data[year_data["entity"] == ent]["value"].values
            if len(val_arr) > 0:
                iso_values[iso] = float(val_arr[0])

    all_vals = df_seg["value"].values
    val_min = min(all_vals) if len(all_vals) > 0 else 0.0
    val_max = max(all_vals) if len(all_vals) > 0 else 1.0

    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 85)
    ax.set_axis_off()

    world_copy = world.copy()
    world_copy["_value"] = world_copy["iso_a3"].map(iso_values)

    # Base background (unmatched)
    world_copy[world_copy["_value"].isna()].plot(
        ax=ax, color="#1a1a2e", edgecolor="#2a2a3e", linewidth=0.2
    )

    # Matched countries
    matched = world_copy[world_copy["_value"].notna()]
    if not matched.empty:
        matched.plot(
            ax=ax, column="_value", cmap="plasma",
            vmin=val_min, vmax=val_max,
            edgecolor="#666666", linewidth=0.5
        )


# ── Main Generator ────────────────────────────────────────────────────────────

def generate_thumbnail(
    df_yearly: pd.DataFrame,
    chart_type: str,
    topic_info: dict,
    extreme_segment: dict,
    entity_colors: dict,
) -> Path:
    """Generate a thumbnail PNG for the Short video.

    Args:
        df_yearly: Full yearly DataFrame.
        chart_type: The chart type used.
        topic_info: Topic metadata dict.
        extreme_segment: Extreme segment dict with start_year, end_year, hook.
        entity_colors: Entity-to-color mapping.

    Returns:
        Path to the saved thumbnail PNG.
    """
    print(f"[thumbnail] Generating thumbnail with background chart snapshot ({chart_type})...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    thumb_path = OUTPUT_DIR / "thumbnail.png"

    # Setup figure
    fig = plt.figure(figsize=(10.8, 19.2), dpi=100)
    fig.patch.set_facecolor("#000000")

    # Safe boundaries extraction
    start_yr = int(extreme_segment.get("start_year", df_yearly["date"].dt.year.min()))
    end_yr   = int(extreme_segment.get("end_year", df_yearly["date"].dt.year.max()))
    hook = extreme_segment.get("hook", topic_info.get("topic", "Data Visualization"))
    short_unit = topic_info.get("short_unit", "")

    # Extract key stats
    stat_text = _compute_key_stat(df_yearly, extreme_segment, short_unit)
    hook_wrapped = "\n".join(textwrap.wrap(hook, width=20))

    # ── 1. Draw Clean Chart Snapshot (zorder=1) ───────────────────────────
    ax_chart = fig.add_axes([0.05, 0.1, 0.9, 0.8], zorder=1)
    ax_chart.patch.set_facecolor("none")

    seg_data = df_yearly[
        (df_yearly["date"].dt.year >= start_yr) &
        (df_yearly["date"].dt.year <= end_yr)
    ].copy()

    try:
        if chart_type == "line_chart_race":
            _draw_lines_clean(ax_chart, seg_data, entity_colors, start_yr, end_yr, short_unit)
        elif chart_type == "area_chart":
            _draw_area_clean(ax_chart, seg_data, entity_colors, start_yr, end_yr, short_unit)
        elif chart_type == "bubble_chart":
            _draw_bubbles_clean(ax_chart, seg_data, entity_colors, end_yr, short_unit)
        elif chart_type == "map_animation":
            _draw_map_clean(ax_chart, seg_data, entity_colors, end_yr, short_unit)
        else:
            # default: bar_chart_race
            _draw_bars_clean(ax_chart, seg_data, entity_colors, end_yr, short_unit)
    except Exception as ex:
        print(f"[thumbnail] Warning: Failed to draw background chart snapshot: {ex}")
        import traceback
        traceback.print_exc()

    # ── 2. Heavily Dimmed Translucent Black Overlay (zorder=2) ────────────
    dim_overlay = plt.Rectangle(
        (0, 0), 1, 1, facecolor="#000000", alpha=0.85,
        transform=fig.transFigure, zorder=2,
    )
    fig.patches.append(dim_overlay)

    # ── 3. High-Tech Color Gradient (zorder=3) ────────────────────────────
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    gradient = np.hstack([gradient] * 256)
    ax_bg = fig.add_axes([0, 0, 1, 1], zorder=3)
    ax_bg.patch.set_facecolor("none")
    ax_bg.imshow(
        gradient, aspect="auto", cmap="RdPu", alpha=0.15,
        extent=[0, 1, 0, 1], origin="lower", zorder=3,
    )
    ax_bg.set_axis_off()

    # ── 4. Brand Watermark Logo (zorder=100) ──────────────────────────────
    overlay_watermark(fig, x=0.35, y=0.72, size=320, alpha=0.08)

    # ── 5. Hook Text Overlay (zorder=5) ───────────────────────────────────
    fig.text(
        0.5, 0.65, hook_wrapped,
        ha="center", va="center",
        color="white", fontsize=44, fontproperties=FONT_BOLD,
        path_effects=OUTLINE_THICK,
        transform=fig.transFigure, zorder=5,
        linespacing=1.3,
    )

    # ── 6. Key Stat Overlay (zorder=5) ────────────────────────────────────
    if stat_text:
        # Pick the bright coral/pink accent color from the config
        accent = ACCENT_COLORS[0]
        fig.text(
            0.5, 0.42, stat_text,
            ha="center", va="center",
            color=accent, fontsize=64, fontproperties=FONT_BOLD,
            path_effects=OUTLINE_THICK,
            transform=fig.transFigure, zorder=5,
        )

    # ── 7. Year Range Badge (zorder=5) ────────────────────────────────────
    year_text = f"{start_yr}  →  {end_yr}"
    fig.text(
        0.5, 0.28, year_text,
        ha="center", va="center",
        color="white", fontsize=36, fontproperties=FONT_BOLD,
        transform=fig.transFigure, zorder=5,
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="#1a1a2e",
            edgecolor="#444444",
            linewidth=2,
        ),
    )

    # ── 8. Chart Type Label (zorder=5) ────────────────────────────────────
    chart_label = chart_type.replace("_", " ").title()
    fig.text(
        0.5, 0.18, chart_label,
        ha="center", va="center",
        color="#888888", fontsize=18, fontproperties=FONT_REGULAR,
        transform=fig.transFigure, zorder=5,
    )

    # ── 9. Top Entities List (zorder=5) ───────────────────────────────────
    if not seg_data.empty:
        last_year = seg_data["date"].dt.year.max()
        last_vals = seg_data[seg_data["date"].dt.year == last_year]
        top_5 = last_vals.nlargest(5, "value")

        y_pos = 0.13
        for _, row in top_5.iterrows():
            name = re.sub(r"\s*\([^)]*\)\s*$", "", str(row["entity"])).strip()
            if len(name) > 20:
                name = name[:17] + "..."
            val = format_value(float(row["value"]), short_unit)
            color = entity_colors.get(row["entity"], "#ffffff")

            fig.text(
                0.3, y_pos, f"- {name}",
                ha="left", va="center",
                color=color, fontsize=16, fontproperties=FONT_BOLD,
                transform=fig.transFigure, zorder=5,
            )
            fig.text(
                0.75, y_pos, val,
                ha="right", va="center",
                color="white", fontsize=16, fontproperties=FONT_REGULAR,
                transform=fig.transFigure, zorder=5,
            )
            y_pos -= 0.025

    # Save output
    fig.savefig(
        thumb_path, dpi=100, facecolor="#000000", pad_inches=0,
        pil_kwargs={"compress_level": 1},
    )
    plt.close(fig)
    print(f"[thumbnail] Saved: {thumb_path}")
    return thumb_path


def _compute_key_stat(
    df_yearly: pd.DataFrame,
    extreme_segment: dict,
    short_unit: str,
) -> Optional[str]:
    """Compute a compelling stat from the extreme segment data.

    Returns something like '↑ 1,400%' or '$2.3T → $18.7T'.
    """
    start_yr = extreme_segment.get("start_year")
    end_yr   = extreme_segment.get("end_year")

    if not start_yr or not end_yr:
        return None

    seg_data = df_yearly[
        (df_yearly["date"].dt.year >= start_yr) &
        (df_yearly["date"].dt.year <= end_yr)
    ]
    if seg_data.empty:
        return None

    # Find the entity with the most dramatic change
    start_data = seg_data[seg_data["date"].dt.year == start_yr]
    end_data   = seg_data[seg_data["date"].dt.year == end_yr]

    if start_data.empty or end_data.empty:
        return None

    # Merge to find matching entities
    merged = start_data.merge(end_data, on="entity", suffixes=("_start", "_end"))
    if merged.empty:
        return None

    # Calculate percentage change
    merged["pct_change"] = (
        (merged["value_end"] - merged["value_start"]) / merged["value_start"].abs().clip(lower=0.001)
    ) * 100

    # Pick the most dramatic
    best = merged.loc[merged["pct_change"].abs().idxmax()]
    pct = float(best["pct_change"])

    if abs(pct) > 50:
        arrow = "↑" if pct > 0 else "↓"
        return f"{arrow} {abs(pct):.0f}%"
    else:
        # Show start → end values
        v_start = format_value(float(best["value_start"]), short_unit)
        v_end   = format_value(float(best["value_end"]), short_unit)
        return f"{v_start} → {v_end}"
