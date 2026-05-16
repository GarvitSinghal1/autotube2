"""
renderer_long.py — Renders the full long-form animated data visualization video.
Uses a fixed slot system for perfectly smooth transitions and no bouncing axes.
"""

import subprocess
import random
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.config import (
    FPS, LONG_FRAMES_PER_STEP, LONG_FORM_MIN_DURATION, LONG_FORM_MAX_DURATION,
    BG_COLOR, TEXT_COLOR, SUBTITLE_COLOR, ACCENT_COLORS,
    FRAMES_LONG_DIR, LONG_FORM_FINAL, TOP_N_ENTITIES, TMP_DIR,
    MUSIC_DIR, DEFAULT_VOLUME, PROJECT_ROOT
)

# Fixed slot configuration
SLOTS = {i: float(9 - i) for i in range(10)}

def assign_entity_colors(entities: list[str]) -> dict[str, str]:
    """Assign consistent colors to entities."""
    return {entity: ACCENT_COLORS[i % len(ACCENT_COLORS)] for i, entity in enumerate(sorted(entities))}

def format_value(value: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return f"{value:.0f}"

def draw_hook_frame(fig, hook_text, alpha_text, title, title_alpha, frames_dir, frame_number):
    """Draw Phase 1 & 2: Intro frames with fading hook and title."""
    fig.clf()
    fig.patch.set_facecolor('#000000')
    
    # Hook text box
    if alpha_text > 0:
        # Calculate y position: moves from 0.5 to 0.95 in Phase 2
        # Phase 2 starts when title_alpha > 0
        hook_y = 0.5 + (0.45 * title_alpha) if title_alpha > 0 else 0.5
        
        fig.text(0.5, hook_y, hook_text,
                 ha='center', va='center',
                 color=(1, 1, 1, alpha_text),
                 fontsize=28, fontweight='bold',
                 wrap=True,
                 transform=fig.transFigure,
                 bbox=dict(boxstyle='round,pad=0.8',
                          facecolor='#111111',
                          edgecolor=(1, 1, 1, alpha_text * 0.4),
                          linewidth=1))
    
    # Title fades in at the top
    if title_alpha > 0:
        fig.text(0.02, 0.97, title,
                 ha='left', va='top',
                 color=(1, 1, 1, title_alpha), 
                 fontsize=13, fontweight='bold',
                 transform=fig.transFigure)

    plt.savefig(f'{frames_dir}/frame_{frame_number:05d}.png',
                dpi=100, facecolor='#000000', pad_inches=0)

def draw_frame(ax, fig, entities_data, title, source, date_label, frames_dir, frame_number):
    """Draw Phase 3: The actual animated chart frame."""
    ax.cla()
    ax.set_facecolor('#000000')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 9.6)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#444444')
    ax.tick_params(axis='x', colors='white', labelsize=8)

    if not entities_data:
        max_value = 1
    else:
        max_value = max(d['value'] for d in entities_data) * 1.1
    
    for d in entities_data:
        entity = d['entity']
        norm_value = d['value'] / max_value if max_value > 0 else 0
        y = d['y_pos']
        color = d['color']
        
        # Draw bar
        ax.barh(y, norm_value, height=0.65, color=color, alpha=0.9, left=0)
        
        # Label inside left margin
        ax.text(-0.01, y, entity, ha='right', va='center',
                color='white', fontsize=10, fontweight='bold',
                transform=ax.transData)
        
        # Value label at end of bar
        ax.text(norm_value + 0.01, y, format_value(d['value']),
                ha='left', va='center', color='white', fontsize=9)
    
    # Update x-axis ticks
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([format_value(t * max_value) for t in ticks], color='white', fontsize=8)
    
    # Ghost year counter
    fig.text(0.95, 0.08, date_label,
             ha='right', va='bottom',
             color='white', alpha=0.15,
             fontsize=52, fontweight='bold',
             transform=fig.transFigure)
    
    # Static Title
    fig.text(0.02, 0.97, title,
             ha='left', va='top',
             color='white', fontsize=13, fontweight='bold',
             transform=fig.transFigure)
    
    # Static Source
    fig.text(0.02, 0.93, f'Source: {source}',
             ha='left', va='top',
             color='#888888', fontsize=9, style='italic',
             transform=fig.transFigure)
    
    plt.savefig(
        f'{frames_dir}/frame_{frame_number:05d}.png',
        dpi=100,
        bbox_inches=None,
        facecolor='#000000',
        pad_inches=0
    )

def render_long_form(df, chart_type, topic_info, entity_colors=None):
    """Main rendering entry point for long-form video."""
    print(f"[renderer_long] Starting render...")
    FRAMES_LONG_DIR.mkdir(parents=True, exist_ok=True)
    
    all_entities = sorted(df["entity"].unique())
    if entity_colors is None:
        entity_colors = assign_entity_colors(all_entities)
    
    time_steps = sorted(df["date"].unique())
    n_steps = len(time_steps)
    
    # Setup Figure
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_axes([0.15, 0.10, 0.80, 0.78])
    fig.patch.set_facecolor('#000000')
    
    hook_text = topic_info.get("hook", "Discover the data behind the story.")
    title = topic_info.get("topic", "Data Visualization")
    source = topic_info.get("source", "Public Data")
    
    frame_number = 0
    
    # Phase 1: Static Hook (1.5s = 45 frames)
    for f in range(45):
        draw_hook_frame(fig, hook_text, 1.0, title, 0.0, FRAMES_LONG_DIR, frame_number)
        frame_number += 1
        
    # Phase 2: Hook Slides/Fades, Title Fades In (1.5s = 45 frames)
    for f in range(45):
        t = f / 44
        eased_t = t * t * (3 - 2 * t)
        draw_hook_frame(fig, hook_text, 1.0 - eased_t, title, eased_t, FRAMES_LONG_DIR, frame_number)
        frame_number += 1

    # Phase 3: Chart Animation
    prev_ranks = {} # entity -> slot_index (0-9)
    
    for step_idx in range(len(time_steps) - 1):
        curr_ts = time_steps[step_idx]
        next_ts = time_steps[step_idx+1]
        
        curr_df = df[df["date"] == curr_ts]
        next_df = df[df["date"] == next_ts]
        
        # Calculate actual rankings at start and end of this step
        curr_sorted = curr_df.sort_values("value", ascending=False).head(10)
        next_sorted = next_df.sort_values("value", ascending=False).head(10)
        
        curr_step_ranks = {row.entity: i for i, row in enumerate(curr_sorted.itertuples())}
        next_step_ranks = {row.entity: i for i, row in enumerate(next_sorted.itertuples())}
        
        all_involved = set(curr_step_ranks.keys()).union(set(next_step_ranks.keys()))
        
        date_label = pd.Timestamp(curr_ts).strftime("%Y") if pd.Timestamp(next_ts).year != pd.Timestamp(curr_ts).year else pd.Timestamp(curr_ts).strftime("%b %Y")

        for f in range(LONG_FRAMES_PER_STEP):
            alpha = f / LONG_FRAMES_PER_STEP
            eased_alpha = alpha * alpha * (3 - 2 * alpha)
            
            entities_data = []
            for entity in all_involved:
                # Value interpolation
                v0 = curr_df[curr_df["entity"] == entity]["value"].values
                v1 = next_df[next_df["entity"] == entity]["value"].values
                v0 = v0[0] if len(v0) > 0 else 0
                v1 = v1[0] if len(v1) > 0 else 0
                interp_val = v0 * (1 - eased_alpha) + v1 * eased_alpha
                
                # Y Position interpolation (Slot movement)
                # Slot 0 is y=9, Slot 9 is y=0.
                r0 = curr_step_ranks.get(entity, 11) # 11 = off screen below
                r1 = next_step_ranks.get(entity, 11)
                
                y0 = SLOTS.get(r0, -2.0)
                y1 = SLOTS.get(r1, -2.0)
                interp_y = y0 * (1 - eased_alpha) + y1 * eased_alpha
                
                if interp_y > -1.5:
                    entities_data.append({
                        "entity": entity,
                        "value": interp_val,
                        "y_pos": interp_y,
                        "color": entity_colors.get(entity, ACCENT_COLORS[0])
                    })
            
            draw_frame(ax, fig, entities_data, title, source, date_label, FRAMES_LONG_DIR, frame_number)
            frame_number += 1
            
        if step_idx % 5 == 0:
            print(f"[renderer_long] Progress: {step_idx}/{len(time_steps)}")

    plt.close(fig)
    _encode_video(FRAMES_LONG_DIR, LONG_FORM_FINAL)
    return LONG_FORM_FINAL, entity_colors

def _encode_video(frames_dir: Path, output_path: Path) -> None:
    """Encode frames into MP4 using FFmpeg with music."""
    print(f"[renderer_long] Encoding...")
    import random
    music_files = list(MUSIC_DIR.glob("*.mp3")) + list(MUSIC_DIR.glob("*.wav"))
    
    if music_files:
        bg_music = random.choice(music_files)
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(FPS),
            '-i', f'{frames_dir}/frame_%05d.png',
            '-stream_loop', '-1',
            '-i', str(bg_music),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            '-preset', 'slow',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-filter:a', f'volume={DEFAULT_VOLUME}',
            '-shortest',
            '-ac', '2',
            '-ar', '44100',
            str(output_path)
        ]
    else:
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(FPS),
            '-i', f'{frames_dir}/frame_%05d.png',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            '-preset', 'slow',
            str(output_path)
        ]
        
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(f"[renderer_long] Done: {output_path}")
