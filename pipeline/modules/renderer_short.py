"""
renderer_short.py — Renders the YouTube Short from the extreme segment.
Uses a vertical format and a fixed slot system for smooth transitions.
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
    FPS, SHORT_FRAMES_PER_STEP, SHORT_MIN_DURATION, SHORT_MAX_DURATION,
    BG_COLOR, TEXT_COLOR, SUBTITLE_COLOR, ACCENT_COLORS,
    FRAMES_SHORT_DIR, SHORT_FINAL, TOP_N_ENTITIES, TMP_DIR,
    MUSIC_DIR, DEFAULT_VOLUME
)
from pipeline.modules.renderer_long import SLOTS, assign_entity_colors, format_value, draw_hook_frame

def draw_short_frame(ax, fig, entities_data, title, source, date_label, frames_dir, frame_number):
    """Draw a vertical frame for YouTube Shorts."""
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
        
        # Label inside left margin (vertical adjustment)
        ax.text(-0.01, y, entity, ha='right', va='center',
                color='white', fontsize=12, fontweight='bold',
                transform=ax.transData)
        
        # Value label at end of bar
        ax.text(norm_value + 0.01, y, format_value(d['value']),
                ha='left', va='center', color='white', fontsize=11)
    
    # Update x-axis ticks
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([format_value(t * max_value) for t in ticks], color='white', fontsize=8)
    
    # Ghost year counter (Vertical Short center-ish bottom)
    fig.text(0.5, 0.15, date_label,
             ha='center', va='bottom',
             color='white', alpha=0.15,
             fontsize=120, fontweight='bold',
             transform=fig.transFigure)
    
    # Static Title
    fig.text(0.5, 0.97, title,
             ha='center', va='top',
             color='white', fontsize=24, fontweight='bold',
             wrap=True, transform=fig.transFigure)
    
    # Static Source
    fig.text(0.5, 0.91, f'Source: {source}',
             ha='center', va='top',
             color='#888888', fontsize=10, style='italic',
             transform=fig.transFigure)
    
    plt.savefig(
        f'{frames_dir}/frame_{frame_number:05d}.png',
        dpi=100,
        bbox_inches=None,
        facecolor='#000000',
        pad_inches=0
    )

def render_short(df_yearly, chart_type, topic_info, extreme_segment, entity_colors=None):
    """Main rendering entry point for YouTube Shorts."""
    print(f"[renderer_short] Starting render...")
    FRAMES_SHORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Filter to extreme segment
    start_yr = extreme_segment["start_year"]
    end_yr = extreme_segment["end_year"]
    df_segment = df_yearly[
        (df_yearly["date"].dt.year >= start_yr) &
        (df_yearly["date"].dt.year <= end_yr)
    ].copy()
    
    if df_segment.empty:
        raise RuntimeError(f"No data in segment {start_yr}–{end_yr}")

    all_entities = sorted(df_yearly["entity"].unique())
    if entity_colors is None:
        entity_colors = assign_entity_colors(all_entities)
        
    time_steps = sorted(df_segment["date"].unique())
    
    # Setup Figure (Vertical)
    fig = plt.figure(figsize=(10.8, 19.2), dpi=100)
    ax = fig.add_axes([0.20, 0.08, 0.75, 0.78])
    fig.patch.set_facecolor('#000000')
    
    hook_text = extreme_segment.get("hook", "This changed everything.")
    title = topic_info.get("topic", "Data Visualization")
    source = topic_info.get("source", "Public Data")
    
    frame_number = 0
    
    # Phase 1: Static Hook (1.5s = 45 frames)
    for f in range(45):
        draw_hook_frame(fig, hook_text, 1.0, title, 0.0, FRAMES_SHORT_DIR, frame_number)
        frame_number += 1
        
    # Phase 2: Hook Slides/Fades, Title Fades In (1.5s = 45 frames)
    for f in range(45):
        t = f / 44
        eased_t = t * t * (3 - 2 * t)
        draw_hook_frame(fig, hook_text, 1.0 - eased_t, title, eased_t, FRAMES_SHORT_DIR, frame_number)
        frame_number += 1

    # Phase 3: Chart Animation
    for step_idx in range(len(time_steps) - 1):
        curr_ts = time_steps[step_idx]
        next_ts = time_steps[step_idx+1]
        
        curr_df = df_segment[df_segment["date"] == curr_ts]
        next_df = df_segment[df_segment["date"] == next_ts]
        
        curr_step_ranks = {row.entity: i for i, row in enumerate(curr_df.sort_values("value", ascending=False).head(10).itertuples())}
        next_step_ranks = {row.entity: i for i, row in enumerate(next_df.sort_values("value", ascending=False).head(10).itertuples())}
        
        all_involved = set(curr_step_ranks.keys()).union(set(next_step_ranks.keys()))
        date_label = str(pd.Timestamp(curr_ts).year)

        # Using a fixed 15 frames per step for shorts to keep pacing high
        STEP_FRAMES = 15
        
        for f in range(STEP_FRAMES):
            alpha = f / STEP_FRAMES
            eased_alpha = alpha * alpha * (3 - 2 * alpha)
            
            entities_data = []
            for entity in all_involved:
                v0 = curr_df[curr_df["entity"] == entity]["value"].values
                v1 = next_df[next_df["entity"] == entity]["value"].values
                v0 = v0[0] if len(v0) > 0 else 0
                v1 = v1[0] if len(v1) > 0 else 0
                interp_val = v0 * (1 - eased_alpha) + v1 * eased_alpha
                
                r0 = curr_step_ranks.get(entity, 11)
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
            
            draw_short_frame(ax, fig, entities_data, title, source, date_label, FRAMES_SHORT_DIR, frame_number)
            frame_number += 1

    plt.close(fig)
    _encode_short(FRAMES_SHORT_DIR, SHORT_FINAL)
    return SHORT_FINAL, entity_colors

def _encode_short(frames_dir: Path, output_path: Path) -> None:
    """Encode frames into vertical MP4 using FFmpeg with music."""
    print(f"[renderer_short] Encoding...")
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
    print(f"[renderer_short] Done: {output_path}")
