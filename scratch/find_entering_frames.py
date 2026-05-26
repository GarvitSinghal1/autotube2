import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path("/Users/garvitsinghal/Library/CloudStorage/GoogleDrive-progarvit000@gmail.com/Other computers/My Computer/Codes/Robotics/CurrentlyWorking/autotube2")
sys.path.insert(0, str(PROJECT_ROOT))

from scratch.test_short_render import generate_mock_data

def find_entering():
    df_data = generate_mock_data()
    start_yr = 2013
    end_yr = 2018
    df_seg = df_data[
        (df_data["date"].dt.year >= start_yr) &
        (df_data["date"].dt.year <= end_yr)
    ].copy()
    
    time_steps = sorted(df_seg["date"].unique())
    step_values = []
    for ts in time_steps:
        row = df_seg[df_seg["date"] == ts]
        step_values.append({r.entity: float(r.value) for r in row.itertuples()})
        
    prev_ranks = {e: r for r, e in enumerate(sorted(step_values[0].keys(), key=lambda e: step_values[0][e], reverse=True))}
    
    frame_number = 36 # skip intro
    
    for step_idx in range(len(time_steps) - 1):
        prev_vals = step_values[step_idx]
        next_vals = step_values[step_idx + 1]
        all_shown = set(prev_vals.keys()) | set(next_vals.keys())
        
        # we know frames_per_step in test is 14
        frames_per_step = 14
        
        for interp_frame in range(frames_per_step):
            alpha = interp_frame / frames_per_step
            interp_vals = {}
            for entity in all_shown:
                v0 = prev_vals.get(entity, 0.0)
                v1 = next_vals.get(entity, 0.0)
                interp_vals[entity] = v0 + (v1 - v0) * alpha
                
            sorted_ents = sorted(interp_vals.keys(), key=lambda e: interp_vals[e], reverse=True)
            current_top10 = sorted_ents[:10]
            
            for rank, entity in enumerate(current_top10):
                prev_slot_rank = prev_ranks.get(entity, 10)
                if prev_slot_rank == 10:
                    print(f"Frame {frame_number}: {entity} is entering! (alpha={alpha:.2f}, target rank={rank})")
            
            frame_number += 1
            
        prev_ranks = {e: r for r, e in enumerate(
            sorted(next_vals.keys(), key=lambda k: next_vals[k], reverse=True)
        )}

if __name__ == "__main__":
    find_entering()
