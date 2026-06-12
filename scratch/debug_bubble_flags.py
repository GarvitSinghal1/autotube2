import pandas as pd
from pathlib import Path
from pipeline.modules.renderer_short import _render_bubble_chart, _add_flag_to_axes
import matplotlib.pyplot as plt

# Test if _add_flag_to_axes works directly
fig, ax = plt.subplots()
val = _add_flag_to_axes(ax, "India", 0.5, 0.5)
print(f"Direct call to _add_flag_to_axes for India returned: {val}")

# Run bubble chart rendering with printing
data = [
    {"date": "2020-01-01", "entity": "India", "value": 90.0},
    {"date": "2020-01-01", "entity": "United States", "value": 70.0},
    {"date": "2020-01-01", "entity": "Germany", "value": 12.0},
    {"date": "2021-01-01", "entity": "India", "value": 100.0},
    {"date": "2021-01-01", "entity": "United States", "value": 75.0},
    {"date": "2021-01-01", "entity": "Germany", "value": 13.0},
]
df_seg = pd.DataFrame(data)
topic_info = {"topic": "Bubble Debug", "source": "Debug", "short_unit": "M"}
extreme_segment = {"start_year": 2020, "end_year": 2021, "hook": "Debug"}
entity_colors = {"India": "#ff9933", "United States": "#0000ff", "Germany": "#ffff00"}

frames_dir = Path("tmp/debug_bubble_frames")
frames_dir.mkdir(parents=True, exist_ok=True)
output_video = Path("tmp/debug_bubble.mp4")

print("Rendering debug bubble chart...")
_render_bubble_chart(df_seg, topic_info, extreme_segment, entity_colors, frames_dir, output_video)
print("Debug bubble chart rendering finished.")
