import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from pipeline.modules.renderer_short import (
    _draw_bar_chart_frame,
    _render_line_chart,
    _render_bubble_chart,
    FONT_BOLD,
    OUTLINE,
)
from pipeline.modules.font_loader import overlay_watermark

def test_bar_chart():
    print("Testing bar chart race frame flag drawing...")
    fig, ax = plt.subplots(figsize=(10.8, 19.2), dpi=100)
    
    # 2 top entities (norm_val > 0.5) and 2 lower ones (norm_val <= 0.5)
    entities_data = [
        {"entity": "India", "value": 95.0, "y_pos": 9.0, "color": "#ff9933"},
        {"entity": "United States", "value": 85.0, "y_pos": 8.0, "color": "#0000ff"},
        {"entity": "China", "value": 40.0, "y_pos": 7.0, "color": "#ff0000"},
        {"entity": "Brazil", "value": 25.0, "y_pos": 6.0, "color": "#008000"},
        {"entity": "Unknown Entity", "value": 15.0, "y_pos": 5.0, "color": "#808080"},  # Should not have a flag
    ]
    
    _draw_bar_chart_frame(
        ax, fig, entities_data,
        title="Testing Bar Chart Flags",
        source="FlagCDN",
        date_label="2026",
        topic_info={"short_unit": "M", "full_unit": "Millions"},
        save=True
    )
    
    output_path = Path("tmp/test_bar_flags.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="#000000", bbox_inches=None)
    plt.close(fig)
    print(f"Bar chart test frame saved to {output_path}")

def test_line_chart():
    print("Testing line chart race flag drawing...")
    data = []
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    countries = ["India", "United States", "China"]
    for year in years:
        for country in countries:
            val = 0.0
            if country == "India":
                val = 10.0 + (year - 2020) * 5.0
            elif country == "United States":
                val = 25.0 + (year - 2020) * 2.0
            elif country == "China":
                val = 18.0 + (year - 2020) * 3.5
            data.append({"date": f"{year}-01-01", "entity": country, "value": val})
            
    df_seg = pd.DataFrame(data)
    topic_info = {"topic": "Line Chart Flags Test", "source": "FlagCDN", "short_unit": "B"}
    extreme_segment = {"start_year": 2020, "end_year": 2025, "hook": "Look at the flags!"}
    entity_colors = {"India": "#ff9933", "United States": "#0000ff", "China": "#ff0000"}
    
    frames_dir = Path("tmp/line_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_video = Path("tmp/test_line_flags.mp4")
    
    _render_line_chart(
        df_seg, topic_info, extreme_segment, entity_colors,
        frames_dir, output_video
    )
    
    # Copy the last generated frame as a verification image
    last_frame = sorted(frames_dir.glob("*.png"))[-1]
    verification_path = Path("tmp/test_line_flags.png")
    verification_path.write_bytes(last_frame.read_bytes())
    print(f"Line chart test frame saved to {verification_path}")

def test_bubble_chart():
    print("Testing bubble chart flags...")
    data = []
    years = [2020, 2021, 2022]
    # We want some large bubbles (radius >= 15) and medium bubbles (radius >= 8)
    countries = ["India", "United States", "China", "Brazil", "Germany"]
    for year in years:
        for country in countries:
            val = 10.0
            if country == "India":
                val = 90.0 + (year - 2020) * 10.0
            elif country == "United States":
                val = 70.0 + (year - 2020) * 5.0
            elif country == "China":
                val = 40.0 + (year - 2020) * 4.0
            elif country == "Brazil":
                val = 22.0 + (year - 2020) * 2.0
            elif country == "Germany":
                val = 12.0 + (year - 2020) * 1.0
            data.append({"date": f"{year}-01-01", "entity": country, "value": val})
            
    df_seg = pd.DataFrame(data)
    topic_info = {"topic": "Bubble Chart Flags Test", "source": "FlagCDN", "short_unit": "M"}
    extreme_segment = {"start_year": 2020, "end_year": 2022, "hook": "Bubble physics and flags"}
    entity_colors = {"India": "#ff9933", "United States": "#0000ff", "China": "#ff0000", "Brazil": "#008000", "Germany": "#ffff00"}
    
    frames_dir = Path("tmp/bubble_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_video = Path("tmp/test_bubble_flags.mp4")
    
    _render_bubble_chart(
        df_seg, topic_info, extreme_segment, entity_colors,
        frames_dir, output_video
    )
    
    # Copy the last generated frame as a verification image
    last_frame = sorted(frames_dir.glob("*.png"))[-1]
    verification_path = Path("tmp/test_bubble_flags.png")
    verification_path.write_bytes(last_frame.read_bytes())
    print(f"Bubble chart test frame saved to {verification_path}")

if __name__ == "__main__":
    test_bar_chart()
    test_line_chart()
    test_bubble_chart()
    print("All tests finished.")
