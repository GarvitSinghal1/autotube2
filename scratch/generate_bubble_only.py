"""Generate only the bubble chart example video and thumbnail."""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import matplotlib
matplotlib.use("Agg")

from pipeline.modules.renderer_short import render_short
from pipeline.modules.thumbnail import generate_thumbnail
from pipeline.config import OUTPUT_DIR
from scratch.generate_examples import _make_bubble_chart_data

def main():
    print("Generating only bubble_chart...")
    df, topic_info, extreme_segment = _make_bubble_chart_data()
    video_path, colors = render_short(df, "bubble_chart", topic_info, extreme_segment)
    dest_video = OUTPUT_DIR / "example_bubble_chart.mp4"
    shutil.copy2(video_path, dest_video)
    print(f"✓ Video saved to: {dest_video}")

    thumb_path = generate_thumbnail(df, "bubble_chart", topic_info, extreme_segment, colors)
    dest_thumb = OUTPUT_DIR / "example_bubble_chart.png"
    shutil.copy2(thumb_path, dest_thumb)
    print(f"✓ Thumbnail saved to: {dest_thumb}")

if __name__ == "__main__":
    main()
