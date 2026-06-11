"""Regenerate only bubble_chart and map_animation examples."""

import matplotlib
matplotlib.use("Agg")
import sys
sys.path.insert(0, ".")

from scratch.generate_examples import _make_bubble_chart_data, _make_map_data
from pipeline.modules.renderer_short import render_short
from pipeline.modules.thumbnail import generate_thumbnail
from pipeline.config import OUTPUT_DIR
import shutil

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for ct, data_fn in [("bubble_chart", _make_bubble_chart_data), ("map_animation", _make_map_data)]:
    print(f"\n{'='*60}")
    print(f"--- Regenerating: {ct} ---")
    print(f"{'='*60}")
    try:
        df, topic_info, extreme_segment = data_fn()
        video_path, colors = render_short(df, ct, topic_info, extreme_segment)
        dest_video = OUTPUT_DIR / f"example_{ct}.mp4"
        shutil.copy2(video_path, dest_video)
        print(f"  ✓ Video: {dest_video}")

        thumb_path = generate_thumbnail(df, ct, topic_info, extreme_segment, colors)
        dest_thumb = OUTPUT_DIR / f"example_{ct}.png"
        shutil.copy2(thumb_path, dest_thumb)
        print(f"  ✓ Thumb: {dest_thumb}")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\nDone!")
