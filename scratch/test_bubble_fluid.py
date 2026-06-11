import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.modules.renderer_short import render_short

def generate_mock_bubble_data():
    print("[*] Generating mock bubble data...")
    np.random.seed(42)
    dates = pd.date_range("2010-01-01", "2015-01-01", freq="MS")
    entities = ["Apple", "Microsoft", "Google", "Amazon", "Meta", "Tesla", "Nvidia"]
    
    rows = []
    current_vals = {e: float(np.random.randint(100, 500)) for e in entities}
    
    for dt in dates:
        for e in entities:
            change = np.random.normal(5, 15)
            current_vals[e] = max(10.0, current_vals[e] + change)
            rows.append({
                "date": dt,
                "entity": e,
                "value": current_vals[e]
            })
            
    return pd.DataFrame(rows)

def generate_mock_map_data():
    print("[*] Generating mock map data...")
    dates = pd.date_range("2010-01-01", "2012-01-01", freq="MS")
    entities = ["India", "Pakistan", "China", "United States", "Germany"]
    
    rows = []
    current_vals = {e: float(np.random.randint(100, 500)) for e in entities}
    
    for dt in dates:
        for e in entities:
            change = np.random.normal(5, 10)
            current_vals[e] = max(10.0, current_vals[e] + change)
            rows.append({
                "date": dt,
                "entity": e,
                "value": current_vals[e]
            })
            
    return pd.DataFrame(rows)

def test_bubble():
    df_data = generate_mock_bubble_data()
    topic_info = {
        "topic": "Bubble Tech Giant Valuations",
        "description": "Fluid bubble chart race of tech companies.",
        "source": "Market Cap Data",
        "short_unit": "B",
        "full_unit": "Billion USD",
    }
    extreme_segment = {
        "start_year": 2011,
        "end_year": 2014,
        "reason": "Dramatic valuations shift",
        "hook": "Who took the crown?"
    }
    print("[*] Rendering bubble chart short...")
    output_path, colors = render_short(
        df_data,
        "bubble_chart",
        topic_info,
        extreme_segment
    )
    import shutil
    dest = output_path.parent / "test_bubble.mp4"
    shutil.copy2(output_path, dest)
    print(f"[+] Bubble chart rendering successful: {dest}")

def test_map():
    df_data = generate_mock_map_data()
    topic_info = {
        "topic": "Regional Economic Metrics",
        "description": "Geographic map visualization.",
        "source": "Global Economic Survey",
        "short_unit": "B",
        "full_unit": "Billion USD",
    }
    extreme_segment = {
        "start_year": 2010,
        "end_year": 2012,
        "reason": "Regional shift",
        "hook": "Look at J&K representation!"
    }
    print("[*] Rendering map animation short...")
    output_path, colors = render_short(
        df_data,
        "map_animation",
        topic_info,
        extreme_segment
    )
    import shutil
    dest = output_path.parent / "test_map.mp4"
    shutil.copy2(output_path, dest)
    print(f"[+] Map animation rendering successful: {dest}")

if __name__ == "__main__":
    test_bubble()
    test_map()
