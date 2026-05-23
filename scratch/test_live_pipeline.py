import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.modules.topic import discover_topic
from pipeline.modules.fetcher import fetch_dataset
from pipeline.modules.cleaner import clean_dataframe
from pipeline.modules.analyzer import find_extreme_segment
from pipeline.modules.chart_selector import select_chart_type
from pipeline.modules.renderer_short import render_short
from pipeline.modules.metadata import generate_metadata

def run_test():
    print("=" * 60)
    print("RUNNING END-TO-END PIPELINE TEST WITH LIVE DATASETS")
    print("=" * 60)
    
    # 1. Topic Discovery
    print("\n[*] Discovering topic...")
    topic_info = discover_topic()
    print(f"[+] Chosen Dataset: {topic_info['dataset_name']}")
    print(f"[+] Topic Title: {topic_info['topic']}")
    print(f"[+] CSV URL: {topic_info['url']}")
    
    # 2. Fetch Dataset
    print("\n[*] Fetching dataset...")
    df_raw = fetch_dataset(topic_info["url"], topic_info["format"])
    
    # 3. Clean Data
    print("\n[*] Cleaning dataset...")
    df_monthly, df_yearly = clean_dataframe(df_raw, topic_info)
    topic_info["start_year"] = int(df_yearly["date"].dt.year.min())
    topic_info["end_year"] = int(df_yearly["date"].dt.year.max())
    print(f"[+] Cleaned data range: {topic_info['start_year']} to {topic_info['end_year']}")
    
    # 4. Find Extreme Segment
    print("\n[*] Finding extreme segment...")
    extreme_segment = find_extreme_segment(df_yearly)
    print(f"[+] Segment range: {extreme_segment['start_year']} to {extreme_segment['end_year']}")
    print(f"[+] Hook: {extreme_segment['hook']}")
    
    # 5. Select Chart Type
    print("\n[*] Selecting chart type...")
    chart_type = select_chart_type(df_yearly, topic_info)
    print(f"[+] Chart type: {chart_type}")
    
    # 6. Render Short
    print("\n[*] Rendering Short video...")
    short_path, entity_colors = render_short(
        df_monthly, chart_type, topic_info, extreme_segment
    )
    print(f"[+] Short video rendered to: {short_path}")
    
    # 7. Generate Metadata
    print("\n[*] Generating video metadata...")
    video_metadata = generate_metadata(topic_info, extreme_segment)
    print("[+] Title:")
    print(f"    Short: {video_metadata['short']['title']}")
    print("[+] Description:")
    print(f"    Short: {video_metadata['short']['description']}")
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_test()
