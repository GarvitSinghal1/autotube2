import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.modules.renderer_short import render_short

def generate_mock_data():
    print("[*] Generating mock monthly data...")
    np.random.seed(42)
    dates = pd.date_range("2010-01-01", "2020-01-01", freq="MS")
    entities = [
        "Apple", "Microsoft", "Google", "Amazon", "Meta", 
        "Tesla", "Nvidia", "Netflix", "Intel", "AMD",
        "Adobe", "Oracle", "Salesforce", "Cisco", "IBM"
    ]
    
    rows = []
    # Seed values
    current_vals = {e: float(np.random.randint(100, 500)) for e in entities}
    
    # Introduce some dramatic rank changes: Meta grows extremely fast mid-way
    for dt in dates:
        year = dt.year
        for e in entities:
            # Random walk
            change = np.random.normal(5, 15)
            if e == "Meta" and year >= 2014 and year <= 2016:
                change += 25  # Force Meta to climb rapidly
            if e == "Tesla" and year >= 2017:
                change += 30  # Force Tesla to climb rapidly
            
            current_vals[e] = max(10.0, current_vals[e] + change)
            rows.append({
                "date": dt,
                "entity": e,
                "value": current_vals[e]
            })
            
    return pd.DataFrame(rows)

def run_test():
    df_data = generate_mock_data()
    
    topic_info = {
        "topic": "Tech Giant Valuations Over Time",
        "description": "A race of the top 10 technology companies by valuation.",
        "source": "Market Cap Data",
        "short_unit": "B",
        "full_unit": "Billion USD",
    }
    
    extreme_segment = {
        "start_year": 2013,
        "end_year": 2018,
        "reason": "Tesla and Meta experience exponential growth, climbing to the top.",
        "hook": "In just 5 years, one tech giant rose from last place to challenge Apple..."
    }
    
    print("[*] Running render_short...")
    output_path, colors = render_short(
        df_data,
        "bar_chart_race",
        topic_info,
        extreme_segment
    )
    print(f"[+] Test complete! Video rendered to: {output_path}")

if __name__ == "__main__":
    run_test()
