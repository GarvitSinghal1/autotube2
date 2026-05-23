import requests
import pandas as pd
import io

urls = [
    "https://ourworldindata.org/grapher/nuclear-warhead-stockpiles.csv",
    "https://ourworldindata.org/grapher/airliner-hijackings-and-fatalities-from-them.csv"
]

for url in urls:
    print(f"\nFetching {url}...")
    try:
        resp = requests.get(url, timeout=10, verify=False)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text))
            print(f"Success! Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print("First 3 rows:")
            print(df.head(3))
            
            # Check year range
            year_col = None
            for c in df.columns:
                if c.lower() in ("year", "date"):
                    year_col = c
                    break
            if year_col:
                print(f"Years: {df[year_col].min()} to {df[year_col].max()}")
        else:
            print(f"Failed with status: {resp.status_code}")
    except Exception as e:
        print(f"Error: {e}")
