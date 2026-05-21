import requests
import pandas as pd
import io

url = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/20th%20century%20deaths%20in%20US%20-%20CDC/20th%20century%20deaths%20in%20US%20-%20CDC.csv"
try:
    print("Fetching first 2KB with Range header...")
    resp = requests.get(url, headers={"Range": "bytes=0-2047", "User-Agent": "AutoTube2-Pipeline/1.0"}, timeout=10)
    print("Status Code:", resp.status_code)
    print("Content length received:", len(resp.content))
    print("Headers:", dict(resp.headers))
    
    # Try reading the first few lines
    text = resp.text
    print("\nText preview:\n", text[:300])
    
    # Parse CSV of this partial text (ignoring the last truncated line if needed)
    lines = text.splitlines()
    if len(lines) > 2:
        # exclude last line as it might be truncated
        csv_data = "\n".join(lines[:-1])
        df = pd.read_csv(io.StringIO(csv_data))
        print("\nParsed columns:", list(df.columns))
        print("First row:\n", df.head(1))
except Exception as e:
    print("Failed:", e)
