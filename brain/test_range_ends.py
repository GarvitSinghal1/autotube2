import requests
import pandas as pd
import io
import re

url = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/20th%20century%20deaths%20in%20US%20-%20CDC/20th%20century%20deaths%20in%20US%20-%20CDC.csv"
file_size = 32127

CHUNK_SIZE = 16384

try:
    if file_size <= CHUNK_SIZE:
        print(f"File size ({file_size}) is smaller than chunk size ({CHUNK_SIZE}). Fetching entire file...")
        resp = requests.get(url, headers={"User-Agent": "AutoTube2-Pipeline/1.0"}, timeout=10)
        start_text = resp.text
        end_text = resp.text
    else:
        print(f"Fetching start (0-{CHUNK_SIZE-1})...")
        resp_start = requests.get(url, headers={"Range": f"bytes=0-{CHUNK_SIZE-1}", "User-Agent": "AutoTube2-Pipeline/1.0"}, timeout=10)
        print("Start status:", resp_start.status_code)
        
        last_chunk_start = max(0, file_size - CHUNK_SIZE)
        print(f"Fetching end ({last_chunk_start}-{file_size-1})...")
        resp_end = requests.get(url, headers={"Range": f"bytes={last_chunk_start}-{file_size-1}", "User-Agent": "AutoTube2-Pipeline/1.0"}, timeout=10)
        print("End status:", resp_end.status_code)
        
        start_text = resp_start.text
        end_text = resp_end.text
    
    start_lines = start_text.splitlines()
    print("Start lines count:", len(start_lines))
    
    if len(start_lines) > 2:
        # Exclude the last line in case it is truncated
        start_csv = "\n".join(start_lines[:-1])
        df_start = pd.read_csv(io.StringIO(start_csv))
        columns = list(df_start.columns)
        print("Detected columns:", columns)
        
        date_col = None
        for col in columns:
            if col.lower() in ("year", "date", "time", "yr"):
                date_col = col
                break
        
        print("Detected date column:", date_col)
        if date_col:
            start_years = pd.to_numeric(df_start[date_col], errors="coerce").dropna().astype(int)
            if not start_years.empty:
                print("Start year found:", start_years.min())
            
            end_lines = end_text.splitlines()
            print("End lines count:", len(end_lines))
            if len(end_lines) > 1:
                # Header + all lines except the first (which is probably partial)
                end_csv = start_lines[0] + "\n" + "\n".join(end_lines[1:])
                try:
                    df_end = pd.read_csv(io.StringIO(end_csv))
                    end_years = pd.to_numeric(df_end[date_col], errors="coerce").dropna().astype(int)
                    if not end_years.empty:
                        print("End year found:", end_years.max())
                except Exception as parse_err:
                    print("Could not parse end chunk as CSV:", parse_err)
                    years = [int(y) for y in re.findall(r"\b(18\d{2}|19\d{2}|20\d{2})\b", end_text)]
                    if years:
                        print("Regex detected end year:", max(years))
except Exception as e:
    print("Failed:", e)

