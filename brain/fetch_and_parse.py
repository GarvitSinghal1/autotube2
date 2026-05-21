import requests
import json

url = "https://api.github.com/repos/owid/owid-datasets/git/trees/master?recursive=1"
try:
    print("Fetching from GitHub Git Trees API...")
    resp = requests.get(url, timeout=15, headers={"User-Agent": "AutoTube2-Pipeline/1.0"})
    print("Status Code:", resp.status_code)
    if resp.status_code == 200:
        data = resp.json()
        tree = data.get("tree", [])
        print("Total items in tree:", len(tree))
        csv_files = [item["path"] for item in tree if item["type"] == "blob" and item["path"].startswith("datasets/") and item["path"].endswith(".csv")]
        print("Total CSV files in datasets/ folder:", len(csv_files))
        print("Sample of first 10:")
        for path in csv_files[:10]:
            print("  ", path)
    else:
        print("Error response:", resp.text[:200])
except Exception as e:
    print("Request failed:", e)
