import json
import re

content_file = "/Users/garvitsinghal/.gemini/antigravity-ide/brain/7cd57549-4ec2-4e18-b90e-8ff056bd888c/.system_generated/steps/89/content.md"

with open(content_file, "r") as f:
    lines = f.readlines()

# Line 9 contains the JSON data
json_line = None
for line in lines:
    if line.startswith('{"sha":'):
        json_line = line
        break

if not json_line:
    print("Could not find the JSON line in content.md")
else:
    data = json.loads(json_line)
    tree = data.get("tree", [])
    print("Total items in tree:", len(tree))
    csv_items = [
        item for item in tree 
        if item["type"] == "blob" 
        and item["path"].startswith("datasets/") 
        and item["path"].endswith(".csv")
    ]
    print("Total CSV files in datasets/ folder:", len(csv_items))
    print("\nFirst 15 CSV files:")
    for item in csv_items[:15]:
        print(f"Path: {item['path']} (Size: {item.get('size', 0)} bytes)")
