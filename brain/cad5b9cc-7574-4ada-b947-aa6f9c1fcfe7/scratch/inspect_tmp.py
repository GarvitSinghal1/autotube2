import os
import glob

print("Files in /tmp:")
for item in os.listdir("/tmp"):
    if "dataviz" in item or "csv" in item or "owid" in item:
        print(item)

# Search for any downloaded CSV files in /tmp or subdirectories
csv_files = glob.glob("/tmp/**/*.csv", recursive=True)
print("CSV files in /tmp:", csv_files)
