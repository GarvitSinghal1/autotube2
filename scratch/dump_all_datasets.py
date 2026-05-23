import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_INDEX_DB = PROJECT_ROOT / "pipeline" / "datasets_index.db"

def main():
    if not DATASETS_INDEX_DB.exists():
        print(f"Database at {DATASETS_INDEX_DB} does not exist.")
        return
        
    conn = sqlite3.connect(str(DATASETS_INDEX_DB))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM datasets WHERE is_valid = 1")
    rows = cursor.fetchall()
    conn.close()
    
    output_path = PROJECT_ROOT / "scratch" / "all_valid_datasets.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in sorted(r[0] for r in rows):
            f.write(f"{r}\n")
            
    print(f"Wrote {len(rows)} valid dataset names to {output_path}")

if __name__ == "__main__":
    main()
