import sqlite3
import json
from pathlib import Path

DB_PATH = Path("pipeline/datasets_index.db")

def recover_string(s: str) -> str:
    if not s:
        return s
    try:
        # standard recipe to fix UTF-8 string that was read/decoded as ISO-8859-1 (Latin-1)
        return s.encode("latin-1").decode("utf-8")
    except Exception:
        # If it can't be decoded (e.g. it was already clean UTF-8), return as-is
        return s

def migrate():
    if not DB_PATH.exists():
        print(f"Database {DB_PATH} not found.")
        return

    print("Connecting to database...")
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, columns, entity_col, date_col, value_col, error_reason FROM datasets")
    rows = cursor.fetchall()
    
    print(f"Processing {len(rows)} records...")
    updated_count = 0

    for row in rows:
        row_id, name, columns, entity_col, date_col, value_col, error_reason = row
        
        # Recover each string field
        name_new = recover_string(name)
        entity_col_new = recover_string(entity_col)
        date_col_new = recover_string(date_col)
        value_col_new = recover_string(value_col)
        error_reason_new = recover_string(error_reason)
        
        columns_new = columns
        if columns:
            try:
                cols_list = json.loads(columns)
                cols_list_new = [recover_string(c) for c in cols_list]
                columns_new = json.dumps(cols_list_new)
            except Exception as e:
                print(f"Failed to parse columns JSON for ID {row_id}: {e}")

        # Check if anything actually changed
        if (name_new != name or 
            columns_new != columns or 
            entity_col_new != entity_col or 
            date_col_new != date_col or 
            value_col_new != value_col or 
            error_reason_new != error_reason):
            
            cursor.execute(
                """
                UPDATE datasets
                SET name = ?, columns = ?, entity_col = ?, date_col = ?, value_col = ?, error_reason = ?
                WHERE id = ?
                """,
                (name_new, columns_new, entity_col_new, date_col_new, value_col_new, error_reason_new, row_id)
            )
            updated_count += 1
            if updated_count <= 5 or value_col != value_col_new:
                print(f"Fixed ID {row_id}: '{name}' -> '{name_new}'")
                print(f"  value_col: '{value_col}' -> '{value_col_new}'")

    conn.commit()
    conn.close()
    print(f"Migration completed. Updated {updated_count} records.")

if __name__ == "__main__":
    migrate()
