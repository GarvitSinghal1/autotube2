# Spec: Combine Gemini Calls in Cleaner

Unified single Gemini call optimization for data cleaning to conserve the 20 requests/day API limit.

## Proposed Changes

### [cleaner.py](file:///Users/garvitsinghal/Library/CloudStorage/GoogleDrive-progarvit000@gmail.com/Other%20computers/My%20Computer/Codes/Robotics/CurrentlyWorking/autotube2/pipeline/modules/cleaner.py)

#### 1. Candidate Entity Extraction (Python)
Extract candidate entities from all string/object columns before calling Gemini. This is done by checking columns that:
- Are of `object` type (or string type).
- Are not the auto-detected date column.
- Are not completely numeric.

#### 2. Unified Gemini Helper
Create `_gemini_unified_cleaner_call` to replace `_gemini_detect_columns`, `_clean_all_entity_names_with_gemini`, and `_gemini_detect_unit`.
- Prompt requests:
  - Column detection (if not resolved).
  - Unit detection (if not resolved).
  - Entity mapping (always requested).
- Response JSON Schema:
  ```json
  {
    "columns": { "date_col": "...", "entity_col": "...", "value_col": "..." } | null,
    "units": { "full_unit": "...", "short_unit": "..." } | null,
    "entity_mapping": { "raw": "clean", ... }
  }
  ```

#### 3. Update `clean_dataframe`
- Check columns and units.
- Run `_gemini_unified_cleaner_call` once.
- Parse JSON to get resolved columns, units, and entity mappings.
- Clean and structure the DataFrame.

## Verification Plan

- Run `PYTHONPATH=. venv/bin/python scratch/test_short_render.py` to verify mock data cleaning still behaves correctly.
- Check generated output structures.
