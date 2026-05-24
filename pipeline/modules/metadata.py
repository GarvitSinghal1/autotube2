"""
metadata.py — Generates YouTube metadata (title, description, tags) for both videos.

Uses Gemini to create compelling, SEO-optimized metadata.
"""

import json
import re
import time

from google import genai
from google.genai import types

from pipeline.config import GEMINI_API_KEY, GEMINI_MODEL


def generate_metadata(topic_info: dict, extreme_segment: dict) -> dict:
    """Generate metadata for both long-form and Short videos."""
    # Check if metadata was pre-generated during extreme segment analysis to save API calls
    if "metadata" in extreme_segment and extreme_segment["metadata"]:
        print("[metadata] Using pre-calculated metadata from extreme segment analysis.")
        return extreme_segment["metadata"]

    from pipeline.modules.gemini_helper import build_gemini_client
    client = build_gemini_client()

    prompt = f"""Generate YouTube video metadata for TWO videos about the same dataset.

Topic: {topic_info.get('topic', '')}
Description: {topic_info.get('description', '')}
Data source: {topic_info.get('source', '')}
Full date range: {topic_info.get('start_year', '')} to {topic_info.get('end_year', '')}
Extreme segment: {extreme_segment.get('start_year', '')} to {extreme_segment.get('end_year', '')}
Extreme reason: {extreme_segment.get('reason', '')}
Hook: {extreme_segment.get('hook', '')}

VIDEO 1 — LONG FORM (5-10 min, full dataset visualization):
- Title: informative, specific, includes date range. Example style: "How World GDP Changed From 1960 to 2023"
- Description: 3-4 sentences explaining what the data shows, why it matters. Include data source credit at the end.
- Tags: 10-15 relevant tags for SEO

VIDEO 2 — SHORT (60 sec, highlights the extreme segment):
- Title: punchy, hook-driven, highlights the most dramatic moment. Must end with " #Shorts". Example: "China Overtook Japan's Economy In Just 15 Years #Shorts"
- Description: 2 sentences max
- Tags: 10 relevant tags

Return ONLY valid JSON:
{{
  "long_form": {{
    "title": "...",
    "description": "...",
    "tags": ["tag1", "tag2", ...]
  }},
  "short": {{
    "title": "... #Shorts",
    "description": "...",
    "tags": ["tag1", "tag2", ...]
  }}
}}
"""

    from pipeline.modules.gemini_helper import generate_content_with_retry
    max_json_retries = 3
    result = None

    try:
        for attempt in range(max_json_retries):
            try:
                response = generate_content_with_retry(
                    client=client,
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                    ),
                )
                raw = response.text.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re.sub(r"\s*```$", "", raw)

                result = json.loads(raw)
                break  # Success!
            except json.JSONDecodeError as e:
                if attempt == max_json_retries - 1:
                    raise RuntimeError(f"Invalid JSON from Gemini after {max_json_retries} attempts.\nRaw: {raw}\nError: {e}") from e
                print(f"[metadata] Invalid JSON: {e}. Retrying JSON generation (Attempt {attempt+1}/{max_json_retries})...")
                time.sleep(5)
    except Exception as e:
        print(f"[metadata] Warning: Gemini metadata generation failed: {e}. Falling back to programmatic metadata.")
        topic_title = topic_info.get("topic", "Data Visualization")
        source = topic_info.get("source", "Open Numbers / OWID")
        start_year = extreme_segment.get("start_year", "")
        end_year = extreme_segment.get("end_year", "")
        
        # Clean up topic title for tags
        clean_title = re.sub(r'[^\w\s]', '', topic_title)
        tags = [t.lower() for t in clean_title.split() if len(t) > 3][:12]
        if "data" not in tags:
            tags.append("data")
        if "visualization" not in tags:
            tags.append("visualization")
            
        result = {
            "long_form": {
                "title": f"How {topic_title} Changed Over Time",
                "description": f"A comprehensive data visualization tracking {topic_title}.\n\nData source: {source}",
                "tags": tags + ["chart race", "bar chart race", "statistics"]
            },
            "short": {
                "title": f"The Dramatic Shift in {topic_title} ({start_year}-{end_year}) #Shorts",
                "description": f"Highlighting the most extreme changes in {topic_title} from {start_year} to {end_year}.",
                "tags": tags + ["shorts", "trending", "history"]
            }
        }

    # Validate structure (in case the generated JSON was partially valid but missing keys)
    for key in ("long_form", "short"):
        if key not in result:
            result[key] = {}
        for field in ("title", "description", "tags"):
            if field not in result[key]:
                if field == "title":
                    result[key]["title"] = f"{topic_info.get('topic', 'Data Visualization')} #Shorts" if key == "short" else f"{topic_info.get('topic', 'Data Visualization')}"
                elif field == "description":
                    result[key]["description"] = f"Data tracking {topic_info.get('topic', 'Data Visualization')}"
                elif field == "tags":
                    result[key]["tags"] = ["data", "visualization"]

    # Ensure Short title ends with #Shorts
    if not result["short"]["title"].rstrip().endswith("#Shorts"):
        result["short"]["title"] = result["short"]["title"].rstrip() + " #Shorts"

    print(f"[metadata] Long title: {result['long_form']['title']}")
    print(f"[metadata] Short title: {result['short']['title']}")

    return result
