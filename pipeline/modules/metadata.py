"""
metadata.py — Generates YouTube metadata (title, description, tags) for both videos.

Uses Gemini to create compelling, SEO-optimized metadata.
"""

import json
import re

from google import genai
from google.genai import types

from pipeline.config import GEMINI_API_KEY, GEMINI_MODEL


def generate_metadata(topic_info: dict, extreme_segment: dict) -> dict:
    """Generate metadata for both long-form and Short videos.

    Args:
        topic_info: Dict with topic, description, source, url.
        extreme_segment: Dict with start_year, end_year, reason, hook.

    Returns:
        Dict with keys 'long_form' and 'short', each containing
        title, description, and tags.

    Raises:
        RuntimeError: If Gemini fails to generate valid metadata.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""Generate YouTube video metadata for TWO videos about the same dataset.

Topic: {topic_info.get('topic', '')}
Description: {topic_info.get('description', '')}
Data source: {topic_info.get('source', '')}
Full date range: from the dataset
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

    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            result = json.loads(raw)
            break  # Success!
        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Invalid JSON from Gemini after {max_retries} attempts.\nRaw: {raw}\nError: {e}") from e
            print(f"[metadata] Invalid JSON: {e}. Retrying (Attempt {attempt+1}/{max_retries})...")
            time.sleep(5)
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Gemini metadata generation failed after {max_retries} attempts: {e}") from e
            print(f"[metadata] API error or rate limit: {e}. Waiting 30s (Attempt {attempt+1}/{max_retries})...")
            time.sleep(30)

    # Validate structure
    for key in ("long_form", "short"):
        if key not in result:
            raise RuntimeError(f"Missing '{key}' in metadata response.")
        for field in ("title", "description", "tags"):
            if field not in result[key]:
                raise RuntimeError(f"Missing '{field}' in metadata['{key}'].")

    # Ensure Short title ends with #Shorts
    if not result["short"]["title"].rstrip().endswith("#Shorts"):
        result["short"]["title"] = result["short"]["title"].rstrip() + " #Shorts"

    print(f"[metadata] Long title: {result['long_form']['title']}")
    print(f"[metadata] Short title: {result['short']['title']}")

    return result
