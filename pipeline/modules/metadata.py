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
    from pipeline.modules.gemini_helper import clean_banned_words
    result = None
    # Check if metadata was pre-generated during extreme segment analysis to save API calls
    if "metadata" in extreme_segment and extreme_segment["metadata"]:
        print("[metadata] Using pre-calculated metadata from extreme segment analysis.")
        result = extreme_segment["metadata"]

    if not result:
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
- CRITICAL: BANNED WORDS. Do NOT use the words "Witness", "Explode", "Surge" (or their variations, such as "Witnessed", "Exploded", "Explosion", "Surged", "Surging", etc.) anywhere in the title. Choose alternative action/drama verbs and nouns (e.g., Rise, Growth, Climb, Boom, Leap, Battle, Dominance).

VIDEO 2 — SHORT (60 sec, highlights the extreme segment):
- Title: punchy, hook-driven, highlights the most dramatic moment. Must end with " #Shorts". Example: "China Overtook Japan's Economy In Just 15 Years #Shorts"
- CRITICAL: Keep the final winner or outcome UNCLEAR in the title to create suspense and keep viewers watching (e.g. Bad: "USA Dominates GDP #Shorts", Good: "The Battle for Global GDP Dominance #Shorts" or "Who Overtook Japan's Economy? #Shorts").
- CRITICAL: Focus on "emotion + data" (rivalry, rise/fall, nostalgia, or controversy) and highlight extreme growth/decline without spoiling who wins in the end.
- CRITICAL: BANNED WORDS. Do NOT use the words "Witness", "Explode", "Surge" (or their variations, such as "Witnessed", "Exploded", "Explosion", "Surged", "Surging", etc.) anywhere in the title or hooks. Choose alternative action/drama verbs and nouns (e.g., Rise, Growth, Climb, Boom, Leap, Battle, Dominance).
- CRITICAL: VARY TITLE STRUCTURE. Do NOT start titles with predictable, repetitive formulas (e.g. "The Rise of...", "The Rise and Fall of...", "Witness the..."). Force variety by using questions, comparisons, milestones, or action verbs.
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

                    parsed = json.loads(raw)
                    
                    # Validate titles against banned words
                    long_title = parsed.get("long_form", {}).get("title", "")
                    short_title = parsed.get("short", {}).get("title", "")
                    
                    def contains_banned(t: str) -> bool:
                        t_lower = t.lower()
                        return any(w in t_lower for w in ["witness", "explod", "explos", "surg"])
                        
                    if contains_banned(long_title) or contains_banned(short_title):
                        print(f"[metadata] Attempt {attempt+1}: Generated title contains banned words. Retrying...")
                        if attempt == max_json_retries - 1:
                            print("[metadata] Last attempt failed validation. Programmatically cleaning titles.")
                            result = parsed
                            break
                        continue
                    result = parsed
                    break  # Success!
                except json.JSONDecodeError as e:
                    if attempt == max_json_retries - 1:
                        raise RuntimeError(f"Invalid JSON from Gemini after {max_json_retries} attempts.\nRaw: {raw}\nError: {e}") from e
                    print(f"[metadata] Invalid JSON: {e}. Retrying JSON generation (Attempt {attempt+1}/{max_json_retries})...")
                    time.sleep(5)
        except Exception as e:
            print(f"[metadata] Warning: Gemini metadata generation failed: {e}. Falling back to programmatic metadata.")
            topic_title = topic_info.get("topic") or "Data Visualization"
            source = topic_info.get("source") or "Open Numbers / OWID"
            start_year = extreme_segment.get("start_year") or ""
            end_year = extreme_segment.get("end_year") or ""
            
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

    # Validate structure (in case the generated JSON was partially valid but missing keys/values)
    fallback_topic = clean_banned_words(topic_info.get("topic") or "Data Visualization")
    fallback_start = extreme_segment.get("start_year") or ""
    fallback_end = extreme_segment.get("end_year") or ""

    for key in ("long_form", "short"):
        if key not in result or not isinstance(result[key], dict):
            result[key] = {}
        for field in ("title", "description", "tags"):
            val = result[key].get(field)
            is_valid = True
            if field == "tags":
                if not val or not isinstance(val, list):
                    is_valid = False
            else:
                if not val or not isinstance(val, str) or not val.strip():
                    is_valid = False
                elif key == "short" and field == "title" and not val.replace("#Shorts", "").replace("#shorts", "").strip():
                    # Title is empty except for #Shorts tag
                    is_valid = False

            if not is_valid:
                if field == "title":
                    if key == "short":
                        result[key]["title"] = f"The Dramatic Shift in {fallback_topic} ({fallback_start}-{fallback_end}) #Shorts"
                    else:
                        result[key]["title"] = f"How {fallback_topic} Changed Over Time"
                elif field == "description":
                    result[key]["description"] = f"Data visualization tracking {fallback_topic}."
                elif field == "tags":
                    result[key]["tags"] = ["data", "visualization", "history", "statistics"]
            else:
                # Clean up whitespace and programmatically clean banned words for titles
                if field != "tags":
                    val_clean = val.strip()
                    if field == "title":
                        val_clean = clean_banned_words(val_clean)
                    result[key][field] = val_clean

        # Enforce YouTube title limit of 100 characters
        title_val = result[key]["title"]
        if len(title_val) > 100:
            if key == "short":
                title_clean = title_val.replace(" #Shorts", "").replace("#Shorts", "").strip()
                result[key]["title"] = title_clean[:88] + " #Shorts"
            else:
                result[key]["title"] = title_val[:97] + "..."

    # Ensure Short title ends with #Shorts
    if not result["short"]["title"].endswith("#Shorts"):
        result["short"]["title"] = result["short"]["title"] + " #Shorts"

    print(f"[metadata] Long title: {result['long_form']['title']}")
    print(f"[metadata] Short title: {result['short']['title']}")

    return result
