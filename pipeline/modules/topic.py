"""
topic.py — Uses Gemini to discover a compelling topic and locate a free public dataset.

Returns a dict with topic, description, source, URL, and format.
"""

import json
import re
import google.generativeai as genai

from pipeline.config import GEMINI_API_KEY, GEMINI_MODEL

SYSTEM_PROMPT = """\
You are a data journalist who finds compelling, publicly available datasets for YouTube videos. You only select datasets that are:
- Free and directly downloadable (CSV, JSON, or accessible via a public API with no auth key)
- Contain time-series data spanning at least 10 years
- Cover an inherently interesting topic that general audiences would find surprising, dramatic, or emotionally engaging
- Come from reputable sources: World Bank, Our World in Data, UN, Wikipedia, US Census, NOAA, NASA, IMF, or similar

Topics can be anything: economics, population, military, sports records, technology adoption, climate, disease, trade, energy, crime, culture. Fully open.

You always return the actual direct download URL, not a landing page.

IMPORTANT: Prefer datasets from these sources and URL patterns that are known to work:
- Our World in Data: https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/<dataset-name>/<dataset-name>.csv
- Our World in Data (new): https://catalog.ourworldindata.org/garden/... .csv
- World Bank API: https://api.worldbank.org/v2/country/all/indicator/<INDICATOR>?format=json&per_page=10000&date=1960:2023
- GitHub raw CSV files from reputable data repos

AVOID:
- URLs that require clicking through a web UI to download
- ZIP files
- World Bank CSV bulk downloads (the zip format ones)
- URLs that redirect to HTML pages

You MUST respond with ONLY a valid JSON object, no markdown, no explanation:
{
  "topic": "descriptive topic name with date range",
  "description": "one sentence about why this is interesting",
  "source": "source organization name",
  "url": "direct download URL",
  "format": "csv or json or api"
}
"""

USER_PROMPT = """\
Find me a fascinating dataset for a data visualization YouTube video. Pick something that would make a great animated bar chart race, line chart race, or map animation. The data should have multiple entities (countries, companies, teams, etc.) competing or changing over time.

Choose a topic that has NOT been overdone on YouTube. Avoid GDP, population, and CO2 emissions — those have been done to death. Find something more niche and surprising.

Return ONLY the JSON object, nothing else.
"""


def discover_topic() -> dict:
    """Use Gemini to discover a compelling topic and dataset URL.

    Returns:
        dict with keys: topic, description, source, url, format

    Raises:
        RuntimeError: If Gemini fails to return valid JSON or the API call fails.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                USER_PROMPT,
                generation_config=genai.types.GenerationConfig(
                    temperature=1.0,  # high creativity for diverse topics
                    max_output_tokens=1024,
                ),
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Gemini API call failed after {max_retries} attempts: {e}") from e
            print(f"[topic] API error or rate limit: {e}. Waiting 30s (Attempt {attempt+1}/{max_retries})...")
            time.sleep(30)

    raw_text = response.text.strip()

    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Gemini returned invalid JSON.\nRaw response:\n{raw_text}\nError: {e}"
        ) from e

    required_keys = {"topic", "description", "source", "url", "format"}
    missing = required_keys - set(result.keys())
    if missing:
        raise RuntimeError(f"Gemini response missing keys: {missing}\nGot: {result}")

    if result["format"] not in ("csv", "json", "api"):
        raise RuntimeError(f"Unsupported format: {result['format']}")

    print(f"[topic] Discovered: {result['topic']}")
    print(f"[topic] Source: {result['source']}")
    print(f"[topic] URL: {result['url']}")

    return result
