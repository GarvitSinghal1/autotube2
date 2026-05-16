"""
topic.py — Discovers a compelling topic by fetching real datasets and asking Gemini to choose.

Returns a dict with topic, description, source, URL, and format.
"""

import json
import re
import random
import urllib.parse
import requests
from google import genai
from google.genai import types

from pipeline.config import GEMINI_API_KEY, GEMINI_MODEL

SYSTEM_PROMPT = """\
You are an expert data journalist curating topics for YouTube data visualization videos.
You will be provided with a list of real dataset names from Our World in Data.
Your job is to select the single MOST fascinating, surprising, or dramatic dataset from the list.
Choose a topic that would make a great animated bar chart race, line chart race, or map animation.
Avoid boring or overdone topics like basic GDP, population, or CO2 emissions unless there is a very unique angle.

You MUST respond with ONLY a valid JSON object, no markdown, no explanation:
{
  "dataset_name": "the exact name you chose from the list provided",
  "topic": "a catchy, descriptive title for the YouTube video",
  "description": "one sentence explaining why this data is compelling to watch"
}
"""

_FALLBACK_OWID_FOLDERS = [
    "Child Mortality - Gapminder",
    "CO2 emissions - Global Carbon Project",
    "Economic growth - Maddison Project Database",
    "Energy consumption by source - BP",
    "Life expectancy - WHO",
    "Military Expenditure - SIPRI",
    "Population - UN",
    "Urban population - UN",
    "Internet users - World Bank",
    "Homicide rates - UNODC",
    "Nuclear weapons - FAS",
    "Renewable Energy - BP",
    "Space exploration - NASA",
]

def _get_owid_dataset_list() -> list[str]:
    """Fetch the list of dataset directories from Our World in Data's GitHub."""
    url = "https://api.github.com/repos/owid/owid-datasets/contents/datasets"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return [item["name"] for item in data if item["type"] == "dir"]
    except Exception as e:
        print(f"[topic] Failed to fetch OWID repo (offline?): {e}")
    
    return _FALLBACK_OWID_FOLDERS


def discover_topic() -> dict:
    """Use Gemini to select a compelling topic from a real list of datasets.

    Returns:
        dict with keys: topic, description, source, url, format
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    # 1. Get real dataset options
    all_datasets = _get_owid_dataset_list()
    sample_size = min(25, len(all_datasets))
    sample_names = random.sample(all_datasets, sample_size)
    
    user_prompt = "Here are the available datasets. Pick the most fascinating one:\n\n" + "\n".join(sample_names)

    # 2. Ask Gemini to choose
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=1.0,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                ),
            )
            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)

            result = json.loads(raw_text)
            break  # Success!
        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Gemini returned invalid JSON after {max_retries} attempts.\nRaw: {raw_text}\nError: {e}") from e
            print(f"[topic] Invalid JSON: {e}. Retrying (Attempt {attempt+1}/{max_retries})...")
            time.sleep(5)
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Gemini API call failed after {max_retries} attempts: {e}") from e
            print(f"[topic] API error or rate limit: {e}. Waiting 30s (Attempt {attempt+1}/{max_retries})...")
            time.sleep(30)

    chosen_name = result.get("dataset_name")
    if not chosen_name or chosen_name not in sample_names:
        # Fallback if Gemini hallucinates a name not in the list
        chosen_name = sample_names[0]
        print(f"[topic] Gemini selected invalid dataset, falling back to: {chosen_name}")

    # 3. Construct deterministic URL based on OWID repo conventions
    # Structure: master/datasets/Folder Name/Folder Name.csv
    encoded_name = urllib.parse.quote(chosen_name)
    url = f"https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/{encoded_name}/{encoded_name}.csv"

    final_result = {
        "topic": result.get("topic", chosen_name),
        "description": result.get("description", "A fascinating dataset from Our World in Data."),
        "source": "Our World in Data",
        "url": url,
        "format": "csv"
    }

    print(f"[topic] Selected Dataset: {chosen_name}")
    print(f"[topic] Topic Title: {final_result['topic']}")
    print(f"[topic] URL: {final_result['url']}")

    return final_result
