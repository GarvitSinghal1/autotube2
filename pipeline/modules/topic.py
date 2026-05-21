"""
topic.py — Discovers a compelling topic by fetching real datasets and asking Gemini to choose.

Returns a dict with topic, description, source, URL, and format.
"""

import json
import re
import random
import urllib.parse
import requests
from typing import Optional
from google import genai
from google.genai import types

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from pipeline.config import GEMINI_API_KEY, GEMINI_MODEL

SYSTEM_PROMPT = """\
You are an expert data journalist curating topics for YouTube data visualization videos.
You will be provided with a list of real dataset names from Our World in Data.
Your job is to select the single MOST fascinating, surprising, or dramatic dataset from the list.
Choose a topic that would make a great animated bar chart race, line chart race, or map animation.

CRITICAL: Avoid dry, boring academic topics, niche development/clinical metrics (e.g. Guinea worm cases, basic agricultural yields, specific nutrient deficiencies, diarrhea rates). 
Instead, prioritize dramatic human stories, global conflicts, technological transitions, changes in lifestyles/addictions (food, alcohol, tech), existential threats (nuclear weapons, natural disasters), or major historical shifts.

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

# Keyword filters to guarantee interesting topics on YouTube
INTERESTING_KEYWORDS = [
    "military", "weapon", "nuclear", "war", "conflict", "battle", "defense", "armaments",
    "space", "nasa", "rocket", "exploration", "satellite",
    "internet", "mobile", "phone", "technology", "computer", "ai", "robot", "patent", "innovation",
    "homicide", "crime", "murder", "suicide", "terrorism", "disaster", "earthquake", "tsunami", "volcano",
    "diet", "food", "nutrition", "sugar", "meat", "alcohol", "beer", "wine", "smoking", "tobacco", "drug", "addiction",
    "democracy", "regime", "election", "political", "government", "freedom", "human rights",
    "deforestation", "forest", "extinction", "species", "whale", "animal", "wildlife",
    "energy", "oil", "coal", "gas", "solar", "wind", "renewable", "electricity",
    "billionaire", "wealth", "poverty", "inequality",
    "olympic", "medal", "sports", "leisure",
    "pandemic", "epidemic", "plague", "influenza", "covid", "death"
]

BORING_KEYWORDS = [
    "guinea", "worm", "diarrhea", "diarrheal", "deficiency", "malnutrition", "micronutrient",
    "tuberculosis", "malaria", "tetanus", "measles", "hepatitis", "meningitis", "encephalitis",
    "pertussis", "diphtheria", "leprosy", "trachoma", "onchocerciasis", "filariasis", "rabies",
    "dengue", "fever", "chagas", "leishmaniasis", "trypanosomiasis", "hookworm", "trichuriasis",
    "ascariasis", "nematode", "fluke", "sanitation", "hygiene", "wastewater", "treatment",
    "deworming", "iodized", "vitamin", "breastfeeding", "stunting", "wasting", "anaemia"
]

def _get_valid_datasets_from_db() -> list[dict]:
    """Load valid datasets from SQLite index database if available."""
    from pipeline.config import DATASETS_INDEX_DB
    import sqlite3
    
    if not DATASETS_INDEX_DB.exists():
        print(f"[topic] Database at {DATASETS_INDEX_DB} does not exist.")
        return []
        
    try:
        conn = sqlite3.connect(str(DATASETS_INDEX_DB))
        cursor = conn.cursor()
        # Query valid datasets
        cursor.execute("SELECT name, path, csv_url, entity_col, date_col, value_col, start_year, end_year, span_years, entity_count FROM datasets WHERE is_valid = 1")
        rows = cursor.fetchall()
        conn.close()
        
        datasets = []
        for r in rows:
            datasets.append({
                "name": r[0],
                "path": r[1],
                "csv_url": r[2],
                "entity_col": r[3],
                "date_col": r[4],
                "value_col": r[5],
                "start_year": r[6],
                "end_year": r[7],
                "span_years": r[8],
                "entity_count": r[9]
            })
        return datasets
    except Exception as e:
        print(f"[topic] Failed to query database: {e}")
        return []


def _get_owid_dataset_list() -> list[str]:
    """Fetch the list of dataset directories from Our World in Data's GitHub."""
    url = "https://api.github.com/repos/owid/owid-datasets/contents/datasets"
    try:
        resp = requests.get(url, timeout=10, verify=False)
        if resp.status_code == 200:
            data = resp.json()
            return [item["name"] for item in data if item["type"] == "dir"]
    except Exception as e:
        print(f"[topic] Failed to fetch OWID repo (offline?): {e}")
    
    return _FALLBACK_OWID_FOLDERS


def discover_topic(blacklist: Optional[set[str]] = None) -> dict:
    """Use Gemini to select a compelling topic from a real list of datasets.

    Returns:
        dict with keys: topic, description, source, url, format
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    # 1. Try to load from database
    db_datasets = _get_valid_datasets_from_db()
    
    if db_datasets:
        print(f"[topic] Loaded {len(db_datasets)} valid datasets from database.")
        all_datasets = db_datasets
        # Map dataset name to details
        dataset_map = {d["name"]: d for d in all_datasets}
        dataset_names = list(dataset_map.keys())
    else:
        print("[topic] Falling back to fetching dataset list from GitHub API / fallback list.")
        fallback_names = _get_owid_dataset_list()
        dataset_names = fallback_names
        dataset_map = {}

    if blacklist:
        dataset_names = [d for d in dataset_names if d not in blacklist]
        
    # Filter datasets to keep only potentially interesting ones
    interesting_names = []
    for d in dataset_names:
        name_lower = d.lower()
        has_interesting = any(w in name_lower for w in INTERESTING_KEYWORDS)
        has_boring = any(w in name_lower for w in BORING_KEYWORDS)
        if has_interesting and not has_boring:
            interesting_names.append(d)
            
    print(f"[topic] Filtered {len(dataset_names)} datasets down to {len(interesting_names)} interesting ones.")
    
    # Fallback to all datasets if filtering left too few
    if len(interesting_names) >= 10:
        candidate_names = interesting_names
    else:
        candidate_names = dataset_names

    sample_size = min(25, len(candidate_names))
    sample_names = random.sample(candidate_names, sample_size)
    
    user_prompt = "Here are the available datasets. Pick the most fascinating one:\n\n" + "\n".join(sample_names)

    # 2. Define the Pydantic schema for structured output with a dynamic Enum
    from pydantic import BaseModel
    from enum import Enum
    
    # Create the dynamic Enum of the sampled names
    DatasetEnum = Enum("DatasetEnum", {f"item_{i}": name for i, name in enumerate(sample_names)})
    
    class TopicSelection(BaseModel):
        dataset_name: DatasetEnum
        topic: str
        description: str

    # 3. Ask Gemini to choose
    from pipeline.modules.gemini_helper import generate_content_with_retry
    chosen_name = None
    topic_title = None
    topic_desc = None
    
    try:
        response = generate_content_with_retry(
            client=client,
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=TopicSelection,
                temperature=0.2,
            ),
        )
        raw_text = response.text.strip()
        result = json.loads(raw_text)
        
        chosen_name = result.get("dataset_name")
        # Extract name from dynamic Enum if returned as object or enum member
        if hasattr(chosen_name, "value"):
            chosen_name = chosen_name.value
            
        topic_title = result.get("topic")
        topic_desc = result.get("description")
    except Exception as e:
        print(f"[topic] Gemini selection failed: {e}. Falling back to default selection.")


    # Fallback if Gemini failed or selection is invalid
    if not chosen_name or chosen_name not in sample_names:
        chosen_name = sample_names[0]
        topic_title = chosen_name
        topic_desc = "A fascinating dataset from Our World in Data."
        print(f"[topic] Gemini selected invalid dataset or failed, falling back to: {chosen_name}")

    # 4. Construct deterministic URL based on database details or fallback
    if chosen_name in dataset_map:
        url = dataset_map[chosen_name]["csv_url"]
    else:
        encoded_name = urllib.parse.quote(chosen_name)
        url = f"https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/{encoded_name}/{encoded_name}.csv"

    final_result = {
        "dataset_name": chosen_name,
        "topic": topic_title,
        "description": topic_desc,
        "source": "Our World in Data",
        "url": url,
        "format": "csv"
    }

    print(f"[topic] Selected Dataset: {chosen_name}")
    print(f"[topic] Topic Title: {final_result['topic']}")
    print(f"[topic] URL: {final_result['url']}")

    return final_result

