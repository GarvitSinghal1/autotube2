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
You are a viral YouTube Shorts and video producer curating data visualization topics.
You will be provided with a list of real dataset names from Our World in Data.
Your job is to select the single MOST dramatic, surprising, and human-interest dataset from the list.
Choose a topic that will hook a general audience in the first 2 seconds.

CRITICAL GUIDELINES:
1. Avoid dry academic, political development, or clinical metrics (e.g. government spending, energy efficiency, Gini coefficients, pollution rates, standard disease prevalence/treatment rates).
2. Prioritize high-retention concepts: global conflicts, weapon/nuclear proliferation, space exploration milestones, lifestyles/addictions (alcohol, cigarette sales), historical crime/suicide trends, natural disasters, major epidemics/plagues, or massive wealth concentration (billionaires).
3. The title ("topic") MUST be an engaging, clickable YouTube hook (e.g., use formats like "The Rise and Fall of...", "The Deadliest...", "The Shocking Truth About...", "Inside the...") rather than repeating the dry dataset name.
4. Based on your knowledge of the selected dataset, suggest the unit of measurement:
   - suggested_full_unit: the complete, formal unit name (e.g., "liters of pure alcohol per capita", "number of nuclear weapons", "metric tons per capita")
   - suggested_short_unit: a very short version (1 word or abbreviation/symbol, e.g., "liters", "weapons", "tons", "%", "$") to display next to numbers on the chart.

You MUST respond with ONLY a valid JSON object, no markdown, no explanation:
{
  "dataset_name": "the exact name you chose from the list provided",
  "topic": "a catchy, dramatic YouTube video hook title",
  "description": "one sentence explaining why this data is highly compelling or shocking to watch",
  "suggested_full_unit": "complete formal unit name",
  "suggested_short_unit": "short abbreviation/symbol/word"
}
"""

_FALLBACK_OWID_FOLDERS = [
    "Child Mortality - Gapminder",
    "Population - UN",
    "Urban population - UN",
    "Nuclear weapons - FAS",
]

# Keyword filters to guarantee interesting topics on YouTube
INTERESTING_KEYWORDS = [
    "military", "weapon", "nuclear", "war", "conflict", "battle", "defense", "armaments",
    "space", "nasa", "rocket", "exploration", "satellite",
    "homicide", "crime", "murder", "suicide", "terrorism", "terrorist", "poaching", "fatality", "fatalities", "accident", "aviation",
    "disaster", "earthquake", "tsunami", "volcano",
    "alcohol", "beer", "wine", "drinking", "smoking", "tobacco", "cigarette", "drug", "addiction",
    "billionaire", "top 1%", "wealth shares",
    "olympic", "medal", "sports",
    "pandemic", "epidemic", "plague", "influenza", "covid", "smallpox", "polio",
    "whale", "extinction", "rhino",
    "media coverage", "causes of death",
    "fertility", "births",
    "iq data", "intelligence",
    "plastic waste",
    # Demographic & global growth additions
    "population", "urbanization", "city", "mortality"
]

BORING_KEYWORDS = [
    "guinea", "worm", "diarrhea", "diarrheal", "deficiency", "malnutrition", "micronutrient",
    "tuberculosis", "malaria", "tetanus", "measles", "hepatitis", "meningitis", "encephalitis",
    "pertussis", "diphtheria", "leprosy", "trachoma", "onchocerciasis", "filariasis", "rabies",
    "dengue", "fever", "chagas", "leishmaniasis", "trypanosomiasis", "hookworm", "trichuriasis",
    "ascariasis", "nematode", "fluke", "sanitation", "hygiene", "wastewater", "treatment",
    "deworming", "iodized", "vitamin", "breastfeeding", "stunting", "wasting", "anaemia",
    "education", "expenditure", "spending", "transparency", "yield", "attainment", "schooling", "literacy",
    "diet compositions", "macronutrient", "calorie", "protein", "fat supply", "fat intake", "nutrition", "food supply", "cereal allocation",
    "pm2.5", "pollution", "emission", "air quality", "pollutant", "co2", "ghg", "methane", "nitrous",
    "gdp", "inequality", "poverty", "gini", "index", "povcalnet", "development", "mpi", "national poverty",
    "regime", "democracy", "political rights", "human rights", "civil liberties", "freedom house",
    "prevalence", "incidence", "case rate", "treatment", "coverage", "at birth", "life stage"
]

def _get_used_dataset_names() -> set[str]:
    """Return the set of dataset names that have already been uploaded."""
    from pipeline.config import DATASETS_INDEX_DB
    import sqlite3

    if not DATASETS_INDEX_DB.exists():
        return set()

    try:
        conn = sqlite3.connect(str(DATASETS_INDEX_DB))
        cursor = conn.cursor()
        # Check if the uploads table exists (it may not on first run)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='uploads'")
        if not cursor.fetchone():
            conn.close()
            return set()
        cursor.execute("SELECT DISTINCT dataset_name FROM uploads")
        names = {row[0] for row in cursor.fetchall()}
        conn.close()
        return names
    except Exception as e:
        print(f"[topic] Failed to query uploads table: {e}")
        return set()


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
        cursor.execute("SELECT name, path, csv_url, entity_col, date_col, value_col, start_year, end_year, span_years, entity_count FROM datasets WHERE is_valid = 1 AND end_year >= 2022")
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
    from pipeline.modules.gemini_helper import build_gemini_client
    client = build_gemini_client()

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

    # Merge ephemeral blacklist with persistently-used datasets
    used_names = _get_used_dataset_names()
    if used_names:
        print(f"[topic] Filtering out {len(used_names)} previously-uploaded datasets.")
    exclude = (blacklist or set()) | used_names
    if exclude:
        dataset_names = [d for d in dataset_names if d not in exclude]
        
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

    sample_size = min(40, len(candidate_names))
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
        suggested_full_unit: str
        suggested_short_unit: str

    # 3. Ask Gemini to choose
    from pipeline.modules.gemini_helper import generate_content_with_retry
    chosen_name = None
    topic_title = None
    topic_desc = None
    suggested_full_unit = ""
    suggested_short_unit = ""
    
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
        suggested_full_unit = result.get("suggested_full_unit", "")
        suggested_short_unit = result.get("suggested_short_unit", "")
    except Exception as e:
        print(f"[topic] Gemini selection failed: {e}. Falling back to default selection.")

    # Fallback if Gemini failed or selection is invalid
    if not chosen_name or chosen_name not in sample_names:
        chosen_name = sample_names[0]
        topic_title = chosen_name
        topic_desc = "A fascinating dataset from Our World in Data."
        suggested_full_unit = ""
        suggested_short_unit = ""
        print(f"[topic] Gemini selected invalid dataset or failed, falling back to: {chosen_name}")
    else:
        if not topic_title or not isinstance(topic_title, str) or not topic_title.strip():
            topic_title = chosen_name
        else:
            topic_title = topic_title.strip()

        if not topic_desc or not isinstance(topic_desc, str) or not topic_desc.strip():
            topic_desc = f"A fascinating dataset about {topic_title}."
        else:
            topic_desc = topic_desc.strip()

    # 4. Construct deterministic URL based on database details or fallback
    entity_col = None
    date_col = None
    value_col = None
    
    if chosen_name in dataset_map:
        url = dataset_map[chosen_name]["csv_url"]
        entity_col = dataset_map[chosen_name]["entity_col"]
        date_col = dataset_map[chosen_name]["date_col"]
        value_col = dataset_map[chosen_name]["value_col"]
    else:
        encoded_name = urllib.parse.quote(chosen_name)
        url = f"https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/{encoded_name}/{encoded_name}.csv"

    final_result = {
        "dataset_name": chosen_name,
        "topic": topic_title,
        "description": topic_desc,
        "source": "Our World in Data",
        "url": url,
        "format": "csv",
        "suggested_full_unit": suggested_full_unit,
        "suggested_short_unit": suggested_short_unit,
        "entity_col": entity_col,
        "date_col": date_col,
        "value_col": value_col
    }

    print(f"[topic] Selected Dataset: {chosen_name}")
    print(f"[topic] Topic Title: {final_result['topic']}")
    print(f"[topic] URL: {final_result['url']}")

    return final_result

