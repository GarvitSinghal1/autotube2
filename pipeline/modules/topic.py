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
2. STRONGLY PREFER these high-engagement categories in this priority order:
   - TIER 1 (most wanted): space exploration, satellites, rocket launches, AI/tech adoption, internet/mobile growth, billionaires & wealth, economic dominance races (GDP, trade), sports records
   - TIER 2: population milestones, urbanization booms, historical pandemics/plagues, drug/alcohol/tobacco trends, natural disasters, extinction/wildlife crises
   - TIER 3 (use only if nothing better available): crime, homicide, suicide trends
   - AVOID (exhausted/overused): nuclear weapons, nuclear tests, armed conflict deaths, war casualties, terrorism — these have been done repeatedly.
3. Focus on "Emotion + Data": Prioritize datasets that trigger strong feelings of rivalry, nostalgia, controversy, or dramatic rise-and-fall stories.
4. Keep the winner or outcome UNCLEAR in the title: Avoid titles that spoil the ending. Build suspense.
5. The title ("topic") MUST be an engaging, clickable YouTube hook rather than repeating the dry dataset name.
6. Based on your knowledge of the selected dataset, suggest the unit of measurement:
   - suggested_full_unit: the complete, formal unit name (e.g., "liters of pure alcohol per capita", "number of satellites", "metric tons per capita")
   - suggested_short_unit: a very short version (1 word or abbreviation/symbol, e.g., "liters", "satellites", "tons", "%", "$") to display next to numbers on the chart.
7. CRITICAL: BANNED WORDS. Do NOT use the words "Witness", "Explode", "Surge" (or their variations, such as "Witnessed", "Exploded", "Explosion", "Surged", "Surging", etc.) anywhere in the topic title. Choose alternative action/drama verbs and nouns (e.g., Rise, Growth, Climb, Boom, Leap, Battle, Dominance).

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
# TIER 1: Space & Tech (highest priority — most wanted)
INTERESTING_KEYWORDS = [
    # Space & technology (TIER 1 — most wanted)
    "space", "nasa", "rocket", "satellite", "spacecraft", "launch", "orbit", "iss", "astronaut",
    "internet", "mobile", "smartphone", "broadband", "technology", "computer", "software", "ai",
    # Economics & wealth (TIER 1)
    "billionaire", "top 1%", "wealth shares", "gdp growth", "trade", "export", "import", "stock",
    "economic", "richest",
    # Sports (TIER 1)
    "olympic", "medal", "sports", "football", "soccer", "basketball", "tennis", "chess",
    # Population & demographics (TIER 2)
    "population", "urbanization", "city", "birth rate", "fertility", "births", "mortality",
    "migration", "immigration",
    # Historical disasters & disease (TIER 2)
    "disaster", "earthquake", "tsunami", "volcano", "hurricane", "famine",
    "pandemic", "epidemic", "plague", "influenza", "covid", "smallpox", "polio",
    # Lifestyle & addiction (TIER 2)
    "alcohol", "beer", "wine", "drinking", "smoking", "tobacco", "cigarette", "drug", "addiction",
    # Wildlife & environment drama (TIER 2)
    "whale", "extinction", "rhino", "poaching", "deforestation",
    # Crime & social (TIER 3 — use sparingly)
    "homicide", "crime", "murder", "suicide", "accident", "aviation",
    "media coverage", "causes of death",
    "iq data", "intelligence",
    "plastic waste",
]

BORING_KEYWORDS = [
    # Exhausted topics — pipeline has done too many of these, avoid
    "nuclear", "weapon", "warfare", "military", "armed conflict", "conflict death", "war death",
    "terrorism", "terrorist", "bomb",
    # Dry academic / low-engagement
    "guinea", "worm", "diarrhea", "diarrheal", "deficiency", "malnutrition", "micronutrient",
    "tuberculosis", "malaria", "tetanus", "measles", "hepatitis", "meningitis", "encephalitis",
    "pertussis", "diphtheria", "leprosy", "trachoma", "onchocerciasis", "filariasis", "rabies",
    "dengue", "fever", "chagas", "leishmaniasis", "trypanosomiasis", "hookworm", "trichuriasis",
    "ascariasis", "nematode", "fluke", "sanitation", "hygiene", "wastewater", "treatment",
    "deworming", "iodized", "vitamin", "breastfeeding", "stunting", "wasting", "anaemia",
    "education", "expenditure", "spending", "transparency", "yield", "attainment", "schooling", "literacy",
    "diet compositions", "macronutrient", "calorie", "protein", "fat supply", "fat intake", "nutrition", "food supply", "cereal allocation",
    "pm2.5", "pollution", "emission", "air quality", "pollutant", "co2", "ghg", "methane", "nitrous",
    "inequality", "poverty", "gini", "povcalnet", "development", "mpi", "national poverty",
    "regime", "democracy", "political rights", "human rights", "civil liberties", "freedom house",
    "prevalence", "incidence", "case rate", "coverage", "at birth", "life stage"
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


def _get_recent_topics_from_log(n: int = 30) -> tuple[list[str], list[str]]:
    """Read the last n run entries from run_log.json.

    Returns:
        (recent_topics, recent_urls): two lists of strings for context injection.
        Topics are the human-readable video title strings.
        URLs are the dataset URLs (used to cross-check dataset_map for name dedup).
    """
    from pipeline.config import RUN_LOG_PATH
    import json

    if not RUN_LOG_PATH.exists():
        return [], []

    try:
        with open(RUN_LOG_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        print(f"[topic] Could not read run_log.json: {e}")
        return [], []

    recent_topics: list[str] = []
    recent_urls: list[str] = []
    # Walk newest-first
    for record in reversed(records):
        topic = record.get("topic")
        url = record.get("dataset_url")
        if topic and isinstance(topic, str):
            recent_topics.append(topic)
        if url and isinstance(url, str):
            recent_urls.append(url)
        if len(recent_topics) >= n:
            break

    print(f"[topic] Loaded {len(recent_topics)} recent topics from run_log.json for dedup context.")
    return recent_topics, recent_urls


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

    # 2. Load recent topic memory from run_log.json (works even when uploads table is empty)
    recent_topics, recent_urls = _get_recent_topics_from_log(n=30)

    # Build a set of recently-used dataset names by reverse-mapping URLs through dataset_map
    url_to_name = {d["csv_url"]: d["name"] for d in db_datasets} if db_datasets else {}
    recent_dataset_names_from_log = set()
    for url in recent_urls:
        name = url_to_name.get(url)
        if name:
            recent_dataset_names_from_log.add(name)

    # Merge ephemeral blacklist + SQLite uploads + run_log recent names
    used_names = _get_used_dataset_names()
    if used_names:
        print(f"[topic] Filtering out {len(used_names)} previously-uploaded datasets (uploads table).")
    exclude = (blacklist or set()) | used_names | recent_dataset_names_from_log
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

    # Build the user prompt with recent topic memory injected
    recent_topics_block = ""
    if recent_topics:
        recent_topics_block = (
            "\n\nRECENTLY USED TOPICS (DO NOT REPEAT these or anything thematically similar):\n"
            + "\n".join(f"- {t}" for t in recent_topics)
            + "\n\nPick something COMPLETELY DIFFERENT from the above list.\n"
        )

    user_prompt = (
        "Here are the available datasets. Pick the most fascinating one:"
        + recent_topics_block
        + "\nAVAILABLE DATASETS:\n"
        + "\n".join(sample_names)
    )

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

    from pipeline.modules.gemini_helper import clean_banned_words
    cleaned_topic = clean_banned_words(topic_title)

    final_result = {
        "dataset_name": chosen_name,
        "topic": cleaned_topic,
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

