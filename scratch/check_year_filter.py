import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_INDEX_DB = PROJECT_ROOT / "pipeline" / "datasets_index.db"

# Refined keywords with demographic expansion
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

def main():
    if not DATASETS_INDEX_DB.exists():
        print(f"Database at {DATASETS_INDEX_DB} does not exist.")
        return
        
    conn = sqlite3.connect(str(DATASETS_INDEX_DB))
    cursor = conn.cursor()
    cursor.execute("SELECT name, end_year, is_valid FROM datasets")
    rows = cursor.fetchall()
    conn.close()
    
    valid_datasets = [r for r in rows if r[2] == 1]
    
    matching_recent = []
    
    for name, end_year, _ in valid_datasets:
        name_lower = name.lower()
        has_interesting = any(w in name_lower for w in INTERESTING_KEYWORDS)
        has_boring = any(w in name_lower for w in BORING_KEYWORDS)
        if has_interesting and not has_boring:
            if end_year is not None and end_year >= 2022:
                matching_recent.append((name, end_year))
                
    print(f"Total valid datasets: {len(valid_datasets)}")
    print(f"Matching interesting and >= 2022: {len(matching_recent)}")
    
    print("\n--- INTERESTING DATASETS ENDING IN 2022 OR LATER ---")
    for name, end_year in sorted(matching_recent, key=lambda x: x[0]):
        print(f" - {name} (ends: {end_year})")

if __name__ == "__main__":
    main()
