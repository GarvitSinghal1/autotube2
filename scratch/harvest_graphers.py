import requests
import xml.etree.ElementTree as ET
import re
from pathlib import Path

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
    sitemap_path = Path(__file__).resolve().parent.parent / "logs" / "sitemap.xml"
    
    # If not locally cached, fetch it
    if not sitemap_path.exists():
        print("Fetching sitemap from Our World in Data...")
        resp = requests.get("https://ourworldindata.org/sitemap.xml", verify=False)
        sitemap_content = resp.content
        sitemap_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sitemap_path, "wb") as f:
            f.write(sitemap_content)
    else:
        print("Loading sitemap from cache...")
        with open(sitemap_path, "rb") as f:
            sitemap_content = f.read()

    # Parse XML
    try:
        root = ET.fromstring(sitemap_content)
        # XML namespaces
        ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        locs = [elem.text for elem in root.findall('.//ns:loc', ns)]
    except Exception as e:
        print(f"Failed to parse XML: {e}")
        # Fallback to simple regex parsing if string structure differs
        locs = re.findall(r'<loc>(https://ourworldindata.org/[^<]+)</loc>', sitemap_content.decode('utf-8'))
        
    print(f"Total links found in sitemap: {len(locs)}")
    
    grapher_urls = [url for url in locs if "/grapher/" in url]
    print(f"Total grapher links: {len(grapher_urls)}")
    
    matching = []
    for url in grapher_urls:
        slug = url.split("/grapher/")[-1]
        slug_clean = slug.replace("-", " ").replace("_", " ").lower()
        has_interesting = any(w in slug_clean for w in INTERESTING_KEYWORDS)
        has_boring = any(w in slug_clean for w in BORING_KEYWORDS)
        if has_interesting and not has_boring:
            matching.append((slug, url))
            
    print(f"Matching interesting graphers: {len(matching)}")
    print("\n--- SAMPLE OF NEW LIVE GRAPHERS ---")
    for slug, url in sorted(matching[:60], key=lambda x: x[0]):
        print(f" - {slug} -> {url}.csv")

if __name__ == "__main__":
    main()
