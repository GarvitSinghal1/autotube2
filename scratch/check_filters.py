import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Refined high-drama keywords
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
    "plastic waste"
]

# Tighter filters for boring/academic metrics
BORING_KEYWORDS = [
    "guinea", "worm", "diarrhea", "diarrheal", "deficiency", "malnutrition", "micronutrient",
    "tuberculosis", "malaria", "tetanus", "measles", "hepatitis", "meningitis", "encephalitis",
    "pertussis", "diphtheria", "leprosy", "trachoma", "onchocerciasis", "filariasis", "rabies",
    "dengue", "fever", "chagas", "leishmaniasis", "trypanosomiasis", "hookworm", "trichuriasis",
    "ascariasis", "nematode", "fluke", "sanitation", "hygiene", "wastewater", "treatment",
    "deworming", "iodized", "vitamin", "breastfeeding", "stunting", "wasting", "anaemia",
    "education", "transparency", "yield", "attainment", "schooling", "literacy",
    "diet", "macronutrient", "calorie", "protein", "fat supply", "fat intake", "nutrition", "food supply", "cereal allocation",
    "pm2.5", "pollution", "emission", "air quality", "pollutant", "co2", "ghg", "methane", "nitrous",
    "gdp", "inequality", "poverty", "gini", "index", "povcalnet", "development", "mpi", "national poverty",
    "regime", "democracy", "political rights", "human rights", "civil liberties", "freedom house",
    "prevalence", "incidence", "case rate", "treatment", "coverage", "at birth", "life stage"
]

test_names = [
    "Aviation accidents and fatalities by flight phase (ASN, 2019)",
    "Fatal aviation accidents & fatalties - Aviation Safety Network (ASN)",
    "Natural disasters (EMDAT)",
    "Natural disasters (EMDAT – decadal)",
    "Natural disasters from 1900 to 2019 - EMDAT (2020)"
]

for name in test_names:
    name_lower = name.lower()
    has_interesting = any(w in name_lower for w in INTERESTING_KEYWORDS)
    matching_int = [w for w in INTERESTING_KEYWORDS if w in name_lower]
    has_boring = any(w in name_lower for w in BORING_KEYWORDS)
    matching_boring = [w for w in BORING_KEYWORDS if w in name_lower]
    
    print(f"Name: {name}")
    print(f"  Interesting match? {has_interesting} (matched: {matching_int})")
    print(f"  Boring match? {has_boring} (matched: {matching_boring})")
    print()
