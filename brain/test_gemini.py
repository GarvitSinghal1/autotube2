import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import GEMINI_API_KEY, GEMINI_MODEL
from google import genai
from google.genai import types
from pydantic import BaseModel
from enum import Enum

client = genai.Client(api_key=GEMINI_API_KEY)

sample_names = ["Nuclear weapons - FAS", "Military Expenditure - SIPRI", "Population - UN"]
DatasetEnum = Enum("DatasetEnum", {f"item_{i}": name for i, name in enumerate(sample_names)})

class TopicSelection(BaseModel):
    dataset_name: DatasetEnum
    topic: str
    description: str

try:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents="Pick the nuclear one from: " + str(sample_names),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TopicSelection,
            temperature=0.1,
        )
    )
    print("Response text:", response.text)
except Exception as e:
    import traceback
    traceback.print_exc()
