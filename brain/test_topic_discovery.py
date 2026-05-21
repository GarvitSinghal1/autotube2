import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.modules.topic import discover_topic
import json

try:
    topic_info = discover_topic()
    print("\n--- Successful Test ---")
    print(json.dumps(topic_info, indent=2))
except Exception as e:
    import traceback
    traceback.print_exc()
