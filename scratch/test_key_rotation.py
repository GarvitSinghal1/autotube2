import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
from unittest.mock import MagicMock
from google.genai.errors import ClientError, APIError

# Set multiple keys in environment
os.environ["GEMINI_API_KEY"] = "KEY_PRIMARY,KEY_SECONDARY"

from pipeline.modules.gemini_helper import build_gemini_client, generate_content_with_retry, API_KEYS

print("Loaded API keys:", API_KEYS)

# Build client
client = build_gemini_client()

# Mock the underlying client calls to raise a 429 once, then succeed
call_count = 0

def mock_generate_content(*args, **kwargs):
    global call_count
    call_count += 1
    if call_count == 1:
        print("Mock: Call 1 -> raising 429 RESOURCE_EXHAUSTED")
        # Raise APIError with required arguments
        raise APIError(code=429, response_json='RESOURCE_EXHAUSTED')
    else:
        print(f"Mock: Call {call_count} -> success")
        # Return a mock response
        mock_response = MagicMock()
        mock_response.text = f"Success after rotation! Active index is now {os.environ.get('GEMINI_API_KEY')}"
        return mock_response

# Replace the wrapper's _generate_content
client._generate_content = mock_generate_content

# Run the retry wrapper
try:
    res = generate_content_with_retry(client, model="gemini-2.5-flash", contents="test prompt")
    print("Result text:", res.text)
    print("Test passed successfully!")
except Exception as e:
    print("Test failed with error:", e)
