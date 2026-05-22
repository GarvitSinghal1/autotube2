"""
gemini_helper.py — Provides resilient wrapper functions for Gemini API calls.

Automatically handles 429 RESOURCE_EXHAUSTED errors with precise sleeps or
rotating to alternative API keys if multiple keys are provided.
"""

import os
import re
import time
import httpx
from google import genai
from google.genai import types
from google.genai.errors import ClientError, APIError

# Sourced keys
def _load_api_keys() -> list[str]:
    from dotenv import load_dotenv
    from pathlib import Path
    # Load .env relative to project root
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(env_path)

    keys = []
    # 1. From GEMINI_API_KEY (could be comma or semicolon separated)
    raw_key = os.environ.get("GEMINI_API_KEY", "")
    if raw_key:
        for part in re.split(r'[;,]', raw_key):
            part = part.strip()
            if part and part not in keys:
                keys.append(part)
                
    # 2. From GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
    for i in range(1, 51):
        k1 = os.environ.get(f"GEMINI_API_KEY_{i}", "")
        if k1:
            k1 = k1.strip()
            if k1 and k1 not in keys:
                keys.append(k1)
        k2 = os.environ.get(f"GEMINI_API_KEY{i}", "")
        if k2:
            k2 = k2.strip()
            if k2 and k2 not in keys:
                keys.append(k2)
                
    return keys

API_KEYS = _load_api_keys()
ACTIVE_KEY_INDEX = 0

class GeminiModelsWrapper:
    def __init__(self, wrapper):
        self.wrapper = wrapper

    def generate_content(self, *args, **kwargs):
        return self.wrapper._generate_content(*args, **kwargs)

class GeminiClientWrapper:
    def __init__(self):
        self.models = GeminiModelsWrapper(self)
        self.httpx_client = httpx.Client(verify=False)
        self.current_client = None
        self._init_current_client()

    def _init_current_client(self):
        global ACTIVE_KEY_INDEX, API_KEYS
        if not API_KEYS:
            # Fallback to whatever is in GEMINI_API_KEY or raise error
            fallback = os.environ.get("GEMINI_API_KEY", "").strip()
            if fallback:
                API_KEYS = [fallback]
            else:
                raise RuntimeError("GEMINI_API_KEY environment variable is not set and no alternative keys found.")
        
        ACTIVE_KEY_INDEX = ACTIVE_KEY_INDEX % len(API_KEYS)
        key = API_KEYS[ACTIVE_KEY_INDEX]
        masked_key = key[:6] + "..." + key[-4:] if len(key) > 10 else "..."
        print(f"[gemini_helper] Initializing Gemini client with key index {ACTIVE_KEY_INDEX} ({masked_key})")
        self.current_client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(httpx_client=self.httpx_client),
        )

    def rotate_key(self) -> bool:
        global ACTIVE_KEY_INDEX, API_KEYS
        if not API_KEYS or len(API_KEYS) <= 1:
            print("[gemini_helper] Key rotation requested, but only 1 key is available. No rotation possible.")
            return False
            
        ACTIVE_KEY_INDEX = (ACTIVE_KEY_INDEX + 1) % len(API_KEYS)
        key = API_KEYS[ACTIVE_KEY_INDEX]
        masked_key = key[:6] + "..." + key[-4:] if len(key) > 10 else "..."
        print(f"[gemini_helper] 🔄 Rotating Gemini API key to index {ACTIVE_KEY_INDEX} ({masked_key})...")
        self.current_client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(httpx_client=self.httpx_client),
        )
        return True

    def _generate_content(self, *args, **kwargs):
        return self.current_client.models.generate_content(*args, **kwargs)

def build_gemini_client() -> GeminiClientWrapper:
    """Build a Gemini API client wrapper supporting key rotation and custom HTTP client to bypass SSL verification."""
    return GeminiClientWrapper()

def generate_content_with_retry(client, model: str, contents, config=None, max_retries: int = 5, initial_delay: float = 5.0):
    """Call client.models.generate_content with retry logic for rate limits.

    If a 429 RESOURCE_EXHAUSTED or 403/400 authentication error is raised,
    rotates keys first. If all keys are exhausted, sleeps and backs off.
    """
    delay = initial_delay
    num_keys = len(API_KEYS) if API_KEYS else 1

    for attempt in range(max_retries):
        consecutive_rotations = 0
        while True:
            try:
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except (ClientError, APIError) as e:
                err_msg = str(e)
                is_rate_limit = (
                    getattr(e, "code", None) == 429
                    or "429" in err_msg
                    or "RESOURCE_EXHAUSTED" in err_msg
                    or "quota" in err_msg.lower()
                )
                is_auth_error = (
                    getattr(e, "code", None) in (400, 401, 403)
                    and any(w in err_msg.lower() for w in ("key", "permission", "denied", "not valid", "unauthorized", "forbidden"))
                )

                if (is_rate_limit or is_auth_error) and consecutive_rotations < num_keys:
                    if hasattr(client, "rotate_key"):
                        rotated = client.rotate_key()
                        if rotated:
                            consecutive_rotations += 1
                            error_type = "Rate limit 429" if is_rate_limit else "Auth error"
                            print(f"[gemini_helper] 🔄 {error_type} hit. Key rotated successfully ({consecutive_rotations}/{num_keys}). Retrying immediately...")
                            continue

                # If rotation is not possible or keys are exhausted, handle sleep and retry/raise
                print(f"[gemini_helper] Error: {e} (Attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    raise e

                if is_rate_limit:
                    match = re.search(r"[Pp]lease retry in (\d+\.?\d*)s", err_msg)
                    if match:
                        wait_time = float(match.group(1))
                        sleep_time = wait_time + 2.0
                        print(
                            f"[gemini_helper] Rate limit 429 hit. Server requested wait of {wait_time}s. "
                            f"Sleeping for {sleep_time:.2f}s..."
                        )
                    else:
                        sleep_time = delay
                        print(f"[gemini_helper] Sleeping for {sleep_time:.2f}s...")
                        delay *= 2.0
                else:
                    sleep_time = delay
                    print(f"[gemini_helper] Sleeping for {sleep_time:.2f}s...")
                    delay *= 2.0

                time.sleep(sleep_time)
                break  # Exit inner while loop to move to the next outer attempt

            except Exception as e:
                print(f"[gemini_helper] Unexpected error: {e} (Attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    raise e
                sleep_time = delay
                print(f"[gemini_helper] Sleeping for {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                delay *= 2.0
                break  # Exit inner while loop to move to the next outer attempt


