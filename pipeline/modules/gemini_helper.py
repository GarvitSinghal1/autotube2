"""
gemini_helper.py — Provides resilient wrapper functions for Gemini API calls.

Automatically handles 429 RESOURCE_EXHAUSTED errors with precise sleeps.
"""

import re
import time
from google.genai.errors import ClientError, APIError


def generate_content_with_retry(client, model: str, contents, config=None, max_retries: int = 5, initial_delay: float = 5.0):
    """Call client.models.generate_content with retry logic for rate limits.

    If a 429 RESOURCE_EXHAUSTED error is raised, parses the requested wait time
    from the exception message and sleeps accordingly. Otherwise falls back to
    exponential backoff.
    """
    delay = initial_delay
    for attempt in range(max_retries):
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

            if is_rate_limit and attempt < max_retries - 1:
                # Search for retry duration: e.g. "Please retry in 46.369671979s."
                match = re.search(r"[Pp]lease retry in (\d+\.?\d*)s", err_msg)
                if match:
                    wait_time = float(match.group(1))
                    # Add a 2.0s safety buffer
                    sleep_time = wait_time + 2.0
                    print(
                        f"[gemini_helper] Rate limit 429 hit. Server requested wait of {wait_time}s. "
                        f"Sleeping for {sleep_time:.2f}s... (Attempt {attempt + 1}/{max_retries})"
                    )
                else:
                    sleep_time = delay
                    print(
                        f"[gemini_helper] Rate limit 429 hit (no wait time parsed). "
                        f"Sleeping for {sleep_time:.2f}s... (Attempt {attempt + 1}/{max_retries})"
                    )
                    delay *= 2.0

                time.sleep(sleep_time)
                continue

            # If it's a regular error or we're out of retries, raise it
            print(f"[gemini_helper] Error: {e} (Attempt {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                raise e

        except Exception as e:
            # General connection or unexpected errors
            if attempt < max_retries - 1:
                sleep_time = delay
                print(
                    f"[gemini_helper] Unexpected error: {e}. "
                    f"Sleeping for {sleep_time:.2f}s... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_time)
                delay *= 2.0
                continue
            raise e
