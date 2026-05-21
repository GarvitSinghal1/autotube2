"""
uploader.py — Uploads both videos to YouTube using the Data API v3.

Uses OAuth2 refresh token flow. Schedules long-form for 9:00 AM UTC
and Short for 12:00 PM UTC the next day.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from typing import Optional
import httplib2
import google_auth_httplib2
import urllib3
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Disable warnings for self-signed certificates in proxied environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from pipeline.config import (
    YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN,
    YOUTUBE_API_SERVICE, YOUTUBE_API_VERSION, CATEGORY_EDUCATION,
)

TOKEN_URI = "https://oauth2.googleapis.com/token"


def upload_both_videos(
    long_form_path: Optional[Path],
    short_path: Path,
    metadata: dict,
) -> dict:
    """Upload both long-form and Short videos to YouTube.

    Args:
        long_form_path: Optional path to the long-form MP4 file.
        short_path: Path to the Short MP4 file.
        metadata: Dict with 'long_form' and 'short' sub-dicts, each containing
                  title, description, and tags.

    Returns:
        Dict with 'long_form_url' and 'short_url'.

    Raises:
        RuntimeError: If upload fails or credentials are missing.
    """
    if not all([YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN]):
        raise RuntimeError(
            "Missing YouTube credentials. Set YOUTUBE_CLIENT_ID, "
            "YOUTUBE_CLIENT_SECRET, and YOUTUBE_REFRESH_TOKEN env vars."
        )

    youtube = _build_youtube_client()

    # Schedule times: next day
    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    long_publish = tomorrow.replace(hour=9, minute=0, second=0, microsecond=0)
    short_publish = tomorrow.replace(hour=12, minute=0, second=0, microsecond=0)

    # Upload long form if provided
    long_url = None
    if long_form_path:
        print("[uploader] Uploading long-form video...")
        long_url = _upload_video(
            youtube=youtube,
            file_path=long_form_path,
            title=metadata["long_form"]["title"],
            description=metadata["long_form"]["description"],
            tags=metadata["long_form"]["tags"],
            category_id=CATEGORY_EDUCATION,
            publish_at=long_publish,
            is_short=False,
        )
    else:
        print("[uploader] Long-form video upload skipped (ONLY_SHORTS mode active).")

    # Upload Short
    print("[uploader] Uploading Short...")
    short_url = _upload_video(
        youtube=youtube,
        file_path=short_path,
        title=metadata["short"]["title"],
        description=metadata["short"]["description"],
        tags=metadata["short"]["tags"],
        category_id=CATEGORY_EDUCATION,
        publish_at=short_publish,
        is_short=True,
    )

    return {
        "long_form_url": long_url or "Skipped",
        "short_url": short_url,
    }


def _build_youtube_client():
    """Build an authenticated YouTube API client with SSL verification disabled for proxy compatibility."""
    credentials = Credentials(
        token=None,
        refresh_token=YOUTUBE_REFRESH_TOKEN,
        token_uri=TOKEN_URI,
        client_id=YOUTUBE_CLIENT_ID,
        client_secret=YOUTUBE_CLIENT_SECRET,
    )
    
    # Disable SSL certificate verification to handle proxy routing/SSL intercept issues
    http_client = httplib2.Http(disable_ssl_certificate_validation=True)
    authorized_http = google_auth_httplib2.AuthorizedHttp(credentials, http=http_client)
    
    return build(
        YOUTUBE_API_SERVICE,
        YOUTUBE_API_VERSION,
        http=authorized_http,
        static_discovery=True
    )


def _upload_video(
    youtube,
    file_path: Path,
    title: str,
    description: str,
    tags: list[str],
    category_id: str,
    publish_at: datetime,
    is_short: bool,
) -> str:
    """Upload a single video to YouTube.

    Args:
        youtube: Authenticated YouTube API client.
        file_path: Path to the video file.
        title: Video title.
        description: Video description.
        tags: List of tag strings.
        category_id: YouTube category ID.
        publish_at: Scheduled publish datetime (UTC).
        is_short: Whether this is a Short.

    Returns:
        Full YouTube URL of the uploaded video.

    Raises:
        RuntimeError: If the upload fails.
    """
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": "private",  # Will become public at publish_at
            "publishAt": publish_at.isoformat(),
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        str(file_path),
        mimetype="video/mp4",
        resumable=True,
        chunksize=10 * 1024 * 1024,  # 10MB chunks
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"[uploader] Upload progress: {int(status.progress() * 100)}%")

    video_id = response.get("id")
    if not video_id:
        raise RuntimeError(f"Upload succeeded but no video ID returned: {response}")

    if is_short:
        url = f"https://youtube.com/shorts/{video_id}"
    else:
        url = f"https://youtube.com/watch?v={video_id}"

    print(f"[uploader] Uploaded: {url}")
    return url
