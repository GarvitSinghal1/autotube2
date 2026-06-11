"""
config.py — All constants, paths, and settings for the AutoTube2 pipeline.

Secrets are read from environment variables only; nothing is hardcoded.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Directories ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"
OUTPUT_DIR = PROJECT_ROOT / "output"
FRAMES_LONG_DIR = TMP_DIR / "frames_long"
FRAMES_SHORT_DIR = TMP_DIR / "frames_short"
LONG_FORM_RAW = TMP_DIR / "long_form_raw.mp4"
LONG_FORM_FINAL = TMP_DIR / "long_form.mp4"
SHORT_FINAL = TMP_DIR / "short.mp4"
LOGS_DIR = PROJECT_ROOT / "logs"
RUN_LOG_PATH = LOGS_DIR / "run_log.json"
ASSETS_DIR = PROJECT_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
FONT_BOLD_PATH = FONTS_DIR / "Outfit-Bold.ttf"
FONT_REGULAR_PATH = FONTS_DIR / "Outfit-Regular.ttf"
MUSIC_DIR = ASSETS_DIR / "music"
CHANNEL_LOGO_PATH = ASSETS_DIR / "channel_logo.png"
DEFAULT_VOLUME = 0.15

# ── Video specs ──────────────────────────────────────────────────────────────
LONG_FORM_WIDTH = 1920
LONG_FORM_HEIGHT = 1080
SHORT_WIDTH = 1080
SHORT_HEIGHT = 1920
FPS = 30

# Frames per time-step (interpolation smoothness)
LONG_FRAMES_PER_STEP = 20   # ~0.67 s per month/year in long form
SHORT_FRAMES_PER_STEP = 6   # faster pacing for Shorts

# Duration targets (seconds)
LONG_FORM_MIN_DURATION = 300   # 5 min
LONG_FORM_MAX_DURATION = 600   # 10 min
SHORT_MIN_DURATION = 20
SHORT_MAX_DURATION = 35

# ── Design tokens ────────────────────────────────────────────────────────────
BG_COLOR = "#0f0f0f"
TEXT_COLOR = "#ffffff"
SUBTITLE_COLOR = "#aaaaaa"
ACCENT_COLORS = [
    "#FF6B6B",  # coral
    "#4ECDC4",  # teal
    "#FFD93D",  # gold
    "#6C5CE7",  # purple
    "#A8E6CF",  # mint
    "#FF8B94",  # salmon
    "#45B7D1",  # sky
    "#F9CA24",  # amber
    "#E056A0",  # magenta
    "#00CEC9",  # cyan
    "#FDA7DF",  # pink
    "#55E6C1",  # green
    "#778BEB",  # periwinkle
    "#F8A5C2",  # rose
    "#63CDDA",  # light cyan
    "#CF6A87",  # dusty rose
    "#786FA6",  # muted purple
    "#F19066",  # peach
    "#3DC1D3",  # turquoise
    "#E77F67",  # burnt sienna
]

# ── API keys & secrets ───────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# YouTube Credentials (must be authorized for the second channel)
YOUTUBE_CLIENT_ID = os.environ.get("YOUTUBE_CLIENT_ID", "")
YOUTUBE_CLIENT_SECRET = os.environ.get("YOUTUBE_CLIENT_SECRET", "")
YOUTUBE_REFRESH_TOKEN = os.environ.get("YOUTUBE_REFRESH_TOKEN", "")

# ── Gemini model ─────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"

# ── Pipeline Modes ───────────────────────────────────────────────────────────
ONLY_SHORTS = True

# ── Chart types ──────────────────────────────────────────────────────────────
CHART_TYPES = [
    "bar_chart_race",
    "map_animation",
    "line_chart_race",
    "area_chart",
    "bubble_chart",
]

# ── YouTube upload ───────────────────────────────────────────────────────────
YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
YOUTUBE_API_SERVICE = "youtube"
YOUTUBE_API_VERSION = "v3"
# Category IDs
CATEGORY_EDUCATION = "27"
CATEGORY_NEWS = "25"

# ── Minimum data requirements ───────────────────────────────────────────────
MIN_YEARS_REQUIRED = 10
TOP_N_ENTITIES = 10  # shown in bar chart race at any given frame

# ── Database index ──────────────────────────────────────────────────────────
DATASETS_INDEX_DB = PROJECT_ROOT / "pipeline" / "datasets_index.db"

