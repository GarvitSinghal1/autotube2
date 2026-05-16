# AutoTube2 — Automated Data Visualization YouTube Pipeline

Fully automated pipeline that discovers public datasets, renders animated data visualizations, and uploads two video versions — a **YouTube Short** and a **long-form video** — with zero human intervention.

## How It Works

Every pipeline run:
1. **Discovers** a compelling public dataset using Gemini AI
2. **Fetches** and **cleans** the data into standardized time-series format
3. **Analyzes** the data to find the most dramatic segment
4. **Selects** the best chart type (bar chart race, line chart, area chart, etc.)
5. **Renders** two videos:
   - 🎬 **Long-form** (5–10 min, 1920×1080) — full dataset story, month-wise granularity
   - ⚡ **Short** (50–59s, 1080×1920) — dramatic highlight, year-wise granularity
6. **Uploads** both to YouTube with generated metadata, scheduled 3 hours apart

## Setup

### Prerequisites
- Python 3.11+
- FFmpeg (`brew install ffmpeg` on macOS)
- Google Gemini API key
- YouTube API OAuth2 credentials

### Installation
```bash
pip install -r requirements.txt
```

### Environment Variables
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export YOUTUBE_CLIENT_ID="your-youtube-client-id"
export YOUTUBE_CLIENT_SECRET="your-youtube-client-secret"
export YOUTUBE_REFRESH_TOKEN="your-youtube-refresh-token"
```

### Run Locally
```bash
python -m pipeline.main
```

## GitHub Actions

The pipeline runs automatically via `.github/workflows/daily_run.yml`:
- **Schedule**: Daily at 6:00 AM UTC
- **Manual trigger**: Available via `workflow_dispatch`

### Required Secrets
Add these to your GitHub repo → Settings → Secrets:
- `GEMINI_API_KEY`
- `YOUTUBE_CLIENT_ID`
- `YOUTUBE_CLIENT_SECRET`
- `YOUTUBE_REFRESH_TOKEN`

## Project Structure
```
├── pipeline/
│   ├── main.py              # Orchestrator
│   ├── config.py            # Constants, paths, settings
│   └── modules/
│       ├── topic.py          # Gemini topic & dataset discovery
│       ├── fetcher.py        # Dataset download & parsing
│       ├── cleaner.py        # Data cleaning & normalization
│       ├── analyzer.py       # Extreme segment detection
│       ├── chart_selector.py # Chart type selection
│       ├── renderer_long.py  # Long-form video rendering
│       ├── renderer_short.py # Short video rendering
│       ├── metadata.py       # YouTube metadata generation
│       ├── uploader.py       # YouTube upload
│       └── logger.py         # Run logging
├── assets/fonts/             # Custom fonts (optional)
├── logs/run_log.json         # Pipeline run history
├── .github/workflows/        # CI/CD
├── requirements.txt
└── README.md
```

## Architecture

```
Gemini AI ──→ Topic Discovery ──→ Fetch Data ──→ Clean & Normalize
                                                       │
                                                       ▼
                                              ┌─── df_yearly
                                              │    df_monthly
                                              ▼
                                     Analyze Extremes
                                     Select Chart Type
                                              │
                                    ┌─────────┴──────────┐
                                    ▼                    ▼
                              Long-Form              Short
                             (5-10 min)            (50-59s)
                             1920×1080             1080×1920
                              monthly               yearly
                                    │                    │
                                    ▼                    ▼
                              FFmpeg Encode        FFmpeg Encode
                                    │                    │
                                    └────────┬───────────┘
                                             ▼
                                     YouTube Upload
                                    (scheduled publish)
```
