# AutoTube2 — Automated Data Visualization YouTube Shorts Pipeline

Fully automated pipeline that discovers compelling public datasets, renders animated data visualizations as **YouTube Shorts**, and uploads them — with zero human intervention. Runs 4× daily via GitHub Actions.

## How It Works

Every pipeline run:
1. **Discovers** a compelling dataset from 346+ indexed OWID datasets using Gemini AI
2. **Deduplicates** — checks the SQLite `uploads` table to ensure no dataset is ever reused
3. **Fetches** and **cleans** the data into standardized time-series format
4. **Analyzes** the data to find the most dramatic segment (biggest rise/fall)
5. **Generates** a viral hook line via Gemini AI (context-aware, tone-matched)
6. **Selects** the best chart type (bar chart race, line chart, area chart, bubble chart, map)
7. **Renders** a 20–35s Short (1080×1920) with animated intro card, chart race, and outro
8. **Uploads** to YouTube with AI-generated title, description, tags, and hashtags
9. **Records** the upload in the database to prevent future duplicates

## Features

- 🤖 **AI-Powered Topic Selection** — Gemini 2.5 Flash picks the most viral-worthy dataset from a curated pool
- 🎣 **AI Hook Generation** — Gemini writes a punchy opening hook based on actual data insights
- 📊 **Live Dataset Index** — Scrapes OWID sitemap directly; 346+ validated datasets with 10+ year spans
- 🔄 **Redundancy Checker** — SQLite-backed upload tracking ensures no dataset is ever repeated
- 🎨 **5 Chart Types** — Bar chart race, line chart race, area chart, bubble chart, map animation
- ⚡ **Shorts-Only Mode** — Optimized for vertical video (1080×1920) at 20–35 seconds
- 🔑 **Multi-Key Rotation** — Supports comma-separated Gemini API keys with automatic failover
- 📈 **Smart Filtering** — Keyword-based interest/boring filters ensure only engaging topics are picked

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
export GEMINI_API_KEY="your-gemini-api-key"          # supports comma-separated keys for rotation
export YOUTUBE_CLIENT_ID="your-youtube-client-id"
export YOUTUBE_CLIENT_SECRET="your-youtube-client-secret"
export YOUTUBE_REFRESH_TOKEN="your-youtube-refresh-token"
```

### Run Locally
```bash
python -m pipeline.main
```

### Re-index Datasets
```bash
python -m pipeline.modules.index_datasets
```

## GitHub Actions

The pipeline runs automatically via `.github/workflows/daily_run.yml`:
- **Schedule**: 4× daily — every 6 hours (00:00, 06:00, 12:00, 18:00 UTC)
- **IST times**: 5:30 AM, 11:30 AM, 5:30 PM, 11:30 PM
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
│   ├── main.py                # Orchestrator — runs all 9 steps sequentially
│   ├── config.py              # Constants, paths, design tokens, API settings
│   ├── datasets_index.db      # SQLite database (datasets + uploads tracking)
│   └── modules/
│       ├── topic.py           # AI topic discovery + redundancy filtering
│       ├── fetcher.py         # Dataset download & parsing
│       ├── cleaner.py         # Data cleaning, normalization, aggregate filtering
│       ├── analyzer.py        # Extreme segment detection + AI hook generation
│       ├── chart_selector.py  # Chart type selection
│       ├── renderer_short.py  # Short video rendering (1080×1920)
│       ├── renderer_long.py   # Long-form video rendering (1920×1080, disabled)
│       ├── metadata.py        # YouTube metadata generation (title, desc, tags)
│       ├── uploader.py        # YouTube API v3 upload
│       ├── gemini_helper.py   # Gemini client builder + retry logic + key rotation
│       ├── index_datasets.py  # OWID sitemap scraper + dataset validator + DB schema
│       └── logger.py          # Run logging to JSON
├── assets/
│   ├── fonts/                 # Custom fonts (Bebas Neue, Roboto, etc.)
│   └── music/                 # Background music tracks
├── output/                    # Generated videos & metadata (gitignored)
├── logs/run_log.json          # Pipeline run history
├── .github/workflows/         # CI/CD
├── requirements.txt
└── README.md
```

## Architecture

```
                          ┌──────────────────┐
                          │  OWID Sitemap    │
                          │  (346+ datasets) │
                          └────────┬─────────┘
                                   │
                          ┌────────▼─────────┐
                          │  SQLite Index DB  │◄── uploads table (dedup)
                          └────────┬─────────┘
                                   │
                          ┌────────▼─────────┐
                          │  Gemini AI       │
                          │  Topic Selection │
                          └────────┬─────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Fetch → Clean → Analyze    │
                    │  (data pipeline)            │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Gemini AI Hook Generation  │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Render Short (1080×1920)   │
                    │  Intro Card → Chart → Outro │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  YouTube Upload (API v3)    │
                    │  100 credits/upload          │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Record in uploads table    │
                    │  (never reuse this dataset) │
                    └─────────────────────────────┘
```

## Database Schema

The SQLite database (`pipeline/datasets_index.db`) has two tables:

### `datasets` — Indexed OWID datasets
| Column | Description |
|---|---|
| `name` | Human-readable dataset name |
| `csv_url` | Direct CSV download URL |
| `entity_col`, `date_col`, `value_col` | Auto-detected column roles |
| `start_year`, `end_year`, `span_years` | Temporal coverage |
| `entity_count` | Number of unique countries/entities |
| `is_valid` | Whether the dataset meets pipeline requirements |

### `uploads` — Upload tracking (redundancy prevention)
| Column | Description |
|---|---|
| `dataset_name` | Which dataset was used |
| `topic_title` | The generated video title |
| `youtube_video_id` | YouTube video ID |
| `youtube_url` | Full YouTube URL |
| `video_type` | `short` or `long` |
| `uploaded_at` | UTC timestamp |

## License

MIT
