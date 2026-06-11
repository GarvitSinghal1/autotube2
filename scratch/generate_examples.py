"""Generate example videos for every chart type with realistic, differentiated data."""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import matplotlib
matplotlib.use("Agg")

from pipeline.modules.renderer_short import render_short
from pipeline.modules.thumbnail import generate_thumbnail
from pipeline.config import OUTPUT_DIR


def _make_bar_chart_data():
    """10 entities with dramatic rank swaps over time."""
    np.random.seed(42)
    dates = pd.date_range('2000', '2023', freq='YS')
    entities = [
        'United States', 'China', 'Japan', 'Germany', 'India',
        'United Kingdom', 'France', 'Brazil', 'Canada', 'South Korea',
    ]
    # Start values — US dominant, China small but will grow
    start_vals = {
        'United States': 10000, 'Japan': 4800, 'Germany': 2000, 'France': 1400,
        'United Kingdom': 1600, 'China': 1200, 'Canada': 750, 'Brazil': 600,
        'India': 450, 'South Korea': 560,
    }
    # End values — China overtakes everyone, India rockets up
    end_vals = {
        'United States': 25500, 'China': 17800, 'Japan': 4200, 'Germany': 4100,
        'India': 3400, 'United Kingdom': 3100, 'France': 2800, 'Brazil': 1900,
        'Canada': 2100, 'South Korea': 1700,
    }
    rows = []
    for dt in dates:
        t = (dt.year - 2000) / 23.0
        for ent in entities:
            v0 = start_vals[ent]
            v1 = end_vals[ent]
            # S-curve growth for China/India, linear for others
            if ent in ('China', 'India'):
                s = 1 / (1 + np.exp(-10 * (t - 0.45)))
                val = v0 + (v1 - v0) * s
            else:
                val = v0 + (v1 - v0) * t
            val += np.random.normal(0, val * 0.03)
            rows.append({"date": dt, "entity": ent, "value": max(10, val)})
    topic = {
        "topic": "GDP by Country (Nominal, Billions USD)",
        "description": "Nominal GDP growth showing China and India's rise.",
        "source": "World Bank", "hook": "Watch China's explosive rise!",
        "short_unit": "B", "full_unit": "Billion USD",
    }
    seg = {"start_year": 2000, "end_year": 2023, "reason": "GDP shift",
           "hook": "This is how global power shifted!"}
    return pd.DataFrame(rows), topic, seg


def _make_line_chart_data():
    """4 entities with clear crossover patterns — great for line charts."""
    np.random.seed(7)
    dates = pd.date_range('2010', '2023', freq='YS')
    entities = ['Netflix', 'Disney+', 'HBO Max', 'Amazon Prime']
    rows = []
    for dt in dates:
        yr = dt.year - 2010
        netflix = 20 + yr * 18 - max(0, (yr - 9)) * 8 + np.random.normal(0, 3)
        disney = max(0, (yr - 9) * 45 - 20) + np.random.normal(0, 2)
        hbo = max(0, (yr - 10) * 25 - 10) + np.random.normal(0, 2)
        prime = 10 + yr * 12 + np.random.normal(0, 3)
        for ent, val in [('Netflix', netflix), ('Disney+', disney),
                         ('HBO Max', hbo), ('Amazon Prime', prime)]:
            rows.append({"date": dt, "entity": ent, "value": max(0.1, val)})
    topic = {
        "topic": "Streaming Subscribers (Millions)",
        "description": "How streaming giants battled for subscribers.",
        "source": "Industry Reports", "hook": "Who won the streaming war?",
        "short_unit": "M", "full_unit": "Million Subscribers",
    }
    seg = {"start_year": 2010, "end_year": 2023, "reason": "Streaming wars",
           "hook": "Netflix wasn't always #1..."}
    return pd.DataFrame(rows), topic, seg


def _make_area_chart_data():
    """Single entity with dramatic exponential then plateau growth."""
    np.random.seed(99)
    dates = pd.date_range('2005', '2023', freq='YS')
    rows = []
    for dt in dates:
        yr = dt.year - 2005
        # Smartphone adoption: S-curve from ~0 to ~6.8B
        t = yr / 18.0
        s = 1 / (1 + np.exp(-8 * (t - 0.4)))
        val = 50 + 6750 * s + np.random.normal(0, 80)
        rows.append({"date": dt, "entity": "Global Smartphones", "value": max(50, val)})
    topic = {
        "topic": "Global Smartphone Users",
        "description": "The incredible rise of smartphone adoption worldwide.",
        "source": "Statista", "hook": "From zero to 6.8 BILLION!",
        "short_unit": "M", "full_unit": "Million Users",
    }
    seg = {"start_year": 2005, "end_year": 2023, "reason": "Smartphone revolution",
           "hook": "The device that changed everything"}
    return pd.DataFrame(rows), topic, seg


def _make_bubble_chart_data():
    """8 entities with wide value spread for clear bubble size differences."""
    np.random.seed(13)
    dates = pd.date_range('2010', '2022', freq='YS')
    entities = {
        'Apple': (300, 2900),
        'Microsoft': (220, 2500),
        'Google': (180, 1800),
        'Amazon': (100, 1600),
        'Tesla': (5, 800),
        'Meta': (50, 550),
        'Netflix': (8, 160),
        'Spotify': (1, 45),
    }
    rows = []
    for dt in dates:
        t = (dt.year - 2010) / 12.0
        for ent, (v0, v1) in entities.items():
            if ent in ('Tesla', 'Netflix', 'Spotify'):
                s = 1 / (1 + np.exp(-8 * (t - 0.5)))
                val = v0 + (v1 - v0) * s
            else:
                val = v0 + (v1 - v0) * (t ** 1.3)
            val += np.random.normal(0, val * 0.04)
            rows.append({"date": dt, "entity": ent, "value": max(1, val)})
    topic = {
        "topic": "Tech Company Market Cap (Billions)",
        "description": "Market capitalization of major tech companies.",
        "source": "Yahoo Finance", "hook": "Apple hit $3 TRILLION!",
        "short_unit": "B", "full_unit": "Billion USD",
    }
    seg = {"start_year": 2010, "end_year": 2022, "reason": "Tech boom",
           "hook": "The trillion-dollar race"}
    return pd.DataFrame(rows), topic, seg


def _make_map_data():
    """15 real countries with GDP-like data — meaningful for a choropleth."""
    np.random.seed(21)
    dates = pd.date_range('2000', '2023', freq='YS')
    countries = {
        'United States': (10300, 25500),
        'China':         (1200, 17800),
        'Japan':         (4900, 4200),
        'Germany':       (1950, 4100),
        'India':         (460, 3400),
        'United Kingdom':(1660, 3100),
        'France':        (1360, 2800),
        'Brazil':        (650, 1900),
        'Canada':        (740, 2100),
        'South Korea':   (560, 1700),
        'Russia':        (260, 1800),
        'Australia':     (410, 1700),
        'Mexico':        (680, 1300),
        'Indonesia':     (170, 1300),
        'Saudi Arabia':  (190, 1100),
    }
    rows = []
    for dt in dates:
        t = (dt.year - 2000) / 23.0
        for ent, (v0, v1) in countries.items():
            if ent in ('China', 'India', 'Indonesia'):
                s = 1 / (1 + np.exp(-8 * (t - 0.4)))
                val = v0 + (v1 - v0) * s
            elif ent == 'Japan':
                val = v0 + (v1 - v0) * t + 400 * np.sin(t * np.pi)
            else:
                val = v0 + (v1 - v0) * t
            val += np.random.normal(0, val * 0.02)
            rows.append({"date": dt, "entity": ent, "value": max(50, val)})
    topic = {
        "topic": "GDP by Country (Nominal, Billions USD)",
        "description": "World GDP distribution animated on a map.",
        "source": "World Bank", "hook": "The world economy is shifting East",
        "short_unit": "B", "full_unit": "Billion USD",
    }
    seg = {"start_year": 2000, "end_year": 2023, "reason": "Global economic shift",
           "hook": "Asia's economic revolution"}
    return pd.DataFrame(rows), topic, seg


def run_generate_examples():
    print("Starting example generation with meaningful data...")

    chart_configs = {
        "bar_chart_race": _make_bar_chart_data,
        "line_chart_race": _make_line_chart_data,
        "area_chart": _make_area_chart_data,
        "bubble_chart": _make_bubble_chart_data,
        "map_animation": _make_map_data,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ct, data_fn in chart_configs.items():
        print(f"\n{'='*60}")
        print(f"--- Generating example for: {ct} ---")
        print(f"{'='*60}")
        try:
            df, topic_info, extreme_segment = data_fn()
            print(f"  Data: {len(df)} rows, {df['entity'].nunique()} entities, "
                  f"years {df['date'].dt.year.min()}-{df['date'].dt.year.max()}")

            # Render video
            video_path, colors = render_short(df, ct, topic_info, extreme_segment)
            dest_video = OUTPUT_DIR / f"example_{ct}.mp4"
            shutil.copy2(video_path, dest_video)
            print(f"  ✓ Video saved: {dest_video}")

            # Generate thumbnail
            thumb_path = generate_thumbnail(df, ct, topic_info, extreme_segment, colors)
            dest_thumb = OUTPUT_DIR / f"example_{ct}.png"
            shutil.copy2(thumb_path, dest_thumb)
            print(f"  ✓ Thumbnail saved: {dest_thumb}")

        except Exception as e:
            print(f"  ✗ ERROR rendering {ct}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Example generation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_generate_examples()
