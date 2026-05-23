"""
main.py — Pipeline orchestrator.

Runs all steps in order, passing a shared state dict between modules.
Cleans up /tmp/dataviz/ at the end of every run whether it succeeded or failed.
"""

import shutil
import sys
import traceback
from pathlib import Path

# Add project root to sys.path so 'pipeline' is importable when running python pipeline/main.py directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import TMP_DIR, ONLY_SHORTS
from pipeline.modules.logger import PipelineLogger


def run_pipeline() -> None:
    """Execute the full data visualization pipeline end-to-end."""
    logger = PipelineLogger()

    try:
        _execute_steps(logger)
    except Exception as e:
        logger.set_error(f"{type(e).__name__}: {e}")
        print(f"\n[main] ❌ Pipeline failed: {e}")
        traceback.print_exc()
    finally:
        logger.save()
        _cleanup()


def _execute_steps(logger: PipelineLogger) -> None:
    """Run each pipeline step sequentially."""
    state: dict = {}

    blacklist = set()
    max_discovery_attempts = 5
    discovered = False

    for attempt in range(max_discovery_attempts):
        print(f"\n--- Data Acquisition Attempt {attempt + 1}/{max_discovery_attempts} ---")
        try:
            # ── Step 1: Topic Discovery ──────────────────────────────────────────
            print("\n" + "=" * 60)
            print("STEP 1: Topic Discovery")
            print("=" * 60)
            from pipeline.modules.topic import discover_topic
            topic_info = discover_topic(blacklist=blacklist)
            state["topic_info"] = topic_info
            logger.set_field("topic", topic_info["topic"])
            logger.set_field("dataset_url", topic_info["url"])

            # ── Step 2: Fetch Dataset ────────────────────────────────────────────
            print("\n" + "=" * 60)
            print("STEP 2: Fetch Dataset")
            print("=" * 60)
            from pipeline.modules.fetcher import fetch_dataset
            df_raw = fetch_dataset(topic_info["url"], topic_info["format"])
            state["df_raw"] = df_raw

            # ── Step 3: Clean Data ───────────────────────────────────────────────
            print("\n" + "=" * 60)
            print("STEP 3: Clean & Normalize Data")
            print("=" * 60)
            from pipeline.modules.cleaner import clean_dataframe
            df_monthly, df_yearly = clean_dataframe(df_raw, topic_info)
            state["df_monthly"] = df_monthly
            state["df_yearly"] = df_yearly
            topic_info["start_year"] = int(df_yearly["date"].dt.year.min())
            topic_info["end_year"] = int(df_yearly["date"].dt.year.max())

            logger.mark_step("topic", "pass")
            logger.mark_step("fetcher", "pass")
            logger.mark_step("cleaner", "pass")
            discovered = True
            break
        except Exception as e:
            dataset_name = state.get("topic_info", {}).get("dataset_name")
            if dataset_name:
                print(f"\n[main] Attempt failed for dataset '{dataset_name}': {e}")
                blacklist.add(dataset_name)
            else:
                print(f"\n[main] Attempt failed: {e}")

            # Reset state for next attempt
            state.pop("topic_info", None)
            state.pop("df_raw", None)
            state.pop("df_monthly", None)
            state.pop("df_yearly", None)

            if attempt == max_discovery_attempts - 1:
                logger.mark_step("topic", "fail")
                logger.mark_step("fetcher", "fail")
                logger.mark_step("cleaner", "fail")
                raise RuntimeError(f"Failed to acquire a valid dataset after {max_discovery_attempts} attempts.") from e
            print("Retrying with another topic...")

    # Extract df_monthly/df_yearly for subsequent steps
    df_monthly = state["df_monthly"]
    df_yearly = state["df_yearly"]
    topic_info = state["topic_info"]

    # ── Step 4: Analyze Extremes ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Find Extreme Segment")
    print("=" * 60)
    try:
        from pipeline.modules.analyzer import find_extreme_segment
        extreme_segment = find_extreme_segment(df_yearly)
        state["extreme_segment"] = extreme_segment
        logger.set_field("extreme_segment", {
            "start_year": extreme_segment["start_year"],
            "end_year": extreme_segment["end_year"],
        })
        logger.mark_step("analyzer", "pass")
    except Exception as e:
        logger.mark_step("analyzer", "fail")
        raise RuntimeError(f"Analysis failed: {e}") from e

    # ── Step 5: Select Chart Type ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Select Chart Type")
    print("=" * 60)
    try:
        from pipeline.modules.chart_selector import select_chart_type
        chart_type = select_chart_type(df_yearly, topic_info)
        state["chart_type"] = chart_type
        logger.set_field("chart_type", chart_type)
        logger.mark_step("chart_selector", "pass")
    except Exception as e:
        logger.mark_step("chart_selector", "fail")
        raise RuntimeError(f"Chart selection failed: {e}") from e

    # ── Step 6: Render Short ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Render Short")
    print("=" * 60)
    try:
        from pipeline.modules.renderer_short import render_short
        short_path, entity_colors = render_short(
            df_monthly, chart_type, topic_info, extreme_segment
        )
        state["short_path"] = short_path
        logger.mark_step("renderer_short", "pass")
    except Exception as e:
        logger.mark_step("renderer_short", "fail")
        raise RuntimeError(f"Short rendering failed: {e}") from e

    # ── Step 7: Render Long-Form Video ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Render Long-Form Video")
    print("=" * 60)
    long_path = None
    if ONLY_SHORTS:
        print("[main] ONLY_SHORTS mode active. Skipping long-form rendering.")
        logger.mark_step("renderer_long", "pass")
    else:
        try:
            from pipeline.modules.renderer_long import render_long_form
            from pipeline.config import FPS
            
            # Decide whether to use monthly or yearly data for long-form to avoid rendering bottleneck
            n_months = len(df_monthly["date"].unique())
            target_duration = 300  # 5 minutes
            target_frames = target_duration * FPS
            usable_frames = target_frames - 90 - (FPS * 2)  # minus intro/outro
            
            if n_months > 1 and (usable_frames // (n_months - 1)) < 12:
                print(f"[main] Monthly steps ({n_months}) would result in too many frames / too fast transitions. Using yearly data for long-form.")
                df_long = df_yearly
            else:
                print(f"[main] Using monthly data for long-form ({n_months} steps).")
                df_long = df_monthly

            long_path, _ = render_long_form(
                df_long, chart_type, topic_info, entity_colors=entity_colors
            )
            state["long_path"] = long_path
            logger.mark_step("renderer_long", "pass")
        except Exception as e:
            logger.mark_step("renderer_long", "fail")
            raise RuntimeError(f"Long-form rendering failed: {e}") from e

    # ── Step 8: Generate Metadata ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8: Generate Metadata")
    print("=" * 60)
    try:
        from pipeline.modules.metadata import generate_metadata
        video_metadata = generate_metadata(topic_info, extreme_segment)
        state["metadata"] = video_metadata
        logger.mark_step("metadata", "pass")
    except Exception as e:
        logger.mark_step("metadata", "fail")
        raise RuntimeError(f"Metadata generation failed: {e}") from e

    # ── Step 8.5: Save Output Locally ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8.5: Save Output Locally")
    print("=" * 60)
    try:
        import json
        output_dir = PROJECT_ROOT / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Copy videos
        saved_short = output_dir / "short.mp4"
        shutil.copy2(short_path, saved_short)
        print(f"[main] Saved short video to: {saved_short}")

        if long_path:
            saved_long = output_dir / "long_form.mp4"
            shutil.copy2(long_path, saved_long)
            print(f"[main] Saved long-form video to: {saved_long}")
        
        # Save metadata
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(video_metadata, f, indent=2)
            
        print(f"[main] Saved metadata to: {metadata_file}")
    except Exception as e:
        print(f"[main] Warning: Failed to save outputs locally: {e}")

    # ── Step 9: Upload to YouTube ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 9: Upload to YouTube")
    print("=" * 60)
    try:
        from pipeline.modules.uploader import upload_both_videos
        urls = upload_both_videos(long_path, short_path, video_metadata)
        logger.set_field("long_form_url", urls["long_form_url"])
        logger.set_field("short_url", urls["short_url"])
        logger.mark_step("uploader", "pass")
    except Exception as e:
        logger.mark_step("uploader", "fail")
        raise RuntimeError(f"Upload failed: {e}") from e

    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    if long_path:
        print(f"   Long-form: {urls['long_form_url']}")
    print(f"   Short:     {urls['short_url']}")
    print("=" * 60)


def _cleanup() -> None:
    """Remove /tmp/dataviz/ directory and all contents."""
    if TMP_DIR.exists():
        print(f"[main] Cleaning up {TMP_DIR}...")
        shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    run_pipeline()
