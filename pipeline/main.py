"""
main.py — Pipeline orchestrator.

Runs all steps in order, passing a shared state dict between modules.
Cleans up /tmp/dataviz/ at the end of every run whether it succeeded or failed.
"""

import shutil
import sys
import traceback

from pipeline.config import TMP_DIR
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

    # ── Step 1: Topic Discovery ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Topic Discovery")
    print("=" * 60)
    try:
        from pipeline.modules.topic import discover_topic
        topic_info = discover_topic()
        state["topic_info"] = topic_info
        logger.set_field("topic", topic_info["topic"])
        logger.set_field("dataset_url", topic_info["url"])
        logger.mark_step("topic", "pass")
    except Exception as e:
        logger.mark_step("topic", "fail")
        raise RuntimeError(f"Topic discovery failed: {e}") from e

    # ── Step 2: Fetch Dataset ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Fetch Dataset")
    print("=" * 60)
    try:
        from pipeline.modules.fetcher import fetch_dataset
        df_raw = fetch_dataset(topic_info["url"], topic_info["format"])
        state["df_raw"] = df_raw
        logger.mark_step("fetcher", "pass")
    except Exception as e:
        logger.mark_step("fetcher", "fail")
        raise RuntimeError(f"Dataset fetch failed: {e}") from e

    # ── Step 3: Clean Data ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Clean & Normalize Data")
    print("=" * 60)
    try:
        from pipeline.modules.cleaner import clean_dataframe
        df_monthly, df_yearly = clean_dataframe(df_raw, topic_info)
        state["df_monthly"] = df_monthly
        state["df_yearly"] = df_yearly
        logger.mark_step("cleaner", "pass")
    except Exception as e:
        logger.mark_step("cleaner", "fail")
        raise RuntimeError(f"Data cleaning failed: {e}") from e

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

    # ── Step 6: Render Long-Form Video ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Render Long-Form Video")
    print("=" * 60)
    try:
        from pipeline.modules.renderer_long import render_long_form
        # Use monthly data if available, fall back to yearly
        render_df = df_monthly if not df_monthly.empty else df_yearly
        long_path, entity_colors = render_long_form(
            render_df, chart_type, topic_info
        )
        state["long_path"] = long_path
        state["entity_colors"] = entity_colors
        logger.mark_step("renderer_long", "pass")
    except Exception as e:
        logger.mark_step("renderer_long", "fail")
        raise RuntimeError(f"Long-form rendering failed: {e}") from e

    # ── Step 7: Render Short ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Render Short")
    print("=" * 60)
    try:
        from pipeline.modules.renderer_short import render_short
        short_path = render_short(
            df_yearly, chart_type, topic_info,
            extreme_segment, entity_colors,
        )
        state["short_path"] = short_path
        logger.mark_step("renderer_short", "pass")
    except Exception as e:
        logger.mark_step("renderer_short", "fail")
        raise RuntimeError(f"Short rendering failed: {e}") from e

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
