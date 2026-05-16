"""
logger.py — Structured logging for every pipeline run.

Appends a run record to logs/run_log.json after each pipeline execution.
Also provides step-level status tracking during a run.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pipeline.config import RUN_LOG_PATH, LOGS_DIR


class PipelineLogger:
    """Tracks step results during a pipeline run and writes the final log entry."""

    STEP_NAMES = [
        "topic",
        "fetcher",
        "cleaner",
        "analyzer",
        "chart_selector",
        "renderer_long",
        "renderer_short",
        "metadata",
        "uploader",
    ]

    def __init__(self) -> None:
        self.record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topic": None,
            "dataset_url": None,
            "chart_type": None,
            "extreme_segment": None,
            "long_form_url": None,
            "short_url": None,
            "steps": {name: "pending" for name in self.STEP_NAMES},
            "error": None,
        }

    def mark_step(self, step: str, status: str = "pass") -> None:
        """Mark a pipeline step as pass or fail.

        Args:
            step: One of the STEP_NAMES.
            status: 'pass' or 'fail'.
        """
        if step in self.record["steps"]:
            self.record["steps"][step] = status

    def set_field(self, key: str, value: Any) -> None:
        """Set a top-level field in the run record.

        Args:
            key: Field name (e.g. 'topic', 'chart_type').
            value: The value to store.
        """
        if key in self.record:
            self.record[key] = value

    def set_error(self, error: str) -> None:
        """Record a fatal error message.

        Args:
            error: Human-readable error description.
        """
        self.record["error"] = error

    def save(self) -> None:
        """Append the current run record to the run log JSON file.

        Creates the log file and parent directories if they don't exist.
        The log file is a JSON array of run records.
        """
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        existing: list[dict] = []
        if RUN_LOG_PATH.exists():
            try:
                with open(RUN_LOG_PATH, "r") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = []

        existing.append(self.record)

        with open(RUN_LOG_PATH, "w") as f:
            json.dump(existing, f, indent=2, default=str)

        print(f"[logger] Run log saved to {RUN_LOG_PATH}")
