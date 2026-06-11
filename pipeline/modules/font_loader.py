"""
font_loader.py — Centralized font loading and watermark utility.

Loads Outfit-Bold and Outfit-Regular from assets/fonts/ and provides
FontProperties objects for matplotlib. Also loads and caches the channel
logo for watermark overlay.
"""

import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional

from pipeline.config import FONT_BOLD_PATH, FONT_REGULAR_PATH, CHANNEL_LOGO_PATH


# ── Font loading ─────────────────────────────────────────────────────────────

def _load_font(path: Path, fallback_weight: str = "normal") -> FontProperties:
    """Load a font file and return a FontProperties object.

    Falls back to system sans-serif if the font file doesn't exist.
    """
    if path.exists():
        fm.fontManager.addfont(str(path))
        return FontProperties(fname=str(path))
    else:
        print(f"[font_loader] Warning: Font not found at {path}, using system default.")
        return FontProperties(family="sans-serif", weight=fallback_weight)


FONT_BOLD: FontProperties = _load_font(FONT_BOLD_PATH, fallback_weight="bold")
FONT_REGULAR: FontProperties = _load_font(FONT_REGULAR_PATH, fallback_weight="normal")


# ── Logo / watermark ────────────────────────────────────────────────────────

_logo_cache: dict[str, Optional[np.ndarray]] = {"data": None, "loaded": False}


def _load_logo(size: int = 60) -> Optional[np.ndarray]:
    """Load and cache the channel logo as a numpy array, resized for watermark use."""
    if _logo_cache["loaded"]:
        return _logo_cache["data"]

    _logo_cache["loaded"] = True

    if not CHANNEL_LOGO_PATH.exists():
        print(f"[font_loader] Warning: Channel logo not found at {CHANNEL_LOGO_PATH}")
        return None

    try:
        img = Image.open(CHANNEL_LOGO_PATH).convert("RGBA")
        img = img.resize((size, size), Image.LANCZOS)
        _logo_cache["data"] = np.array(img)
        return _logo_cache["data"]
    except Exception as e:
        print(f"[font_loader] Warning: Failed to load channel logo: {e}")
        return None


def overlay_watermark(
    fig,
    x: float = 0.04,
    y: float = 0.04,
    size: int = 60,
    alpha: float = 0.25,
) -> None:
    """Overlay the DataDrift channel logo on a matplotlib figure.

    Args:
        fig: matplotlib Figure object.
        x: X position in figure coordinates (0-1). Default: bottom-left.
        y: Y position in figure coordinates (0-1). Default: bottom-left.
        size: Logo size in pixels.
        alpha: Transparency (0 = invisible, 1 = opaque).
    """
    logo_data = _load_logo(size)
    if logo_data is None:
        return

    # Apply alpha to the image
    logo_rgba = logo_data.copy().astype(float) / 255.0
    logo_rgba[:, :, 3] *= alpha

    # Calculate extent in figure coordinates
    fig_w, fig_h = fig.get_size_inches()
    dpi = fig.dpi
    pw = size / (fig_w * dpi)   # width in figure fraction
    ph = size / (fig_h * dpi)   # height in figure fraction

    ax_logo = fig.add_axes([x, y, pw, ph], zorder=100)
    ax_logo.imshow(logo_rgba)
    ax_logo.set_axis_off()
