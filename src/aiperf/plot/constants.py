# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Constants for the visualization package.

This module defines file patterns, default paths, plot settings, and other
configuration constants used throughout the visualization functionality.
"""

from enum import Enum
from pathlib import Path

# File patterns for AIPerf profiling output files
PROFILE_EXPORT_JSONL = "profile_export.jsonl"
PROFILE_EXPORT_AIPERF_JSON = "profile_export_aiperf.json"
PROFILE_EXPORT_AIPERF_CSV = "profile_export_aiperf.csv"
PROFILE_EXPORT_TIMESLICES_CSV = "profile_export_aiperf_timeslices.csv"
PROFILE_EXPORT_TIMESLICES_JSON = "profile_export_aiperf_timeslices.json"
PROFILE_EXPORT_GPU_TELEMETRY_JSONL = "gpu_telemetry_export.jsonl"
INPUTS_JSON = "inputs.json"

# Default output directory and filenames
DEFAULT_OUTPUT_DIR = Path("plot_export")
DEFAULT_PNG_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "png"
DEFAULT_HTML_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "html"
PLOT_LOG_FILE = "aiperf_plot.log"


class PlotMode(str, Enum):
    """Available output modes for plot generation."""

    PNG = "png"
    HTML = "html"
    SERVER = "server"


class PlotTheme(str, Enum):
    """Available themes for plot styling."""

    LIGHT = "light"
    DARK = "dark"


DEFAULT_PLOT_WIDTH = 1600
DEFAULT_PLOT_HEIGHT = 800
DEFAULT_PLOT_DPI = 150

PLOT_FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif"

NVIDIA_GREEN = "#76B900"
NVIDIA_DARK = "#0a0a0a"
NVIDIA_GOLD = "#F4E5C3"
NVIDIA_WHITE = "#FFFFFF"
NVIDIA_DARK_BG = "#1a1a1a"
NVIDIA_GRAY = "#999999"
NVIDIA_BORDER_DARK = "#333333"
NVIDIA_BORDER_LIGHT = "#CCCCCC"
NVIDIA_TEXT_LIGHT = "#E0E0E0"
NVIDIA_CARD_BG = "#252525"

DARK_THEME_COLORS = {
    "primary": NVIDIA_GREEN,
    "secondary": NVIDIA_GOLD,
    "background": NVIDIA_DARK_BG,
    "paper": NVIDIA_CARD_BG,
    "text": NVIDIA_TEXT_LIGHT,
    "grid": NVIDIA_BORDER_DARK,
    "border": NVIDIA_BORDER_DARK,
}

LIGHT_THEME_COLORS = {
    "primary": NVIDIA_GREEN,
    "secondary": NVIDIA_GRAY,
    "background": NVIDIA_WHITE,
    "paper": NVIDIA_WHITE,
    "text": NVIDIA_DARK,
    "grid": NVIDIA_BORDER_LIGHT,
    "border": NVIDIA_BORDER_LIGHT,
}


DEFAULT_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
DEFAULT_PERCENTILE = "p50"
AVAILABLE_STATS = ["avg", "min", "max", "std"]

# All available statistic keys as they appear in metric data
# Useful for iteration, validation, and dynamic plot axis selection
ALL_STAT_KEYS = AVAILABLE_STATS + [f"p{p}" for p in DEFAULT_PERCENTILES]

# Non-metric keys in the aggregated JSON (used for filtering)
NON_METRIC_KEYS = {
    "input_config",
    "telemetry_data",
    "start_time",
    "end_time",
    "was_cancelled",
    "error_summary",
}
