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
INPUTS_JSON = "inputs.json"

# Default output directory and filenames
DEFAULT_OUTPUT_DIR = Path("plot_export")
DEFAULT_PNG_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "png"
DEFAULT_HTML_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "html"
PLOT_LOG_FILE = "aiperf_plot.log"


# Plot modes
class PlotMode(str, Enum):
    """Available output modes for plot generation."""

    PNG = "png"
    HTML = "html"
    SERVER = "server"


# Plot settings
DEFAULT_PLOT_WIDTH = 1600
DEFAULT_PLOT_HEIGHT = 800
DEFAULT_PLOT_DPI = 150

# NVIDIA Brand Colors - Dark Mode Theme
# Based on generate_dashboard_keynote.py styling
NVIDIA_GREEN = "#76B900"
NVIDIA_DARK = "#0a0a0a"
NVIDIA_GOLD = "#F4E5C3"  # Light gold for contrast
NVIDIA_WHITE = "#FFFFFF"
NVIDIA_DARK_BG = "#1a1a1a"  # Dark background
NVIDIA_GRAY = "#999999"  # Light gray for dark mode
NVIDIA_BORDER = "#333333"  # Dark border
NVIDIA_TEXT_LIGHT = "#E0E0E0"  # Light text for dark backgrounds
NVIDIA_CARD_BG = "#252525"  # Card backgrounds

# Roboto Mono is NVIDIA preferred, but monospace is fallback for all as a generic CSS keyword for each OS to use its preferred monospace font.
# TODO [AIP-546]: Add font loading for HTML plots if needed.
PLOT_FONT_FAMILY = "'Roboto Mono', monospace"


# Percentiles for statistical analysis
# Numeric percentile values used to generate "p1", "p5", etc. keys dynamically
DEFAULT_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
DEFAULT_PERCENTILE = "p50"

# Other statistical measures available in metric data (non-percentile)
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

# Re-export metric display name functions from metric_names module
# These functions dynamically load display names from MetricRegistry and GPU telemetry config
