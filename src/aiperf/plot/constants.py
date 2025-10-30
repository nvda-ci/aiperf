# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Constants for the visualization package.

This module defines file patterns, default paths, plot settings, and other
configuration constants used throughout the visualization functionality.
"""

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

# Plot settings
DEFAULT_PLOT_WIDTH = 1200
DEFAULT_PLOT_HEIGHT = 600
DEFAULT_PLOT_DPI = 100

# Plot size presets
PLOT_SIZE_SMALL = (800, 400)
PLOT_SIZE_MEDIUM = (1200, 600)
PLOT_SIZE_LARGE = (1600, 800)

# Color schemes for visualizations
COLOR_SCHEME_DEFAULT = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]

# NVIDIA brand colors (optional alternative color scheme)
COLOR_SCHEME_NVIDIA = [
    "#76B900",  # NVIDIA Green
    "#000000",  # Black
    "#FFFFFF",  # White
    "#1A1A1A",  # Dark Gray
    "#333333",  # Medium Gray
]

# Percentiles for statistical analysis
DEFAULT_PERCENTILES = [50, 90, 95, 99]
