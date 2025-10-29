# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Constants for the visualization package.

This module defines file patterns, default paths, plot settings, and other
configuration constants used throughout the visualization functionality.

Metric-related constants (METRIC_DISPLAY_NAMES, METRIC_UNITS, DEFAULT_METRICS)
are dynamically generated from the MetricRegistry at import time, mirroring
the pattern used by GPU telemetry metrics. This ensures consistency with
metric definitions and automatic updates when new metrics are added.
"""

from pathlib import Path

from aiperf.common.enums import MetricFlags
from aiperf.metrics.metric_registry import MetricRegistry

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


def _build_metric_display_names() -> dict[str, str]:
    """
    Dynamically build metric display names from the MetricRegistry.

    This function iterates over all registered metrics and creates a mapping
    of metric tags to their human-readable display names (header field).
    Metrics with INTERNAL or EXPERIMENTAL flags are excluded.

    Returns:
        Dictionary mapping metric tag to display name (e.g., {"time_to_first_token": "Time to First Token"})
    """
    result = {}
    for metric_class in MetricRegistry.all_classes():
        if metric_class.has_any_flags(MetricFlags.INTERNAL | MetricFlags.EXPERIMENTAL):
            continue
        result[metric_class.tag] = metric_class.header
    return result


def _build_metric_units() -> dict[str, str]:
    """
    Dynamically build metric units from the MetricRegistry.

    This function iterates over all registered metrics and creates a mapping
    of metric tags to their display unit strings. Uses display_unit if available,
    otherwise falls back to the base unit. Metrics with INTERNAL or EXPERIMENTAL
    flags are excluded.

    Returns:
        Dictionary mapping metric tag to unit string (e.g., {"time_to_first_token": "ms"})
    """
    result = {}
    for metric_class in MetricRegistry.all_classes():
        if metric_class.has_any_flags(MetricFlags.INTERNAL | MetricFlags.EXPERIMENTAL):
            continue
        unit = metric_class.display_unit or metric_class.unit
        result[metric_class.tag] = str(unit)
    return result


def _build_default_metrics() -> list[str]:
    """
    Dynamically build the default metrics list from the MetricRegistry.

    This function returns all metric tags that pass the filter criteria
    (excludes INTERNAL and EXPERIMENTAL metrics). The list is sorted by
    display_order if defined, with unordered metrics appearing last.

    Returns:
        List of metric tags to plot by default
    """
    metrics_with_order = []
    metrics_without_order = []

    for metric_class in MetricRegistry.all_classes():
        if metric_class.has_any_flags(MetricFlags.INTERNAL | MetricFlags.EXPERIMENTAL):
            continue

        if metric_class.display_order is not None:
            metrics_with_order.append((metric_class.display_order, metric_class.tag))
        else:
            metrics_without_order.append(metric_class.tag)

    # Sort by display_order, then append unordered metrics
    metrics_with_order.sort()
    ordered_tags = [tag for _, tag in metrics_with_order]

    return ordered_tags + sorted(metrics_without_order)


# Metric display names mapping (dynamically generated from MetricRegistry)
# When loading data files, take the intersection of available metrics in the data
# with the metrics defined here to only plot metrics that exist in both.
METRIC_DISPLAY_NAMES = _build_metric_display_names()

# Units for metrics (dynamically generated from MetricRegistry)
METRIC_UNITS = _build_metric_units()

# Default metrics to plot (dynamically generated from MetricRegistry)
# Includes all non-INTERNAL/EXPERIMENTAL metrics, sorted by display_order
DEFAULT_METRICS = _build_default_metrics()

# Percentiles for statistical analysis
DEFAULT_PERCENTILES = [50, 90, 95, 99]
