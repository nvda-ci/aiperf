# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot package for AIPerf profiling results.

This package provides tools for plotting and analyzing AIPerf profiling data,
including data loading, plot generation, and various output modes (PNG, HTML, dashboard).
"""

__version__ = "0.1.0"
from aiperf.plot.constants import (
    COLOR_SCHEME_DEFAULT,
    COLOR_SCHEME_NVIDIA,
    DEFAULT_HTML_OUTPUT_DIR,
    DEFAULT_METRICS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PERCENTILES,
    DEFAULT_PLOT_DPI,
    DEFAULT_PLOT_HEIGHT,
    DEFAULT_PLOT_WIDTH,
    DEFAULT_PNG_OUTPUT_DIR,
    INPUTS_JSON,
    METRIC_DISPLAY_NAMES,
    METRIC_UNITS,
    PLOT_LOG_FILE,
    PLOT_SIZE_LARGE,
    PLOT_SIZE_MEDIUM,
    PLOT_SIZE_SMALL,
    PROFILE_EXPORT_AIPERF_CSV,
    PROFILE_EXPORT_AIPERF_JSON,
    PROFILE_EXPORT_JSONL,
)
from aiperf.plot.core import (
    DataLoader,
    ModeDetector,
    RunData,
    RunMetadata,
    VisualizationMode,
)
from aiperf.plot.exceptions import (
    ConfigError,
    DataLoadError,
    ModeDetectionError,
    PlotError,
    PlotGenerationError,
)
from aiperf.plot.logging import (
    setup_plot_logging,
)

__all__ = [
    "COLOR_SCHEME_DEFAULT",
    "COLOR_SCHEME_NVIDIA",
    "ConfigError",
    "DEFAULT_HTML_OUTPUT_DIR",
    "DEFAULT_METRICS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PERCENTILES",
    "DEFAULT_PLOT_DPI",
    "DEFAULT_PLOT_HEIGHT",
    "DEFAULT_PLOT_WIDTH",
    "DEFAULT_PNG_OUTPUT_DIR",
    "DataLoadError",
    "DataLoader",
    "INPUTS_JSON",
    "METRIC_DISPLAY_NAMES",
    "METRIC_UNITS",
    "ModeDetectionError",
    "ModeDetector",
    "PLOT_LOG_FILE",
    "PLOT_SIZE_LARGE",
    "PLOT_SIZE_MEDIUM",
    "PLOT_SIZE_SMALL",
    "PROFILE_EXPORT_AIPERF_CSV",
    "PROFILE_EXPORT_AIPERF_JSON",
    "PROFILE_EXPORT_JSONL",
    "PlotError",
    "PlotGenerationError",
    "RunData",
    "RunMetadata",
    "VisualizationMode",
    "setup_plot_logging",
]
