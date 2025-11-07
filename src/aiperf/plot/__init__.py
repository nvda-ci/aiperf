# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot package for AIPerf profiling results.

This package provides tools for plotting and analyzing AIPerf profiling data,
including data loading, plot generation, and various output modes (PNG, HTML, dashboard).
"""

__version__ = "0.1.0"
from aiperf.plot.cli_runner import (
    run_plot_controller,
)
from aiperf.plot.constants import (
    ALL_STAT_KEYS,
    AVAILABLE_STATS,
    DARK_THEME_COLORS,
    DEFAULT_HTML_OUTPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PERCENTILE,
    DEFAULT_PERCENTILES,
    DEFAULT_PLOT_DPI,
    DEFAULT_PLOT_HEIGHT,
    DEFAULT_PLOT_WIDTH,
    DEFAULT_PNG_OUTPUT_DIR,
    INPUTS_JSON,
    LIGHT_THEME_COLORS,
    NON_METRIC_KEYS,
    NVIDIA_BORDER,
    NVIDIA_CARD_BG,
    NVIDIA_DARK,
    NVIDIA_DARK_BG,
    NVIDIA_GOLD,
    NVIDIA_GRAY,
    NVIDIA_GREEN,
    NVIDIA_TEXT_LIGHT,
    NVIDIA_WHITE,
    PLOT_FONT_FAMILY,
    PLOT_LOG_FILE,
    PROFILE_EXPORT_AIPERF_CSV,
    PROFILE_EXPORT_AIPERF_JSON,
    PROFILE_EXPORT_GPU_TELEMETRY_JSONL,
    PROFILE_EXPORT_JSONL,
    PROFILE_EXPORT_TIMESLICES_CSV,
    PROFILE_EXPORT_TIMESLICES_JSON,
    PlotMode,
    PlotTheme,
)
from aiperf.plot.core import (
    DataLoader,
    ModeDetector,
    PlotGenerator,
    RunData,
    RunMetadata,
    VisualizationMode,
    get_nvidia_color_scheme,
)
from aiperf.plot.exceptions import (
    ConfigError,
    DataLoadError,
    ModeDetectionError,
    PlotError,
    PlotGenerationError,
)
from aiperf.plot.exporters import (
    BaseExporter,
    BasePNGExporter,
    MultiRunPNGExporter,
    SingleRunPNGExporter,
)
from aiperf.plot.logging import (
    setup_plot_logging,
)
from aiperf.plot.metric_names import (
    get_all_metric_display_names,
    get_metric_display_name,
)
from aiperf.plot.plot_controller import (
    PlotController,
)

__all__ = [
    "ALL_STAT_KEYS",
    "AVAILABLE_STATS",
    "BaseExporter",
    "BasePNGExporter",
    "ConfigError",
    "DARK_THEME_COLORS",
    "DEFAULT_HTML_OUTPUT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PERCENTILE",
    "DEFAULT_PERCENTILES",
    "DEFAULT_PLOT_DPI",
    "DEFAULT_PLOT_HEIGHT",
    "DEFAULT_PLOT_WIDTH",
    "DEFAULT_PNG_OUTPUT_DIR",
    "DataLoadError",
    "DataLoader",
    "INPUTS_JSON",
    "LIGHT_THEME_COLORS",
    "ModeDetectionError",
    "ModeDetector",
    "MultiRunPNGExporter",
    "NON_METRIC_KEYS",
    "NVIDIA_BORDER",
    "NVIDIA_CARD_BG",
    "NVIDIA_DARK",
    "NVIDIA_DARK_BG",
    "NVIDIA_GOLD",
    "NVIDIA_GRAY",
    "NVIDIA_GREEN",
    "NVIDIA_TEXT_LIGHT",
    "NVIDIA_WHITE",
    "PLOT_FONT_FAMILY",
    "PLOT_LOG_FILE",
    "PROFILE_EXPORT_AIPERF_CSV",
    "PROFILE_EXPORT_AIPERF_JSON",
    "PROFILE_EXPORT_GPU_TELEMETRY_JSONL",
    "PROFILE_EXPORT_JSONL",
    "PROFILE_EXPORT_TIMESLICES_CSV",
    "PROFILE_EXPORT_TIMESLICES_JSON",
    "PlotController",
    "PlotError",
    "PlotGenerationError",
    "PlotGenerator",
    "PlotMode",
    "PlotTheme",
    "RunData",
    "RunMetadata",
    "SingleRunPNGExporter",
    "VisualizationMode",
    "get_all_metric_display_names",
    "get_metric_display_name",
    "get_nvidia_color_scheme",
    "run_plot_controller",
    "setup_plot_logging",
]
