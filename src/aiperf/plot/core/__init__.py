# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Core plot functionality including data loading and mode detection.
"""

from aiperf.plot.core.data_loader import (
    DataLoader,
    RunData,
    RunMetadata,
)
from aiperf.plot.core.mode_detector import (
    ModeDetector,
    VisualizationMode,
)
from aiperf.plot.core.plot_generator import (
    PlotGenerator,
    get_nvidia_color_scheme,
)
from aiperf.plot.core.plot_specs import (
    DataSource,
    MetricSpec,
    PlotSpec,
    PlotType,
    TimeSlicePlotSpec,
)

__all__ = [
    "DataLoader",
    "DataSource",
    "MetricSpec",
    "ModeDetector",
    "PlotGenerator",
    "PlotSpec",
    "PlotType",
    "RunData",
    "RunMetadata",
    "TimeSlicePlotSpec",
    "VisualizationMode",
    "get_nvidia_color_scheme",
]
