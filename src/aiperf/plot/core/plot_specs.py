# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plot specifications for configurable plot generation."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class DataSource(Enum):
    """Data sources for plot metrics."""

    REQUESTS = "requests"
    TIMESLICES = "timeslices"
    GPU_TELEMETRY = "gpu_telemetry"
    AGGREGATED = "aggregated"
    SERVER_TELEMETRY = "server_telemetry"  # Future


class PlotType(Enum):
    """Types of plots that can be generated."""

    SCATTER = "scatter"
    AREA = "area"
    HISTOGRAM = "histogram"
    PARETO = "pareto"
    SCATTER_LINE = "scatter_line"
    DUAL_AXIS = "dual_axis"
    STACKED_AREA = "stacked_area"
    SCATTER_WITH_PERCENTILES = "scatter_with_percentiles"


@dataclass
class MetricSpec:
    """Specification for a single metric in a plot.

    Args:
        name: Name of the metric (column name in DataFrame)
        source: Data source where the metric comes from
        axis: Which axis the metric should be plotted on
        stat: Optional statistic to filter/extract (e.g., "avg", "p50", "p95")
              Applies to timeslices, aggregated data, and any source with stats
    """

    name: str
    source: DataSource
    axis: Literal["x", "y", "y2"]
    stat: str | None = None


@dataclass
class PlotSpec:
    """Base specification for a plot.

    Args:
        name: Unique identifier for the plot
        plot_type: Type of plot to generate
        metrics: List of metrics to plot
        title: Plot title (auto-generated if None)
        filename: Output filename (auto-generated from name if None)
        label_by: Column to use for labeling points (for multi-series plots)
        group_by: Column to use for grouping data (for multi-series plots)
        primary_mode: Visualization mode for primary (y) axis ("lines", "markers", "lines+markers")
        primary_line_shape: Line shape for primary axis ("linear", "hv", "spline", None)
        primary_fill: Fill mode for primary axis ("tozeroy", "tonexty", None)
        secondary_mode: Visualization mode for secondary (y2) axis
        secondary_line_shape: Line shape for secondary axis
        secondary_fill: Fill mode for secondary axis
        supplementary_col: Optional supplementary column name (e.g., "active_requests")
    """

    name: str
    plot_type: PlotType
    metrics: list[MetricSpec]
    title: str | None = None
    filename: str | None = None
    label_by: str | None = None
    group_by: str | None = None
    primary_mode: str = "lines"
    primary_line_shape: str | None = None
    primary_fill: str | None = None
    secondary_mode: str = "lines"
    secondary_line_shape: str | None = None
    secondary_fill: str | None = None
    supplementary_col: str | None = None


@dataclass
class TimeSlicePlotSpec(PlotSpec):
    """Specification for timeslice histogram plots.

    Args:
        use_slice_duration: Whether to pass slice_duration to the plot generator
                           for proper time-based x-axis formatting
    """

    use_slice_duration: bool = True


# Single-run plot specifications
SINGLE_RUN_PLOT_SPECS: list[PlotSpec] = [
    PlotSpec(
        name="ttft_over_time",
        plot_type=PlotType.SCATTER,
        metrics=[
            MetricSpec("request_number", DataSource.REQUESTS, "x"),
            MetricSpec("time_to_first_token", DataSource.REQUESTS, "y"),
        ],
        title="TTFT Per Request Over Time",
        filename="ttft_over_time.png",
    ),
    PlotSpec(
        name="itl_over_time",
        plot_type=PlotType.SCATTER,
        metrics=[
            MetricSpec("request_number", DataSource.REQUESTS, "x"),
            MetricSpec("inter_token_latency", DataSource.REQUESTS, "y"),
        ],
        title="Inter-Token Latency Per Request Over Time",
        filename="itl_over_time.png",
    ),
    PlotSpec(
        name="latency_over_time",
        plot_type=PlotType.SCATTER_WITH_PERCENTILES,
        metrics=[
            MetricSpec("timestamp", DataSource.REQUESTS, "x"),
            MetricSpec("request_latency", DataSource.REQUESTS, "y"),
        ],
        title="Request Latency Over Time with Percentiles",
        filename="latency_over_time.png",
    ),
    PlotSpec(
        name="dispersed_throughput_over_time",
        plot_type=PlotType.AREA,
        metrics=[
            MetricSpec("timestamp_s", DataSource.REQUESTS, "x"),
            MetricSpec("throughput_tokens_per_sec", DataSource.REQUESTS, "y"),
        ],
        title="Dispersed Output Token Throughput Over Time",
        filename="dispersed_throughput_over_time.png",
    ),
]


# Timeslice plot specifications
TIMESLICE_PLOT_SPECS: list[TimeSlicePlotSpec] = [
    TimeSlicePlotSpec(
        name="timeslices_ttft",
        plot_type=PlotType.HISTOGRAM,
        metrics=[
            MetricSpec("Timeslice", DataSource.TIMESLICES, "x"),
            MetricSpec("Time to First Token", DataSource.TIMESLICES, "y", stat="avg"),
        ],
        title="Time to First Token Across Time Slices",
        filename="timeslices_ttft.png",
        use_slice_duration=True,
    ),
    TimeSlicePlotSpec(
        name="timeslices_itl",
        plot_type=PlotType.HISTOGRAM,
        metrics=[
            MetricSpec("Timeslice", DataSource.TIMESLICES, "x"),
            MetricSpec("Inter Token Latency", DataSource.TIMESLICES, "y", stat="avg"),
        ],
        title="Inter Token Latency Across Time Slices",
        filename="timeslices_itl.png",
        use_slice_duration=True,
    ),
    TimeSlicePlotSpec(
        name="timeslices_throughput",
        plot_type=PlotType.HISTOGRAM,
        metrics=[
            MetricSpec("Timeslice", DataSource.TIMESLICES, "x"),
            MetricSpec("Request Throughput", DataSource.TIMESLICES, "y", stat="avg"),
        ],
        title="Request Throughput Across Time Slices",
        filename="timeslices_throughput.png",
        use_slice_duration=True,
    ),
    TimeSlicePlotSpec(
        name="timeslices_latency",
        plot_type=PlotType.HISTOGRAM,
        metrics=[
            MetricSpec("Timeslice", DataSource.TIMESLICES, "x"),
            MetricSpec("Request Latency", DataSource.TIMESLICES, "y", stat="avg"),
        ],
        title="Request Latency Across Time Slices",
        filename="timeslices_latency.png",
        use_slice_duration=True,
    ),
]


# GPU plot specifications
GPU_PLOT_SPECS: list[PlotSpec] = [
    PlotSpec(
        name="gpu_utilization_and_throughput_over_time",
        plot_type=PlotType.DUAL_AXIS,
        metrics=[
            MetricSpec("timestamp_s", DataSource.REQUESTS, "x"),
            MetricSpec("throughput_tokens_per_sec", DataSource.REQUESTS, "y"),
            MetricSpec("gpu_utilization", DataSource.GPU_TELEMETRY, "y2"),
        ],
        title="Output Token Throughput with GPU Utilization",
        filename="gpu_utilization_and_throughput_over_time.png",
        primary_mode="lines",
        primary_line_shape="hv",
        primary_fill=None,
        secondary_mode="lines",
        secondary_line_shape=None,
        secondary_fill="tozeroy",
        supplementary_col="active_requests",
    ),
]


# Multi-run comparison plot specifications
MULTI_RUN_PLOT_SPECS: list[PlotSpec] = [
    PlotSpec(
        name="pareto_curve_throughput_per_gpu_vs_latency",
        plot_type=PlotType.PARETO,
        metrics=[
            MetricSpec("request_latency", DataSource.AGGREGATED, "x", stat="avg"),
            MetricSpec(
                "output_token_throughput_per_gpu",
                DataSource.AGGREGATED,
                "y",
                stat="avg",
            ),
        ],
        title="Pareto Curve: Token Throughput per GPU vs Latency",
        filename="pareto_curve_throughput_per_gpu_vs_latency.png",
        label_by="concurrency",
        group_by="model",
    ),
    PlotSpec(
        name="ttft_vs_throughput",
        plot_type=PlotType.SCATTER_LINE,
        metrics=[
            MetricSpec("time_to_first_token", DataSource.AGGREGATED, "x", stat="p50"),
            MetricSpec("request_throughput", DataSource.AGGREGATED, "y", stat="avg"),
        ],
        title="TTFT vs Throughput",
        filename="ttft_vs_throughput.png",
        label_by="concurrency",
        group_by="model",
    ),
    PlotSpec(
        name="pareto_curve_throughput_per_gpu_vs_interactivity",
        plot_type=PlotType.SCATTER_LINE,
        metrics=[
            MetricSpec(
                "output_token_throughput_per_gpu",
                DataSource.AGGREGATED,
                "x",
                stat="avg",
            ),
            MetricSpec(
                "output_token_throughput_per_user",
                DataSource.AGGREGATED,
                "y",
                stat="avg",
            ),
        ],
        title="Pareto Curve: Token Throughput per GPU vs Interactivity",
        filename="pareto_curve_throughput_per_gpu_vs_interactivity.png",
        label_by="concurrency",
        group_by="model",
    ),
]
