# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot specifications for configurable plot generation."""

from enum import Enum
from typing import Literal

from pydantic import Field

from aiperf.common.models import AIPerfBaseModel


class Style(AIPerfBaseModel):
    """Styling configuration for a plot trace."""

    mode: str = Field(
        default="lines",
        description="Plotly visualization mode ('lines', 'markers', 'lines+markers')",
    )
    line_shape: str | None = Field(
        default=None,
        description="Line shape for the trace ('linear', 'hv' for step, 'spline', or None)",
    )
    fill: str | None = Field(
        default=None,
        description="Fill pattern for the trace ('tozeroy', 'tonexty', or None for no fill)",
    )
    line_width: int = Field(
        default=2,
        description="Width of the line in pixels",
    )
    marker_size: int = Field(
        default=8,
        description="Size of markers in pixels",
    )
    marker_opacity: float = Field(
        default=1.0,
        description="Opacity of markers (0.0 to 1.0)",
    )
    fill_opacity: float = Field(
        default=0.3,
        description="Opacity of fill area (0.0 to 1.0)",
    )


class DataSource(Enum):
    """Data sources for plot metrics."""

    REQUESTS = "requests"
    TIMESLICES = "timeslices"
    GPU_TELEMETRY = "gpu_telemetry"
    AGGREGATED = "aggregated"


class PlotType(Enum):
    """Types of plots that can be generated."""

    SCATTER = "scatter"
    AREA = "area"
    HISTOGRAM = "histogram"
    PARETO = "pareto"
    SCATTER_LINE = "scatter_line"
    DUAL_AXIS = "dual_axis"
    SCATTER_WITH_PERCENTILES = "scatter_with_percentiles"


class MetricSpec(AIPerfBaseModel):
    """Specification for a single metric in a plot."""

    name: str = Field(description="Name of the metric (column name in DataFrame)")
    source: DataSource = Field(description="Data source where the metric comes from")
    axis: Literal["x", "y", "y2"] = Field(
        description="Which axis the metric should be plotted on"
    )
    stat: str | None = Field(
        default=None,
        description="Optional statistic to filter/extract (e.g., 'avg', 'p50', 'p95'). "
        "Applies to timeslices, aggregated data, and any source with stats",
    )


class PlotSpec(AIPerfBaseModel):
    """Base specification for a plot."""

    name: str = Field(description="Unique identifier for the plot")
    plot_type: PlotType = Field(description="Type of plot to generate")
    metrics: list[MetricSpec] = Field(description="List of metrics to plot")
    title: str | None = Field(
        default=None, description="Plot title (auto-generated if None)"
    )
    filename: str | None = Field(
        default=None, description="Output filename (auto-generated from name if None)"
    )
    label_by: str | None = Field(
        default=None,
        description="Column to use for labeling points (for multi-series plots)",
    )
    group_by: str | None = Field(
        default=None,
        description="Column to use for grouping data (for multi-series plots)",
    )
    primary_style: Style | None = Field(
        default=None,
        description="Style configuration for primary (y) axis trace",
    )
    secondary_style: Style | None = Field(
        default=None,
        description="Style configuration for secondary (y2) axis trace",
    )
    supplementary_col: str | None = Field(
        default=None,
        description="Optional supplementary column name (e.g., 'active_requests')",
    )


class TimeSlicePlotSpec(PlotSpec):
    """Specification for timeslice histogram plots."""

    use_slice_duration: bool = Field(
        default=True,
        description="Whether to pass slice_duration to the plot generator "
        "for proper time-based x-axis formatting",
    )


# Single-run plot specifications
SINGLE_RUN_PLOT_SPECS: list[PlotSpec] = [
    PlotSpec(
        name="ttft_over_time",
        plot_type=PlotType.SCATTER,
        metrics=[
            MetricSpec(name="request_number", source=DataSource.REQUESTS, axis="x"),
            MetricSpec(
                name="time_to_first_token", source=DataSource.REQUESTS, axis="y"
            ),
        ],
        title="TTFT Per Request Over Time",
        filename="ttft_over_time.png",
    ),
    PlotSpec(
        name="itl_over_time",
        plot_type=PlotType.SCATTER,
        metrics=[
            MetricSpec(name="request_number", source=DataSource.REQUESTS, axis="x"),
            MetricSpec(
                name="inter_token_latency", source=DataSource.REQUESTS, axis="y"
            ),
        ],
        title="Inter-Token Latency Per Request Over Time",
        filename="itl_over_time.png",
    ),
    PlotSpec(
        name="latency_over_time",
        plot_type=PlotType.SCATTER_WITH_PERCENTILES,
        metrics=[
            MetricSpec(name="timestamp", source=DataSource.REQUESTS, axis="x"),
            MetricSpec(name="request_latency", source=DataSource.REQUESTS, axis="y"),
        ],
        title="Request Latency Over Time with Percentiles",
        filename="latency_over_time.png",
    ),
    PlotSpec(
        name="dispersed_throughput_over_time",
        plot_type=PlotType.AREA,
        metrics=[
            MetricSpec(name="timestamp_s", source=DataSource.REQUESTS, axis="x"),
            MetricSpec(
                name="throughput_tokens_per_sec", source=DataSource.REQUESTS, axis="y"
            ),
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
            MetricSpec(name="Timeslice", source=DataSource.TIMESLICES, axis="x"),
            MetricSpec(
                name="Time to First Token",
                source=DataSource.TIMESLICES,
                axis="y",
                stat="avg",
            ),
        ],
        title="Time to First Token Across Time Slices",
        filename="timeslices_ttft.png",
        use_slice_duration=True,
    ),
    TimeSlicePlotSpec(
        name="timeslices_itl",
        plot_type=PlotType.HISTOGRAM,
        metrics=[
            MetricSpec(name="Timeslice", source=DataSource.TIMESLICES, axis="x"),
            MetricSpec(
                name="Inter Token Latency",
                source=DataSource.TIMESLICES,
                axis="y",
                stat="avg",
            ),
        ],
        title="Inter Token Latency Across Time Slices",
        filename="timeslices_itl.png",
        use_slice_duration=True,
    ),
    TimeSlicePlotSpec(
        name="timeslices_throughput",
        plot_type=PlotType.HISTOGRAM,
        metrics=[
            MetricSpec(name="Timeslice", source=DataSource.TIMESLICES, axis="x"),
            MetricSpec(
                name="Request Throughput",
                source=DataSource.TIMESLICES,
                axis="y",
                stat="avg",
            ),
        ],
        title="Request Throughput Across Time Slices",
        filename="timeslices_throughput.png",
        use_slice_duration=True,
    ),
    TimeSlicePlotSpec(
        name="timeslices_latency",
        plot_type=PlotType.HISTOGRAM,
        metrics=[
            MetricSpec(name="Timeslice", source=DataSource.TIMESLICES, axis="x"),
            MetricSpec(
                name="Request Latency",
                source=DataSource.TIMESLICES,
                axis="y",
                stat="avg",
            ),
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
            MetricSpec(name="timestamp_s", source=DataSource.REQUESTS, axis="x"),
            MetricSpec(
                name="throughput_tokens_per_sec", source=DataSource.REQUESTS, axis="y"
            ),
            MetricSpec(
                name="gpu_utilization", source=DataSource.GPU_TELEMETRY, axis="y2"
            ),
        ],
        title="Output Token Throughput with GPU Utilization",
        filename="gpu_utilization_and_throughput_over_time.png",
        primary_style=Style(mode="lines", line_shape="hv", fill=None),
        secondary_style=Style(mode="lines", line_shape=None, fill="tozeroy"),
        supplementary_col="active_requests",
    ),
]


# Multi-run comparison plot specifications
MULTI_RUN_PLOT_SPECS: list[PlotSpec] = [
    PlotSpec(
        name="pareto_curve_throughput_per_gpu_vs_latency",
        plot_type=PlotType.PARETO,
        metrics=[
            MetricSpec(
                name="request_latency",
                source=DataSource.AGGREGATED,
                axis="x",
                stat="avg",
            ),
            MetricSpec(
                name="output_token_throughput_per_gpu",
                source=DataSource.AGGREGATED,
                axis="y",
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
            MetricSpec(
                name="time_to_first_token",
                source=DataSource.AGGREGATED,
                axis="x",
                stat="p50",
            ),
            MetricSpec(
                name="request_throughput",
                source=DataSource.AGGREGATED,
                axis="y",
                stat="avg",
            ),
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
                name="output_token_throughput_per_gpu",
                source=DataSource.AGGREGATED,
                axis="x",
                stat="avg",
            ),
            MetricSpec(
                name="output_token_throughput_per_user",
                source=DataSource.AGGREGATED,
                axis="y",
                stat="avg",
            ),
        ],
        title="Pareto Curve: Token Throughput per GPU vs Interactivity",
        filename="pareto_curve_throughput_per_gpu_vs_interactivity.png",
        label_by="concurrency",
        group_by="model",
    ),
]
