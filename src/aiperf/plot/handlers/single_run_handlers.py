# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-run plot type handlers.

Handlers for creating plots from single profiling run data.
"""

import pandas as pd
import plotly.graph_objects as go

from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.data_preparation import (
    aggregate_gpu_telemetry,
    calculate_rolling_percentiles,
    calculate_throughput_events,
    prepare_request_timeseries,
    prepare_timeslice_metrics,
    validate_request_uniformity,
)
from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.core.plot_specs import (
    DataSource,
    PlotSpec,
    PlotType,
    TimeSlicePlotSpec,
)
from aiperf.plot.core.plot_type_handlers import PlotTypeHandlerFactory


class BaseSingleRunHandler:
    """
    Base class for single-run plot handlers.

    Provides common functionality for data preparation and validation.
    """

    def __init__(self, plot_generator: PlotGenerator, logger=None) -> None:
        """
        Initialize the handler.

        Args:
            plot_generator: PlotGenerator instance for rendering plots
            logger: Optional logger instance
        """
        self.plot_generator = plot_generator
        self.logger = logger

    def _get_axis_label(self, metric_spec, available_metrics: dict) -> str:
        """
        Get axis label for a metric.

        Args:
            metric_spec: MetricSpec object
            available_metrics: Dictionary with display_names and units

        Returns:
            Formatted axis label
        """
        if metric_spec.name == "request_number":
            return "Request Number"
        elif metric_spec.name == "timestamp":
            return "Time (seconds)"
        elif metric_spec.name == "timestamp_s" or metric_spec.name == "Timeslice":
            return "Time (s)"
        else:
            return self._get_metric_label(
                metric_spec.name, metric_spec.stat, available_metrics
            )

    def _get_metric_label(
        self, metric_name: str, stat: str | None, available_metrics: dict
    ) -> str:
        """
        Get formatted metric label.

        Args:
            metric_name: Name of the metric
            stat: Statistic (e.g., "avg", "p50")
            available_metrics: Dictionary with display_names and units

        Returns:
            Formatted metric label
        """
        if metric_name in available_metrics:
            display_name = available_metrics[metric_name].get(
                "display_name", metric_name
            )
            unit = available_metrics[metric_name].get("unit", "")
            if stat and stat not in ["avg", "value"]:
                display_name = f"{display_name} ({stat})"
            if unit:
                return f"{display_name} ({unit})"
            return display_name
        return metric_name

    def _prepare_data_for_source(
        self, source: DataSource, run: RunData
    ) -> pd.DataFrame:
        """
        Prepare data from a specific source.

        Args:
            source: Data source to prepare
            run: RunData object

        Returns:
            Prepared DataFrame
        """
        if source == DataSource.REQUESTS:
            return prepare_request_timeseries(run)
        elif source == DataSource.TIMESLICES:
            return run.timeslices
        elif source == DataSource.GPU_TELEMETRY:
            return run.gpu_telemetry
        else:
            raise ValueError(f"Unsupported data source: {source}")


@PlotTypeHandlerFactory.register(PlotType.SCATTER)
class ScatterHandler(BaseSingleRunHandler):
    """Handler for scatter plot type."""

    def can_handle(self, spec: PlotSpec, data: RunData) -> bool:
        """Check if scatter plot can be generated."""
        for metric in spec.metrics:
            if metric.source == DataSource.REQUESTS and (
                data.requests is None or data.requests.empty
            ):
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create a scatter plot."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        df = self._prepare_data_for_source(x_metric.source, data)

        return self.plot_generator.create_time_series_scatter(
            df=df,
            x_col=x_metric.name,
            y_metric=y_metric.name,
            title=spec.title,
            x_label=self._get_axis_label(x_metric, available_metrics),
            y_label=self._get_axis_label(y_metric, available_metrics),
        )


@PlotTypeHandlerFactory.register(PlotType.AREA)
class AreaHandler(BaseSingleRunHandler):
    """Handler for area plot type."""

    def can_handle(self, spec: PlotSpec, data: RunData) -> bool:
        """Check if area plot can be generated."""
        for metric in spec.metrics:
            if metric.source == DataSource.REQUESTS and (
                data.requests is None or data.requests.empty
            ):
                return False

        # Special validation for dispersed throughput plot
        if spec.name == "dispersed_throughput_over_time":
            if data.requests is None or data.requests.empty:
                return False
            required_cols = [
                "request_start_ns",
                "request_end_ns",
                "output_sequence_length",
            ]
            if not all(col in data.requests.columns for col in required_cols):
                return False

        return True

    def create_plot(
        self, spec: PlotSpec, data: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create an area plot."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        # Special handling for dispersed throughput
        if y_metric.name == "throughput_tokens_per_sec":
            df = prepare_request_timeseries(data)
            throughput_df = calculate_throughput_events(df)
        else:
            throughput_df = self._prepare_data_for_source(x_metric.source, data)

        return self.plot_generator.create_time_series_area(
            df=throughput_df,
            x_col=x_metric.name,
            y_metric=y_metric.name,
            title=spec.title,
            x_label=self._get_axis_label(x_metric, available_metrics),
            y_label=self._get_axis_label(y_metric, available_metrics),
        )


@PlotTypeHandlerFactory.register(PlotType.HISTOGRAM)
class HistogramHandler(BaseSingleRunHandler):
    """Handler for histogram plot type (timeslices)."""

    def can_handle(self, spec: PlotSpec, data: RunData) -> bool:
        """Check if histogram plot can be generated."""
        for metric in spec.metrics:
            if metric.source == DataSource.TIMESLICES and (
                data.timeslices is None or data.timeslices.empty
            ):
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create a histogram plot."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        # Prepare timeslice data
        plot_df = prepare_timeslice_metrics(data, y_metric.name, y_metric.stat)

        # Get unit for y-axis label
        metric_data = data.timeslices[
            (data.timeslices["Metric"] == y_metric.name)
            & (data.timeslices["Stat"] == y_metric.stat)
        ]
        unit = metric_data["Unit"].iloc[0] if not metric_data.empty else ""
        y_label = f"{y_metric.name} ({unit})" if unit else y_metric.name

        # Check if we need warning for throughput plot
        warning_message = None
        if "throughput" in spec.name.lower():
            _, warning_message = validate_request_uniformity(data, self.logger)

        # Determine if we should use slice_duration
        use_slice_duration = (
            isinstance(spec, TimeSlicePlotSpec) and spec.use_slice_duration
        )

        return self.plot_generator.create_time_series_histogram(
            df=plot_df,
            x_col=x_metric.name,
            y_col=y_metric.stat,
            title=spec.title,
            x_label=self._get_axis_label(x_metric, available_metrics),
            y_label=y_label,
            slice_duration=data.slice_duration if use_slice_duration else None,
            warning_text=warning_message,
        )


@PlotTypeHandlerFactory.register(PlotType.DUAL_AXIS)
class DualAxisHandler(BaseSingleRunHandler):
    """Handler for dual-axis plot type."""

    def can_handle(self, spec: PlotSpec, data: RunData) -> bool:
        """Check if dual-axis plot can be generated."""
        for metric in spec.metrics:
            if metric.source == DataSource.GPU_TELEMETRY and (
                data.gpu_telemetry is None or data.gpu_telemetry.empty
            ):
                return False

        # Special validation for GPU utilization with throughput
        if spec.name == "gpu_utilization_and_throughput_over_time":
            if data.requests is None or data.requests.empty:
                return False
            required_cols = [
                "request_start_ns",
                "request_end_ns",
                "output_sequence_length",
            ]
            if not all(col in data.requests.columns for col in required_cols):
                return False

        return True

    def create_plot(
        self, spec: PlotSpec, data: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create a dual-axis plot."""
        y1_metric = next(m for m in spec.metrics if m.axis == "y")
        y2_metric = next(m for m in spec.metrics if m.axis == "y2")

        # Handle GPU utilization with throughput overlay
        if spec.name == "gpu_utilization_and_throughput_over_time":
            # Prepare GPU data
            gpu_df = aggregate_gpu_telemetry(data)

            # Prepare throughput data
            requests_df = prepare_request_timeseries(data)
            throughput_df = calculate_throughput_events(requests_df)

            if throughput_df.empty:
                raise ValueError("No throughput data available")

            return self.plot_generator.create_dual_axis_plot(
                df_primary=throughput_df,
                df_secondary=gpu_df,
                x_col_primary="timestamp_s",
                x_col_secondary="timestamp_s",
                y1_metric=y1_metric.name,
                y2_metric=y2_metric.name,
                primary_mode="lines",
                primary_line_shape="hv",
                primary_fill=None,
                secondary_mode="lines",
                secondary_line_shape=None,
                secondary_fill="tozeroy",
                active_count_col="active_requests",
                title=spec.title,
                x_label="Time (s)",
                y1_label="Output Tokens/sec",
                y2_label="GPU Utilization (%)",
            )
        else:
            raise ValueError(f"Unsupported dual-axis plot: {spec.name}")


@PlotTypeHandlerFactory.register(PlotType.SCATTER_WITH_PERCENTILES)
class ScatterWithPercentilesHandler(BaseSingleRunHandler):
    """Handler for scatter plot with percentile overlays."""

    def can_handle(self, spec: PlotSpec, data: RunData) -> bool:
        """Check if scatter with percentiles plot can be generated."""
        for metric in spec.metrics:
            if metric.source == DataSource.REQUESTS and (
                data.requests is None or data.requests.empty
            ):
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create a scatter plot with percentile overlays."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        df = self._prepare_data_for_source(x_metric.source, data)
        df_sorted = df.sort_values(x_metric.name).copy()

        # Calculate rolling percentiles
        df_sorted = calculate_rolling_percentiles(df_sorted, y_metric.name)

        return self.plot_generator.create_latency_scatter_with_percentiles(
            df=df_sorted,
            x_col=x_metric.name,
            y_metric=y_metric.name,
            percentile_cols=["p50", "p95", "p99"],
            title=spec.title,
            x_label=self._get_axis_label(x_metric, available_metrics),
            y_label=self._get_axis_label(y_metric, available_metrics),
        )
