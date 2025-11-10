# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the PlotGenerator class.

This module tests the plot generation functionality, ensuring that each plot
type is created correctly with proper styling and data handling.
"""

import pandas as pd
import plotly.graph_objects as go
import pytest

from aiperf.plot.constants import NVIDIA_GREEN, NVIDIA_WHITE
from aiperf.plot.core.plot_generator import PlotGenerator


@pytest.fixture
def plot_generator():
    """Create a PlotGenerator instance for testing."""
    return PlotGenerator()


@pytest.fixture
def multi_run_df():
    """Create sample multi-run DataFrame for testing."""
    return pd.DataFrame(
        {
            "model": ["Qwen/Qwen3-0.6B"] * 3 + ["meta-llama/Meta-Llama-3-8B"] * 3,
            "concurrency": [1, 4, 8, 1, 4, 8],
            "request_latency": [100, 150, 200, 120, 180, 250],
            "request_throughput": [10, 25, 35, 8, 20, 28],
            "time_to_first_token": [45, 55, 70, 50, 65, 85],
            "inter_token_latency": [15, 18, 22, 16, 20, 25],
            "output_token_throughput_per_user": [100, 90, 80, 95, 85, 75],
        }
    )


@pytest.fixture
def single_run_df():
    """Create sample single-run DataFrame for testing."""
    return pd.DataFrame(
        {
            "request_number": list(range(10)),
            "timestamp": [i * 0.5 for i in range(10)],
            "time_to_first_token": [45 + i * 2 for i in range(10)],
            "inter_token_latency": [18 + i * 0.5 for i in range(10)],
            "request_latency": [900 + i * 10 for i in range(10)],
        }
    )


class TestPlotGenerator:
    """Tests for PlotGenerator class."""

    def test_initialization(self, plot_generator):
        """Test that PlotGenerator can be instantiated."""
        assert isinstance(plot_generator, PlotGenerator)

    def test_create_pareto_plot_basic(self, plot_generator, multi_run_df):
        """Test basic Pareto plot creation."""
        fig = plot_generator.create_pareto_plot(
            df=multi_run_df,
            x_metric="request_latency",
            y_metric="request_throughput",
            label_by="concurrency",
            group_by="model",
        )

        # Verify return type
        assert isinstance(fig, go.Figure)

        # Verify figure has traces (data points + lines + shadows)
        assert len(fig.data) > 0

        # Verify layout properties (colors for light mode)
        assert fig.layout.plot_bgcolor == NVIDIA_WHITE
        assert fig.layout.paper_bgcolor == NVIDIA_WHITE

    def test_create_pareto_plot_custom_labels(self, plot_generator, multi_run_df):
        """Test Pareto plot with custom labels."""
        title = "Custom Title"
        x_label = "Custom X"
        y_label = "Custom Y"

        fig = plot_generator.create_pareto_plot(
            df=multi_run_df,
            x_metric="request_latency",
            y_metric="request_throughput",
            label_by="concurrency",
            group_by="model",
            title=title,
            x_label=x_label,
            y_label=y_label,
        )

        # Verify custom labels are used
        assert fig.layout.title.text == title
        assert fig.layout.xaxis.title.text == x_label
        assert fig.layout.yaxis.title.text == y_label

    def test_create_pareto_plot_no_grouping(self, plot_generator):
        """Test Pareto plot without grouping."""
        df = pd.DataFrame(
            {
                "concurrency": [1, 4, 8],
                "request_latency": [100, 150, 200],
                "request_throughput": [10, 25, 35],
            }
        )

        fig = plot_generator.create_pareto_plot(
            df=df,
            x_metric="request_latency",
            y_metric="request_throughput",
            label_by="concurrency",
            group_by=None,  # No grouping
        )

        # Should still create a valid figure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_scatter_line_plot_basic(self, plot_generator, multi_run_df):
        """Test basic scatter line plot creation."""
        fig = plot_generator.create_scatter_line_plot(
            df=multi_run_df,
            x_metric="time_to_first_token",
            y_metric="inter_token_latency",
            label_by="concurrency",
            group_by="model",
        )

        # Verify return type
        assert isinstance(fig, go.Figure)

        # Verify figure has traces
        assert len(fig.data) > 0

        # Verify styling (colors for light mode)
        assert fig.layout.plot_bgcolor == NVIDIA_WHITE

    def test_create_scatter_line_plot_auto_labels(self, plot_generator, multi_run_df):
        """Test scatter line plot with auto-generated labels."""
        fig = plot_generator.create_scatter_line_plot(
            df=multi_run_df,
            x_metric="time_to_first_token",
            y_metric="inter_token_latency",
            label_by="concurrency",
            group_by="model",
        )

        # Verify auto-generated labels contain metric names
        assert "Time To First Token" in fig.layout.xaxis.title.text
        assert "Inter Token Latency" in fig.layout.yaxis.title.text

    def test_create_time_series_scatter(self, plot_generator, single_run_df):
        """Test time series scatter plot creation."""
        fig = plot_generator.create_time_series_scatter(
            df=single_run_df,
            x_col="request_number",
            y_metric="time_to_first_token",
        )

        # Verify return type
        assert isinstance(fig, go.Figure)

        # Verify has scatter trace
        assert len(fig.data) > 0
        assert fig.data[0].mode == "markers"

        # Verify styling
        assert fig.layout.plot_bgcolor == NVIDIA_WHITE
        assert fig.layout.hovermode == "x unified"

    def test_create_time_series_scatter_custom_labels(
        self, plot_generator, single_run_df
    ):
        """Test time series scatter with custom labels."""
        title = "Custom Time Series"
        x_label = "Time"
        y_label = "Latency (ms)"

        fig = plot_generator.create_time_series_scatter(
            df=single_run_df,
            x_col="request_number",
            y_metric="time_to_first_token",
            title=title,
            x_label=x_label,
            y_label=y_label,
        )

        # Verify custom labels
        assert fig.layout.title.text == title
        assert fig.layout.xaxis.title.text == x_label
        assert fig.layout.yaxis.title.text == y_label

    def test_create_time_series_area(self, plot_generator, single_run_df):
        """Test time series area plot creation."""
        fig = plot_generator.create_time_series_area(
            df=single_run_df,
            x_col="timestamp",
            y_metric="request_latency",
        )

        # Verify return type
        assert isinstance(fig, go.Figure)

        # Verify has filled area
        assert len(fig.data) > 0
        assert fig.data[0].fill == "tozeroy"
        assert fig.data[0].mode == "lines"

        # Verify NVIDIA green color
        assert NVIDIA_GREEN in fig.data[0].line.color

    def test_create_time_series_area_auto_labels(self, plot_generator, single_run_df):
        """Test time series area with auto-generated labels."""
        fig = plot_generator.create_time_series_area(
            df=single_run_df,
            x_col="timestamp",
            y_metric="request_latency",
        )

        # Verify auto-generated labels
        assert "Timestamp" in fig.layout.xaxis.title.text
        assert "Request Latency" in fig.layout.yaxis.title.text

    def test_plots_have_proper_height(self, plot_generator, multi_run_df):
        """Test that all plots have the expected height."""
        plots = [
            plot_generator.create_pareto_plot(
                multi_run_df, "request_latency", "request_throughput"
            ),
            plot_generator.create_scatter_line_plot(
                multi_run_df, "time_to_first_token", "inter_token_latency"
            ),
        ]

        for fig in plots:
            assert fig.layout.height == 600

    def test_plots_have_nvidia_branding(self, plot_generator, multi_run_df):
        """Test that plots use NVIDIA brand colors."""
        fig = plot_generator.create_pareto_plot(
            df=multi_run_df,
            x_metric="request_latency",
            y_metric="request_throughput",
        )

        # Check layout colors (light mode by default)
        assert fig.layout.plot_bgcolor == NVIDIA_WHITE
        assert fig.layout.paper_bgcolor == NVIDIA_WHITE

    def test_empty_dataframe_handling(self, plot_generator):
        """Test that generator handles empty DataFrames gracefully."""
        empty_df = pd.DataFrame()

        # Should not raise an exception
        try:
            fig = plot_generator.create_scatter_line_plot(
                df=empty_df,
                x_metric="request_latency",
                y_metric="request_throughput",
            )
            assert isinstance(fig, go.Figure)
        except KeyError:
            # Expected if columns don't exist in empty DataFrame
            pass

    def test_single_data_point(self, plot_generator):
        """Test plots with single data point."""
        df = pd.DataFrame(
            {
                "concurrency": [1],
                "request_latency": [100],
                "request_throughput": [10],
            }
        )

        fig = plot_generator.create_pareto_plot(
            df=df,
            x_metric="request_latency",
            y_metric="request_throughput",
            label_by="concurrency",
            group_by=None,
        )

        # Should create valid figure with single point
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_with_missing_group_column(self, plot_generator):
        """Test plot when group_by column doesn't exist."""
        df = pd.DataFrame(
            {
                "concurrency": [1, 4, 8],
                "request_latency": [100, 150, 200],
                "request_throughput": [10, 25, 35],
            }
        )

        fig = plot_generator.create_pareto_plot(
            df=df,
            x_metric="request_latency",
            y_metric="request_throughput",
            label_by="concurrency",
            group_by="nonexistent_column",  # Column doesn't exist
        )

        # Should fall back to treating all as single group
        assert isinstance(fig, go.Figure)

    def test_dynamic_model_color_assignment(self, plot_generator):
        """Test that colors are assigned dynamically to any model names."""
        # Use arbitrary model names (not hardcoded)
        df = pd.DataFrame(
            {
                "model": ["ModelA", "ModelB", "ModelC"] * 2,
                "concurrency": [1, 1, 1, 4, 4, 4],
                "request_latency": [100, 110, 120, 150, 160, 170],
                "request_throughput": [10, 9, 8, 25, 23, 21],
            }
        )

        fig = plot_generator.create_pareto_plot(
            df=df,
            x_metric="request_latency",
            y_metric="request_throughput",
            label_by="concurrency",
            group_by="model",
        )

        # Should work with any model names
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

        # Test the helper method directly
        models = ["ModelA", "ModelB", "ModelC"]
        color_map = plot_generator._assign_group_colors(models)

        # Verify all models get colors
        assert len(color_map) == 3
        assert "ModelA" in color_map
        assert "ModelB" in color_map
        assert "ModelC" in color_map

        # Verify colors are hex codes
        for color in color_map.values():
            assert isinstance(color, str)
            assert color.startswith("#")

    def test_color_consistency_across_models(self, plot_generator):
        """Test that same model gets same color across different calls."""
        models1 = ["ModelX", "ModelY", "ModelZ"]
        colors1 = plot_generator._assign_group_colors(models1)

        models2 = ["ModelX", "ModelY", "ModelZ"]
        colors2 = plot_generator._assign_group_colors(models2)

        # Same models in same order should get same colors
        assert colors1 == colors2

    def test_color_assignment_with_many_models(self, plot_generator):
        """Test that color assignment cycles when there are more models than colors."""
        # Create more models than available colors
        models = [f"Model{i}" for i in range(15)]
        color_map = plot_generator._assign_group_colors(models)

        # All models should get a color
        assert len(color_map) == 15

        # Colors should cycle (some will repeat)
        unique_colors = set(color_map.values())
        # Should have fewer unique colors than models (due to cycling)
        assert len(unique_colors) < len(models)


class TestTimeSeriesHistogram:
    """Tests for create_time_series_histogram method."""

    @pytest.fixture
    def timeslice_df(self):
        """Create sample timeslice DataFrame for testing."""
        return pd.DataFrame(
            {
                "timeslice": [0, 1, 2, 3, 4],
                "avg": [100.5, 120.3, 115.7, 130.2, 125.8],
                "p50": [95.0, 115.0, 110.0, 125.0, 120.0],
                "p90": [150.0, 180.0, 170.0, 195.0, 185.0],
            }
        )

    def test_histogram_basic(self, plot_generator, timeslice_df):
        """Test basic histogram creation."""
        fig = plot_generator.create_time_series_histogram(
            df=timeslice_df, x_col="timeslice", y_col="avg"
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == "bar"

    def test_histogram_with_slice_duration(self, plot_generator, timeslice_df):
        """Test histogram with slice duration for time-based x-axis."""
        slice_duration = 10.0
        fig = plot_generator.create_time_series_histogram(
            df=timeslice_df,
            x_col="timeslice",
            y_col="avg",
            slice_duration=slice_duration,
        )

        assert isinstance(fig, go.Figure)
        assert fig.data[0].type == "bar"
        assert fig.data[0].width == slice_duration
        assert fig.layout.xaxis.dtick == slice_duration
        assert fig.layout.bargap == 0

    def test_histogram_with_annotations(self, plot_generator, timeslice_df):
        """Test that slice indices are annotated when slice_duration is provided."""
        fig = plot_generator.create_time_series_histogram(
            df=timeslice_df,
            x_col="timeslice",
            y_col="avg",
            slice_duration=10.0,
        )

        assert fig.layout.annotations is not None
        assert len(fig.layout.annotations) == len(timeslice_df)

        for i, annotation in enumerate(fig.layout.annotations):
            assert annotation["text"] == str(i)

    def test_histogram_with_warning_text(self, plot_generator, timeslice_df):
        """Test histogram with warning text annotation."""
        warning_text = "Warning: Non-uniform request distribution detected"
        fig = plot_generator.create_time_series_histogram(
            df=timeslice_df,
            x_col="timeslice",
            y_col="avg",
            slice_duration=10.0,
            warning_text=warning_text,
        )

        assert len(fig.layout.annotations) > len(timeslice_df)

        warning_annotation = fig.layout.annotations[-1]
        assert warning_text in warning_annotation["text"]
        assert warning_annotation["yref"] == "paper"
        assert fig.layout.margin.b == 140

    def test_histogram_custom_labels(self, plot_generator, timeslice_df):
        """Test histogram with custom labels."""
        title = "Custom Histogram Title"
        x_label = "Custom X Label"
        y_label = "Custom Y Label"

        fig = plot_generator.create_time_series_histogram(
            df=timeslice_df,
            x_col="timeslice",
            y_col="avg",
            title=title,
            x_label=x_label,
            y_label=y_label,
        )

        assert fig.layout.title.text == title
        assert fig.layout.xaxis.title.text == x_label
        assert fig.layout.yaxis.title.text == y_label

    def test_histogram_auto_labels(self, plot_generator, timeslice_df):
        """Test histogram with auto-generated labels."""
        fig = plot_generator.create_time_series_histogram(
            df=timeslice_df, x_col="timeslice", y_col="avg"
        )

        assert "avg" in fig.layout.title.text.lower()
        assert fig.layout.yaxis.title.text == "Avg"

    def test_histogram_auto_labels_with_slice_duration(
        self, plot_generator, timeslice_df
    ):
        """Test that x-axis label is 'Time (s)' when slice_duration is provided."""
        fig = plot_generator.create_time_series_histogram(
            df=timeslice_df,
            x_col="timeslice",
            y_col="avg",
            slice_duration=10.0,
        )

        assert fig.layout.xaxis.title.text == "Time (s)"

    def test_histogram_with_empty_dataframe(self, plot_generator):
        """Test histogram with empty DataFrame."""
        empty_df = pd.DataFrame({"timeslice": [], "avg": []})
        fig = plot_generator.create_time_series_histogram(
            df=empty_df, x_col="timeslice", y_col="avg"
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_histogram_marker_config_with_slice_duration(
        self, plot_generator, timeslice_df
    ):
        """Test that histogram uses transparent bars with borders when slice_duration is provided."""
        fig = plot_generator.create_time_series_histogram(
            df=timeslice_df,
            x_col="timeslice",
            y_col="avg",
            slice_duration=10.0,
        )

        marker = fig.data[0].marker
        assert "rgba(118, 185, 0, 0.7)" in marker.color
        assert marker.line.color == NVIDIA_GREEN
        assert marker.line.width == 2

    def test_histogram_marker_config_without_slice_duration(
        self, plot_generator, timeslice_df
    ):
        """Test that histogram uses solid bars when slice_duration is not provided."""
        fig = plot_generator.create_time_series_histogram(
            df=timeslice_df, x_col="timeslice", y_col="avg"
        )

        marker = fig.data[0].marker
        assert marker.color == NVIDIA_GREEN


class TestDualAxisPlots:
    """Tests for dual-axis plotting functions."""

    @pytest.fixture
    def gpu_metrics_df(self):
        """Create sample GPU metrics DataFrame for testing."""
        return pd.DataFrame(
            {
                "timestamp_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "gpu_utilization": [45.5, 67.2, 78.9, 82.3, 75.4, 68.1],
                "throughput": [100.0, 150.0, 180.0, 190.0, 170.0, 155.0],
                "power_draw_w": [120.0, 180.0, 220.0, 240.0, 210.0, 190.0],
            }
        )

    @pytest.fixture
    def gpu_memory_df(self):
        """Create sample GPU memory DataFrame for testing."""
        return pd.DataFrame(
            {
                "timestamp_s": [0.0, 1.0, 2.0, 3.0, 4.0],
                "memory_used_gb": [2.5, 4.0, 5.5, 6.0, 5.0],
                "memory_free_gb": [5.5, 4.0, 2.5, 2.0, 3.0],
            }
        )

    def test_gpu_dual_axis_plot_basic(self, plot_generator, gpu_metrics_df):
        """Test basic GPU dual-axis plot creation with separate DataFrames."""
        throughput_df = gpu_metrics_df[["timestamp_s", "throughput"]].copy()
        throughput_df["active_requests"] = [2, 3, 4, 5, 4, 3]
        gpu_df = gpu_metrics_df[["timestamp_s", "gpu_utilization"]].copy()

        fig = plot_generator.create_dual_axis_plot(
            df_primary=throughput_df,
            df_secondary=gpu_df,
            x_col_primary="timestamp_s",
            x_col_secondary="timestamp_s",
            y1_metric="throughput",
            y2_metric="gpu_utilization",
            primary_mode="lines",
            primary_line_shape="hv",
            primary_fill=None,
            secondary_mode="lines",
            secondary_line_shape=None,
            secondary_fill="tozeroy",
            active_count_col="active_requests",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

        assert fig.data[0].yaxis == "y"
        assert fig.data[1].yaxis == "y2"

        assert fig.data[0].line.shape == "hv"
        assert fig.data[1].fill == "tozeroy"

    def test_gpu_dual_axis_plot_custom_labels(self, plot_generator, gpu_metrics_df):
        """Test GPU dual-axis plot with custom labels."""
        title = "Throughput with GPU Utilization"
        x_label = "Time"
        y1_label = "Tokens/s"
        y2_label = "GPU %"

        throughput_df = gpu_metrics_df[["timestamp_s", "throughput"]].copy()
        gpu_df = gpu_metrics_df[["timestamp_s", "gpu_utilization"]].copy()

        fig = plot_generator.create_dual_axis_plot(
            df_primary=throughput_df,
            df_secondary=gpu_df,
            x_col_primary="timestamp_s",
            x_col_secondary="timestamp_s",
            y1_metric="throughput",
            y2_metric="gpu_utilization",
            primary_mode="lines",
            primary_line_shape="hv",
            primary_fill=None,
            secondary_mode="lines",
            secondary_line_shape=None,
            secondary_fill="tozeroy",
            title=title,
            x_label=x_label,
            y1_label=y1_label,
            y2_label=y2_label,
        )

        assert fig.layout.title.text == title
        assert fig.layout.xaxis.title.text == x_label
        assert fig.layout.yaxis.title.text == y1_label
        assert fig.layout.yaxis2.title.text == y2_label

    def test_gpu_dual_axis_plot_auto_labels(self, plot_generator, gpu_metrics_df):
        """Test dual-axis plot with auto-generated labels."""
        throughput_df = gpu_metrics_df[["timestamp_s", "throughput"]].copy()
        gpu_df = gpu_metrics_df[["timestamp_s", "gpu_utilization"]].copy()

        fig = plot_generator.create_dual_axis_plot(
            df_primary=throughput_df,
            df_secondary=gpu_df,
            x_col_primary="timestamp_s",
            x_col_secondary="timestamp_s",
            y1_metric="throughput",
            y2_metric="gpu_utilization",
            primary_mode="lines",
            primary_line_shape="hv",
            primary_fill=None,
            secondary_mode="lines",
            secondary_line_shape=None,
            secondary_fill="tozeroy",
        )

        assert "Throughput" in fig.layout.title.text
        assert "Gpu Utilization" in fig.layout.title.text
        assert fig.layout.xaxis.title.text == "Time (s)"

    def test_gpu_dual_axis_plot_styling(self, plot_generator, gpu_metrics_df):
        """Test dual-axis plot styling (colors, line widths)."""
        throughput_df = gpu_metrics_df[["timestamp_s", "throughput"]].copy()
        gpu_df = gpu_metrics_df[["timestamp_s", "gpu_utilization"]].copy()

        fig = plot_generator.create_dual_axis_plot(
            df_primary=throughput_df,
            df_secondary=gpu_df,
            x_col_primary="timestamp_s",
            x_col_secondary="timestamp_s",
            y1_metric="throughput",
            y2_metric="gpu_utilization",
            primary_mode="lines",
            primary_line_shape="hv",
            primary_fill=None,
            secondary_mode="lines",
            secondary_line_shape=None,
            secondary_fill="tozeroy",
        )

        assert fig.data[0].line.color == NVIDIA_GREEN
        assert fig.data[0].line.width == 2
        assert fig.data[1].line.width == 2

    def test_gpu_dual_axis_layout(self, plot_generator, gpu_metrics_df):
        """Test that secondary y-axis is configured correctly with theme colors."""
        throughput_df = gpu_metrics_df[["timestamp_s", "throughput"]].copy()
        gpu_df = gpu_metrics_df[["timestamp_s", "gpu_utilization"]].copy()

        fig = plot_generator.create_dual_axis_plot(
            df_primary=throughput_df,
            df_secondary=gpu_df,
            x_col_primary="timestamp_s",
            x_col_secondary="timestamp_s",
            y1_metric="throughput",
            y2_metric="gpu_utilization",
            primary_mode="lines",
            primary_line_shape="hv",
            primary_fill=None,
            secondary_mode="lines",
            secondary_line_shape=None,
            secondary_fill="tozeroy",
        )

        assert fig.layout.yaxis2 is not None
        assert fig.layout.yaxis2.overlaying == "y"
        assert fig.layout.yaxis2.side == "right"

        # Verify theme consistency for yaxis2
        from aiperf.plot.constants import LIGHT_THEME_COLORS

        assert fig.layout.yaxis2.gridcolor == LIGHT_THEME_COLORS["grid"]
        assert fig.layout.yaxis2.linecolor == LIGHT_THEME_COLORS["border"]
        assert fig.layout.yaxis2.color == LIGHT_THEME_COLORS["text"]

    def test_gpu_dual_axis_with_empty_dataframe(self, plot_generator):
        """Test dual-axis plot with empty DataFrame."""
        empty_throughput_df = pd.DataFrame({"timestamp_s": [], "throughput": []})
        empty_gpu_df = pd.DataFrame({"timestamp_s": [], "gpu_utilization": []})
        fig = plot_generator.create_dual_axis_plot(
            df_primary=empty_throughput_df,
            df_secondary=empty_gpu_df,
            x_col_primary="timestamp_s",
            x_col_secondary="timestamp_s",
            y1_metric="throughput",
            y2_metric="gpu_utilization",
            primary_mode="lines",
            primary_line_shape="hv",
            primary_fill=None,
            secondary_mode="lines",
            secondary_line_shape=None,
            secondary_fill="tozeroy",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_gpu_memory_stacked_area_basic(self, plot_generator, gpu_memory_df):
        """Test basic GPU memory stacked area plot creation."""
        fig = plot_generator.create_gpu_memory_stacked_area(
            df=gpu_memory_df,
            x_col="timestamp_s",
            used_col="memory_used_gb",
            free_col="memory_free_gb",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

        assert fig.data[0].fill == "tozeroy"
        assert fig.data[1].fill == "tonexty"
        assert fig.data[0].stackgroup == "one"
        assert fig.data[1].stackgroup == "one"

    def test_gpu_memory_stacked_area_custom_labels(self, plot_generator, gpu_memory_df):
        """Test GPU memory stacked area plot with custom labels."""
        title = "Memory Usage"
        x_label = "Time"
        y_label = "Memory (GB)"

        fig = plot_generator.create_gpu_memory_stacked_area(
            df=gpu_memory_df,
            x_col="timestamp_s",
            used_col="memory_used_gb",
            free_col="memory_free_gb",
            title=title,
            x_label=x_label,
            y_label=y_label,
        )

        assert fig.layout.title.text == title
        assert fig.layout.xaxis.title.text == x_label
        assert fig.layout.yaxis.title.text == y_label

    def test_gpu_memory_stacked_area_auto_labels(self, plot_generator, gpu_memory_df):
        """Test GPU memory stacked area plot with auto-generated labels."""
        fig = plot_generator.create_gpu_memory_stacked_area(
            df=gpu_memory_df,
            x_col="timestamp_s",
            used_col="memory_used_gb",
            free_col="memory_free_gb",
        )

        assert fig.layout.title.text == "GPU Memory Usage Over Time"
        assert fig.layout.xaxis.title.text == "Time (s)"
        assert fig.layout.yaxis.title.text == "Memory (GB)"

    def test_gpu_memory_stacked_area_traces(self, plot_generator, gpu_memory_df):
        """Test that stacked area has correct trace names and order."""
        fig = plot_generator.create_gpu_memory_stacked_area(
            df=gpu_memory_df,
            x_col="timestamp_s",
            used_col="memory_used_gb",
            free_col="memory_free_gb",
        )

        assert fig.data[0].name == "Used"
        assert fig.data[1].name == "Free"

    def test_gpu_memory_stacked_area_with_empty_dataframe(self, plot_generator):
        """Test GPU memory stacked area plot with empty DataFrame."""
        empty_df = pd.DataFrame(
            {"timestamp_s": [], "memory_used_gb": [], "memory_free_gb": []}
        )
        fig = plot_generator.create_gpu_memory_stacked_area(
            df=empty_df,
            x_col="timestamp_s",
            used_col="memory_used_gb",
            free_col="memory_free_gb",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2


class TestLatencyScatterWithPercentiles:
    """Tests for create_latency_scatter_with_percentiles method."""

    @pytest.fixture
    def latency_df(self):
        """Create sample latency DataFrame with percentiles for testing."""
        return pd.DataFrame(
            {
                "timestamp": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                "request_latency": [900, 920, 950, 880, 1100, 910, 940, 930, 960, 920],
                "p50": [900, 910, 920, 915, 920, 915, 920, 920, 925, 920],
                "p95": [900, 920, 950, 950, 1100, 1100, 1100, 1100, 1100, 1100],
                "p99": [900, 920, 950, 950, 1100, 1100, 1100, 1100, 1100, 1100],
            }
        )

    def test_latency_scatter_with_percentiles_basic(self, plot_generator, latency_df):
        """Test basic latency scatter with percentiles plot creation."""
        fig = plot_generator.create_latency_scatter_with_percentiles(
            df=latency_df,
            x_col="timestamp",
            y_metric="request_latency",
            percentile_cols=["p50", "p95", "p99"],
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4

        assert fig.data[0].mode == "markers"
        assert fig.data[0].name == "Individual Requests"

        assert fig.data[1].mode == "lines"
        assert fig.data[1].name == "P50"

        assert fig.data[2].mode == "lines"
        assert fig.data[2].name == "P95"

        assert fig.data[3].mode == "lines"
        assert fig.data[3].name == "P99"

    def test_latency_scatter_with_percentiles_custom_labels(
        self, plot_generator, latency_df
    ):
        """Test latency scatter with percentiles plot with custom labels."""
        title = "Custom Latency Plot"
        x_label = "Time (s)"
        y_label = "Latency (ms)"

        fig = plot_generator.create_latency_scatter_with_percentiles(
            df=latency_df,
            x_col="timestamp",
            y_metric="request_latency",
            percentile_cols=["p50", "p95", "p99"],
            title=title,
            x_label=x_label,
            y_label=y_label,
        )

        assert fig.layout.title.text == title
        assert fig.layout.xaxis.title.text == x_label
        assert fig.layout.yaxis.title.text == y_label

    def test_latency_scatter_with_percentiles_auto_labels(
        self, plot_generator, latency_df
    ):
        """Test latency scatter with percentiles plot with auto-generated labels."""
        fig = plot_generator.create_latency_scatter_with_percentiles(
            df=latency_df,
            x_col="timestamp",
            y_metric="request_latency",
            percentile_cols=["p50", "p95", "p99"],
        )

        assert "Request Latency" in fig.layout.title.text
        assert "Percentiles" in fig.layout.title.text
        assert "Timestamp" in fig.layout.xaxis.title.text
        assert "Request Latency" in fig.layout.yaxis.title.text

    def test_latency_scatter_with_percentiles_colors(self, plot_generator, latency_df):
        """Test that percentile lines use NVIDIA color palette."""
        fig = plot_generator.create_latency_scatter_with_percentiles(
            df=latency_df,
            x_col="timestamp",
            y_metric="request_latency",
            percentile_cols=["p50", "p95", "p99"],
        )

        assert fig.data[1].line.color == NVIDIA_GREEN
        assert fig.data[2].line.color is not None
        assert fig.data[3].line.color is not None

        for i in range(1, 4):
            assert isinstance(fig.data[i].line.color, str)
            assert fig.data[i].line.color.startswith("#")

    def test_latency_scatter_with_percentiles_styling(self, plot_generator, latency_df):
        """Test styling of scatter points and percentile lines."""
        fig = plot_generator.create_latency_scatter_with_percentiles(
            df=latency_df,
            x_col="timestamp",
            y_metric="request_latency",
            percentile_cols=["p50", "p95", "p99"],
        )

        assert fig.data[0].marker.opacity == 0.4
        assert fig.data[0].marker.size == 6

        for i in range(1, 4):
            assert fig.data[i].line.width == 2.5

    def test_latency_scatter_with_percentiles_subset(self, plot_generator, latency_df):
        """Test with subset of percentiles."""
        fig = plot_generator.create_latency_scatter_with_percentiles(
            df=latency_df,
            x_col="timestamp",
            y_metric="request_latency",
            percentile_cols=["p50", "p95"],
        )

        assert len(fig.data) == 3
        assert fig.data[0].name == "Individual Requests"
        assert fig.data[1].name == "P50"
        assert fig.data[2].name == "P95"

    def test_latency_scatter_with_missing_percentile_column(
        self, plot_generator, latency_df
    ):
        """Test that missing percentile columns are gracefully skipped."""
        fig = plot_generator.create_latency_scatter_with_percentiles(
            df=latency_df,
            x_col="timestamp",
            y_metric="request_latency",
            percentile_cols=["p50", "p90", "p99"],
        )

        assert len(fig.data) == 3
        assert fig.data[0].name == "Individual Requests"
        assert fig.data[1].name == "P50"
        assert fig.data[2].name == "P99"

    def test_latency_scatter_with_percentiles_hover_mode(
        self, plot_generator, latency_df
    ):
        """Test that hover mode is set to x unified."""
        fig = plot_generator.create_latency_scatter_with_percentiles(
            df=latency_df,
            x_col="timestamp",
            y_metric="request_latency",
            percentile_cols=["p50", "p95", "p99"],
        )

        assert fig.layout.hovermode == "x unified"

    def test_latency_scatter_with_empty_dataframe(self, plot_generator):
        """Test latency scatter with percentiles plot with empty DataFrame."""
        empty_df = pd.DataFrame(
            {
                "timestamp": [],
                "request_latency": [],
                "p50": [],
                "p95": [],
                "p99": [],
            }
        )
        fig = plot_generator.create_latency_scatter_with_percentiles(
            df=empty_df,
            x_col="timestamp",
            y_metric="request_latency",
            percentile_cols=["p50", "p95", "p99"],
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4
