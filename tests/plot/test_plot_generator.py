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

from aiperf.plot.constants import NVIDIA_DARK_BG, NVIDIA_GREEN, NVIDIA_TEXT_LIGHT
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

        # Verify layout properties (colors, not template object)
        assert fig.layout.plot_bgcolor == NVIDIA_DARK_BG
        assert fig.layout.paper_bgcolor == NVIDIA_DARK_BG

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

        # Verify styling (colors, not template object)
        assert fig.layout.plot_bgcolor == NVIDIA_DARK_BG

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
        assert fig.layout.plot_bgcolor == NVIDIA_DARK_BG
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

        # Check layout colors
        assert fig.layout.plot_bgcolor == NVIDIA_DARK_BG
        assert fig.layout.paper_bgcolor == NVIDIA_DARK_BG
        assert fig.layout.title.font.color == NVIDIA_TEXT_LIGHT

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
        color_map = plot_generator._assign_model_colors(models)

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
        colors1 = plot_generator._assign_model_colors(models1)

        models2 = ["ModelX", "ModelY", "ModelZ"]
        colors2 = plot_generator._assign_model_colors(models2)

        # Same models in same order should get same colors
        assert colors1 == colors2

    def test_color_assignment_with_many_models(self, plot_generator):
        """Test that color assignment cycles when there are more models than colors."""
        # Create more models than available colors
        models = [f"Model{i}" for i in range(15)]
        color_map = plot_generator._assign_model_colors(models)

        # All models should get a color
        assert len(color_map) == 15

        # Colors should cycle (some will repeat)
        unique_colors = set(color_map.values())
        # Should have fewer unique colors than models (due to cycling)
        assert len(unique_colors) < len(models)
