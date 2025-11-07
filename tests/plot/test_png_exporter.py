# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the PNG Exporter classes.

This module tests the PNG export functionality, ensuring that plots are
correctly generated and saved as PNG files with proper metadata.
"""

from pathlib import Path

import pandas as pd
import pytest

from aiperf.common.models.record_models import MetricResult
from aiperf.plot.core.data_loader import RunData, RunMetadata
from aiperf.plot.exporters.png import MultiRunPNGExporter, SingleRunPNGExporter


@pytest.fixture
def multi_run_exporter(tmp_path):
    """Create a MultiRunPNGExporter instance for testing."""
    output_dir = tmp_path / "plot_export"
    return MultiRunPNGExporter(output_dir)


@pytest.fixture
def single_run_exporter(tmp_path):
    """Create a SingleRunPNGExporter instance for testing."""
    output_dir = tmp_path / "plot_export"
    return SingleRunPNGExporter(output_dir)


@pytest.fixture
def sample_multi_run_data(tmp_path):
    """Create sample multi-run data for testing."""
    return [
        RunData(
            metadata=RunMetadata(
                run_name="run_001",
                run_path=tmp_path / "run_001",
                model="Qwen/Qwen3-0.6B",
                concurrency=1,
            ),
            requests=None,
            aggregated={
                "request_latency": {"p50": 100.0, "avg": 105.0, "unit": "ms"},
                "request_throughput": {"avg": 10.0, "unit": "req/s"},
                "time_to_first_token": {"p50": 45.0, "unit": "ms"},
                "inter_token_latency": {"p50": 18.0, "unit": "ms"},
                "output_token_throughput_per_user": {
                    "avg": 100.0,
                    "unit": "tokens/s/user",
                },
            },
            timeslices=None,
            slice_duration=None,
        ),
        RunData(
            metadata=RunMetadata(
                run_name="run_002",
                run_path=tmp_path / "run_002",
                model="Qwen/Qwen3-0.6B",
                concurrency=4,
            ),
            requests=None,
            aggregated={
                "request_latency": {"p50": 150.0, "avg": 155.0, "unit": "ms"},
                "request_throughput": {"avg": 25.0, "unit": "req/s"},
                "time_to_first_token": {"p50": 55.0, "unit": "ms"},
                "inter_token_latency": {"p50": 20.0, "unit": "ms"},
                "output_token_throughput_per_user": {
                    "avg": 90.0,
                    "unit": "tokens/s/user",
                },
            },
            timeslices=None,
            slice_duration=None,
        ),
    ]


@pytest.fixture
def sample_single_run_data(tmp_path):
    """Create sample single-run data for testing."""
    # Create per-request DataFrame
    per_request_data = pd.DataFrame(
        {
            "request_end_ns": [1000000000000 + i * 500000000 for i in range(10)],
            "time_to_first_token": [45.0 + i * 2 for i in range(10)],
            "inter_token_latency": [18.0 + i * 0.5 for i in range(10)],
            "request_latency": [900.0 + i * 10 for i in range(10)],
        }
    )

    return RunData(
        metadata=RunMetadata(
            run_name="run_003",
            run_path=tmp_path / "run_003",
            model="Qwen/Qwen3-0.6B",
            concurrency=8,
        ),
        requests=per_request_data,
        aggregated={
            "request_latency": {"p50": 200.0, "avg": 205.0, "unit": "ms"},
            "request_throughput": {"avg": 35.0, "unit": "req/s"},
        },
        timeslices=None,
    )


@pytest.fixture
def sample_timeslice_data():
    """Create sample timeslice data for testing."""
    return pd.DataFrame(
        {
            "Timeslice": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "Metric": [
                "Time to First Token",
                "Time to First Token",
                "Time to First Token",
                "Time to First Token",
                "Time to First Token",
                "Time to First Token",
                "Time to First Token",
                "Time to First Token",
                "Time to First Token",
            ],
            "Unit": ["ms"] * 9,
            "Stat": ["avg", "min", "max", "avg", "min", "max", "avg", "min", "max"],
            "Value": [45.0, 30.0, 60.0, 47.0, 32.0, 65.0, 46.0, 31.0, 62.0],
        }
    )


@pytest.fixture
def sample_available_metrics():
    """Create sample available metrics dictionary."""
    return {
        "display_names": {
            "request_latency": "Request Latency",
            "request_throughput": "Request Throughput",
            "time_to_first_token": "Time to First Token",
            "inter_token_latency": "Inter Token Latency",
            "output_token_throughput_per_user": "Output Token Throughput per User",
        },
        "units": {
            "request_latency": "ms",
            "request_throughput": "req/s",
            "time_to_first_token": "ms",
            "inter_token_latency": "ms",
            "output_token_throughput_per_user": "tokens/s/user",
        },
    }


class TestMultiRunPNGExporter:
    """Tests for MultiRunPNGExporter class."""

    def test_initialization(self, multi_run_exporter):
        """Test that MultiRunPNGExporter can be instantiated."""
        assert isinstance(multi_run_exporter, MultiRunPNGExporter)
        assert isinstance(multi_run_exporter.output_dir, Path)

    def test_export_multi_run_creates_files(
        self,
        multi_run_exporter,
        sample_multi_run_data,
        sample_available_metrics,
    ):
        """Test that multi-run export creates PNG files."""
        generated_files = multi_run_exporter.export(
            sample_multi_run_data, sample_available_metrics
        )

        # Should generate 3 plots for multi-run
        assert len(generated_files) == 3

        # Check that files exist
        for file_path in generated_files:
            assert file_path.exists()
            assert file_path.suffix == ".png"

    def test_export_multi_run_creates_expected_plots(
        self,
        multi_run_exporter,
        sample_multi_run_data,
        sample_available_metrics,
    ):
        """Test that expected plot files are created."""
        generated_files = multi_run_exporter.export(
            sample_multi_run_data, sample_available_metrics
        )

        # Get filenames
        filenames = {f.name for f in generated_files}

        # Check expected files
        assert "pareto_curve.png" in filenames
        assert "ttft_vs_throughput.png" in filenames
        assert "throughput_per_user_vs_concurrency.png" in filenames

    def test_export_multi_run_creates_summary(
        self,
        multi_run_exporter,
        sample_multi_run_data,
        sample_available_metrics,
    ):
        """Test that summary file is created."""
        multi_run_exporter.export(sample_multi_run_data, sample_available_metrics)

        summary_path = multi_run_exporter.output_dir / "summary.txt"
        assert summary_path.exists()

        # Check summary content
        content = summary_path.read_text()
        assert "AIPerf Plot Export Summary" in content
        assert "Generated 3 plots" in content

    def test_runs_to_dataframe_with_metric_result_objects(
        self, multi_run_exporter, tmp_path
    ):
        """Test conversion with MetricResult objects instead of dicts."""
        runs = [
            RunData(
                metadata=RunMetadata(
                    run_name="run_001",
                    run_path=tmp_path / "run_001",
                    model="TestModel",
                    concurrency=4,
                ),
                requests=None,
                aggregated={
                    "request_latency": MetricResult(
                        tag="request_latency",
                        header="Request Latency",
                        unit="ms",
                        p50=100.0,
                        avg=105.0,
                    ),
                    "request_throughput": MetricResult(
                        tag="request_throughput",
                        header="Request Throughput",
                        unit="req/s",
                        avg=25.0,
                    ),
                },
                timeslices=None,
                slice_duration=None,
            )
        ]

        df = multi_run_exporter._runs_to_dataframe(
            runs, {"display_names": {}, "units": {}}
        )

        # Verify DataFrame structure
        assert len(df) == 1
        assert "model" in df.columns
        assert "concurrency" in df.columns
        assert "request_latency" in df.columns
        assert "request_throughput" in df.columns

        # Verify values were extracted correctly from MetricResult objects
        assert df["request_latency"].iloc[0] == 100.0  # p50 preferred
        assert df["request_throughput"].iloc[0] == 25.0  # avg fallback

    def test_runs_to_dataframe_with_mixed_types(self, multi_run_exporter, tmp_path):
        """Test conversion with both MetricResult objects and dicts."""
        runs = [
            RunData(
                metadata=RunMetadata(
                    run_name="run_001",
                    run_path=tmp_path / "run_001",
                    model="TestModel",
                    concurrency=4,
                ),
                requests=None,
                aggregated={
                    "request_latency": MetricResult(
                        tag="request_latency",
                        header="Request Latency",
                        unit="ms",
                        p50=100.0,
                    ),
                    "request_throughput": {
                        "avg": 25.0,
                        "unit": "req/s",
                    },  # Dict fallback
                },
                timeslices=None,
                slice_duration=None,
            )
        ]

        df = multi_run_exporter._runs_to_dataframe(
            runs, {"display_names": {}, "units": {}}
        )

        # Both MetricResult and dict should work
        assert df["request_latency"].iloc[0] == 100.0
        assert df["request_throughput"].iloc[0] == 25.0

    def test_runs_to_dataframe(
        self, multi_run_exporter, sample_multi_run_data, sample_available_metrics
    ):
        """Test conversion of runs to DataFrame."""
        df = multi_run_exporter._runs_to_dataframe(
            sample_multi_run_data, sample_available_metrics
        )

        # Check DataFrame structure
        assert len(df) == 2  # Two runs
        assert "model" in df.columns
        assert "concurrency" in df.columns
        assert "request_latency" in df.columns
        assert "request_throughput" in df.columns

        # Check values
        assert df["concurrency"].tolist() == [1, 4]
        assert df["request_latency"].tolist() == [100.0, 150.0]  # p50 values


class TestSingleRunPNGExporter:
    """Tests for SingleRunPNGExporter class."""

    def test_initialization(self, single_run_exporter):
        """Test that SingleRunPNGExporter can be instantiated."""
        assert isinstance(single_run_exporter, SingleRunPNGExporter)
        assert isinstance(single_run_exporter.output_dir, Path)

    def test_export_single_run_creates_files(
        self,
        single_run_exporter,
        sample_single_run_data,
        sample_available_metrics,
    ):
        """Test that single-run export creates PNG files."""
        generated_files = single_run_exporter.export(
            sample_single_run_data, sample_available_metrics
        )

        # Should generate 3 plots for single-run
        assert len(generated_files) == 3

        # Check that files exist
        for file_path in generated_files:
            assert file_path.exists()
            assert file_path.suffix == ".png"

    def test_export_single_run_creates_expected_plots(
        self,
        single_run_exporter,
        sample_single_run_data,
        sample_available_metrics,
    ):
        """Test that expected plot files are created for single run."""
        generated_files = single_run_exporter.export(
            sample_single_run_data, sample_available_metrics
        )

        # Get filenames
        filenames = {f.name for f in generated_files}

        # Check expected files
        assert "ttft_over_time.png" in filenames
        assert "itl_over_time.png" in filenames
        assert "latency_over_time.png" in filenames

    def test_export_single_run_with_no_per_request_data(
        self,
        single_run_exporter,
        sample_available_metrics,
        tmp_path,
    ):
        """Test handling of single run with no per-request data."""
        run_data = RunData(
            metadata=RunMetadata(
                run_name="test_run",
                run_path=tmp_path / "test_run",
                model="Test",
                concurrency=1,
            ),
            requests=None,  # No per-request data
            aggregated={},
            timeslices=None,
            slice_duration=None,
        )

        generated_files = single_run_exporter.export(run_data, sample_available_metrics)

        # Should return empty list when no data available
        assert len(generated_files) == 0

    def test_per_request_to_dataframe(
        self, single_run_exporter, sample_single_run_data
    ):
        """Test conversion of per-request data to DataFrame."""
        df = single_run_exporter._per_request_to_dataframe(sample_single_run_data)

        # Check DataFrame structure
        assert len(df) == 10  # 10 requests
        assert "request_number" in df.columns
        assert "timestamp" in df.columns
        assert "time_to_first_token" in df.columns

        # Check timestamp normalization (should start from 0)
        assert df["timestamp"].min() == 0.0

    def test_get_metric_label(self, single_run_exporter, sample_available_metrics):
        """Test metric label formatting."""
        # With stat
        label = single_run_exporter._get_metric_label(
            "request_latency", "p50", sample_available_metrics
        )
        assert "Request Latency" in label
        assert "P50" in label
        assert "ms" in label

        # Without stat
        label = single_run_exporter._get_metric_label(
            "request_latency", None, sample_available_metrics
        )
        assert "Request Latency" in label
        assert "P50" not in label
        assert "ms" in label

    def test_get_metric_label_unknown_metric(self, single_run_exporter):
        """Test metric label for unknown metric."""
        label = single_run_exporter._get_metric_label(
            "unknown_metric", "p50", {"display_names": {}, "units": {}}
        )

        # Should use formatted metric tag as fallback
        assert "Unknown Metric" in label

    def test_export_single_run_with_timeslice_data(
        self,
        single_run_exporter,
        sample_single_run_data,
        sample_timeslice_data,
        sample_available_metrics,
        tmp_path,
    ):
        """Test that timeslice plots are generated when timeslice data is available."""
        # Add timeslice data to the run
        run_with_timeslices = RunData(
            metadata=sample_single_run_data.metadata,
            requests=sample_single_run_data.requests,
            aggregated=sample_single_run_data.aggregated,
            timeslices=sample_timeslice_data,
            slice_duration=10.0,
        )

        generated_files = single_run_exporter.export(
            run_with_timeslices, sample_available_metrics
        )

        # Should generate 3 regular plots + 1 timeslice plot (only TTFT has data)
        assert len(generated_files) == 4

        # Check that timeslice plot is in the generated files
        filenames = {f.name for f in generated_files}
        assert "timeslices_ttft.png" in filenames

        # Validate that the timeslice plot has correct annotations for slice labels
        ttft_data = sample_timeslice_data[
            sample_timeslice_data["Metric"] == "Time to First Token"
        ]
        unique_slices = ttft_data["Timeslice"].unique()

        # Create a figure to test annotation content
        plot_df = ttft_data[ttft_data["Stat"] == "avg"][["Timeslice", "Value"]]
        plot_df = plot_df.rename(columns={"Value": "avg"})
        fig = single_run_exporter.plot_generator.create_time_series_histogram(
            df=plot_df,
            x_col="Timeslice",
            y_col="avg",
            title="Test",
            x_label="Time (s)",
            y_label="TTFT (ms)",
            slice_duration=10.0,
        )

        # Verify annotations exist
        assert hasattr(fig.layout, "annotations"), "Figure should have annotations"
        assert len(fig.layout.annotations) > 0, "Should have slice label annotations"

        # Verify correct number of annotations (one per slice)
        assert len(fig.layout.annotations) == len(unique_slices), (
            f"Should have {len(unique_slices)} annotations"
        )

        # Verify annotation content (should be just numbers, not "Slice X")
        annotation_texts = [ann.text for ann in fig.layout.annotations]
        for slice_idx in unique_slices:
            expected_text = str(int(slice_idx))
            assert expected_text in annotation_texts, (
                f"Should have annotation '{expected_text}'"
            )

        # Verify annotation positioning (should be at top of bars)
        for ann in fig.layout.annotations:
            assert ann.yref == "y", "Annotations should use data coordinates"
            assert ann.y > 0, "Annotations should be at bar height (> 0)"
            assert hasattr(ann, "yshift"), "Annotations should have yshift"
            assert ann.yshift > 0, "Annotations should be shifted up above bar"

        # Verify legend exists with slice explanation
        assert fig.layout.showlegend is True, "Legend should be shown"
        legend_names = [trace.name for trace in fig.data if trace.showlegend]
        assert any(
            "bar numbers" in name.lower() and "slice index" in name.lower()
            for name in legend_names
        ), "Legend should explain bar numbers and slice index"

    def test_timeslices_plot_handles_missing_data_gracefully(
        self,
        single_run_exporter,
        sample_single_run_data,
        sample_available_metrics,
    ):
        """Test that missing timeslice data is handled gracefully."""
        # Run without timeslice data (None)
        generated_files = single_run_exporter.export(
            sample_single_run_data, sample_available_metrics
        )

        # Should generate 3 plots without timeslice plots
        assert len(generated_files) == 3

        filenames = {f.name for f in generated_files}
        assert "ttft_over_time.png" in filenames
        assert "itl_over_time.png" in filenames
        assert "latency_over_time.png" in filenames
        # No timeslice plots
        assert "timeslices_ttft.png" not in filenames

    def test_uniform_requests_no_warning(
        self,
        single_run_exporter,
        tmp_path,
        sample_available_metrics,
    ):
        """Test that uniform requests (identical ISL/OSL) show no warning."""
        per_request_data = pd.DataFrame(
            {
                "request_end_ns": [1000000000000 + i * 500000000 for i in range(10)],
                "time_to_first_token": [45.0 + i * 2 for i in range(10)],
                "inter_token_latency": [18.0 + i * 0.5 for i in range(10)],
                "request_latency": [900.0 + i * 10 for i in range(10)],
                "input_sequence_length": [100] * 10,
                "output_sequence_length": [200] * 10,
            }
        )

        timeslice_data = pd.DataFrame(
            {
                "Timeslice": [0, 0, 1, 1],
                "Metric": ["Request Throughput", "Request Throughput"] * 2,
                "Unit": ["req/s"] * 4,
                "Stat": ["avg", "min", "avg", "min"],
                "Value": [10.0, 8.0, 12.0, 9.0],
            }
        )

        run_data = RunData(
            metadata=RunMetadata(
                run_name="uniform_run",
                run_path=tmp_path / "uniform_run",
                model="Test",
                concurrency=1,
            ),
            requests=per_request_data,
            aggregated={},
            timeslices=timeslice_data,
            slice_duration=10.0,
        )

        is_uniform, warning = single_run_exporter._check_request_uniformity(run_data)
        assert is_uniform is True
        assert warning is None

        generated_files = single_run_exporter.export(run_data, sample_available_metrics)
        throughput_plot = [
            f for f in generated_files if "timeslices_throughput" in f.name
        ]
        assert len(throughput_plot) == 1

        fig = single_run_exporter.plot_generator.create_time_series_histogram(
            df=timeslice_data[timeslice_data["Stat"] == "avg"][
                ["Timeslice", "Value"]
            ].rename(columns={"Value": "avg"}),
            x_col="Timeslice",
            y_col="avg",
            title="Request Throughput Across Time Slices",
            x_label="Time (s)",
            y_label="Request Throughput (req/s)",
            slice_duration=10.0,
            warning_text=warning,
        )

        if hasattr(fig.layout, "annotations") and fig.layout.annotations:
            for ann in fig.layout.annotations:
                assert "varying ISL/OSL" not in ann.text

    def test_non_uniform_isl_shows_warning(
        self,
        single_run_exporter,
        tmp_path,
        sample_available_metrics,
    ):
        """Test that non-uniform ISL (varying input lengths) shows warning."""
        per_request_data = pd.DataFrame(
            {
                "request_end_ns": [1000000000000 + i * 500000000 for i in range(10)],
                "time_to_first_token": [45.0 + i * 2 for i in range(10)],
                "inter_token_latency": [18.0 + i * 0.5 for i in range(10)],
                "request_latency": [900.0 + i * 10 for i in range(10)],
                "input_sequence_length": [
                    100,
                    200,
                    100,
                    300,
                    100,
                    200,
                    100,
                    300,
                    100,
                    200,
                ],
                "output_sequence_length": [200] * 10,
            }
        )

        timeslice_data = pd.DataFrame(
            {
                "Timeslice": [0, 0, 1, 1],
                "Metric": ["Request Throughput", "Request Throughput"] * 2,
                "Unit": ["req/s"] * 4,
                "Stat": ["avg", "min", "avg", "min"],
                "Value": [10.0, 8.0, 12.0, 9.0],
            }
        )

        run_data = RunData(
            metadata=RunMetadata(
                run_name="non_uniform_isl_run",
                run_path=tmp_path / "non_uniform_isl_run",
                model="Test",
                concurrency=1,
            ),
            requests=per_request_data,
            aggregated={},
            timeslices=timeslice_data,
            slice_duration=10.0,
        )

        is_uniform, warning = single_run_exporter._check_request_uniformity(run_data)
        assert is_uniform is False
        assert warning is not None
        assert "varying ISL/OSL" in warning
        assert (
            "Req/sec throughput may not accurately represent workload capacity"
            in warning
        )

        fig = single_run_exporter.plot_generator.create_time_series_histogram(
            df=timeslice_data[timeslice_data["Stat"] == "avg"][
                ["Timeslice", "Value"]
            ].rename(columns={"Value": "avg"}),
            x_col="Timeslice",
            y_col="avg",
            title="Request Throughput Across Time Slices",
            x_label="Time (s)",
            y_label="Request Throughput (req/s)",
            slice_duration=10.0,
            warning_text=warning,
        )

        assert hasattr(fig.layout, "annotations")
        assert len(fig.layout.annotations) > 0

        warning_annotations = [
            ann for ann in fig.layout.annotations if "varying ISL/OSL" in ann.text
        ]
        assert len(warning_annotations) == 1

        warning_ann = warning_annotations[0]
        assert warning_ann.xref == "paper"
        assert warning_ann.yref == "paper"
        assert warning_ann.x == 0.5
        assert (
            warning_ann.y < 0
        )  # Warning is below x-axis (negative y in paper coordinates)
        assert warning_ann.y > -0.3  # But not too far below
        assert warning_ann.xanchor == "center"
        assert warning_ann.yanchor == "top"  # Top edge anchored to y position

    def test_non_uniform_osl_shows_warning(
        self,
        single_run_exporter,
        tmp_path,
        sample_available_metrics,
    ):
        """Test that non-uniform OSL (varying output lengths) shows warning."""
        per_request_data = pd.DataFrame(
            {
                "request_end_ns": [1000000000000 + i * 500000000 for i in range(10)],
                "time_to_first_token": [45.0 + i * 2 for i in range(10)],
                "inter_token_latency": [18.0 + i * 0.5 for i in range(10)],
                "request_latency": [900.0 + i * 10 for i in range(10)],
                "input_sequence_length": [100] * 10,
                "output_sequence_length": [
                    200,
                    300,
                    200,
                    400,
                    200,
                    300,
                    200,
                    400,
                    200,
                    300,
                ],
            }
        )

        run_data = RunData(
            metadata=RunMetadata(
                run_name="non_uniform_osl_run",
                run_path=tmp_path / "non_uniform_osl_run",
                model="Test",
                concurrency=1,
            ),
            requests=per_request_data,
            aggregated={},
            timeslices=None,
            slice_duration=None,
        )

        is_uniform, warning = single_run_exporter._check_request_uniformity(run_data)
        assert is_uniform is False
        assert warning is not None
        assert "varying ISL/OSL" in warning

    def test_warning_only_on_throughput_plot(
        self,
        single_run_exporter,
        tmp_path,
        sample_available_metrics,
    ):
        """Test that warning appears only on throughput plot, not other metrics."""
        per_request_data = pd.DataFrame(
            {
                "request_end_ns": [1000000000000 + i * 500000000 for i in range(10)],
                "time_to_first_token": [45.0 + i * 2 for i in range(10)],
                "inter_token_latency": [18.0 + i * 0.5 for i in range(10)],
                "request_latency": [900.0 + i * 10 for i in range(10)],
                "input_sequence_length": [
                    100,
                    200,
                    100,
                    200,
                    100,
                    200,
                    100,
                    200,
                    100,
                    200,
                ],
                "output_sequence_length": [200] * 10,
            }
        )

        timeslice_data = pd.DataFrame(
            {
                "Timeslice": [0, 0, 1, 1] * 4,
                "Metric": (
                    ["Time to First Token"] * 2
                    + ["Inter Token Latency"] * 2
                    + ["Request Throughput"] * 2
                    + ["Request Latency"] * 2
                )
                * 2,
                "Unit": ["ms"] * 4 + ["req/s"] * 4 + ["ms"] * 8,
                "Stat": ["avg", "min"] * 8,
                "Value": [45.0, 40.0, 18.0, 15.0, 10.0, 8.0, 900.0, 850.0] * 2,
            }
        )

        run_data = RunData(
            metadata=RunMetadata(
                run_name="multi_metric_run",
                run_path=tmp_path / "multi_metric_run",
                model="Test",
                concurrency=1,
            ),
            requests=per_request_data,
            aggregated={},
            timeslices=timeslice_data,
            slice_duration=10.0,
        )

        generated_files = single_run_exporter.export(run_data, sample_available_metrics)

        assert len(generated_files) > 0

    def test_on_demand_loading_non_uniform_requests(
        self,
        single_run_exporter,
        tmp_path,
    ):
        """Test that warning works when loading ISL/OSL from disk (requests=None)."""
        import json

        run_path = tmp_path / "on_demand_test_run"
        run_path.mkdir()

        profile_data = []
        for i in range(10):
            profile_data.append(
                {
                    "metrics": {
                        "input_sequence_length": {"value": 100, "unit": "tokens"},
                        "output_sequence_length": {
                            "value": 200 + i * 50,
                            "unit": "tokens",
                        },
                    }
                }
            )

        with open(run_path / "profile_export.jsonl", "w") as f:
            for record in profile_data:
                f.write(json.dumps(record) + "\n")

        run_data = RunData(
            metadata=RunMetadata(
                run_name="on_demand_test_run",
                run_path=run_path,
                model="Test",
                concurrency=1,
            ),
            requests=None,
            aggregated={},
            timeslices=None,
            slice_duration=None,
        )

        is_uniform, warning = single_run_exporter._check_request_uniformity(run_data)
        assert is_uniform is False
        assert warning is not None
        assert "varying ISL/OSL" in warning

    def test_on_demand_loading_uniform_requests(
        self,
        single_run_exporter,
        tmp_path,
    ):
        """Test that no warning appears when on-demand loading finds uniform requests."""
        import json

        run_path = tmp_path / "on_demand_uniform_run"
        run_path.mkdir()

        profile_data = []
        for _ in range(10):
            profile_data.append(
                {
                    "metrics": {
                        "input_sequence_length": {"value": 100, "unit": "tokens"},
                        "output_sequence_length": {"value": 200, "unit": "tokens"},
                    }
                }
            )

        with open(run_path / "profile_export.jsonl", "w") as f:
            for record in profile_data:
                f.write(json.dumps(record) + "\n")

        run_data = RunData(
            metadata=RunMetadata(
                run_name="on_demand_uniform_run",
                run_path=run_path,
                model="Test",
                concurrency=1,
            ),
            requests=None,
            aggregated={},
            timeslices=None,
            slice_duration=None,
        )

        is_uniform, warning = single_run_exporter._check_request_uniformity(run_data)
        assert is_uniform is True
        assert warning is None

    def test_on_demand_loading_missing_file(
        self,
        single_run_exporter,
        tmp_path,
    ):
        """Test that missing profile_export.jsonl returns uniform (no warning)."""
        run_path = tmp_path / "missing_file_run"
        run_path.mkdir()

        run_data = RunData(
            metadata=RunMetadata(
                run_name="missing_file_run",
                run_path=run_path,
                model="Test",
                concurrency=1,
            ),
            requests=None,
            aggregated={},
            timeslices=None,
            slice_duration=None,
        )

        is_uniform, warning = single_run_exporter._check_request_uniformity(run_data)
        assert is_uniform is True
        assert warning is None


class TestSharedExporterFunctionality:
    """Tests for shared functionality across both exporters."""

    def test_output_directory_created(self, tmp_path, sample_multi_run_data):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "new_directory" / "plot_export"
        assert not output_dir.exists()

        exporter = MultiRunPNGExporter(output_dir)
        exporter.export(sample_multi_run_data, {"display_names": {}, "units": {}})

        # Directory should be created
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_export_handles_missing_metrics_gracefully(
        self, tmp_path, sample_available_metrics
    ):
        """Test that export handles missing metrics without crashing."""
        output_dir = tmp_path / "plot_export"
        exporter = MultiRunPNGExporter(output_dir)

        incomplete_data = [
            RunData(
                metadata=RunMetadata(
                    run_name="incomplete_run",
                    run_path=tmp_path / "incomplete_run",
                    model="Test",
                    concurrency=1,
                ),
                requests=None,
                aggregated={
                    "request_latency": {"p50": 100.0, "unit": "ms"},
                    # Missing other metrics
                },
                timeslices=None,
                slice_duration=None,
            )
        ]

        # Should not raise an exception
        generated_files = exporter.export(incomplete_data, sample_available_metrics)

        # May generate fewer plots if metrics are missing
        assert isinstance(generated_files, list)
