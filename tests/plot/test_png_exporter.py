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
            )
        ]

        # Should not raise an exception
        generated_files = exporter.export(incomplete_data, sample_available_metrics)

        # May generate fewer plots if metrics are missing
        assert isinstance(generated_files, list)
