# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for data loading functionality.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from aiperf.plot.core.data_loader import DataLoader, RunData, RunMetadata
from aiperf.plot.exceptions import DataLoadError


class TestDataLoaderLoadRun:
    """Tests for DataLoader.load_run method."""

    def test_load_single_run_success(self, populated_run_dir: Path) -> None:
        """Test successfully loading a single run."""
        loader = DataLoader()
        run = loader.load_run(populated_run_dir)

        assert isinstance(run, RunData)
        assert isinstance(run.metadata, RunMetadata)
        assert isinstance(run.requests, pd.DataFrame)
        assert isinstance(run.aggregated, dict)

        # Check that requests were loaded
        assert len(run.requests) == 2
        assert "time_to_first_token" in run.requests.columns
        assert "request_latency" in run.requests.columns

    def test_load_run_nonexistent_path(self) -> None:
        """Test loading from nonexistent path raises error."""
        loader = DataLoader()
        fake_path = Path("/nonexistent/path")

        with pytest.raises(DataLoadError, match="does not exist"):
            loader.load_run(fake_path)

    def test_load_run_file_path(self, tmp_path: Path) -> None:
        """Test loading from file path raises error."""
        loader = DataLoader()
        file_path = tmp_path / "test.txt"
        file_path.write_text("test")

        with pytest.raises(DataLoadError, match="not a directory"):
            loader.load_run(file_path)

    def test_load_run_missing_jsonl(self, tmp_path: Path) -> None:
        """Test loading run without JSONL file raises error."""
        loader = DataLoader()
        run_dir = tmp_path / "incomplete_run"
        run_dir.mkdir()

        with pytest.raises(DataLoadError, match="JSONL file not found"):
            loader.load_run(run_dir)

    def test_metadata_extraction(self, populated_run_dir: Path) -> None:
        """Test that metadata is correctly extracted."""
        loader = DataLoader()
        run = loader.load_run(populated_run_dir)

        assert run.metadata.run_name == populated_run_dir.name
        assert run.metadata.run_path == populated_run_dir
        assert run.metadata.model == "test-model"
        assert run.metadata.concurrency == 4
        assert run.metadata.request_count == 100
        assert run.metadata.endpoint_type == "chat"

    def test_timestamp_conversion_to_utc(self, populated_run_dir: Path) -> None:
        """Test that timestamps are converted to UTC datetime."""
        loader = DataLoader()
        run = loader.load_run(populated_run_dir)

        # Check that timestamp columns are datetime with UTC timezone
        assert pd.api.types.is_datetime64_any_dtype(run.requests["request_start_ns"])
        assert run.requests["request_start_ns"].dt.tz is not None


class TestDataLoaderLoadMultipleRuns:
    """Tests for DataLoader.load_multiple_runs method."""

    def test_load_multiple_runs_success(self, multiple_run_dirs: list[Path]) -> None:
        """Test successfully loading multiple runs."""
        loader = DataLoader()
        runs = loader.load_multiple_runs(multiple_run_dirs)

        assert len(runs) == 3
        assert all(isinstance(r, RunData) for r in runs)

    def test_swept_parameter_detection(self, multiple_run_dirs: list[Path]) -> None:
        """Test detection of swept parameters across runs."""
        loader = DataLoader()
        runs = loader.load_multiple_runs(multiple_run_dirs)

        # Concurrency is swept (2, 4, 8)
        concurrencies = [r.metadata.concurrency for r in runs]
        assert set(concurrencies) == {2, 4, 8}

        # Check that swept_params are populated
        for run in runs:
            assert "loadgen.concurrency" in run.metadata.swept_params

    def test_load_empty_list_raises_error(self) -> None:
        """Test that empty run list raises error."""
        loader = DataLoader()

        with pytest.raises(DataLoadError, match="No run paths provided"):
            loader.load_multiple_runs([])

    def test_load_with_invalid_run_raises_error(
        self, populated_run_dir: Path, tmp_path: Path
    ) -> None:
        """Test that invalid run in list raises error."""
        loader = DataLoader()
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()

        with pytest.raises(DataLoadError):
            loader.load_multiple_runs([populated_run_dir, invalid_dir])


class TestDataLoaderLoadJsonl:
    """Tests for DataLoader._load_jsonl method."""

    def test_load_valid_jsonl(self, populated_run_dir: Path) -> None:
        """Test loading valid JSONL file."""
        loader = DataLoader()
        jsonl_path = populated_run_dir / "profile_export.jsonl"
        df = loader._load_jsonl(jsonl_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "time_to_first_token" in df.columns
        assert "session_num" in df.columns

    def test_load_jsonl_missing_file(self, tmp_path: Path) -> None:
        """Test loading nonexistent JSONL file raises error."""
        loader = DataLoader()
        fake_path = tmp_path / "nonexistent.jsonl"

        with pytest.raises(DataLoadError, match="JSONL file not found"):
            loader._load_jsonl(fake_path)

    def test_load_jsonl_with_corrupted_lines(self, tmp_path: Path) -> None:
        """Test loading JSONL with corrupted lines."""
        loader = DataLoader()
        jsonl_path = tmp_path / "corrupted.jsonl"

        # Write JSONL with one good line and one bad line
        with open(jsonl_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "metadata": {"session_num": 0},
                        "metrics": {"time_to_first_token": {"value": 45.0}},
                    }
                )
                + "\n"
            )
            f.write("{ invalid json }\n")
            f.write(
                json.dumps(
                    {
                        "metadata": {"session_num": 1},
                        "metrics": {"time_to_first_token": {"value": 50.0}},
                    }
                )
                + "\n"
            )

        df = loader._load_jsonl(jsonl_path)
        # Should load 2 valid records, skip 1 corrupted
        assert len(df) == 2

    def test_load_jsonl_empty_file_raises_error(self, tmp_path: Path) -> None:
        """Test loading empty JSONL file raises error."""
        loader = DataLoader()
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")

        with pytest.raises(DataLoadError, match="No valid records found"):
            loader._load_jsonl(jsonl_path)


class TestDataLoaderComputeInterChunkLatencyStats:
    """Tests for DataLoader._compute_inter_chunk_latency_stats method."""

    def test_compute_stats_with_valid_data(self) -> None:
        """Test computing statistics from valid inter_chunk_latency data."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = DataLoader._compute_inter_chunk_latency_stats(values)

        assert "inter_chunk_latency_avg" in stats
        assert "inter_chunk_latency_p50" in stats
        assert "inter_chunk_latency_p95" in stats
        assert "inter_chunk_latency_std" in stats
        assert "inter_chunk_latency_min" in stats
        assert "inter_chunk_latency_max" in stats
        assert "inter_chunk_latency_range" in stats

        # Verify computed values
        assert stats["inter_chunk_latency_avg"] == 30.0
        assert stats["inter_chunk_latency_p50"] == 30.0
        assert stats["inter_chunk_latency_min"] == 10.0
        assert stats["inter_chunk_latency_max"] == 50.0
        assert stats["inter_chunk_latency_range"] == 40.0

    def test_compute_stats_with_empty_array(self) -> None:
        """Test that empty array returns empty dict."""
        stats = DataLoader._compute_inter_chunk_latency_stats([])
        assert stats == {}

    def test_compute_stats_with_single_value(self) -> None:
        """Test computing statistics from single value."""
        values = [25.0]
        stats = DataLoader._compute_inter_chunk_latency_stats(values)

        assert stats["inter_chunk_latency_avg"] == 25.0
        assert stats["inter_chunk_latency_p50"] == 25.0
        assert stats["inter_chunk_latency_min"] == 25.0
        assert stats["inter_chunk_latency_max"] == 25.0
        assert stats["inter_chunk_latency_range"] == 0.0
        assert stats["inter_chunk_latency_std"] == 0.0

    def test_compute_stats_with_jitter(self) -> None:
        """Test statistics capture jitter/variance in stream health."""
        # Stable stream
        stable_values = [20.0, 20.0, 20.0, 20.0, 20.0]
        stable_stats = DataLoader._compute_inter_chunk_latency_stats(stable_values)

        # Jittery stream with spike
        jittery_values = [10.0, 10.0, 10.0, 50.0, 10.0]
        jittery_stats = DataLoader._compute_inter_chunk_latency_stats(jittery_values)

        # Stable stream should have zero std and range
        assert stable_stats["inter_chunk_latency_std"] == 0.0
        assert stable_stats["inter_chunk_latency_range"] == 0.0

        # Jittery stream should have higher std and range
        assert jittery_stats["inter_chunk_latency_std"] > 0.0
        assert jittery_stats["inter_chunk_latency_range"] == 40.0
        assert jittery_stats["inter_chunk_latency_max"] == 50.0


class TestDataLoaderFlattenJsonlRecord:
    """Tests for DataLoader._flatten_jsonl_record method."""

    def test_flatten_complete_record(self) -> None:
        """Test flattening a complete JSONL record."""
        loader = DataLoader()
        record = {
            "metadata": {"session_num": 0, "x_request_id": "req-1"},
            "metrics": {
                "time_to_first_token": {"value": 45.5, "unit": "ms"},
                "output_sequence_length": {"value": 100, "unit": "tokens"},
            },
            "error": None,
        }

        flat = loader._flatten_jsonl_record(record)

        assert flat["session_num"] == 0
        assert flat["x_request_id"] == "req-1"
        assert flat["time_to_first_token"] == 45.5
        assert flat["output_sequence_length"] == 100
        assert flat["error"] is None

    def test_flatten_computes_inter_chunk_latency_statistics(self) -> None:
        """Test that inter_chunk_latency array is converted to statistics."""
        loader = DataLoader()
        record = {
            "metadata": {"session_num": 0},
            "metrics": {
                "inter_chunk_latency": {
                    "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                    "unit": "ms",
                },
                "time_to_first_token": {"value": 45.5, "unit": "ms"},
            },
        }

        flat = loader._flatten_jsonl_record(record)

        # Raw array should not be present
        assert "inter_chunk_latency" not in flat
        # But statistics should be computed
        assert "inter_chunk_latency_avg" in flat
        assert "inter_chunk_latency_p50" in flat
        assert "inter_chunk_latency_p95" in flat
        assert "inter_chunk_latency_std" in flat
        assert "inter_chunk_latency_min" in flat
        assert "inter_chunk_latency_max" in flat
        assert "inter_chunk_latency_range" in flat

        # Verify values are correct
        assert flat["inter_chunk_latency_avg"] == 30.0
        assert flat["inter_chunk_latency_p50"] == 30.0
        assert flat["inter_chunk_latency_min"] == 10.0
        assert flat["inter_chunk_latency_max"] == 50.0
        assert flat["inter_chunk_latency_range"] == 40.0

        # Other metrics should still be present
        assert "time_to_first_token" in flat

    def test_flatten_inter_chunk_latency_empty_array(self) -> None:
        """Test that empty inter_chunk_latency array is handled gracefully."""
        loader = DataLoader()
        record = {
            "metadata": {"session_num": 0},
            "metrics": {
                "inter_chunk_latency": {
                    "value": [],
                    "unit": "ms",
                },
                "time_to_first_token": {"value": 45.5, "unit": "ms"},
            },
        }

        flat = loader._flatten_jsonl_record(record)

        # No statistics should be added for empty array
        assert "inter_chunk_latency_avg" not in flat
        assert "inter_chunk_latency_p50" not in flat
        assert "time_to_first_token" in flat


class TestDataLoaderLoadAggregatedJson:
    """Tests for DataLoader._load_aggregated_json method."""

    def test_load_valid_aggregated_json(self, populated_run_dir: Path) -> None:
        """Test loading valid aggregated JSON."""
        loader = DataLoader()
        json_path = populated_run_dir / "profile_export_aiperf.json"
        data = loader._load_aggregated_json(json_path)

        assert isinstance(data, dict)
        assert "input_config" in data

    def test_load_missing_aggregated_json(self, tmp_path: Path) -> None:
        """Test loading missing aggregated JSON raises DataLoadError."""
        loader = DataLoader()
        fake_path = tmp_path / "nonexistent.json"

        with pytest.raises(DataLoadError, match="JSON file not found"):
            loader._load_aggregated_json(fake_path)

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """Test loading invalid JSON raises DataLoadError."""
        loader = DataLoader()
        json_path = tmp_path / "invalid.json"
        json_path.write_text("{ invalid json }")

        with pytest.raises(DataLoadError, match="Failed to parse JSON"):
            loader._load_aggregated_json(json_path)


class TestDataLoaderLoadInputConfig:
    """Tests for DataLoader._load_input_config method."""

    def test_load_valid_input_config(self, populated_run_dir: Path) -> None:
        """Test loading valid inputs.json."""
        loader = DataLoader()
        inputs_path = populated_run_dir / "inputs.json"
        data = loader._load_input_config(inputs_path)

        assert isinstance(data, dict)
        assert "data" in data

    def test_load_missing_input_config(self, tmp_path: Path) -> None:
        """Test loading missing inputs.json raises DataLoadError."""
        loader = DataLoader()
        fake_path = tmp_path / "nonexistent.json"

        with pytest.raises(DataLoadError, match="inputs file not found"):
            loader._load_input_config(fake_path)


class TestDataLoaderExtractMetadata:
    """Tests for DataLoader._extract_metadata method."""

    def test_extract_metadata_with_all_data(
        self, tmp_path: Path, sample_aggregated_data: dict[str, Any]
    ) -> None:
        """Test metadata extraction with complete data."""
        loader = DataLoader()
        run_path = tmp_path / "test_run"
        requests_df = pd.DataFrame(
            {
                "request_start_ns": pd.to_datetime(
                    [1000000000, 2000000000], unit="ns", utc=True
                ),
                "request_end_ns": pd.to_datetime(
                    [1500000000, 2500000000], unit="ns", utc=True
                ),
            }
        )

        metadata = loader._extract_metadata(
            run_path, requests_df, sample_aggregated_data
        )

        assert metadata.run_name == "test_run"
        assert metadata.run_path == run_path
        assert metadata.model == "test-model"
        assert metadata.concurrency == 4
        assert metadata.request_count == 100
        assert metadata.endpoint_type == "chat"
        assert metadata.duration_seconds is not None

    def test_extract_metadata_missing_aggregated_data(self, tmp_path: Path) -> None:
        """Test metadata extraction without aggregated data."""
        loader = DataLoader()
        run_path = tmp_path / "test_run"
        requests_df = pd.DataFrame()

        metadata = loader._extract_metadata(run_path, requests_df, {})

        assert metadata.run_name == "test_run"
        assert metadata.model is None
        assert metadata.concurrency is None


class TestDataLoaderDetectSweptParameters:
    """Tests for DataLoader._detect_swept_parameters method."""

    def test_detect_single_swept_parameter(self) -> None:
        """Test detection of a single swept parameter."""
        loader = DataLoader()

        # Create runs with different concurrency
        runs = [
            RunData(
                metadata=RunMetadata(run_name=f"run{i}", run_path=Path(f"run{i}")),
                requests=pd.DataFrame(),
                aggregated={"input_config": {"loadgen": {"concurrency": conc}}},
            )
            for i, conc in enumerate([2, 4, 8])
        ]

        swept = loader._detect_swept_parameters(runs)

        assert "loadgen.concurrency" in swept
        assert len(swept["loadgen.concurrency"]) == 3

    def test_detect_multiple_swept_parameters(self) -> None:
        """Test detection of multiple swept parameters."""
        loader = DataLoader()

        runs = [
            RunData(
                metadata=RunMetadata(run_name=f"run{i}", run_path=Path(f"run{i}")),
                requests=pd.DataFrame(),
                aggregated={
                    "input_config": {
                        "loadgen": {"concurrency": conc},
                        "endpoint": {"model_names": [model]},
                    }
                },
            )
            for i, (conc, model) in enumerate([(2, "model-a"), (4, "model-b")])
        ]

        swept = loader._detect_swept_parameters(runs)

        assert "loadgen.concurrency" in swept
        assert "endpoint.model_names" in swept

    def test_no_swept_parameters_single_run(self) -> None:
        """Test that single run has no swept parameters."""
        loader = DataLoader()

        runs = [
            RunData(
                metadata=RunMetadata(run_name="run1", run_path=Path("run1")),
                requests=pd.DataFrame(),
                aggregated={"input_config": {"loadgen": {"concurrency": 4}}},
            )
        ]

        swept = loader._detect_swept_parameters(runs)
        assert swept == {}

    def test_no_swept_parameters_identical_runs(self) -> None:
        """Test that identical runs have no swept parameters."""
        loader = DataLoader()

        runs = [
            RunData(
                metadata=RunMetadata(run_name=f"run{i}", run_path=Path(f"run{i}")),
                requests=pd.DataFrame(),
                aggregated={"input_config": {"loadgen": {"concurrency": 4}}},
            )
            for i in range(3)
        ]

        swept = loader._detect_swept_parameters(runs)
        assert swept == {}


class TestDataLoaderFlattenConfig:
    """Tests for DataLoader._flatten_config method."""

    def test_flatten_nested_config(self) -> None:
        """Test flattening nested configuration."""
        loader = DataLoader()
        config = {
            "loadgen": {"concurrency": 4, "request_count": 100},
            "endpoint": {"model_names": ["test-model"], "type": "chat"},
        }

        flat = loader._flatten_config(config)

        assert flat["loadgen.concurrency"] == 4
        assert flat["loadgen.request_count"] == 100
        assert flat["endpoint.model_names"] == ["test-model"]
        assert flat["endpoint.type"] == "chat"

    def test_flatten_deeply_nested_config(self) -> None:
        """Test flattening deeply nested configuration."""
        loader = DataLoader()
        config = {"a": {"b": {"c": {"d": "value"}}}}

        flat = loader._flatten_config(config)

        assert flat["a.b.c.d"] == "value"


class TestDataLoaderReloadWithDetails:
    """Tests for DataLoader.reload_with_details method."""

    def test_reload_adds_per_request_data(self, populated_run_dir: Path) -> None:
        """Test that reload_with_details loads per-request data."""
        loader = DataLoader()
        run = loader.load_run(populated_run_dir, load_per_request_data=False)

        assert run.requests is None

        reloaded_run = loader.reload_with_details(populated_run_dir)

        assert reloaded_run.requests is not None
        assert not reloaded_run.requests.empty
        assert reloaded_run.metadata.run_name == run.metadata.run_name
        assert reloaded_run.metadata.model == run.metadata.model
        assert reloaded_run.metadata.concurrency == run.metadata.concurrency
        assert reloaded_run.aggregated == run.aggregated

    def test_reload_nonexistent_path_raises_error(self, tmp_path: Path) -> None:
        """Test reload_with_details with nonexistent path."""
        loader = DataLoader()
        fake_path = tmp_path / "nonexistent_run"

        with pytest.raises(DataLoadError, match="Run path does not exist"):
            loader.reload_with_details(fake_path)


class TestDataLoaderLoadPerRequestData:
    """Tests for DataLoader.load_run with load_per_request_data parameter."""

    def test_load_run_without_per_request_data(self, populated_run_dir: Path) -> None:
        """Test loading run without per-request data."""
        loader = DataLoader()
        run = loader.load_run(populated_run_dir, load_per_request_data=False)

        assert run.metadata is not None
        assert run.aggregated is not None
        assert run.requests is None

    def test_load_run_with_per_request_data(self, populated_run_dir: Path) -> None:
        """Test loading run with per-request data."""
        loader = DataLoader()
        run = loader.load_run(populated_run_dir, load_per_request_data=True)

        assert run.metadata is not None
        assert run.aggregated is not None
        assert run.requests is not None
        assert not run.requests.empty

    def test_load_without_per_request_data_missing_jsonl_raises_error(
        self, tmp_path: Path, sample_aggregated_data: dict[str, Any]
    ) -> None:
        """Test that load_per_request_data=False still validates JSONL exists."""
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()

        json_path = run_dir / "profile_export_aiperf.json"
        json_path.write_text(json.dumps(sample_aggregated_data))

        loader = DataLoader()
        with pytest.raises(DataLoadError, match="Required JSONL file not found"):
            loader.load_run(run_dir, load_per_request_data=False)


class TestDataLoaderExtractSweptParams:
    """Tests for DataLoader._extract_swept_params method."""

    def test_extract_swept_params_from_run(
        self, sample_aggregated_data: dict[str, Any]
    ) -> None:
        """Test extracting swept parameter values from a run."""
        loader = DataLoader()
        run = RunData(
            metadata=RunMetadata(
                run_name="test_run",
                run_path=Path("/test"),
                model="test-model",
                concurrency=4,
                request_count=100,
                duration_seconds=10.0,
                endpoint_type="chat",
                swept_params={},
            ),
            requests=None,
            aggregated=sample_aggregated_data,
        )

        swept_params = {"loadgen.concurrency", "endpoint.model_names"}
        result = loader._extract_swept_params(run, swept_params)

        assert "loadgen.concurrency" in result
        assert result["loadgen.concurrency"] == 4
        assert "endpoint.model_names" in result

    def test_extract_swept_params_missing_in_config(self) -> None:
        """Test extracting swept params when config is missing."""
        loader = DataLoader()
        run = RunData(
            metadata=RunMetadata(
                run_name="test_run",
                run_path=Path("/test"),
                model=None,
                concurrency=None,
                request_count=None,
                duration_seconds=None,
                endpoint_type=None,
                swept_params={},
            ),
            requests=None,
            aggregated={},
        )

        swept_params = {"loadgen.concurrency": set([2, 4])}
        result = loader._extract_swept_params(run, swept_params)

        assert result == {}


class TestDataLoaderDurationCalculation:
    """Tests for duration calculation in metadata extraction."""

    def test_duration_calculated_from_requests(self, tmp_path: Path) -> None:
        """Test duration is calculated from request timestamps."""
        loader = DataLoader()
        run_path = tmp_path / "test_run"

        requests_df = pd.DataFrame(
            {
                "request_start_ns": pd.to_datetime(
                    [1000000000, 1500000000, 2000000000], unit="ns", utc=True
                ),
                "request_end_ns": pd.to_datetime(
                    [1200000000, 1700000000, 3000000000], unit="ns", utc=True
                ),
            }
        )

        metadata = loader._extract_metadata(run_path, requests_df, {})

        assert metadata.duration_seconds is not None
        assert metadata.duration_seconds == 2.0

    def test_duration_none_when_no_requests(self, tmp_path: Path) -> None:
        """Test duration is None when no request data available."""
        loader = DataLoader()
        run_path = tmp_path / "test_run"

        metadata = loader._extract_metadata(run_path, None, {})

        assert metadata.duration_seconds is None

    def test_duration_none_when_empty_dataframe(self, tmp_path: Path) -> None:
        """Test duration is None with empty DataFrame."""
        loader = DataLoader()
        run_path = tmp_path / "test_run"
        requests_df = pd.DataFrame()

        metadata = loader._extract_metadata(run_path, requests_df, {})

        assert metadata.duration_seconds is None

    def test_duration_with_missing_timestamp_columns(self, tmp_path: Path) -> None:
        """Test duration is None when timestamp columns are missing."""
        loader = DataLoader()
        run_path = tmp_path / "test_run"
        requests_df = pd.DataFrame({"other_column": [1, 2, 3]})

        metadata = loader._extract_metadata(run_path, requests_df, {})

        assert metadata.duration_seconds is None
