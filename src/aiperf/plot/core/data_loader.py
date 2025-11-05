# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data loading functionality for visualization.

This module provides classes and functions to load AIPerf profiling data
from various output files (JSONL, JSON) and parse them into structured
formats suitable for visualization and analysis.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models.record_models import MetricRecordInfo, MetricResult
from aiperf.plot.constants import (
    PROFILE_EXPORT_AIPERF_JSON,
    PROFILE_EXPORT_JSONL,
)
from aiperf.plot.exceptions import DataLoadError


@dataclass
class RunMetadata:
    """
    Metadata for a single profiling run.

    Args:
        run_name: Name of the run (typically directory name).
        run_path: Path to the run directory.
        model: Model name used in the run.
        concurrency: Concurrency level used.
        request_count: Total number of requests.
        duration_seconds: Duration of the run in seconds.
        endpoint_type: Type of endpoint (e.g., "chat", "completions").
        start_time: ISO timestamp when the profiling run started.
        end_time: ISO timestamp when the profiling run ended.
        was_cancelled: Whether the profiling run was cancelled early.
    """

    run_name: str
    run_path: Path
    model: str | None = None
    concurrency: int | None = None
    request_count: int | None = None
    duration_seconds: float | None = None
    endpoint_type: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    was_cancelled: bool = False


@dataclass
class RunData:
    """
    Complete data for a single profiling run.

    Args:
        metadata: Metadata for the run.
        requests: DataFrame containing per-request data, or None if not loaded.
        aggregated: Dictionary containing aggregated statistics. The "metrics" key
            contains a dict mapping metric tags to MetricResult objects.
    """

    metadata: RunMetadata
    requests: pd.DataFrame | None
    aggregated: dict[str, Any]


class DataLoader(AIPerfLoggerMixin):
    """
    Loader for AIPerf profiling data.

    This class provides methods to load profiling data from various files
    and parse them into structured formats for visualization.

    Examples:
        >>> loader = DataLoader()
        >>> run_data = loader.load_run(Path("results/run1"))
        >>> print(run_data.metadata.model)
        'my-model'

        >>> runs = loader.load_multiple_runs([Path("run1"), Path("run2")])
        >>> for run in runs:
        ...     print(run.metadata.run_name)
    """

    def __init__(self):
        super().__init__()

    def load_run(self, run_path: Path, load_per_request_data: bool = True) -> RunData:
        """
        Load data from a single profiling run.

        Args:
            run_path: Path to the run directory.
            load_per_request_data: Whether to load per-request data from JSONL.
                Defaults to True. Set to False for multi-run comparisons where
                per-request data is not needed.

        Returns:
            RunData object containing metadata, per-request data, and aggregated
            statistics.

        Raises:
            DataLoadError: If data cannot be loaded from the run directory.
        """
        if not run_path.exists():
            raise DataLoadError("Run path does not exist", path=str(run_path))

        if not run_path.is_dir():
            raise DataLoadError("Run path is not a directory", path=str(run_path))

        self.info(f"Loading run from: {run_path}")

        jsonl_path = run_path / PROFILE_EXPORT_JSONL
        if not jsonl_path.exists():
            raise DataLoadError("Required JSONL file not found", path=str(jsonl_path))

        # Load JSONL per-request data (conditionally)
        requests_df = self._load_jsonl(jsonl_path) if load_per_request_data else None

        # Load aggregated JSON
        aggregated = self._load_aggregated_json(run_path / PROFILE_EXPORT_AIPERF_JSON)

        # Extract metadata
        metadata = self._extract_metadata(run_path, requests_df, aggregated)

        return RunData(metadata=metadata, requests=requests_df, aggregated=aggregated)

    def load_multiple_runs(self, run_paths: list[Path]) -> list[RunData]:
        """
        Load data from multiple profiling runs.

        This method also detects swept parameters across runs. Per-request
        data (JSONL) is not loaded for multi-run comparisons as only aggregated
        statistics are needed.

        Args:
            run_paths: List of paths to run directories.

        Returns:
            List of RunData objects, one for each run. The requests field will
            be None for all runs.

        Raises:
            DataLoadError: If any run cannot be loaded.
        """
        if not run_paths:
            raise DataLoadError("No run paths provided")

        # Load all runs with only aggregated summary data, not per-request data
        runs = []
        for path in run_paths:
            try:
                run = self.load_run(path, load_per_request_data=False)
                runs.append(run)
            except DataLoadError as e:
                self.error(f"Failed to load run from {path}: {e}")
                raise

        # Note: swept parameter detection removed until configuration system is implemented

        return runs

    def reload_with_details(self, run_path: Path) -> RunData:
        """
        Reload a run with full per-request data.

        This method is useful in interactive mode (HTML or hosted dashboard) where a run was initially
        loaded as part of a multi-run comparison (without per-request data), but
        now detailed analysis is needed for a specific run.

        Args:
            run_path: Path to the run directory to reload.

        Returns:
            RunData object with full per-request data loaded.

        Raises:
            DataLoadError: If data cannot be loaded from the run directory.

        Examples:
            >>> loader = DataLoader()
            >>> runs = loader.load_multiple_runs([Path("run1"), Path("run2")])
            >>> # User selects run1 for detailed analysis
            >>> detailed_run = loader.reload_with_details(Path("run1"))
            >>> print(detailed_run.requests.shape)
            (100, 15)
        """
        return self.load_run(run_path, load_per_request_data=True)

    def extract_telemetry_data(
        self, aggregated: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Extract telemetry data from aggregated statistics.

        Args:
            aggregated: The aggregated data dictionary from profile_export_aiperf.json

        Returns:
            Telemetry data dictionary with 'summary' and 'endpoints' keys, or None if
            telemetry data is not available.

        Examples:
            >>> loader = DataLoader()
            >>> run = loader.load_run(Path("results/run1"))
            >>> telemetry = loader.extract_telemetry_data(run.aggregated)
            >>> if telemetry:
            ...     print(telemetry['summary']['start_time'])
            ...     print(telemetry['endpoints'].keys())
        """
        if not aggregated or "telemetry_data" not in aggregated:
            self.debug("No telemetry data found in aggregated statistics")
            return None

        telemetry = aggregated.get("telemetry_data")

        if not isinstance(telemetry, dict):
            self.warning("Telemetry data exists but has unexpected structure")
            return None

        if "summary" not in telemetry or "endpoints" not in telemetry:
            self.warning("Telemetry data missing expected keys (summary, endpoints)")
            return None

        self.info(
            f"Extracted telemetry data with {len(telemetry.get('endpoints', {}))} endpoints"
        )
        return telemetry

    def get_telemetry_summary(
        self, aggregated: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Get telemetry summary information (start_time, end_time, endpoints).

        Args:
            aggregated: The aggregated data dictionary

        Returns:
            Dictionary with keys: start_time, end_time, endpoints_configured,
            endpoints_successful, or None if not available.

        Examples:
            >>> loader = DataLoader()
            >>> run = loader.load_run(Path("results/run1"))
            >>> summary = loader.get_telemetry_summary(run.aggregated)
            >>> if summary:
            ...     print(f"Started: {summary['start_time']}")
            ...     print(f"Ended: {summary['end_time']}")
        """
        telemetry = self.extract_telemetry_data(aggregated)
        return telemetry.get("summary") if telemetry else None

    def calculate_gpu_count_from_telemetry(
        self, aggregated: dict[str, Any]
    ) -> int | None:
        """
        Calculate total GPU count from telemetry data.

        Counts unique GPUs across all endpoints in the telemetry data structure.

        Args:
            aggregated: The aggregated data dictionary from profile_export_aiperf.json

        Returns:
            Total GPU count across all endpoints, or None if telemetry data is not
            available.

        Examples:
            >>> loader = DataLoader()
            >>> run = loader.load_run(Path("results/run1"))
            >>> gpu_count = loader.calculate_gpu_count_from_telemetry(run.aggregated)
            >>> if gpu_count:
            ...     print(f"Total GPUs: {gpu_count}")
        """
        telemetry = self.extract_telemetry_data(aggregated)
        if not telemetry:
            return None

        endpoints = telemetry.get("endpoints", {})
        if not isinstance(endpoints, dict):
            self.warning("Telemetry endpoints data has unexpected structure")
            return None

        gpu_count = 0
        for _endpoint_name, endpoint_data in endpoints.items():
            if not isinstance(endpoint_data, dict):
                continue

            gpus = endpoint_data.get("gpus", {})
            if isinstance(gpus, dict):
                gpu_count += len(gpus)

        if gpu_count == 0:
            self.debug("No GPUs found in telemetry data")
            return None

        self.info(f"Calculated GPU count from telemetry: {gpu_count}")
        return gpu_count

    def add_derived_gpu_metrics(self, aggregated: dict[str, Any]) -> dict[str, Any]:
        """
        Add derived GPU metrics to aggregated data when telemetry is available.

        Creates output_token_throughput_per_gpu by dividing output_token_throughput
        by the GPU count extracted from telemetry data. If telemetry data is not
        available or GPU count cannot be determined, no metrics are added.

        Args:
            aggregated: The aggregated data dictionary (will be modified in-place)

        Returns:
            The modified aggregated dictionary (same object as input)

        Examples:
            >>> loader = DataLoader()
            >>> run = loader.load_run(Path("results/run1"))
            >>> loader.add_derived_gpu_metrics(run.aggregated)
            >>> if "output_token_throughput_per_gpu" in run.aggregated:
            ...     print("GPU metric available")
        """
        gpu_count = self.calculate_gpu_count_from_telemetry(aggregated)

        if gpu_count is None or gpu_count == 0:
            self.debug(
                "Skipping derived GPU metrics: telemetry data not available or no GPUs found"
            )
            return aggregated

        if "output_token_throughput" not in aggregated:
            self.debug(
                "Skipping derived GPU metrics: output_token_throughput metric not found"
            )
            return aggregated

        throughput_data = aggregated["output_token_throughput"]
        if not isinstance(throughput_data, dict):
            self.warning(
                "output_token_throughput has unexpected structure, skipping derived metrics"
            )
            return aggregated

        per_gpu_data = {}
        per_gpu_data["unit"] = "tokens/sec/gpu"

        for key, value in throughput_data.items():
            if key == "unit":
                continue
            if isinstance(value, int | float):
                per_gpu_data[key] = value / gpu_count

        aggregated["output_token_throughput_per_gpu"] = per_gpu_data
        self.info(
            f"Added derived metric: output_token_throughput_per_gpu (divided by {gpu_count} GPUs)"
        )

        return aggregated

    def get_available_metrics(self, run_data: RunData) -> dict[str, dict[str, str]]:
        """
        Get metrics available in the loaded data.

        Extracts metric information from the aggregated data, which has a flat structure
        where metrics are stored at the top level (not nested under a "metrics" key).

        Args:
            run_data: RunData object with loaded aggregated data.

        Returns:
            Dictionary with two keys:
                - "display_names": dict mapping metric tag to display name
                - "units": dict mapping metric tag to unit string

        Examples:
            >>> loader = DataLoader()
            >>> run = loader.load_run(Path("results/run1"))
            >>> available = loader.get_available_metrics(run)
            >>> print(available["display_names"])
            {'time_to_first_token': 'Time to First Token', 'inter_token_latency': 'Inter Token Latency'}
        """
        from aiperf.plot.constants import NON_METRIC_KEYS
        from aiperf.plot.metric_names import get_metric_display_name

        if not run_data.aggregated:
            self.warning("No aggregated data available")
            return {"display_names": {}, "units": {}}

        display_names = {}
        units = {}

        # Iterate through top-level keys in aggregated data
        for key, value in run_data.aggregated.items():
            # Skip known non-metric keys
            if key in NON_METRIC_KEYS:
                continue

            # Check if this looks like a metric (has unit field)
            if isinstance(value, dict) and "unit" in value and value is not None:
                # Get display name from MetricRegistry/GPU telemetry config
                display_names[key] = get_metric_display_name(key)
                units[key] = value["unit"]

        if not display_names:
            self.warning("No metrics found in aggregated data")
        else:
            self.info(
                f"Found {len(display_names)} available metrics: {sorted(display_names.keys())}"
            )

        return {"display_names": display_names, "units": units}

    def _load_jsonl(self, jsonl_path: Path) -> pd.DataFrame:
        """
        Load per-request data from JSONL file.

        Args:
            jsonl_path: Path to the profile_export.jsonl file.

        Returns:
            DataFrame containing per-request data with flattened metrics.

        Raises:
            DataLoadError: If file cannot be read or parsed.
        """
        if not jsonl_path.exists():
            raise DataLoadError("JSONL file not found", path=str(jsonl_path))

        records = []
        corrupted_lines = 0

        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Parse using Pydantic model for type safety and validation
                        metric_record = MetricRecordInfo.model_validate_json(line)
                        # Convert to flat dict for DataFrame
                        flat_record = self._convert_to_flat_dict(metric_record)
                        records.append(flat_record)
                    except (json.JSONDecodeError, Exception) as e:
                        corrupted_lines += 1
                        self.warning(
                            f"Skipping invalid line {line_num} in {jsonl_path}: {e}"
                        )
                        continue

            if corrupted_lines > 0:
                self.warning(
                    f"Skipped {corrupted_lines} corrupted lines in {jsonl_path}"
                )

            if not records:
                raise DataLoadError(
                    "No valid records found in JSONL file", path=str(jsonl_path)
                )

            df = pd.DataFrame(records)

            # Convert timestamp columns to UTC datetime
            timestamp_columns = [col for col in df.columns if col.endswith("_ns")]
            for col in timestamp_columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], unit="ns", utc=True)

            self.info(f"Loaded {len(df)} records from {jsonl_path}")
            return df

        except OSError as e:
            raise DataLoadError(
                f"Failed to read JSONL file: {e}", path=str(jsonl_path)
            ) from e

    @staticmethod
    def _compute_inter_chunk_latency_stats(values: list[float]) -> dict[str, float]:
        """
        Compute per-request statistics from inter_chunk_latency array.

        These statistics are useful for analyzing stream health, jitter,
        and stability on a per-request basis.

        Args:
            values: List of inter-chunk latency values (in milliseconds).

        Returns:
            Dictionary mapping statistic names to computed values.
            Returns empty dict if values list is empty.
        """
        if not values:
            return {}

        arr = np.array(values)
        return {
            "inter_chunk_latency_avg": float(np.mean(arr)),
            "inter_chunk_latency_p50": float(np.percentile(arr, 50)),
            "inter_chunk_latency_p95": float(np.percentile(arr, 95)),
            "inter_chunk_latency_std": float(np.std(arr)),
            "inter_chunk_latency_min": float(np.min(arr)),
            "inter_chunk_latency_max": float(np.max(arr)),
            "inter_chunk_latency_range": float(np.max(arr) - np.min(arr)),
        }

    def _convert_to_flat_dict(self, record: MetricRecordInfo) -> dict:
        """
        Convert a MetricRecordInfo Pydantic model to a flat dictionary for DataFrame.

        Args:
            record: Pydantic model from JSONL line.

        Returns:
            Flattened dictionary with metrics and metadata at top level.
        """
        flat = {}

        # Flatten metadata (Pydantic model -> dict)
        flat.update(record.metadata.model_dump())

        # Flatten metrics (extract values from MetricValue objects)
        for key, metric_value in record.metrics.items():
            # Compute per-request statistics for inter_chunk_latency (stream health/jitter)
            if key == "inter_chunk_latency" and isinstance(metric_value.value, list):
                stats = self._compute_inter_chunk_latency_stats(metric_value.value)
                flat.update(stats)
                continue
            flat[key] = metric_value.value

        # Include error field if present
        if record.error:
            flat["error"] = record.error.model_dump()

        return flat

    def _load_aggregated_json(self, json_path: Path) -> dict[str, Any]:
        """
        Load aggregated statistics from JSON file.

        Parses the metrics in the JSON file into MetricResult objects for type-safe
        access to metric fields.

        Args:
            json_path: Path to the profile_export_aiperf.json file.

        Returns:
            Dictionary containing aggregated statistics. The "metrics" key contains
            a dict mapping metric tags to MetricResult objects.

        Raises:
            DataLoadError: If file cannot be read or parsed.
        """
        if not json_path.exists():
            raise DataLoadError("Required JSON file not found", path=str(json_path))

        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            # Parse metrics into MetricResult objects
            if "metrics" in data and isinstance(data["metrics"], dict):
                parsed_metrics = {}
                for tag, metric_data in data["metrics"].items():
                    try:
                        parsed_metrics[tag] = MetricResult(**metric_data)
                    except Exception as e:
                        self.warning(f"Failed to parse metric {tag}: {e}")
                        # Keep raw dict as fallback
                        parsed_metrics[tag] = metric_data
                data["metrics"] = parsed_metrics

            self.info(f"Loaded aggregated data from {json_path}")
            return data
        except json.JSONDecodeError as e:
            raise DataLoadError(
                f"Failed to parse JSON file: {e}", path=str(json_path)
            ) from e
        except OSError as e:
            raise DataLoadError(
                f"Failed to read JSON file: {e}", path=str(json_path)
            ) from e

    def _load_input_config(self, inputs_path: Path) -> dict[str, Any]:
        """
        Load input configuration from inputs.json.

        Args:
            inputs_path: Path to the inputs.json file.

        Returns:
            Dictionary containing input configuration.

        Raises:
            DataLoadError: If file cannot be read or parsed.
        """
        if not inputs_path.exists():
            raise DataLoadError("Required inputs file not found", path=str(inputs_path))

        try:
            with open(inputs_path, encoding="utf-8") as f:
                data = json.load(f)
            self.info(f"Loaded input config from {inputs_path}")
            return data
        except json.JSONDecodeError as e:
            raise DataLoadError(
                f"Failed to parse inputs JSON: {e}", path=str(inputs_path)
            ) from e
        except OSError as e:
            raise DataLoadError(
                f"Failed to read inputs JSON: {e}", path=str(inputs_path)
            ) from e

    def _extract_metadata(
        self,
        run_path: Path,
        requests_df: pd.DataFrame | None,
        aggregated: dict[str, Any],
    ) -> RunMetadata:
        """
        Extract metadata from loaded data.

        Args:
            run_path: Path to the run directory.
            requests_df: DataFrame with per-request data, or None if not loaded.
            aggregated: Aggregated statistics dictionary containing input_config.

        Returns:
            RunMetadata object with extracted information.
        """
        run_name = run_path.name

        # Extract from aggregated data (preferred source)
        model = None
        concurrency = None
        request_count = None
        endpoint_type = None
        start_time = None
        end_time = None
        was_cancelled = False

        if aggregated and "input_config" in aggregated:
            config = aggregated["input_config"]

            # Extract model
            if "endpoint" in config and "model_names" in config["endpoint"]:
                models = config["endpoint"]["model_names"]
                if models:
                    model = models[0]

            # Extract concurrency
            if "loadgen" in config and "concurrency" in config["loadgen"]:
                concurrency = config["loadgen"]["concurrency"]

            # Extract request count
            if "loadgen" in config and "request_count" in config["loadgen"]:
                request_count = config["loadgen"]["request_count"]

            # Extract endpoint type
            if "endpoint" in config and "type" in config["endpoint"]:
                endpoint_type = config["endpoint"]["type"]

        # Extract run timing information from top-level keys
        if aggregated:
            start_time = aggregated.get("start_time")
            end_time = aggregated.get("end_time")
            was_cancelled = aggregated.get("was_cancelled", False)

        # Calculate duration from requests (if available)
        duration_seconds = None
        if (
            requests_df is not None
            and not requests_df.empty
            and "request_start_ns" in requests_df.columns
            and "request_end_ns" in requests_df.columns
        ):
            start_times = requests_df["request_start_ns"].dropna()
            end_times = requests_df["request_end_ns"].dropna()
            if not start_times.empty and not end_times.empty:
                duration = end_times.max() - start_times.min()
                # Handle both Timedelta (from datetime subtraction) and int/float (from ns values)
                if isinstance(duration, pd.Timedelta):
                    duration_seconds = duration.total_seconds()
                else:
                    duration_seconds = duration / 1e9

        return RunMetadata(
            run_name=run_name,
            run_path=run_path,
            model=model,
            concurrency=concurrency,
            request_count=request_count,
            duration_seconds=duration_seconds,
            endpoint_type=endpoint_type,
            start_time=start_time,
            end_time=end_time,
            was_cancelled=was_cancelled,
        )
