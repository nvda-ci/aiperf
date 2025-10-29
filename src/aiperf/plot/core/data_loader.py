# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data loading functionality for visualization.

This module provides classes and functions to load AIPerf profiling data
from various output files (JSONL, JSON) and parse them into structured
formats suitable for visualization and analysis.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.plot.constants import (
    METRIC_DISPLAY_NAMES,
    METRIC_UNITS,
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
        swept_params: Dictionary of swept parameters and their values.
    """

    run_name: str
    run_path: Path
    model: str | None = None
    concurrency: int | None = None
    request_count: int | None = None
    duration_seconds: float | None = None
    endpoint_type: str | None = None
    swept_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunData:
    """
    Complete data for a single profiling run.

    Args:
        metadata: Metadata for the run.
        requests: DataFrame containing per-request data, or None if not loaded.
        aggregated: Dictionary containing aggregated statistics.
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

        Examples:
            >>> loader = DataLoader()
            >>> run = loader.load_run(Path("results/my_run"))
            >>> print(run.requests.shape)
            (100, 15)
        """
        if not run_path.exists():
            raise DataLoadError("Run path does not exist", path=str(run_path))

        if not run_path.is_dir():
            raise DataLoadError("Run path is not a directory", path=str(run_path))

        self.info(f"Loading run from: {run_path}")

        # Load JSONL per-request data (conditionally)
        if load_per_request_data:
            requests_df = self._load_jsonl(run_path / PROFILE_EXPORT_JSONL)
        else:
            # Verify JSONL exists but don't load it
            jsonl_path = run_path / PROFILE_EXPORT_JSONL
            if not jsonl_path.exists():
                raise DataLoadError(
                    "Required JSONL file not found", path=str(jsonl_path)
                )
            requests_df = None
            self.info(f"Skipping per-request data load from {jsonl_path}")

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

        Examples:
            >>> loader = DataLoader()
            >>> runs = loader.load_multiple_runs([
            ...     Path("results/run1"),
            ...     Path("results/run2")
            ... ])
            >>> print(len(runs))
            2
        """
        if not run_paths:
            raise DataLoadError("No run paths provided")

        self.info(f"Loading {len(run_paths)} runs (skipping per-request data)")

        # Load all runs with only aggregated summary data, not per-request data
        runs = []
        for path in run_paths:
            try:
                run = self.load_run(path, load_per_request_data=False)
                runs.append(run)
            except DataLoadError as e:
                self.error(f"Failed to load run from {path}: {e}")
                raise

        swept_params = self._detect_swept_parameters(runs)
        self.info(f"Detected swept parameters: {list(swept_params.keys())}")

        # Update metadata with swept params
        for run in runs:
            run.metadata.swept_params = self._extract_swept_params(run, swept_params)

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

    def get_available_metrics(self, run_data: RunData) -> dict[str, dict[str, str]]:
        """
        Get metrics that are both available in the data and defined in MetricRegistry.

        This method takes the intersection of:
        1. Metrics present in the profile_export_aiperf.json data
        2. Metrics defined in the MetricRegistry (via METRIC_DISPLAY_NAMES and METRIC_UNITS)

        This ensures only metrics that exist in both the data and the registry are returned,
        preventing errors when trying to plot metrics that aren't available.

        Args:
            run_data: RunData object with loaded aggregated data.

        Returns:
            Dictionary with two keys:
                - "display_names": dict mapping metric tag to display name
                - "units": dict mapping metric tag to unit string
            Only includes metrics present in both the data and the registry.

        Examples:
            >>> loader = DataLoader()
            >>> run = loader.load_run(Path("results/run1"))
            >>> available = loader.get_available_metrics(run)
            >>> print(available["display_names"])
            {'time_to_first_token': 'Time to First Token', 'inter_token_latency': 'Inter Token Latency'}
        """
        if not run_data.aggregated or "metrics" not in run_data.aggregated:
            self.warning("No metrics found in aggregated data")
            return {"display_names": {}, "units": {}}

        # Get metrics present in the data
        data_metrics = set(run_data.aggregated["metrics"].keys())

        # Get metrics defined in the registry
        registry_metrics = set(METRIC_DISPLAY_NAMES.keys())

        # Take intersection
        available_metrics = data_metrics & registry_metrics

        if not available_metrics:
            self.warning(
                f"No overlap between data metrics ({len(data_metrics)}) and "
                f"registry metrics ({len(registry_metrics)})"
            )
            return {"display_names": {}, "units": {}}

        # Build filtered dictionaries
        display_names = {
            metric: METRIC_DISPLAY_NAMES[metric] for metric in available_metrics
        }
        units = {metric: METRIC_UNITS[metric] for metric in available_metrics}

        self.info(
            f"Found {len(available_metrics)} available metrics: {sorted(available_metrics)}"
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
            raise DataLoadError("Required JSONL file not found", path=str(jsonl_path))

        records = []
        corrupted_lines = 0

        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                        # Flatten the record
                        flat_record = self._flatten_jsonl_record(record)
                        records.append(flat_record)
                    except json.JSONDecodeError as e:
                        corrupted_lines += 1
                        self.warning(
                            f"Skipping corrupted line {line_num} in {jsonl_path}: {e}"
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

            # Convert timestamp columns to datetime with UTC timezone
            timestamp_cols = [
                "request_start_ns",
                "request_ack_ns",
                "request_end_ns",
                "cancellation_time_ns",
            ]
            for col in timestamp_cols:
                if col in df.columns:
                    # Convert nanoseconds to datetime
                    df[col] = pd.to_datetime(
                        df[col], unit="ns", utc=True, errors="coerce"
                    )

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

    def _flatten_jsonl_record(self, record: dict) -> dict:
        """
        Flatten a JSONL record into a single-level dictionary.

        Args:
            record: Nested dictionary from JSONL line.

        Returns:
            Flattened dictionary with metrics and metadata at top level.
        """
        flat = {}

        # Flatten metadata
        if "metadata" in record:
            for key, value in record["metadata"].items():
                flat[key] = value

        # Flatten metrics (extract just the value, drop unit)
        if "metrics" in record:
            for key, value in record["metrics"].items():
                if isinstance(value, dict) and "value" in value:
                    # Compute per-request statistics for inter_chunk_latency (stream health/jitter)
                    if key == "inter_chunk_latency":
                        stats = self._compute_inter_chunk_latency_stats(value["value"])
                        flat.update(stats)
                        continue
                    flat[key] = value["value"]
                else:
                    flat[key] = value

        # Include error field
        if "error" in record:
            flat["error"] = record["error"]

        return flat

    def _load_aggregated_json(self, json_path: Path) -> dict[str, Any]:
        """
        Load aggregated statistics from JSON file.

        Args:
            json_path: Path to the profile_export_aiperf.json file.

        Returns:
            Dictionary containing aggregated statistics.

        Raises:
            DataLoadError: If file cannot be read or parsed.
        """
        if not json_path.exists():
            raise DataLoadError("Required JSON file not found", path=str(json_path))

        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
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
                duration_ns = (
                    end_times.max() - start_times.min()
                ).total_seconds() * 1e9
                duration_seconds = duration_ns / 1e9

        return RunMetadata(
            run_name=run_name,
            run_path=run_path,
            model=model,
            concurrency=concurrency,
            request_count=request_count,
            duration_seconds=duration_seconds,
            endpoint_type=endpoint_type,
        )

    def _detect_swept_parameters(self, runs: list[RunData]) -> dict[str, set[Any]]:
        """
        Detect which parameters were swept across multiple runs.

        A parameter is considered swept if it has different values across runs.

        Args:
            runs: List of RunData objects.

        Returns:
            Dictionary mapping parameter names to sets of values.
        """
        if len(runs) < 2:
            return {}

        # Collect all config parameters from all runs
        param_values: dict[str, list[Any]] = {}

        for run in runs:
            if not run.aggregated or "input_config" not in run.aggregated:
                continue

            config = run.aggregated["input_config"]

            # Flatten config to extract parameters
            flat_config = self._flatten_config(config)

            for key, value in flat_config.items():
                if key not in param_values:
                    param_values[key] = []
                param_values[key].append(value)

        # Identify swept parameters (multiple unique values)
        swept = {}
        for key, values in param_values.items():
            # Get unique values, preserving type
            unique_values = set()
            for v in values:
                # Convert unhashable types to hashable
                if isinstance(v, list):
                    # Single element lists → extract the value (common for model_names)
                    v = v[0] if len(v) == 1 else tuple(v)
                elif isinstance(v, dict):
                    v = tuple(sorted(v.items()))
                unique_values.add(v)

            if len(unique_values) > 1:
                swept[key] = unique_values

        return swept

    def _flatten_config(
        self, config: dict[str, Any], prefix: str = ""
    ) -> dict[str, Any]:
        """
        Flatten nested configuration dictionary.

        Args:
            config: Nested configuration dictionary.
            prefix: Prefix for keys (used in recursion).

        Returns:
            Flattened dictionary.
        """
        flat = {}
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Recurse into nested dicts
                flat.update(self._flatten_config(value, full_key))
            else:
                flat[full_key] = value

        return flat

    def _extract_swept_params(
        self, run: RunData, swept_params: dict[str, set[Any]]
    ) -> dict[str, Any]:
        """
        Extract swept parameter values for a specific run.

        Args:
            run: RunData object.
            swept_params: Dictionary of swept parameters from
                _detect_swept_parameters.

        Returns:
            Dictionary mapping swept parameter names to their values for this
            run.
        """
        if not swept_params:
            return {}

        if not run.aggregated or "input_config" not in run.aggregated:
            return {}

        config = run.aggregated["input_config"]
        flat_config = self._flatten_config(config)

        # Extract values for swept parameters
        swept_values = {}
        for param_name in swept_params:
            if param_name in flat_config:
                value = flat_config[param_name]
                # Apply same normalization as in _detect_swept_parameters
                if isinstance(value, list):
                    value = value[0] if len(value) == 1 else tuple(value)
                elif isinstance(value, dict):
                    value = tuple(sorted(value.items()))
                swept_values[param_name] = value

        return swept_values
