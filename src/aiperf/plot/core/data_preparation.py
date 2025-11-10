# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data preparation utilities for plot generation.

This module contains reusable functions for transforming raw benchmark data
into formats suitable for plotting.
"""

import json
from typing import Any

import pandas as pd

from aiperf.plot.core.data_loader import RunData


def prepare_request_timeseries(run: RunData) -> pd.DataFrame:
    """
    Convert per-request data into a DataFrame for time series plotting.

    Args:
        run: RunData object with requests DataFrame

    Returns:
        DataFrame with per-request metrics including request_number and timestamp
    """
    if run.requests is None or run.requests.empty:
        return pd.DataFrame()

    df = run.requests.copy()

    df["request_number"] = range(len(df))

    if "request_end_ns" in df.columns:
        # Handle both datetime and numeric timestamp formats
        if pd.api.types.is_datetime64_any_dtype(df["request_end_ns"]):
            # Already datetime, convert to seconds
            df["timestamp"] = df["request_end_ns"].astype("int64") / 1_000_000_000
        else:
            # Numeric nanoseconds, divide directly
            df["timestamp"] = df["request_end_ns"] / 1_000_000_000

        # Normalize to start from 0
        if len(df) > 0:
            df["timestamp"] = df["timestamp"] - df["timestamp"].min()

    return df


def calculate_throughput_events(requests_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate throughput using event-based approach with evenly dispersed tokens.

    Tokens are evenly distributed across the generation phase (from TTFT to request_end)
    rather than being counted at a single event. This creates smooth throughput curves
    that accurately represent the token generation rate over time.

    Args:
        requests_df: DataFrame with request data including timestamps and metrics

    Returns:
        DataFrame with columns: timestamp_s, throughput_tokens_per_sec, active_requests
    """
    events = []

    for _, row in requests_df.iterrows():
        request_start_ns = row.get("request_start_ns")
        request_end_ns = row.get("request_end_ns")

        if pd.isna(request_start_ns) or pd.isna(request_end_ns):
            continue

        if isinstance(request_start_ns, pd.Timestamp):
            request_start_ns = request_start_ns.value
        if isinstance(request_end_ns, pd.Timestamp):
            request_end_ns = request_end_ns.value

        request_start_ns = int(request_start_ns)
        request_end_ns = int(request_end_ns)

        ttft_ms = row.get("time_to_first_token", 0)
        if pd.isna(ttft_ms):
            ttft_ms = 0

        generation_start_ns = (
            request_start_ns + int(ttft_ms * 1e6) if ttft_ms > 0 else request_start_ns
        )

        output_tokens = row.get("output_sequence_length", 0)
        if pd.isna(output_tokens):
            output_tokens = 0

        generation_duration_ns = request_end_ns - generation_start_ns

        if generation_duration_ns > 0 and output_tokens > 0:
            token_rate = output_tokens / (generation_duration_ns / 1e9)

            events.append(
                {
                    "timestamp_ns": generation_start_ns,
                    "delta_rate": token_rate,
                    "active_delta": 1,
                }
            )
            events.append(
                {
                    "timestamp_ns": request_end_ns,
                    "delta_rate": -token_rate,
                    "active_delta": -1,
                }
            )

    if not events:
        return pd.DataFrame(
            columns=["timestamp_s", "throughput_tokens_per_sec", "active_requests"]
        )

    events_df = pd.DataFrame(events).sort_values("timestamp_ns")

    events_df["throughput_tokens_per_sec"] = events_df["delta_rate"].cumsum()
    events_df["active_requests"] = events_df["active_delta"].cumsum()

    start_ns = events_df["timestamp_ns"].min()
    events_df["timestamp_s"] = (events_df["timestamp_ns"] - start_ns) / 1e9

    return events_df[
        ["timestamp_s", "throughput_tokens_per_sec", "active_requests"]
    ].reset_index(drop=True)


def validate_request_uniformity(
    run: RunData, logger: Any | None = None
) -> tuple[bool, str | None]:
    """
    Check if requests have uniform ISL/OSL or if they vary.

    If per-request data is not loaded in run.requests, attempts to load
    ISL/OSL values from profile_export.jsonl on-demand.

    Args:
        run: RunData object with requests DataFrame (may be None)
        logger: Optional logger instance for warnings

    Returns:
        Tuple of (is_uniform, warning_message) where:
        - is_uniform: True if all ISL and OSL are identical, False otherwise
        - warning_message: Warning text if non-uniform, None if uniform
    """
    if run.requests is not None and not run.requests.empty:
        df = run.requests
        has_isl = "input_sequence_length" in df.columns
        has_osl = "output_sequence_length" in df.columns

        if not has_isl and not has_osl:
            return True, None

        isl_values = df["input_sequence_length"].dropna() if has_isl else pd.Series()
        osl_values = df["output_sequence_length"].dropna() if has_osl else pd.Series()
    else:
        profile_path = run.metadata.run_path / "profile_export.jsonl"
        if not profile_path.exists():
            return True, None

        try:
            isl_values = []
            osl_values = []

            with open(profile_path) as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        metrics = record.get("metrics", {})

                        isl = metrics.get("input_sequence_length", {})
                        if isinstance(isl, dict) and "value" in isl:
                            isl_values.append(isl["value"])

                        osl = metrics.get("output_sequence_length", {})
                        if isinstance(osl, dict) and "value" in osl:
                            osl_values.append(osl["value"])
                    except json.JSONDecodeError:
                        continue

            if not isl_values and not osl_values:
                return True, None

            isl_values = pd.Series(isl_values) if isl_values else pd.Series()
            osl_values = pd.Series(osl_values) if osl_values else pd.Series()

        except Exception as e:
            if logger:
                logger.warning(f"Could not load ISL/OSL data for uniformity check: {e}")
            return True, None

    is_uniform = True

    if len(isl_values) > 1 and isl_values.nunique() > 1:
        is_uniform = False

    if len(osl_values) > 1 and osl_values.nunique() > 1:
        is_uniform = False

    if not is_uniform:
        warning_message = (
            "⚠ Requests have varying ISL/OSL. "
            "Req/sec throughput may not accurately represent workload capacity."
        )
        return False, warning_message

    return True, None


def calculate_rolling_percentiles(
    df: pd.DataFrame, metric_col: str, window_size: int | None = None
) -> pd.DataFrame:
    """
    Calculate rolling percentiles (p50, p95, p99) for a metric.

    Args:
        df: DataFrame with metric data
        metric_col: Column name for the metric
        window_size: Window size for rolling calculation (defaults to min(10, len(df)))

    Returns:
        DataFrame with additional p50, p95, p99 columns
    """
    df_sorted = df.copy()

    if window_size is None:
        window_size = min(10, len(df_sorted))

    df_sorted["p50"] = (
        df_sorted[metric_col].rolling(window=window_size, min_periods=1).quantile(0.50)
    )
    df_sorted["p95"] = (
        df_sorted[metric_col].rolling(window=window_size, min_periods=1).quantile(0.95)
    )
    df_sorted["p99"] = (
        df_sorted[metric_col].rolling(window=window_size, min_periods=1).quantile(0.99)
    )

    return df_sorted


def prepare_timeslice_metrics(
    run: RunData, metric_name: str, stat: str
) -> pd.DataFrame:
    """
    Extract and prepare timeslice data for a specific metric and stat.

    Args:
        run: RunData object with timeslices DataFrame
        metric_name: Name of the metric to extract
        stat: Statistic to extract (e.g., "avg", "p50", "p95")

    Returns:
        DataFrame with Timeslice and stat columns, ready for histogram plotting
    """
    if run.timeslices is None or run.timeslices.empty:
        return pd.DataFrame()

    # Filter for specific metric and stat
    metric_data = run.timeslices[
        (run.timeslices["Metric"] == metric_name) & (run.timeslices["Stat"] == stat)
    ].copy()

    if metric_data.empty:
        raise ValueError(f"No timeslice data for {metric_name} ({stat})")

    # Prepare data for plotting
    plot_df = metric_data[["Timeslice", "Value"]].copy()
    plot_df = plot_df.rename(columns={"Value": stat})

    return plot_df


def aggregate_gpu_telemetry(run: RunData) -> pd.DataFrame:
    """
    Aggregate GPU telemetry data by averaging across GPUs at each timestamp.

    Args:
        run: RunData object with gpu_telemetry DataFrame

    Returns:
        DataFrame with timestamp_s and averaged gpu_utilization
    """
    if run.gpu_telemetry is None or run.gpu_telemetry.empty:
        return pd.DataFrame()

    gpu_df = run.gpu_telemetry.copy()

    # If gpu_index column exists, group by timestamp and average
    if "gpu_index" in gpu_df.columns:
        gpu_df = (
            gpu_df.groupby("timestamp_s").agg({"gpu_utilization": "mean"}).reset_index()
        )

    return gpu_df


def flatten_config(config: dict, parent_key: str = "") -> dict[str, Any]:
    """
    Flatten nested config dictionary into dot-notation keys.

    Args:
        config: Nested configuration dictionary
        parent_key: Parent key for recursion (used internally)

    Returns:
        Flattened dictionary with dot-notation keys
    """
    items = {}
    for key, value in config.items():
        new_key = f"{parent_key}.{key}" if parent_key else key

        if isinstance(value, dict) and not _is_leaf_config(value):
            # Recursively flatten nested dicts
            items.update(flatten_config(value, new_key))
        elif isinstance(value, list) and len(value) == 1:
            # Unwrap single-item lists
            items[new_key] = value[0]
        else:
            items[new_key] = value

    return items


def _is_leaf_config(value: dict) -> bool:
    """
    Check if a dict is a leaf config node (contains 'value', 'unit', etc).

    Args:
        value: Dictionary to check

    Returns:
        True if this is a leaf config node, False otherwise
    """
    # Leaf nodes typically have keys like 'value', 'unit', 'description'
    leaf_keys = {"value", "unit", "description", "type"}
    return any(k in value for k in leaf_keys)
