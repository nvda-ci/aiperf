# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prometheus utilities."""

# ============================================================================
# Metrics Utilities
# ============================================================================

from typing import Any
from urllib.parse import urlparse

import numpy as np

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import MetricResult


def compute_histogram_delta(
    start_histogram: dict[str, float], end_histogram: dict[str, float]
) -> dict[str, float]:
    """Compute delta between two cumulative histogram snapshots.

    Prometheus histograms are cumulative counters. To get the delta
    for a time window, subtract the start histogram from the end histogram.

    Args:
        start_histogram: Histogram buckets at start of time window
        end_histogram: Histogram buckets at end of time window

    Returns:
        Delta histogram with bucket-by-bucket differences

    Raises:
        ValueError: If buckets don't match between start and end
    """
    if set(start_histogram.keys()) != set(end_histogram.keys()):
        raise ValueError(
            "Histogram bucket boundaries don't match between start and end"
        )

    delta: dict[str, float] = {}
    for le_str in end_histogram:
        delta[le_str] = end_histogram[le_str] - start_histogram.get(le_str, 0)

    return delta


def compute_metric_statistics(
    data_points: list[tuple[float, int]],
    tag: str,
    header: str,
    unit: str,
    metric_name: str | None = None,
) -> Any:
    """Compute comprehensive statistics from time series data points.

    This function is used by both GPU telemetry and server metrics to compute
    statistics (min, max, avg, std, percentiles) from time series data.

    Args:
        data_points: List of (value, timestamp_ns) tuples
        tag: Unique identifier for this metric (used by dashboard, exports, API)
        header: Human-readable name for display
        unit: Unit of measurement (e.g., "W" for Watts, "%" for percentage)
        metric_name: Optional metric name for error messages

    Returns:
        MetricResult with min/max/avg/percentiles computed from time series

    Raises:
        NoMetricValue: If data_points list is empty
    """

    if not data_points:
        msg = "No data points available"
        if metric_name:
            msg = f"No data available for metric '{metric_name}'"
        raise NoMetricValue(msg)

    # Extract values from (value, timestamp) tuples
    values = np.array([point[0] for point in data_points])

    # Compute all percentiles in one call for efficiency
    p1, p5, p10, p25, p50, p75, p90, p95, p99 = np.percentile(
        values, [1, 5, 10, 25, 50, 75, 90, 95, 99]
    )

    return MetricResult(
        tag=tag,
        header=header,
        unit=unit,
        min=np.min(values),
        max=np.max(values),
        avg=float(np.mean(values)),
        std=float(np.std(values)),
        count=len(values),
        current=float(data_points[-1][0]),  # Most recent value
        p1=p1,
        p5=p5,
        p10=p10,
        p25=p25,
        p50=p50,
        p75=p75,
        p90=p90,
        p95=p95,
        p99=p99,
    )


def compute_metric_statistics_from_histogram(
    buckets: dict[str, float],
    sum_value: float,
    count: int,
    tag: str,
    header: str,
    unit: str,
    metric_name: str | None = None,
) -> MetricResult:
    """Create MetricResult from histogram data without estimating percentiles.

    For Prometheus histogram metrics, we store the raw bucket data and only
    compute the average from sum/count. We do NOT estimate percentiles from
    bucket boundaries as this would be inaccurate. The raw histogram delta
    is stored on the MetricResult for export.

    Args:
        buckets: Histogram bucket boundaries to counts
        sum_value: Sum of all observed values
        count: Total number of observations
        tag: Unique identifier for this metric (used by dashboard, exports, API)
        header: Human-readable name for display
        unit: Unit of measurement (e.g., "ms", "s")
        metric_name: Optional metric name for error messages

    Returns:
        MetricResult with only avg computed, percentiles set to 0.0

    Raises:
        NoMetricValue: If count is zero (no observations)
    """
    if count == 0:
        msg = "Histogram has zero observations"
        if metric_name:
            msg = f"Histogram metric '{metric_name}' has zero observations"
        raise NoMetricValue(msg)

    avg = sum_value / count

    # For histograms, we only store the average and count.
    # Percentiles are set to 0.0 since we don't estimate them from buckets.
    # The raw histogram delta is stored separately on the MetricResult.
    return MetricResult(
        tag=tag,
        header=header,
        unit=unit,
        min=0.0,
        max=0.0,
        avg=avg,
        std=0.0,
        count=count,
        current=0.0,
        p1=0.0,
        p5=0.0,
        p10=0.0,
        p25=0.0,
        p50=0.0,
        p75=0.0,
        p90=0.0,
        p95=0.0,
        p99=0.0,
    )


def normalize_metrics_endpoint_url(url: str) -> str:
    """Ensure metrics endpoint URL ends with /metrics suffix.

    Works with Prometheus, DCGM, and other compatible endpoints.
    This utility is used by both TelemetryManager and ServerMetricsManager
    to ensure consistent URL formatting.

    Args:
        url: Base URL or full metrics URL (e.g., "http://localhost:9400" or
             "http://localhost:9400/metrics")

    Returns:
        URL ending with /metrics with trailing slashes removed
        (e.g., "http://localhost:9400/metrics")

    Examples:
        >>> normalize_metrics_endpoint_url("http://localhost:9400")
        "http://localhost:9400/metrics"
        >>> normalize_metrics_endpoint_url("http://localhost:9400/")
        "http://localhost:9400/metrics"
        >>> normalize_metrics_endpoint_url("http://localhost:9400/metrics")
        "http://localhost:9400/metrics"
    """
    url = url.rstrip("/")
    if not url.endswith("/metrics"):
        url = f"{url}/metrics"
    return url


def build_hostname_aware_prometheus_endpoints(
    inference_endpoint_url: str,
    default_ports: list[int],
) -> list[str]:
    """Build hostname-aware Prometheus/DCGM endpoint URLs based on inference endpoint.

    Extracts hostname and scheme from the inference endpoint URL and generates
    Prometheus-compatible URLs for the specified ports on the same hostname.
    This enables zero-config telemetry for distributed deployments.

    Args:
        inference_endpoint_url: The inference endpoint URL (e.g., http://myserver:8000/v1/chat)
        default_ports: List of ports to check on the same hostname (e.g., [9400, 9401])

    Returns:
        List of Prometheus endpoint URLs with /metrics suffix

    Examples:
        >>> build_hostname_aware_prometheus_endpoints("http://localhost:8000", [9400, 9401])
        ['http://localhost:9400/metrics', 'http://localhost:9401/metrics']
        >>> build_hostname_aware_prometheus_endpoints("http://gpu-server:8000", [8081, 6880])
        ['http://gpu-server:8081/metrics', 'http://gpu-server:6880/metrics']
    """
    parsed = urlparse(inference_endpoint_url)
    hostname = parsed.hostname or "localhost"
    scheme = parsed.scheme or "http"

    endpoints = []
    for port in default_ports:
        endpoint_url = f"{scheme}://{hostname}:{port}/metrics"
        endpoints.append(endpoint_url)

    return endpoints
