# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from urllib.parse import urlparse


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

    Raises:
        ValueError: If URL is empty or whitespace-only

    Examples:
        >>> normalize_metrics_endpoint_url("http://localhost:9400")
        "http://localhost:9400/metrics"
        >>> normalize_metrics_endpoint_url("http://localhost:9400/")
        "http://localhost:9400/metrics"
        >>> normalize_metrics_endpoint_url("http://localhost:9400/metrics")
        "http://localhost:9400/metrics"
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty or whitespace-only")

    url = url.rstrip("/")
    if not url.endswith("/metrics"):
        url = f"{url}/metrics"
    return url


def build_hostname_aware_prometheus_endpoints(
    inference_endpoint_url: str,
    default_ports: list[int],
    include_inference_port: bool = True,
) -> list[str]:
    """Build hostname-aware Prometheus/DCGM endpoint URLs based on inference endpoint.

    Extracts hostname and scheme from the inference endpoint URL and generates
    Prometheus-compatible URLs for the specified ports on the same hostname.
    This enables zero-config telemetry for distributed deployments.

    Args:
        inference_endpoint_url: The inference endpoint URL (e.g., http://myserver:8000/v1/chat)
        default_ports: List of ports to check on the same hostname (e.g., [9400, 9401])
        include_inference_port: Whether to include the inference endpoint port in the list of ports to check

    Returns:
        List of Prometheus endpoint URLs with /metrics suffix

    Examples:
        >>> build_hostname_aware_prometheus_endpoints("http://localhost:8000", [9400, 9401])
        ['http://localhost:9400/metrics', 'http://localhost:9401/metrics']
        >>> build_hostname_aware_prometheus_endpoints("http://gpu-server:8000", [8081, 6880])
        ['http://gpu-server:8081/metrics', 'http://gpu-server:6880/metrics']
    """
    if not inference_endpoint_url.startswith("http"):
        inference_endpoint_url = f"http://{inference_endpoint_url}"
    parsed = urlparse(inference_endpoint_url)
    hostname = parsed.hostname or "localhost"
    scheme = parsed.scheme or "http"

    ports_to_check = list(default_ports)
    if include_inference_port:
        ports_to_check.insert(0, parsed.port or (443 if scheme == "https" else 80))

    # Build endpoints and deduplicate while preserving order
    endpoints = [f"{scheme}://{hostname}:{port}/metrics" for port in ports_to_check]
    return list(dict.fromkeys(endpoints))
