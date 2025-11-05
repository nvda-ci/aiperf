# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Metric display name resolution for plotting.

This module provides access to human-readable display names for all metrics
including standard metrics from MetricRegistry, GPU telemetry metrics, and
derived metrics.
"""

from aiperf.gpu_telemetry.constants import GPU_TELEMETRY_METRICS_CONFIG
from aiperf.metrics.metric_registry import MetricRegistry

# Pre-compute all metric display names at module load time
_ALL_METRIC_NAMES: dict[str, str] = {
    # Standard metrics from MetricRegistry
    **{
        metric_class.tag: metric_class.header
        for metric_class in MetricRegistry.all_classes()
        if metric_class.header
    },
    # GPU telemetry metrics
    **{
        field_name: display_name
        for display_name, field_name, _ in GPU_TELEMETRY_METRICS_CONFIG
    },
    # Derived metrics calculated during data processing
    "output_token_throughput_per_gpu": "Output Token Throughput Per GPU",
}


def get_all_metric_display_names() -> dict[str, str]:
    """
    Get display names for all metrics (standard + GPU telemetry + derived).

    Returns:
        Dictionary mapping metric tag/field to display name

    Examples:
        >>> names = get_all_metric_display_names()
        >>> names["time_to_first_token"]
        'Time to First Token'
        >>> names["gpu_power_usage"]
        'GPU Power Usage'
        >>> names["output_token_throughput_per_gpu"]
        'Output Token Throughput Per GPU'
    """
    return _ALL_METRIC_NAMES


def get_metric_display_name(metric_tag: str) -> str:
    """
    Get display name for a metric tag with fallback to title-cased tag.

    Args:
        metric_tag: The metric identifier (e.g., "time_to_first_token")

    Returns:
        Human-readable display name

    Examples:
        >>> get_metric_display_name("time_to_first_token")
        'Time to First Token'
        >>> get_metric_display_name("unknown_metric")
        'Unknown Metric'
    """
    return _ALL_METRIC_NAMES.get(metric_tag, metric_tag.replace("_", " ").title())
