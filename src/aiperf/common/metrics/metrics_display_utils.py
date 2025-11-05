# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities for formatting and displaying metrics."""

from aiperf.common.enums.metric_enums import (
    EnergyMetricUnit,
    FrequencyMetricUnit,
    GenericMetricUnit,
    MetricSizeUnit,
    MetricTimeUnit,
    MetricUnitT,
    PowerMetricUnit,
    TemperatureMetricUnit,
)

# Acronyms that should be uppercase in display names
METRIC_NAME_ACRONYMS = {
    "api", "cpu", "d2d", "d2h", "dcgm", "e2e", "fb", "fds",
    "gpu", "h2d", "http", "id", "kv", "kvbm", "sm", "ttft",
    "url", "uuid", "vllm", "xid",
}  # fmt: skip


def format_metric_display_name(field_name: str) -> str:
    """Format a field name into a human-readable display name.

    Handles acronyms and capitalizes words appropriately.

    Args:
        field_name: The metric field name (e.g., "kvstats_gpu_cache_usage_percent")

    Returns:
        Human-readable display name (e.g., "KV Stats GPU Cache Usage Percent")

    Example:
        >>> format_metric_display_name("vllm_num_requests_running")
        'vLLM Num Requests Running'
        >>> format_metric_display_name("gpu_power_usage")
        'GPU Power Usage'
        >>> format_metric_display_name("kvbm_offload_blocks_d2h")
        'KVBM Offload Blocks D2H'
    """
    return " | ".join(
        " ".join(
            word.upper() if word.lower() in METRIC_NAME_ACRONYMS else word.capitalize()
            for word in key.split("_")
        )
        for key in field_name.split(":")
    )


def infer_metric_unit(field_name: str) -> MetricUnitT:
    """Infer the metric unit from the field name.

    Uses common naming conventions to determine the appropriate unit.
    Checks GPU-specific patterns first, then general patterns.

    Args:
        field_name: The metric field name (e.g., "gpu_power_usage")

    Returns:
        MetricUnitT enum representing the unit

    Example:
        >>> infer_metric_unit("gpu_power_usage")
        PowerMetricUnit.WATT
        >>> infer_metric_unit("gpu_memory_used")
        MetricSizeUnit.GIGABYTES
        >>> infer_metric_unit("request_duration_seconds")
        MetricTimeUnit.SECONDS
    """
    field_lower = field_name.lower()

    # GPU-specific patterns (more specific, check first)
    if "power" in field_lower and (
        "usage" in field_lower or "limit" in field_lower or "management" in field_lower
    ):
        return PowerMetricUnit.WATT

    if "energy" in field_lower or "consumption" in field_lower:
        return EnergyMetricUnit.MEGAJOULE

    if (
        ("memory" in field_lower or "fb" in field_lower)
        and "temperature" not in field_lower
        and "temp" not in field_lower
    ):
        return MetricSizeUnit.GIGABYTES

    if "clock" in field_lower or "frequency" in field_lower:
        return FrequencyMetricUnit.MEGAHERTZ

    if (
        "temperature" in field_lower
        or "temp" in field_lower
        or "thermal" in field_lower
    ):
        return TemperatureMetricUnit.CELSIUS

    if "util" in field_lower or "utilization" in field_lower:
        return GenericMetricUnit.PERCENT

    if "violation" in field_lower:
        return MetricTimeUnit.MICROSECONDS

    # General patterns (server metrics and fallback)
    if field_lower.endswith("_seconds") or "_seconds_" in field_lower:
        return MetricTimeUnit.SECONDS

    if field_lower.endswith("_bytes") or "_bytes_" in field_lower:
        return MetricSizeUnit.BYTES

    if (
        field_lower.endswith(("_percent", "_perc", "_percentage"))
        or "_percent_" in field_lower
        or "_perc_" in field_lower
    ):
        return GenericMetricUnit.PERCENT

    # Default to count
    return GenericMetricUnit.COUNT
