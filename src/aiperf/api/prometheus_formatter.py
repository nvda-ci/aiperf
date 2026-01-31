# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prometheus Exposition Format formatter for AIPerf metrics.

Converts MetricResult objects to Prometheus-compatible text format for scraping.
"""

from __future__ import annotations

import re
from importlib.metadata import version

from aiperf.common.exceptions import MetricTypeError
from aiperf.common.models import MetricResult
from aiperf.metrics.metric_registry import MetricRegistry

# Type alias for info labels dict
InfoLabels = dict[str, str]

# Stats to expose with human-readable names for HELP text
STAT_DISPLAY_NAMES = {
    "avg": "average",
    "sum": "total",
    "p1": "1st percentile",
    "p5": "5th percentile",
    "p10": "10th percentile",
    "p25": "25th percentile",
    "p50": "50th percentile",
    "p75": "75th percentile",
    "p90": "90th percentile",
    "p95": "95th percentile",
    "p99": "99th percentile",
    "min": "minimum",
    "max": "maximum",
    "std": "standard deviation",
}

# Mapping of strings to replace in unit display names
_REPLACE_MAP = {
    "/": "_per_",
    "tokens_per_sec": "tps",
    "__": "_",
}

# Regex for sanitizing metric names to Prometheus format
_METRIC_NAME_RE = re.compile(r"[^a-zA-Z0-9_]")


def sanitize_metric_name(name: str) -> str:
    """Sanitize a metric name for Prometheus compatibility.

    Prometheus metric names must match [a-zA-Z_:][a-zA-Z0-9_:]*.
    Invalid characters are replaced with underscores.

    Args:
        name: The raw metric name/tag.

    Returns:
        A sanitized metric name valid for Prometheus.
    """
    sanitized = _METRIC_NAME_RE.sub("_", name.lower())
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


def format_labels(labels: InfoLabels) -> str:
    """Format labels dict as Prometheus label string.

    Args:
        labels: Dict of label name to value.

    Returns:
        Formatted label string like {key1="value1",key2="value2"}.
    """
    if not labels:
        return ""
    label_pairs = []
    for key, value in labels.items():
        escaped_value = str(value).replace("\\", "\\\\").replace('"', '\\"')
        label_pairs.append(f'{key}="{escaped_value}"')
    return "{" + ",".join(label_pairs) + "}"


def _format_info_metric(info_labels: InfoLabels) -> str:
    """Format the aiperf_info metric with benchmark metadata.

    Args:
        info_labels: Dict of label name to value for the info metric.

    Returns:
        Prometheus Exposition Format text for the info metric.
    """
    if not info_labels:
        return ""

    labels = {"version": version("aiperf"), **info_labels}
    label_pairs = []
    for key, value in labels.items():
        escaped_value = str(value).replace("\\", "\\\\").replace('"', '\\"')
        label_pairs.append(f'{key}="{escaped_value}"')
    labels_str = ",".join(label_pairs)

    lines = [
        "# HELP aiperf_info AIPerf benchmark information",
        "# TYPE aiperf_info gauge",
        f"aiperf_info{{{labels_str}}} 1",
    ]
    return "\n".join(lines) + "\n"


def format_as_prometheus(
    metrics: list[MetricResult],
    info_labels: InfoLabels | None = None,
) -> str:
    """Convert MetricResult list to Prometheus Exposition Format text.

    Generates raw PEF text with gauges for each metric stat.
    Converts metrics to display units for human-readable values.

    Args:
        metrics: List of MetricResult objects from realtime metrics.
        info_labels: Optional dict of labels for the aiperf_info metric.
            Key labels (excluding 'config') are also added to all metrics.

    Returns:
        Prometheus Exposition Format text string.
    """
    lines: list[str] = []

    if info_labels:
        lines.append(_format_info_metric(info_labels))

    metric_labels = {
        k: v for k, v in (info_labels or {}).items() if k not in ("config", "version")
    }
    labels_str = format_labels(metric_labels)

    for raw_metric in metrics:
        try:
            metric = raw_metric.to_display_unit()
            metric_cls = MetricRegistry.get_class(metric.tag)
            display_unit = metric_cls.display_unit or metric_cls.unit
            unit_suffix = str(display_unit)
            for old, new in _REPLACE_MAP.items():
                unit_suffix = unit_suffix.replace(old, new)
            unit_display = display_unit.display_name()
        except MetricTypeError:
            metric = raw_metric
            unit_suffix = ""
            unit_display = metric.unit

        base_name = f"aiperf_{sanitize_metric_name(metric.tag)}"

        for stat, stat_display in STAT_DISPLAY_NAMES.items():
            value = getattr(metric, stat, None)
            if value is None:
                continue

            if stat == "sum":
                metric_name = f"{base_name}_{unit_suffix}_total"
            else:
                metric_name = f"{base_name}_{stat}_{unit_suffix}"

            lines.append(
                f"# HELP {metric_name} {metric.header} {stat_display} (in {unit_display})"
            )
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name}{labels_str} {value}")

    return "\n".join(lines) + "\n" if lines else ""
