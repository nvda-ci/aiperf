# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics utilities for the AIPerf API.

Provides helper functions for building labels and formatting metrics as JSON.
"""

from __future__ import annotations

from importlib.metadata import version

import orjson

from aiperf.api.prometheus_formatter import InfoLabels
from aiperf.common.config import UserConfig, coerce_value
from aiperf.common.exceptions import MetricTypeError
from aiperf.common.models import MetricResult


def build_info_labels(user_config: UserConfig) -> InfoLabels:
    """Build info labels for metrics from UserConfig.

    These labels identify the benchmark and are included in Prometheus metrics.

    Args:
        user_config: The user configuration for the benchmark.

    Returns:
        Dictionary of label names to values for the info metric.
    """
    labels: InfoLabels = {}

    if user_config.benchmark_id:
        labels["benchmark_id"] = user_config.benchmark_id

    labels["model"] = ",".join(user_config.endpoint.model_names)
    labels["endpoint_type"] = user_config.endpoint.type
    labels["streaming"] = str(user_config.endpoint.streaming).lower()

    if user_config.loadgen.concurrency is not None:
        labels["concurrency"] = str(user_config.loadgen.concurrency)

    if user_config.loadgen.request_rate is not None:
        labels["request_rate"] = str(user_config.loadgen.request_rate)

    labels["config"] = user_config.model_dump_json(exclude_unset=True)

    return labels


def format_metrics_json(
    metrics: list[MetricResult],
    info_labels: InfoLabels | None = None,
    benchmark_id: str | None = None,
) -> bytes:
    """Format metrics as JSON.

    Args:
        metrics: List of MetricResult objects from realtime metrics.
        info_labels: Optional dict of labels for additional metadata.
        benchmark_id: Optional benchmark ID to include.

    Returns:
        JSON bytes containing formatted metrics.
    """
    result: dict = {
        "aiperf_version": version("aiperf"),
    }

    if benchmark_id:
        result["benchmark_id"] = benchmark_id

    if info_labels:
        result.update(
            {
                key: coerce_value(value)
                for key, value in info_labels.items()
                if key not in ("config", "version")
            }
        )

    metrics_dict = {}
    for metric in metrics:
        try:
            display_metric = metric.to_display_unit()
        except MetricTypeError:
            display_metric = metric
        metrics_dict[metric.tag] = display_metric.model_dump(
            mode="json", exclude_none=True, exclude={"tag"}
        )

    result["metrics"] = metrics_dict
    return orjson.dumps(result, option=orjson.OPT_INDENT_2)
