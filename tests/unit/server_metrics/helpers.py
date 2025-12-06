# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test helpers for server metrics export stats tests."""

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models.server_metrics_models import (
    HistogramSnapshot,
    ServerMetricsTimeSeries,
    SummarySnapshot,
)


def add_gauge_samples(
    ts: ServerMetricsTimeSeries,
    name: str,
    values: list[float],
    start_ns: int = 0,
    interval_ns: int = NANOS_PER_SECOND,
) -> None:
    """Add gauge samples at regular intervals."""
    for i, value in enumerate(values):
        ts.append_snapshot(start_ns + i * interval_ns, gauge_metrics={name: value})


def add_counter_samples(
    ts: ServerMetricsTimeSeries,
    name: str,
    values: list[float],
    start_ns: int = 0,
    interval_ns: int = NANOS_PER_SECOND,
) -> None:
    """Add counter samples at regular intervals."""
    for i, value in enumerate(values):
        ts.append_snapshot(start_ns + i * interval_ns, counter_metrics={name: value})


def add_histogram_snapshots(
    ts: ServerMetricsTimeSeries,
    name: str,
    snapshots: list[tuple[int, HistogramSnapshot]],
) -> None:
    """Add histogram snapshots at specified timestamps."""
    for timestamp_ns, snapshot in snapshots:
        ts.append_snapshot(timestamp_ns, histogram_metrics={name: snapshot})


def add_summary_snapshots(
    ts: ServerMetricsTimeSeries,
    name: str,
    snapshots: list[tuple[int, SummarySnapshot]],
) -> None:
    """Add summary snapshots at specified timestamps."""
    for timestamp_ns, snapshot in snapshots:
        ts.append_snapshot(timestamp_ns, summary_metrics={name: snapshot})


def hist(buckets: dict[str, float], sum_: float, count: float) -> HistogramSnapshot:
    """Shorthand for creating HistogramSnapshot."""
    return HistogramSnapshot(buckets=buckets, sum=sum_, count=count)


def summary(quantiles: dict[str, float], sum_: float, count: float) -> SummarySnapshot:
    """Shorthand for creating SummarySnapshot."""
    return SummarySnapshot(quantiles=quantiles, sum=sum_, count=count)
