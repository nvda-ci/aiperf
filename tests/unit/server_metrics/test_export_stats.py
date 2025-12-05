# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models.export_models import (
    CounterExportStats,
    GaugeExportStats,
    HistogramExportStats,
    SummaryExportStats,
)
from aiperf.common.models.server_metrics_models import (
    HistogramSnapshot,
    ServerMetricsTimeSeries,
    SummarySnapshot,
    TimeRangeFilter,
)

# =============================================================================
# Test Helpers
# =============================================================================


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


# =============================================================================
# Gauge Tests
# =============================================================================


class TestGaugeExportStats:
    """Test gauge export statistics computation."""

    def test_gauge_basic_statistics(self):
        """Test gauge computes correct avg, min, max, std, time_weighted_avg."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue_depth", [10.0, 20.0, 30.0, 40.0, 50.0])

        stats = GaugeExportStats.from_time_series(ts.gauges["queue_depth"])

        assert stats.sample_count == 5
        assert stats.avg == 30.0  # (10+20+30+40+50)/5
        assert stats.min == 10.0
        assert stats.max == 50.0
        assert stats.std == pytest.approx(14.142, rel=0.01)
        # Time-weighted: (10*1 + 20*1 + 30*1 + 40*1) / 4 = 25 (excludes last sample)
        assert stats.time_weighted_avg == 25.0

    def test_gauge_percentiles_industry_standard(self):
        """Test gauge computes only p50, p90, p95, p99 (not p1, p5, p10, p25, p75)."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "kv_cache", [float(i) for i in range(100)])

        stats = GaugeExportStats.from_time_series(ts.gauges["kv_cache"])

        # Verify only standard percentiles exist
        assert stats.p50 == pytest.approx(49.5, rel=0.01)
        assert stats.p90 == pytest.approx(89.5, rel=0.01)
        assert stats.p95 == pytest.approx(94.5, rel=0.01)
        assert stats.p99 == pytest.approx(98.5, rel=0.01)

        # Verify no excessive percentiles exist
        for attr in ["p1", "p5", "p10", "p25", "p75", "first", "last"]:
            assert not hasattr(stats, attr)

    def test_gauge_with_time_filter(self):
        """Test gauge export stats respect time filtering (warmup exclusion)."""
        ts = ServerMetricsTimeSeries()
        # Warmup (0-9) + actual run (10-19)
        add_gauge_samples(ts, "queue", [float(i) for i in range(20)])

        # Filter excludes warmup (first 10 seconds)
        time_filter = TimeRangeFilter(start_ns=10 * NANOS_PER_SECOND, end_ns=None)
        stats = GaugeExportStats.from_time_series(ts.gauges["queue"], time_filter)

        # Should only include values 10-19
        assert stats.sample_count == 10
        assert stats.avg == 14.5  # (10+11+...+19)/10
        assert stats.min == 10.0
        assert stats.max == 19.0

    def test_gauge_time_weighted_avg_sparse_samples(self):
        """Test time-weighted avg for sparse samples (e.g., KV cache usage).

        Scenario: value=10 held for 9 seconds, then value=100 held for 1 second.
        Simple avg: (10 + 100 + 100) / 3 = 70
        Time-weighted avg: (10*9 + 100*1) / 10 = 19
        """
        ts = ServerMetricsTimeSeries()
        ts.append_snapshot(0, gauge_metrics={"kv_cache": 10.0})
        ts.append_snapshot(9 * NANOS_PER_SECOND, gauge_metrics={"kv_cache": 100.0})
        ts.append_snapshot(10 * NANOS_PER_SECOND, gauge_metrics={"kv_cache": 100.0})

        stats = GaugeExportStats.from_time_series(ts.gauges["kv_cache"])

        assert stats.avg == pytest.approx(70.0, rel=0.01)
        assert stats.time_weighted_avg == pytest.approx(19.0, rel=0.01)

    def test_gauge_time_weighted_avg_uniform_intervals(self):
        """Test time-weighted avg equals simple avg for uniform intervals."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [10.0, 20.0, 30.0, 40.0, 50.0])

        stats = GaugeExportStats.from_time_series(ts.gauges["queue"])

        # With uniform 1s intervals, time-weighted avg should equal simple avg
        # (excluding last sample which has 0 duration)
        # Time-weighted: (10*1 + 20*1 + 30*1 + 40*1) / 4 = 25
        assert stats.avg == 30.0  # Simple: (10+20+30+40+50)/5
        assert stats.time_weighted_avg == 25.0  # Excludes last sample

    def test_gauge_time_weighted_avg_single_sample(self):
        """Test time-weighted avg with single sample returns that value."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [42.0])

        stats = GaugeExportStats.from_time_series(ts.gauges["queue"])

        assert stats.time_weighted_avg == 42.0


# =============================================================================
# Counter Tests
# =============================================================================


class TestCounterExportStats:
    """Test counter export statistics computation."""

    def test_counter_rate_calculation(self):
        """Test counter computes correct delta and rates."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [0.0, 100.0, 250.0, 450.0, 700.0])

        stats = CounterExportStats.from_time_series(ts.counters["requests"])

        assert stats.delta == 700.0
        assert stats.rate_avg == 175.0  # 700 / 4s
        # Point-to-point rates: [100, 150, 200, 250]
        assert stats.rate_min == 100.0
        assert stats.rate_max == 250.0
        assert stats.rate_std == pytest.approx(55.9, rel=0.1)

    def test_counter_no_rate_percentiles(self):
        """Verify counter does NOT have rate percentiles (simplified)."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [100.0, 200.0])

        stats = CounterExportStats.from_time_series(ts.counters["requests"])

        for attr in ["rate_p1", "rate_p50", "rate_p99"]:
            assert not hasattr(stats, attr)

    def test_counter_with_warmup_filter(self):
        """Test counter delta calculation excludes warmup period."""
        ts = ServerMetricsTimeSeries()
        # Values: 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800
        add_counter_samples(ts, "requests", [float(i * 200) for i in range(10)])

        # Exclude warmup (first 5 seconds)
        time_filter = TimeRangeFilter(start_ns=5 * NANOS_PER_SECOND, end_ns=None)
        stats = CounterExportStats.from_time_series(
            ts.counters["requests"], time_filter
        )

        # ref_idx = 4 (value = 800 at 4s), final = 1800 at 9s
        assert stats.delta == 1000.0  # 1800 - 800
        assert stats.rate_avg == 200.0


# =============================================================================
# Histogram Tests
# =============================================================================


class TestHistogramExportStats:
    """Test histogram export statistics computation."""

    def test_histogram_basic_stats(self):
        """Test histogram computes observation_count, sum, avg, observation_rate."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist({"0.1": 10.0, "1.0": 50.0, "+Inf": 100.0}, 25.0, 100.0)),
                (
                    NANOS_PER_SECOND,
                    hist({"0.1": 30.0, "1.0": 120.0, "+Inf": 250.0}, 65.0, 250.0),
                ),
            ],
        )

        stats = HistogramExportStats.from_time_series(ts.histograms["ttft"])

        # Delta: count 250-100=150, sum 65-25=40
        assert stats.observation_count == 150
        assert stats.sum == 40.0
        assert stats.avg == pytest.approx(40.0 / 150, rel=0.01)
        assert stats.observation_rate == 150.0  # 150 obs / 1s

    def test_histogram_bucket_delta_uses_correct_reference(self):
        """CRITICAL: Test histogram bucket deltas use ref_idx, not first snapshot."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                # Warmup snapshot
                (0, hist({"0.1": 100.0, "1.0": 500.0, "+Inf": 1000.0}, 250.0, 1000.0)),
                # Reference snapshot (last before filter starts)
                (
                    5 * NANOS_PER_SECOND,
                    hist({"0.1": 200.0, "1.0": 800.0, "+Inf": 1500.0}, 400.0, 1500.0),
                ),
                # Final snapshot
                (
                    10 * NANOS_PER_SECOND,
                    hist({"0.1": 300.0, "1.0": 1200.0, "+Inf": 2000.0}, 600.0, 2000.0),
                ),
            ],
        )

        # Filter: exclude warmup (first 5 seconds)
        time_filter = TimeRangeFilter(start_ns=6 * NANOS_PER_SECOND, end_ns=None)
        stats = HistogramExportStats.from_time_series(
            ts.histograms["ttft"], time_filter
        )

        # Bucket deltas: final - ref_idx (NOT final - first!)
        assert stats.buckets == {"0.1": 100.0, "1.0": 400.0, "+Inf": 500.0}
        assert stats.observation_count == 500  # 2000 - 1500
        assert stats.sum == 200.0  # 600 - 400

    def test_histogram_estimated_percentiles(self):
        """Test histogram percentile estimation with +Inf handling."""
        ts = ServerMetricsTimeSeries()
        buckets_zero = {"0.1": 0.0, "0.2": 0.0, "0.5": 0.0, "1.0": 0.0, "+Inf": 0.0}
        buckets_final = {
            "0.1": 0.0,
            "0.2": 50.0,
            "0.5": 90.0,
            "1.0": 95.0,
            "+Inf": 100.0,
        }
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist(buckets_zero, 0.0, 0.0)),
                (NANOS_PER_SECOND, hist(buckets_final, 35.0, 100.0)),
            ],
        )

        stats = HistogramExportStats.from_time_series(ts.histograms["ttft"])

        for pct in ["p50", "p90", "p95", "p99"]:
            assert pct in stats.estimated_percentiles
        # p99 should NOT be infinity
        assert stats.estimated_percentiles["p99"] < float("inf")
        assert stats.estimated_percentiles["p99"] > 1.0

    def test_histogram_observation_rate_is_single_float(self):
        """Verify observation_rate is simplified to single float (not nested object)."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist({"1.0": 100.0, "+Inf": 100.0}, 50.0, 100.0)),
                (
                    5 * NANOS_PER_SECOND,
                    hist({"1.0": 300.0, "+Inf": 300.0}, 150.0, 300.0),
                ),
            ],
        )

        stats = HistogramExportStats.from_time_series(ts.histograms["ttft"])

        assert isinstance(stats.observation_rate, float)
        assert stats.observation_rate == 40.0  # 200 obs / 5s
        for attr in ["observation_rate_min", "observation_rate_max"]:
            assert not hasattr(stats, attr)

    def test_histogram_no_data_raises_key_error(self):
        """Test histogram raises KeyError when metric doesn't exist."""
        ts = ServerMetricsTimeSeries()

        with pytest.raises(KeyError):
            _ = ts.histograms["nonexistent"]


# =============================================================================
# Summary Tests
# =============================================================================


class TestSummaryExportStats:
    """Test summary export statistics computation."""

    def test_summary_basic_stats(self):
        """Test summary computes observation stats and quantiles."""
        ts = ServerMetricsTimeSeries()
        add_summary_snapshots(
            ts,
            "latency",
            [
                (
                    0,
                    summary(
                        {"0.5": 0.1, "0.9": 0.5, "0.95": 0.8, "0.99": 1.0}, 50.0, 100.0
                    ),
                ),
                (
                    2 * NANOS_PER_SECOND,
                    summary(
                        {"0.5": 0.12, "0.9": 0.55, "0.95": 0.85, "0.99": 1.1},
                        150.0,
                        300.0,
                    ),
                ),
            ],
        )

        stats = SummaryExportStats.from_time_series(ts.summaries["latency"])

        assert stats.observation_count == 200  # 300 - 100
        assert stats.sum == 100.0  # 150 - 50
        assert stats.avg == 0.5  # 100 / 200
        assert stats.observation_rate == 100.0  # 200 obs / 2s
        assert stats.quantiles["0.5"] == 0.12
        assert stats.quantiles["0.9"] == 0.55
        assert stats.quantiles["0.95"] == 0.85
        assert stats.quantiles["0.99"] == 1.1

    def test_summary_observation_rate_is_single_float(self):
        """Verify observation_rate is simplified to single float."""
        ts = ServerMetricsTimeSeries()
        add_summary_snapshots(
            ts,
            "latency",
            [
                (0, summary({"0.5": 0.1, "0.9": 0.5, "0.99": 1.0}, 50.0, 100.0)),
                (
                    10 * NANOS_PER_SECOND,
                    summary({"0.5": 0.15, "0.9": 0.6, "0.99": 1.2}, 550.0, 600.0),
                ),
            ],
        )

        stats = SummaryExportStats.from_time_series(ts.summaries["latency"])

        assert isinstance(stats.observation_rate, float)
        assert stats.observation_rate == 50.0  # 500 obs / 10s
        assert not hasattr(stats, "quantile_trends")


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_sample_gauge(self):
        """Test gauge with single sample."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [42.0])

        stats = GaugeExportStats.from_time_series(ts.gauges["queue"])

        assert stats.sample_count == 1
        assert stats.avg == 42.0
        assert stats.min == 42.0
        assert stats.max == 42.0
        assert stats.std == 0.0
        assert stats.p50 == 42.0
        assert stats.p90 == 42.0
        assert stats.time_weighted_avg == 42.0

    def test_empty_time_range(self):
        """Test time filter that excludes all data raises an error."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [10.0], start_ns=NANOS_PER_SECOND)  # Data at 1s

        # Time filter starts after all data (2s), excludes the 1s data point
        time_filter = TimeRangeFilter(start_ns=2 * NANOS_PER_SECOND, end_ns=None)

        with pytest.raises((ValueError, IndexError)):
            GaugeExportStats.from_time_series(ts.gauges["queue"], time_filter)

    def test_histogram_with_missing_buckets(self):
        """Test histogram handles missing buckets gracefully."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist({"0.1": 10.0, "1.0": 50.0, "+Inf": 100.0}, 25.0, 100.0)),
                # Second snapshot missing "1.0" bucket
                (NANOS_PER_SECOND, hist({"0.1": 20.0, "+Inf": 150.0}, 35.0, 150.0)),
            ],
        )

        stats = HistogramExportStats.from_time_series(ts.histograms["ttft"])
        assert stats.observation_count == 50  # 150 - 100
