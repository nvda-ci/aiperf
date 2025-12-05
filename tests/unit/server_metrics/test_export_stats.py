# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models.export_models import (
    CounterExportStats,
    GaugeExportStats,
    HistogramExportStats,
    SummaryExportStats,
    histogram_quantile,
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
        """Test gauge computes correct avg, min, max, std."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue_depth", [10.0, 20.0, 30.0, 40.0, 50.0])

        stats = GaugeExportStats.from_time_series(ts.gauges["queue_depth"])

        assert stats.avg == 30.0  # (10+20+30+40+50)/5
        assert stats.min == 10.0
        assert stats.max == 50.0
        # Sample std (ddof=1): sqrt(500/4) = 15.811
        assert stats.std == pytest.approx(15.811, rel=0.01)

    def test_gauge_percentiles(self):
        """Test gauge computes p50, p90, p95, p99."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "kv_cache", [float(i) for i in range(100)])

        stats = GaugeExportStats.from_time_series(ts.gauges["kv_cache"])

        assert stats.p50 == pytest.approx(49.5, rel=0.01)
        assert stats.p90 == pytest.approx(89.5, rel=0.01)
        assert stats.p95 == pytest.approx(94.5, rel=0.01)
        assert stats.p99 == pytest.approx(98.5, rel=0.01)

    def test_gauge_with_time_filter(self):
        """Test gauge export stats respect time filtering (warmup exclusion)."""
        ts = ServerMetricsTimeSeries()
        # Warmup (0-9) + actual run (10-19)
        add_gauge_samples(ts, "queue", [float(i) for i in range(20)])

        # Filter excludes warmup (first 10 seconds)
        time_filter = TimeRangeFilter(start_ns=10 * NANOS_PER_SECOND, end_ns=None)
        stats = GaugeExportStats.from_time_series(ts.gauges["queue"], time_filter)

        # Should only include values 10-19
        assert stats.avg == 14.5  # (10+11+...+19)/10
        assert stats.min == 10.0
        assert stats.max == 19.0


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
        assert stats.rate_overall == 175.0  # 700 / 4s (overall throughput)
        # Point-to-point rates: [100, 150, 200, 250]
        assert stats.rate_avg == 175.0  # time-weighted: (100+150+200+250) / 4
        assert stats.rate_min == 100.0
        assert stats.rate_max == 250.0
        # Sample std (ddof=1): sqrt(12500/3) = 64.55
        assert stats.rate_std == pytest.approx(64.55, rel=0.01)

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
        assert stats.rate_overall == 200.0

    def test_counter_change_point_rates(self):
        """Test counter rates are computed between change points, not every sample.

        This tests the scenario where we sample faster than the server updates.
        Without change-point detection, we'd get misleading rates like:
        0/s, 0/s, 0/s, 1000/s (when the change is finally observed)

        With change-point detection, we correctly compute:
        100/s (the change attributed to the full time period)
        """
        ts = ServerMetricsTimeSeries()

        # Simulate sampling at 10Hz but server updating at ~1Hz
        # t=0.0s: 0, t=0.1s: 0, ..., t=1.0s: 100, t=1.1s: 100, ..., t=2.0s: 250
        interval_ns = NANOS_PER_SECOND // 10  # 0.1s intervals

        # First second: value stays at 0, then jumps to 100
        for i in range(10):
            ts.append_snapshot(i * interval_ns, counter_metrics={"requests": 0.0})
        # At t=1.0s, value becomes 100
        for i in range(10, 20):
            ts.append_snapshot(i * interval_ns, counter_metrics={"requests": 100.0})
        # At t=2.0s, value becomes 250
        ts.append_snapshot(20 * interval_ns, counter_metrics={"requests": 250.0})

        stats = CounterExportStats.from_time_series(ts.counters["requests"])

        # Total: 250 over 2.0s
        assert stats.delta == 250.0
        assert stats.rate_overall == 125.0  # 250 / 2.0s

        # Change points: t=0 (0), t=1.0 (100), t=2.0 (250)
        # Rates: 100/1.0s = 100/s, 150/1.0s = 150/s
        assert stats.rate_avg == 125.0  # (100 + 150) / 2
        assert stats.rate_min == 100.0
        assert stats.rate_max == 150.0

    def test_counter_no_changes_returns_none_rates(self):
        """Test counter with no value changes returns None for rate stats."""
        ts = ServerMetricsTimeSeries()

        # Value never changes
        for i in range(10):
            ts.append_snapshot(
                i * NANOS_PER_SECOND, counter_metrics={"requests": 100.0}
            )

        stats = CounterExportStats.from_time_series(ts.counters["requests"])

        assert stats.delta == 0.0
        assert stats.rate_overall == 0.0
        # No change points means no rate statistics
        assert stats.rate_avg is None
        assert stats.rate_min is None
        assert stats.rate_max is None
        assert stats.rate_std is None

    def test_counter_rate_avg_is_time_weighted(self):
        """Test that rate_avg is time-weighted, not a simple mean.

        With unequal intervals, time-weighted avg differs from simple mean.
        Longer intervals should contribute more to the average.
        """
        ts = ServerMetricsTimeSeries()

        # Change points with unequal intervals:
        # t=0: 0, t=1s: 100 (rate=100/s over 1s), t=5s: 500 (rate=100/s over 4s)
        ts.append_snapshot(0, counter_metrics={"requests": 0.0})
        ts.append_snapshot(1 * NANOS_PER_SECOND, counter_metrics={"requests": 100.0})
        ts.append_snapshot(5 * NANOS_PER_SECOND, counter_metrics={"requests": 500.0})

        _stats = CounterExportStats.from_time_series(ts.counters["requests"])

        # Rates: 100/s (1s interval), 100/s (4s interval)
        # Simple mean would be: (100 + 100) / 2 = 100
        # Time-weighted: (100 + 400) / (1 + 4) = 500 / 5 = 100
        # (In this case they're equal because rates are the same, so we skip assertions)

        # Let's verify with different rates:
        ts2 = ServerMetricsTimeSeries()
        # t=0: 0, t=1s: 200 (rate=200/s over 1s), t=5s: 600 (rate=100/s over 4s)
        ts2.append_snapshot(0, counter_metrics={"requests": 0.0})
        ts2.append_snapshot(1 * NANOS_PER_SECOND, counter_metrics={"requests": 200.0})
        ts2.append_snapshot(5 * NANOS_PER_SECOND, counter_metrics={"requests": 600.0})

        stats2 = CounterExportStats.from_time_series(ts2.counters["requests"])

        # Rates: 200/s (1s interval), 100/s (4s interval)
        # Simple mean: (200 + 100) / 2 = 150
        # Time-weighted: (200 + 400) / (1 + 4) = 600 / 5 = 120
        assert stats2.rate_avg == 120.0  # Time-weighted, NOT 150
        assert stats2.rate_min == 100.0
        assert stats2.rate_max == 200.0


# =============================================================================
# Histogram Tests
# =============================================================================


class TestHistogramExportStats:
    """Test histogram export statistics computation."""

    def test_histogram_basic_stats(self):
        """Test histogram computes count_delta, sum_delta, avg, rate."""
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
        assert stats.count_delta == 150.0
        assert stats.sum_delta == 40.0
        assert stats.avg == pytest.approx(40.0 / 150, rel=0.01)
        assert stats.rate == 150.0  # 150 obs / 1s

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
        assert stats.count_delta == 500.0  # 2000 - 1500
        assert stats.sum_delta == 200.0  # 600 - 400

    def test_histogram_rate(self):
        """Test histogram rate calculation."""
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

        assert isinstance(stats.rate, float)
        assert stats.rate == 40.0  # 200 obs / 5s

    def test_histogram_no_data_raises_key_error(self):
        """Test histogram raises KeyError when metric doesn't exist."""
        ts = ServerMetricsTimeSeries()

        with pytest.raises(KeyError):
            _ = ts.histograms["nonexistent"]

    def test_histogram_bucket_counter_reset(self):
        """Test histogram returns None for buckets when counter reset detected.

        When a counter resets (server restart), the final value will be less than
        the reference. We return None to indicate invalid/incomplete data.
        """
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                # Before reset: high counts
                (
                    0,
                    hist(
                        {"0.1": 1000.0, "1.0": 5000.0, "+Inf": 10000.0}, 2500.0, 10000.0
                    ),
                ),
                # After reset: low counts (server restarted)
                (
                    NANOS_PER_SECOND,
                    hist({"0.1": 50.0, "1.0": 200.0, "+Inf": 500.0}, 125.0, 500.0),
                ),
            ],
        )

        stats = HistogramExportStats.from_time_series(ts.histograms["ttft"])

        # Negative deltas indicate reset - return None for invalid data
        assert stats.buckets is None

    def test_histogram_estimated_percentiles(self):
        """Test histogram computes estimated p50, p90, p95, p99 from buckets."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist({"0.1": 0.0, "0.5": 0.0, "1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                # 1000 observations: 100 in [0,0.1], 400 in (0.1,0.5], 400 in (0.5,1.0], 100 in (1.0,+Inf]
                (
                    NANOS_PER_SECOND,
                    hist(
                        {"0.1": 100.0, "0.5": 500.0, "1.0": 900.0, "+Inf": 1000.0},
                        500.0,
                        1000.0,
                    ),
                ),
            ],
        )

        stats = HistogramExportStats.from_time_series(ts.histograms["ttft"])

        # p50 = 500th observation, falls in (0.1, 0.5] bucket
        # rank_in_bucket = 500 - 100 = 400, bucket_count = 400
        # p50 = 0.1 + (0.5 - 0.1) * (400/400) = 0.5
        assert stats.p50 == pytest.approx(0.5, rel=0.01)

        # p90 = 900th observation, falls in (0.5, 1.0] bucket
        # rank_in_bucket = 900 - 500 = 400, bucket_count = 400
        # p90 = 0.5 + (1.0 - 0.5) * (400/400) = 1.0
        assert stats.p90 == pytest.approx(1.0, rel=0.01)

        # p95 = 950th observation, falls in (1.0, +Inf) bucket
        # When quantile falls in +Inf bucket, return second-highest bound
        assert stats.p95 == pytest.approx(1.0, rel=0.01)

        # p99 = 990th observation, also in +Inf bucket
        assert stats.p99 == pytest.approx(1.0, rel=0.01)

    def test_histogram_percentiles_none_on_counter_reset(self):
        """Test histogram percentiles are None when counter reset detected."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist({"0.1": 1000.0, "+Inf": 2000.0}, 500.0, 2000.0)),
                # Counter reset - values decreased
                (NANOS_PER_SECOND, hist({"0.1": 50.0, "+Inf": 100.0}, 25.0, 100.0)),
            ],
        )

        stats = HistogramExportStats.from_time_series(ts.histograms["ttft"])

        assert stats.buckets is None
        assert stats.p50 is None
        assert stats.p90 is None
        assert stats.p95 is None
        assert stats.p99 is None


# =============================================================================
# histogram_quantile Function Tests
# =============================================================================


class TestHistogramQuantile:
    """Test histogram_quantile function (Prometheus-style linear interpolation)."""

    def test_basic_linear_interpolation(self):
        """Test basic linear interpolation within a bucket."""
        # 100 observations total, evenly distributed:
        # 50 in [0, 0.5], 50 in (0.5, 1.0]
        buckets = {"0.5": 50.0, "1.0": 100.0, "+Inf": 100.0}

        # p50 = 50th observation, at the boundary
        assert histogram_quantile(0.50, buckets) == pytest.approx(0.5, rel=0.01)

        # p25 = 25th observation, in first bucket
        # rank=25, bucket_count=50, bucket_start=0, bucket_end=0.5
        # 0 + (0.5 - 0) * (25/50) = 0.25
        assert histogram_quantile(0.25, buckets) == pytest.approx(0.25, rel=0.01)

        # p75 = 75th observation, in second bucket
        # rank_in_bucket=75-50=25, bucket_count=50
        # 0.5 + (1.0 - 0.5) * (25/50) = 0.75
        assert histogram_quantile(0.75, buckets) == pytest.approx(0.75, rel=0.01)

    def test_quantile_in_inf_bucket_returns_second_highest(self):
        """Test quantile in +Inf bucket returns upper bound of second-highest bucket."""
        buckets = {"0.1": 90.0, "1.0": 95.0, "+Inf": 100.0}

        # p99 = 99th observation, in +Inf bucket (95-100)
        # Should return 1.0 (second-highest bucket upper bound)
        assert histogram_quantile(0.99, buckets) == 1.0

    def test_first_bucket_assumes_zero_lower_bound(self):
        """Test first bucket assumes lower bound of 0 when upper > 0."""
        buckets = {"0.1": 100.0, "+Inf": 100.0}

        # All 100 observations in first bucket [0, 0.1]
        # p50 = 0 + (0.1 - 0) * (50/100) = 0.05
        assert histogram_quantile(0.50, buckets) == pytest.approx(0.05, rel=0.01)

    def test_empty_buckets_returns_none(self):
        """Test empty buckets returns None."""
        assert histogram_quantile(0.50, {}) is None
        assert histogram_quantile(0.50, None) is None  # type: ignore[arg-type]

    def test_zero_observations_returns_none(self):
        """Test zero total observations returns None."""
        buckets = {"0.1": 0.0, "1.0": 0.0, "+Inf": 0.0}
        assert histogram_quantile(0.50, buckets) is None

    def test_invalid_quantile_returns_none(self):
        """Test invalid quantile values return None."""
        buckets = {"0.1": 50.0, "+Inf": 100.0}
        assert histogram_quantile(-0.1, buckets) is None
        assert histogram_quantile(1.1, buckets) is None

    def test_single_real_bucket(self):
        """Test histogram with only one real bucket + Inf."""
        buckets = {"1.0": 100.0, "+Inf": 100.0}

        # All observations in [0, 1.0]
        assert histogram_quantile(0.50, buckets) == pytest.approx(0.5, rel=0.01)
        assert histogram_quantile(0.90, buckets) == pytest.approx(0.9, rel=0.01)

    def test_unsorted_buckets_handled(self):
        """Test buckets are sorted correctly regardless of input order."""
        # Deliberately unsorted
        buckets = {"+Inf": 100.0, "0.1": 25.0, "1.0": 75.0, "0.5": 50.0}

        # Should work the same as sorted input
        assert histogram_quantile(0.50, buckets) == pytest.approx(0.5, rel=0.01)

    def test_boundary_quantiles(self):
        """Test edge quantiles q=0 and q=1."""
        buckets = {"0.5": 50.0, "1.0": 100.0, "+Inf": 100.0}

        # q=0 should return 0 (start of first bucket)
        assert histogram_quantile(0.0, buckets) == pytest.approx(0.0, rel=0.01)

        # q=1 should return upper bound of second-highest (falls in +Inf)
        assert histogram_quantile(1.0, buckets) == 1.0

    def test_realistic_latency_histogram(self):
        """Test with realistic latency bucket boundaries."""
        # Typical latency distribution: most requests fast, long tail
        buckets = {
            "0.005": 100.0,  # 100 requests < 5ms
            "0.01": 400.0,  # 300 more < 10ms (400 cumulative)
            "0.025": 800.0,  # 400 more < 25ms (800 cumulative)
            "0.05": 950.0,  # 150 more < 50ms (950 cumulative)
            "0.1": 990.0,  # 40 more < 100ms (990 cumulative)
            "0.25": 998.0,  # 8 more < 250ms (998 cumulative)
            "0.5": 999.0,  # 1 more < 500ms (999 cumulative)
            "1.0": 1000.0,  # 1 more < 1s (1000 cumulative)
            "+Inf": 1000.0,  # no requests > 1s
        }

        # p50 = 500th request, in (0.01, 0.025] bucket
        # rank_in_bucket = 500 - 400 = 100, bucket_count = 400
        # p50 = 0.01 + (0.025 - 0.01) * (100/400) = 0.01 + 0.00375 = 0.01375
        assert histogram_quantile(0.50, buckets) == pytest.approx(0.01375, rel=0.01)

        # p99 = 990th request, at boundary of (0.05, 0.1] bucket
        assert histogram_quantile(0.99, buckets) == pytest.approx(0.1, rel=0.01)


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

        assert stats.count_delta == 200.0  # 300 - 100
        assert stats.sum_delta == 100.0  # 150 - 50
        assert stats.avg == 0.5  # 100 / 200
        assert stats.rate == 100.0  # 200 obs / 2s
        assert stats.quantiles["0.5"] == 0.12
        assert stats.quantiles["0.9"] == 0.55
        assert stats.quantiles["0.95"] == 0.85
        assert stats.quantiles["0.99"] == 1.1

    def test_summary_rate(self):
        """Test summary rate calculation."""
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

        assert isinstance(stats.rate, float)
        assert stats.rate == 50.0  # 500 obs / 10s


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

        assert stats.avg == 42.0
        assert stats.min == 42.0
        assert stats.max == 42.0
        assert stats.std == 0.0
        assert stats.p50 == 42.0
        assert stats.p90 == 42.0

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
        assert stats.count_delta == 50.0  # 150 - 100
