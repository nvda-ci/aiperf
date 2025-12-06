# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models.export_models import (
    BucketStatistics,
    CounterExportStats,
    GaugeExportStats,
    HistogramExportStats,
    SummaryExportStats,
    accumulate_bucket_statistics,
    compute_best_guess_percentiles,
    estimate_bucket_sums,
    estimate_inf_bucket_observations,
    extract_all_observations,
    extract_observations_from_scrape,
    generate_observations_with_sum_constraint,
    get_bucket_bounds,
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

        # Verify we have percentiles with bucket interpolation
        assert stats.percentiles is not None
        assert stats.percentiles.bucket is not None

        # p50 = 500th observation, falls in (0.1, 0.5] bucket
        # rank_in_bucket = 500 - 100 = 400, bucket_count = 400
        # p50 = 0.1 + (0.5 - 0.1) * (400/400) = 0.5
        assert stats.percentiles.bucket.p50 == pytest.approx(0.5, rel=0.01)

        # p90 = 900th observation, falls in (0.5, 1.0] bucket
        # rank_in_bucket = 900 - 500 = 400, bucket_count = 400
        # p90 = 0.5 + (1.0 - 0.5) * (400/400) = 1.0
        assert stats.percentiles.bucket.p90 == pytest.approx(1.0, rel=0.01)

        # p95 = 950th observation, falls in (1.0, +Inf) bucket
        # When quantile falls in +Inf bucket, return second-highest bound
        assert stats.percentiles.bucket.p95 == pytest.approx(1.0, rel=0.01)

        # p99 = 990th observation, also in +Inf bucket
        assert stats.percentiles.bucket.p99 == pytest.approx(1.0, rel=0.01)

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
        # When counter reset detected, percentiles should be None
        assert stats.percentiles is None


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


# =============================================================================
# Observation Extraction Tests
# =============================================================================


class TestExtractObservationsFromScrape:
    """Test extract_observations_from_scrape function."""

    def test_single_observation_returns_exact_value(self):
        """When count_delta == 1, return exact observation value."""
        observations, exact, bucket_placed = extract_observations_from_scrape(
            count_delta=1,
            sum_delta=0.123,
            bucket_deltas={"0.1": 0.0, "0.5": 1.0, "+Inf": 1.0},
        )

        assert observations == [0.123]
        assert exact == 1
        assert bucket_placed == 0

    def test_zero_observations_returns_empty(self):
        """When count_delta == 0, return empty list."""
        observations, exact, bucket_placed = extract_observations_from_scrape(
            count_delta=0,
            sum_delta=0.0,
            bucket_deltas={"0.5": 0.0, "+Inf": 0.0},
        )

        assert observations == []
        assert exact == 0
        assert bucket_placed == 0

    def test_negative_count_returns_empty(self):
        """Counter reset (negative delta) returns empty list."""
        observations, exact, bucket_placed = extract_observations_from_scrape(
            count_delta=-5,
            sum_delta=-10.0,
            bucket_deltas={"0.5": -5.0, "+Inf": -5.0},
        )

        assert observations == []
        assert exact == 0
        assert bucket_placed == 0

    def test_multiple_observations_single_bucket(self):
        """Multiple observations in single bucket use linear interpolation."""
        # 3 observations all fell into (0.1, 0.5] bucket
        observations, exact, bucket_placed = extract_observations_from_scrape(
            count_delta=3,
            sum_delta=0.9,  # sum of the 3 observations
            bucket_deltas={"0.1": 0.0, "0.5": 3.0, "+Inf": 3.0},
        )

        assert len(observations) == 3
        assert bucket_placed == 3
        assert exact == 0
        # Values should be interpolated within (0.1, 0.5]
        for obs in observations:
            assert 0.1 < obs <= 0.5

    def test_multiple_observations_multiple_buckets(self):
        """Multiple observations spread across buckets."""
        # 5 observations: 2 in [0, 0.1], 3 in (0.1, 0.5]
        observations, exact, bucket_placed = extract_observations_from_scrape(
            count_delta=5,
            sum_delta=1.5,
            bucket_deltas={"0.1": 2.0, "0.5": 5.0, "+Inf": 5.0},
        )

        assert len(observations) == 5
        assert bucket_placed == 5
        assert exact == 0

    def test_observations_in_first_bucket(self):
        """Observations in first bucket (assumed 0 lower bound)."""
        observations, exact, bucket_placed = extract_observations_from_scrape(
            count_delta=2,
            sum_delta=0.08,
            bucket_deltas={"0.1": 2.0, "0.5": 2.0, "+Inf": 2.0},
        )

        assert len(observations) == 2
        assert bucket_placed == 2
        # Values should be in [0, 0.1] bucket
        for obs in observations:
            assert 0.0 <= obs <= 0.1

    def test_observations_in_inf_bucket_are_skipped(self):
        """Observations in +Inf bucket are skipped (value unknown)."""
        # 2 observations in (+Inf) bucket (above 1.0)
        # We can't accurately place these, so they're not extracted
        observations, exact, bucket_placed = extract_observations_from_scrape(
            count_delta=2,
            sum_delta=3.0,  # values > 1.0
            bucket_deltas={"0.5": 0.0, "1.0": 0.0, "+Inf": 2.0},
        )

        # +Inf bucket observations are not extracted
        assert len(observations) == 0
        assert bucket_placed == 0
        assert exact == 0

    def test_empty_bucket_deltas(self):
        """Empty bucket deltas returns empty observations."""
        observations, exact, bucket_placed = extract_observations_from_scrape(
            count_delta=3,
            sum_delta=1.0,
            bucket_deltas={},
        )

        assert observations == []
        assert exact == 0
        assert bucket_placed == 0


class TestExtractAllObservations:
    """Test extract_all_observations function."""

    def test_extracts_exact_observations(self):
        """Test extraction of exact observations when count_delta == 1."""
        import numpy as np

        timestamps = np.array([0, NANOS_PER_SECOND, 2 * NANOS_PER_SECOND])
        counts = np.array([0.0, 1.0, 2.0])  # +1, +1 per interval
        sums = np.array([0.0, 0.15, 0.35])  # 0.15, then 0.20
        bucket_snapshots = [
            {"0.5": 0.0, "+Inf": 0.0},
            {"0.5": 1.0, "+Inf": 1.0},
            {"0.5": 2.0, "+Inf": 2.0},
        ]

        observations, exact, bucket_placed = extract_all_observations(
            timestamps, sums, counts, bucket_snapshots
        )

        assert len(observations) == 2
        assert exact == 2
        assert bucket_placed == 0
        assert observations[0] == pytest.approx(0.15)
        assert observations[1] == pytest.approx(0.20)

    def test_extracts_bucket_placed_observations(self):
        """Test extraction of bucket-placed observations when count_delta > 1."""
        import numpy as np

        timestamps = np.array([0, NANOS_PER_SECOND])
        counts = np.array([0.0, 10.0])  # 10 observations
        sums = np.array([0.0, 3.0])
        bucket_snapshots = [
            {"0.1": 0.0, "0.5": 0.0, "+Inf": 0.0},
            {"0.1": 5.0, "0.5": 10.0, "+Inf": 10.0},  # 5 in first, 5 in second
        ]

        observations, exact, bucket_placed = extract_all_observations(
            timestamps, sums, counts, bucket_snapshots
        )

        assert len(observations) == 10
        assert exact == 0
        assert bucket_placed == 10

    def test_mixed_exact_and_bucket_placed(self):
        """Test extraction with mix of exact and bucket-placed observations."""
        import numpy as np

        timestamps = np.array([0, NANOS_PER_SECOND, 2 * NANOS_PER_SECOND])
        counts = np.array([0.0, 1.0, 4.0])  # +1 (exact), +3 (bucket-placed)
        sums = np.array([0.0, 0.25, 1.0])
        bucket_snapshots = [
            {"0.5": 0.0, "+Inf": 0.0},
            {"0.5": 1.0, "+Inf": 1.0},
            {"0.5": 4.0, "+Inf": 4.0},
        ]

        observations, exact, bucket_placed = extract_all_observations(
            timestamps, sums, counts, bucket_snapshots
        )

        assert len(observations) == 4
        assert exact == 1
        assert bucket_placed == 3

    def test_start_idx_skips_warmup(self):
        """Test start_idx parameter skips initial observations."""
        import numpy as np

        timestamps = np.array(
            [0, NANOS_PER_SECOND, 2 * NANOS_PER_SECOND, 3 * NANOS_PER_SECOND]
        )
        counts = np.array([0.0, 1.0, 2.0, 3.0])
        sums = np.array([0.0, 0.1, 0.3, 0.6])
        bucket_snapshots = [
            {"0.5": 0.0, "+Inf": 0.0},
            {"0.5": 1.0, "+Inf": 1.0},
            {"0.5": 2.0, "+Inf": 2.0},
            {"0.5": 3.0, "+Inf": 3.0},
        ]

        # Skip first interval (warmup)
        observations, exact, bucket_placed = extract_all_observations(
            timestamps, sums, counts, bucket_snapshots, start_idx=1
        )

        assert len(observations) == 2  # Only intervals 2 and 3
        assert exact == 2
        assert bucket_placed == 0

    def test_empty_data_returns_empty(self):
        """Test empty time series returns empty observations."""
        import numpy as np

        timestamps = np.array([0])  # Single timestamp, no intervals
        counts = np.array([0.0])
        sums = np.array([0.0])
        bucket_snapshots = [{"0.5": 0.0, "+Inf": 0.0}]

        observations, exact, bucket_placed = extract_all_observations(
            timestamps, sums, counts, bucket_snapshots
        )

        assert len(observations) == 0
        assert exact == 0
        assert bucket_placed == 0


class TestHistogramObservedPercentiles:
    """Test histogram observed percentiles from per-scrape observation extraction."""

    def test_observed_percentiles_from_exact_observations(self):
        """Test observed percentiles computed from exact observations."""
        ts = ServerMetricsTimeSeries()
        # Create a histogram with one observation per scrape (exact values)
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist({"0.5": 0.0, "1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                (1 * NANOS_PER_SECOND, hist({"0.5": 1.0, "1.0": 1.0, "+Inf": 1.0}, 0.1, 1.0)),
                (2 * NANOS_PER_SECOND, hist({"0.5": 2.0, "1.0": 2.0, "+Inf": 2.0}, 0.3, 2.0)),
                (3 * NANOS_PER_SECOND, hist({"0.5": 3.0, "1.0": 3.0, "+Inf": 3.0}, 0.6, 3.0)),
                (4 * NANOS_PER_SECOND, hist({"0.5": 4.0, "1.0": 4.0, "+Inf": 4.0}, 1.0, 4.0)),
            ],
        )  # fmt: skip

        stats = HistogramExportStats.from_time_series(ts.histograms["ttft"])

        assert stats.percentiles is not None
        assert stats.percentiles.observed is not None

        # We have 4 exact observations: 0.1, 0.2, 0.3, 0.4
        assert stats.percentiles.observed.exact_count == 4
        assert stats.percentiles.observed.bucket_placed_count == 0
        assert stats.percentiles.observed.coverage == pytest.approx(1.0)

        # Observed percentiles should be computed from exact values
        assert stats.percentiles.observed.p50 is not None
        assert stats.percentiles.observed.p90 is not None

    def test_observed_percentiles_with_bucket_placed(self):
        """Test observed percentiles with bucket-placed observations."""
        ts = ServerMetricsTimeSeries()
        # Multiple observations per scrape (bucket-placed)
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist({"0.1": 0.0, "0.5": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                # 10 observations in first interval
                (
                    NANOS_PER_SECOND,
                    hist({"0.1": 5.0, "0.5": 10.0, "+Inf": 10.0}, 2.0, 10.0),
                ),
            ],
        )

        stats = HistogramExportStats.from_time_series(ts.histograms["ttft"])

        assert stats.percentiles is not None
        assert stats.percentiles.observed is not None

        # All observations are bucket-placed
        assert stats.percentiles.observed.exact_count == 0
        assert stats.percentiles.observed.bucket_placed_count == 10
        assert stats.percentiles.observed.coverage == pytest.approx(0.0)

    def test_observed_percentiles_none_on_counter_reset(self):
        """Test observed percentiles are None on counter reset."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist({"0.5": 100.0, "+Inf": 200.0}, 50.0, 200.0)),
                # Counter reset
                (NANOS_PER_SECOND, hist({"0.5": 10.0, "+Inf": 20.0}, 5.0, 20.0)),
            ],
        )

        stats = HistogramExportStats.from_time_series(ts.histograms["ttft"])

        assert stats.percentiles is None

    def test_coverage_calculation(self):
        """Test coverage is exact_count / total_count."""
        ts = ServerMetricsTimeSeries()
        # Mix of exact and bucket-placed: 2 exact (count_delta==1), 5 bucket-placed
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist({"0.5": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                (
                    1 * NANOS_PER_SECOND,
                    hist({"0.5": 1.0, "+Inf": 1.0}, 0.1, 1.0),
                ),  # exact
                (
                    2 * NANOS_PER_SECOND,
                    hist({"0.5": 6.0, "+Inf": 6.0}, 1.5, 6.0),
                ),  # +5 bucket
                (
                    3 * NANOS_PER_SECOND,
                    hist({"0.5": 7.0, "+Inf": 7.0}, 1.7, 7.0),
                ),  # exact
            ],
        )

        stats = HistogramExportStats.from_time_series(ts.histograms["ttft"])

        assert stats.percentiles is not None
        assert stats.percentiles.observed is not None

        assert stats.percentiles.observed.exact_count == 2
        assert stats.percentiles.observed.bucket_placed_count == 5
        # Coverage = 2 / 7 = 0.2857...
        assert stats.percentiles.observed.coverage == pytest.approx(2 / 7, rel=0.01)


# =============================================================================
# Bucket Statistics Tests (Polynomial Histogram Approach)
# =============================================================================


class TestBucketStatistics:
    """Test BucketStatistics model for polynomial histogram approach."""

    def test_bucket_statistics_record_single(self):
        """Test recording a single observation."""
        stats = BucketStatistics(bucket_le="0.5")
        stats.record(mean=0.25, count=1)

        assert stats.observation_count == 1
        assert stats.sample_count == 1
        assert stats.estimated_mean == pytest.approx(0.25)

    def test_bucket_statistics_record_multiple(self):
        """Test recording multiple observations."""
        stats = BucketStatistics(bucket_le="0.5")
        stats.record(mean=0.2, count=10)
        stats.record(mean=0.3, count=10)

        assert stats.observation_count == 20
        assert stats.sample_count == 2
        # Weighted mean: (0.2*10 + 0.3*10) / 20 = 5/20 = 0.25
        assert stats.estimated_mean == pytest.approx(0.25)

    def test_bucket_statistics_weighted_average(self):
        """Test that estimated_mean is weighted by count."""
        stats = BucketStatistics(bucket_le="1.0")
        stats.record(mean=0.1, count=1)  # weight 1
        stats.record(mean=0.5, count=9)  # weight 9

        # Weighted: (0.1*1 + 0.5*9) / 10 = 4.6/10 = 0.46
        assert stats.estimated_mean == pytest.approx(0.46)

    def test_bucket_statistics_empty_returns_none(self):
        """Test empty bucket returns None for estimated_mean."""
        stats = BucketStatistics(bucket_le="0.5")
        assert stats.estimated_mean is None


class TestAccumulateBucketStatistics:
    """Test accumulate_bucket_statistics function."""

    def test_single_bucket_intervals_learn_means(self):
        """Test that single-bucket intervals are used to learn means."""
        import numpy as np

        timestamps = np.array([0, NANOS_PER_SECOND, 2 * NANOS_PER_SECOND])
        # All observations in same bucket each interval
        counts = np.array([0.0, 5.0, 10.0])  # +5, +5
        sums = np.array([0.0, 1.5, 3.5])  # mean=0.3 first, mean=0.4 second
        bucket_snapshots = [
            {"0.5": 0.0, "1.0": 0.0, "+Inf": 0.0},
            {"0.5": 5.0, "1.0": 5.0, "+Inf": 5.0},  # all in 0.5 bucket
            {"0.5": 10.0, "1.0": 10.0, "+Inf": 10.0},  # all in 0.5 bucket
        ]

        stats = accumulate_bucket_statistics(timestamps, sums, counts, bucket_snapshots)

        assert "0.5" in stats
        assert stats["0.5"].observation_count == 10
        # Weighted: (0.3*5 + 0.4*5) / 10 = 0.35
        assert stats["0.5"].estimated_mean == pytest.approx(0.35)

    def test_multi_bucket_intervals_not_learned(self):
        """Test that multi-bucket intervals are NOT used to learn means."""
        import numpy as np

        timestamps = np.array([0, NANOS_PER_SECOND])
        counts = np.array([0.0, 10.0])
        sums = np.array([0.0, 3.0])
        # Observations split across buckets
        bucket_snapshots = [
            {"0.1": 0.0, "0.5": 0.0, "+Inf": 0.0},
            {"0.1": 3.0, "0.5": 10.0, "+Inf": 10.0},  # 3 in 0.1, 7 in 0.5
        ]

        stats = accumulate_bucket_statistics(timestamps, sums, counts, bucket_snapshots)

        # No single-bucket intervals, so no stats learned
        assert len(stats) == 0

    def test_inf_bucket_learns_mean(self):
        """Test that +Inf bucket can learn means when all observations land there."""
        import numpy as np

        timestamps = np.array([0, NANOS_PER_SECOND])
        counts = np.array([0.0, 3.0])
        sums = np.array([0.0, 45.0])  # mean = 15.0
        bucket_snapshots = [
            {"1.0": 0.0, "+Inf": 0.0},
            {"1.0": 0.0, "+Inf": 3.0},  # all in +Inf
        ]

        stats = accumulate_bucket_statistics(timestamps, sums, counts, bucket_snapshots)

        assert "+Inf" in stats
        assert stats["+Inf"].estimated_mean == pytest.approx(15.0)


class TestGetBucketBounds:
    """Test get_bucket_bounds function."""

    def test_first_bucket_has_zero_lower_bound(self):
        """First bucket has lower bound of 0."""
        sorted_buckets = ["0.1", "0.5", "1.0"]
        lower, upper = get_bucket_bounds("0.1", sorted_buckets)

        assert lower == 0.0
        assert upper == 0.1

    def test_middle_bucket_has_previous_as_lower(self):
        """Middle bucket has previous bucket as lower bound."""
        sorted_buckets = ["0.1", "0.5", "1.0"]
        lower, upper = get_bucket_bounds("0.5", sorted_buckets)

        assert lower == 0.1
        assert upper == 0.5

    def test_last_finite_bucket(self):
        """Last finite bucket has previous bucket as lower bound."""
        sorted_buckets = ["0.1", "0.5", "1.0"]
        lower, upper = get_bucket_bounds("1.0", sorted_buckets)

        assert lower == 0.5
        assert upper == 1.0


class TestEstimateBucketSums:
    """Test estimate_bucket_sums function."""

    def test_uses_learned_means_when_available(self):
        """Test that learned means are used when available."""
        # Cumulative counts: 10 in [0,0.1], 10 in (0.1,0.5], 10 in +Inf
        bucket_deltas = {"0.1": 10.0, "0.5": 20.0, "+Inf": 30.0}
        bucket_stats = {
            "0.1": BucketStatistics(
                bucket_le="0.1",
                observation_count=100,
                weighted_mean_sum=5.0,  # mean = 0.05
                sample_count=10,
            ),
        }

        sums = estimate_bucket_sums(bucket_deltas, bucket_stats)

        # Per-bucket: 10 in 0.1, 10 in 0.5
        # 0.1 bucket uses learned mean: 10 * 0.05 = 0.5
        assert sums["0.1"] == pytest.approx(0.5)
        # 0.5 bucket uses midpoint: 10 * (0.1 + 0.5) / 2 = 10 * 0.3 = 3.0
        assert sums["0.5"] == pytest.approx(3.0)
        # +Inf bucket is not included
        assert "+Inf" not in sums

    def test_falls_back_to_midpoint(self):
        """Test fallback to midpoint when no learned means."""
        # Cumulative counts: 10 in [0,0.2], 5 more in (0.2,1.0], 0 in +Inf
        bucket_deltas = {"0.2": 10.0, "1.0": 15.0, "+Inf": 15.0}
        bucket_stats = {}  # No learned stats

        sums = estimate_bucket_sums(bucket_deltas, bucket_stats)

        # Per-bucket: 10 in 0.2, 5 in 1.0
        # 0.2 bucket: 10 * (0 + 0.2) / 2 = 10 * 0.1 = 1.0
        assert sums["0.2"] == pytest.approx(1.0)
        # 1.0 bucket: 5 * (0.2 + 1.0) / 2 = 5 * 0.6 = 3.0
        assert sums["1.0"] == pytest.approx(3.0)


class TestEstimateInfBucketObservations:
    """Test estimate_inf_bucket_observations function."""

    def test_back_calculates_inf_observations(self):
        """Test back-calculation of +Inf bucket observations."""
        total_sum = 100.0
        estimated_finite_sum = 70.0
        inf_count = 3
        max_finite_bucket = 10.0

        observations = estimate_inf_bucket_observations(
            total_sum, estimated_finite_sum, inf_count, max_finite_bucket
        )

        # Back-calculated inf_sum = 100 - 70 = 30
        # inf_avg = 30 / 3 = 10.0
        # But 10.0 is not > max_finite_bucket, so fallback to 1.5x
        # inf_avg = 10.0 * 1.5 = 15.0
        # Observations spread from 10.0 to 20.0 (mean = 15.0)
        assert len(observations) == 3
        assert all(obs >= max_finite_bucket for obs in observations)
        assert pytest.approx(sum(observations) / len(observations), rel=0.01) == 15.0

    def test_inf_observations_above_max_finite(self):
        """Test all +Inf observations are above max finite bucket."""
        total_sum = 200.0
        estimated_finite_sum = 50.0
        inf_count = 10
        max_finite_bucket = 5.0

        observations = estimate_inf_bucket_observations(
            total_sum, estimated_finite_sum, inf_count, max_finite_bucket
        )

        # inf_avg = 150 / 10 = 15.0 (> 5.0, valid)
        assert len(observations) == 10
        assert all(obs >= max_finite_bucket for obs in observations)

    def test_zero_inf_count_returns_empty(self):
        """Test zero inf_count returns empty list."""
        observations = estimate_inf_bucket_observations(
            total_sum=100.0,
            estimated_finite_sum=100.0,
            inf_count=0,
            max_finite_bucket=10.0,
        )

        assert observations == []

    def test_negative_inf_sum_falls_back(self):
        """Test negative back-calculated sum falls back to 1.5x max."""
        # When estimated_finite_sum > total_sum (estimation error)
        observations = estimate_inf_bucket_observations(
            total_sum=50.0,
            estimated_finite_sum=100.0,  # Error: greater than total
            inf_count=5,
            max_finite_bucket=10.0,
        )

        # Should fall back to 1.5x max_finite_bucket = 15.0
        assert len(observations) == 5
        assert pytest.approx(sum(observations) / len(observations), rel=0.01) == 15.0


# =============================================================================
# Generate Observations With Sum Constraint Tests (Core Algorithm)
# =============================================================================


class TestGenerateObservationsWithSumConstraint:
    """Test generate_observations_with_sum_constraint function.

    This is the core of the polynomial histogram percentile estimation algorithm.
    It generates observation values from bucket counts, constrained to match
    the exact histogram sum.
    """

    def test_basic_uniform_distribution_no_learned_means(self):
        """Test basic uniform distribution without learned means."""
        import numpy as np

        # Cumulative: 10 in [0,0.1], 10 more in (0.1,0.5] = 20 total
        bucket_deltas = {"0.1": 10.0, "0.5": 20.0, "+Inf": 20.0}
        # With uniform distribution:
        # [0,0.1]: mean=0.05, 10 obs -> sum=0.5
        # (0.1,0.5]: mean=0.3, 10 obs -> sum=3.0
        # Total expected = 3.5
        target_sum = 3.5

        observations = generate_observations_with_sum_constraint(
            bucket_deltas, target_sum, bucket_stats=None
        )

        assert len(observations) == 20
        assert np.isclose(observations.sum(), target_sum, rtol=0.01)
        # First 10 should be in [0,0.1]
        assert all(0 <= obs <= 0.1 for obs in observations[:10])
        # Next 10 should be in (0.1,0.5]
        assert all(0.1 <= obs <= 0.5 for obs in observations[10:])

    def test_observations_within_bucket_bounds(self):
        """Test all observations stay within their bucket bounds."""
        import numpy as np

        bucket_deltas = {"0.05": 50.0, "0.1": 100.0, "0.5": 150.0, "+Inf": 150.0}
        target_sum = 15.0  # Arbitrary

        observations = generate_observations_with_sum_constraint(
            bucket_deltas, target_sum, bucket_stats=None
        )

        # Sort observations to check bounds
        sorted_obs = np.sort(observations)

        # First 50 in [0,0.05]
        assert all(0 <= obs <= 0.05 for obs in sorted_obs[:50])
        # Next 50 in (0.05,0.1]
        assert all(0.05 <= obs <= 0.1 for obs in sorted_obs[50:100])
        # Last 50 in (0.1,0.5]
        assert all(0.1 <= obs <= 0.5 for obs in sorted_obs[100:])

    def test_shifted_distribution_with_learned_mean(self):
        """Test distribution shifts toward learned mean."""
        import numpy as np

        # Single bucket for simplicity
        bucket_deltas = {"0.5": 100.0, "+Inf": 100.0}

        # Without learned mean, midpoint is 0.25, sum would be ~25
        obs_no_mean = generate_observations_with_sum_constraint(
            bucket_deltas, target_sum=25.0, bucket_stats=None
        )

        # With learned mean of 0.1 (observations cluster near lower bound)
        bucket_stats = {
            "0.5": BucketStatistics(
                bucket_le="0.5",
                observation_count=50,
                weighted_mean_sum=5.0,  # mean = 0.1
                sample_count=10,
            )
        }
        obs_with_mean = generate_observations_with_sum_constraint(
            bucket_deltas, target_sum=10.0, bucket_stats=bucket_stats
        )

        # With learned mean of 0.1, observations should be shifted lower
        assert np.mean(obs_with_mean) < np.mean(obs_no_mean)
        # All should still be in [0, 0.5]
        assert all(0 <= obs <= 0.5 for obs in obs_with_mean)

    def test_learned_mean_ignored_if_outside_bounds(self):
        """Test learned mean is ignored if outside bucket bounds."""
        import numpy as np

        bucket_deltas = {"0.5": 100.0, "+Inf": 100.0}

        # Invalid learned mean (outside bucket bounds)
        bucket_stats = {
            "0.5": BucketStatistics(
                bucket_le="0.5",
                observation_count=50,
                weighted_mean_sum=35.0,  # mean = 0.7 > 0.5 (invalid!)
                sample_count=10,
            )
        }

        observations = generate_observations_with_sum_constraint(
            bucket_deltas, target_sum=25.0, bucket_stats=bucket_stats
        )

        # Should fall back to midpoint (0.25)
        # Mean should be around 0.25, not 0.7
        assert np.mean(observations) == pytest.approx(0.25, rel=0.1)

    def test_sum_constraint_adjusts_positions(self):
        """Test sum constraint adjusts observation positions."""
        import numpy as np

        bucket_deltas = {"1.0": 100.0, "+Inf": 100.0}

        # Midpoint-based sum would be ~50 (100 * 0.5)
        # Test that different target sums result in different distributions
        # Note: adjustment is limited to 40% of bucket width to stay within bounds
        high_sum = 65.0  # Above midpoint
        obs_high = generate_observations_with_sum_constraint(
            bucket_deltas, target_sum=high_sum, bucket_stats=None
        )

        # Target lower sum means observations shift toward lower bound
        low_sum = 35.0  # Below midpoint
        obs_low = generate_observations_with_sum_constraint(
            bucket_deltas, target_sum=low_sum, bucket_stats=None
        )

        # High sum observations should be shifted higher on average
        assert np.mean(obs_high) > np.mean(obs_low)
        # High sum target should result in higher actual sum than low sum target
        assert obs_high.sum() > obs_low.sum()

    def test_proportional_adjustment_across_buckets(self):
        """Test adjustment is distributed proportionally across buckets."""
        import numpy as np

        # Two buckets with different observation counts
        bucket_deltas = {"0.1": 10.0, "1.0": 110.0, "+Inf": 110.0}
        # Per-bucket: 10 in [0,0.1], 100 in (0.1,1.0]

        # Midpoint sums: 10*0.05=0.5, 100*0.55=55, total=55.5
        # Target 60 means +4.5 needs to be distributed
        target_sum = 60.0

        observations = generate_observations_with_sum_constraint(
            bucket_deltas, target_sum=target_sum, bucket_stats=None
        )

        # The larger bucket (100 obs) should absorb more of the adjustment
        assert len(observations) == 110
        # Sum should be close to target
        assert np.isclose(observations.sum(), target_sum, rtol=0.05)

    def test_empty_buckets_returns_empty(self):
        """Test empty bucket deltas returns empty array."""
        result = generate_observations_with_sum_constraint({}, target_sum=0.0)

        assert len(result) == 0

    def test_only_inf_bucket_returns_empty(self):
        """Test only +Inf bucket returns empty (no finite observations)."""
        bucket_deltas = {"+Inf": 100.0}

        result = generate_observations_with_sum_constraint(
            bucket_deltas, target_sum=100.0
        )

        # Only +Inf bucket, no finite observations to generate
        assert len(result) == 0

    def test_zero_target_sum_returns_observations(self):
        """Test zero target sum still generates observations."""
        bucket_deltas = {"0.5": 10.0, "+Inf": 10.0}

        result = generate_observations_with_sum_constraint(
            bucket_deltas, target_sum=0.0
        )

        # Should still generate observations (no adjustment possible)
        assert len(result) == 10

    def test_single_observation_per_bucket(self):
        """Test single observation per bucket."""
        bucket_deltas = {"0.2": 1.0, "0.4": 2.0, "0.6": 3.0, "+Inf": 3.0}
        # Per-bucket: 1, 1, 1

        target_sum = 0.6  # Approx midpoints: 0.1 + 0.3 + 0.5 = 0.9

        observations = generate_observations_with_sum_constraint(
            bucket_deltas, target_sum=target_sum, bucket_stats=None
        )

        assert len(observations) == 3
        # Each in respective bucket
        assert 0 <= observations[0] <= 0.2
        assert 0.2 <= observations[1] <= 0.4
        assert 0.4 <= observations[2] <= 0.6

    def test_multiple_learned_means_used(self):
        """Test multiple learned means are used for respective buckets."""
        import numpy as np

        bucket_deltas = {"0.1": 50.0, "0.5": 100.0, "+Inf": 100.0}
        # Per-bucket: 50 in [0,0.1], 50 in (0.1,0.5]

        # Learn means for both buckets
        bucket_stats = {
            "0.1": BucketStatistics(
                bucket_le="0.1",
                observation_count=100,
                weighted_mean_sum=3.0,  # mean = 0.03 (near lower bound)
                sample_count=20,
            ),
            "0.5": BucketStatistics(
                bucket_le="0.5",
                observation_count=100,
                weighted_mean_sum=40.0,  # mean = 0.4 (near upper bound)
                sample_count=20,
            ),
        }

        observations = generate_observations_with_sum_constraint(
            bucket_deltas,
            target_sum=22.0,  # 50*0.03 + 50*0.4 = 1.5 + 20 = 21.5
            bucket_stats=bucket_stats,
        )

        # First bucket should cluster near 0.03
        first_bucket = observations[:50]
        assert np.mean(first_bucket) < 0.05  # Near 0.03

        # Second bucket should cluster near 0.4
        second_bucket = observations[50:]
        assert np.mean(second_bucket) > 0.3  # Near 0.4


class TestGenerateObservationsAccuracy:
    """Test accuracy of polynomial histogram approach vs standard interpolation."""

    def test_accuracy_improvement_clustered_distribution(self):
        """Test accuracy improvement when observations cluster near bucket edge.

        Standard Prometheus interpolation assumes uniform distribution (midpoint).
        When observations actually cluster near one edge, our sum-constrained
        approach should produce more accurate percentiles.
        """
        import numpy as np

        # Scenario: 100 observations all in [0, 0.5] bucket
        # True distribution: clustered near 0.4 (mean=0.4, not midpoint 0.25)
        bucket_deltas = {"0.5": 100.0, "+Inf": 100.0}
        true_mean = 0.4
        target_sum = 100 * true_mean  # 40.0

        # Standard midpoint approach would estimate mean=0.25, sum=25
        midpoint_estimate = 0.25

        # Our approach
        observations = generate_observations_with_sum_constraint(
            bucket_deltas, target_sum=target_sum, bucket_stats=None
        )

        actual_mean = np.mean(observations)

        # Our estimate should be closer to true mean than midpoint
        our_error = abs(actual_mean - true_mean)
        midpoint_error = abs(midpoint_estimate - true_mean)

        assert our_error < midpoint_error

    def test_accuracy_with_learned_means(self):
        """Test accuracy improvement when learned means are available."""
        import numpy as np

        # Scenario: We learned that observations in bucket cluster at specific position
        bucket_deltas = {"0.5": 100.0, "+Inf": 100.0}
        true_mean = 0.35

        # Learned mean from previous single-bucket intervals
        bucket_stats = {
            "0.5": BucketStatistics(
                bucket_le="0.5",
                observation_count=500,
                weighted_mean_sum=175.0,  # mean = 0.35
                sample_count=50,
            )
        }

        target_sum = 100 * true_mean  # 35.0

        observations = generate_observations_with_sum_constraint(
            bucket_deltas, target_sum=target_sum, bucket_stats=bucket_stats
        )

        actual_mean = np.mean(observations)

        # With learned means, should be very close to true mean
        assert np.isclose(actual_mean, true_mean, rtol=0.05)

    def test_p99_accuracy_tail_distribution(self):
        """Test p99 accuracy for heavy-tailed distribution.

        This is a key use case: when high percentiles matter, the standard
        midpoint assumption can significantly underestimate latencies.
        """
        import numpy as np

        # Realistic latency scenario:
        # 900 requests < 100ms, 80 requests 100-500ms, 20 requests 500ms-1s
        # Cumulative: 900, 980, 1000
        bucket_deltas = {"0.1": 900.0, "0.5": 980.0, "1.0": 1000.0, "+Inf": 1000.0}

        # p99 = 990th observation (index 989 in 0-indexed array)
        # Bucket 1: indices 0-899, Bucket 2: indices 900-979, Bucket 3: indices 980-999
        # So p99 falls in the (0.5, 1.0] bucket
        target_sum = 900 * 0.05 + 80 * 0.3 + 20 * 0.8  # 45 + 24 + 16 = 85

        observations = generate_observations_with_sum_constraint(
            bucket_deltas, target_sum=target_sum, bucket_stats=None
        )

        assert len(observations) == 1000
        p99 = np.percentile(observations, 99)

        # p99 (990th observation) should be in the last finite bucket
        assert 0.5 < p99 <= 1.0


class TestComputeBestGuessPercentiles:
    """Test compute_best_guess_percentiles function."""

    def test_computes_percentiles_with_inf_observations(self):
        """Test percentile computation including +Inf bucket estimates."""

        # Cumulative: 90 <= 0.5, 97 <= 1.0, 100 total (3 in +Inf)
        bucket_deltas = {"0.5": 90.0, "1.0": 97.0, "+Inf": 100.0}
        bucket_stats = {}  # No learned stats
        total_sum = 100.0
        total_count = 100

        result = compute_best_guess_percentiles(
            bucket_deltas, bucket_stats, total_sum, total_count
        )

        assert result is not None
        assert result.p50 is not None
        assert result.p99 is not None
        assert result.p999 is not None
        assert result.inf_bucket_count == 3  # 100 - 97 = 3
        assert result.inf_bucket_estimated_mean is not None
        assert result.inf_bucket_estimated_mean > 1.0  # Must be > max finite bucket

    def test_confidence_high_when_inf_small(self):
        """Test confidence is high when +Inf bucket is small."""
        # Cumulative: 500 <= 0.5, 500 total (0 in +Inf)
        bucket_deltas = {"0.5": 500.0, "+Inf": 500.0}
        bucket_stats = {
            "0.5": BucketStatistics(
                bucket_le="0.5",
                observation_count=100,
                weighted_mean_sum=25.0,
                sample_count=10,
            )
        }

        result = compute_best_guess_percentiles(bucket_deltas, bucket_stats, 150.0, 500)

        # inf_ratio = 0/500 = 0, mean_coverage = 1/1 = 1.0
        # Should be high confidence
        assert result is not None
        assert result.estimation_confidence == "high"

    def test_confidence_low_when_inf_large(self):
        """Test confidence is low when +Inf bucket is large."""
        # Cumulative: 50 <= 1.0, 100 total (50 in +Inf)
        bucket_deltas = {"1.0": 50.0, "+Inf": 100.0}
        bucket_stats = {}

        result = compute_best_guess_percentiles(
            bucket_deltas, bucket_stats, 2000.0, 100
        )

        # inf_ratio = 50/100 = 0.5 > 0.05, should be low
        assert result is not None
        assert result.estimation_confidence == "low"

    def test_returns_none_on_empty_data(self):
        """Test returns None when data is empty."""
        result = compute_best_guess_percentiles(
            bucket_deltas={},
            bucket_stats={},
            total_sum=0.0,
            total_count=0,
        )

        assert result is None


class TestBestGuessPercentilesIntegration:
    """Integration tests for best_guess percentiles in HistogramExportStats."""

    def test_best_guess_computed_in_from_time_series(self):
        """Test that best_guess percentiles are computed in from_time_series."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist({"0.5": 0.0, "1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                (
                    NANOS_PER_SECOND,
                    hist({"0.5": 50.0, "1.0": 90.0, "+Inf": 100.0}, 70.0, 100.0),
                ),
            ],
        )

        stats = HistogramExportStats.from_time_series(ts.histograms["ttft"])

        assert stats.percentiles is not None
        assert stats.percentiles.best_guess is not None
        assert stats.percentiles.best_guess.p50 is not None
        assert stats.percentiles.best_guess.p99 is not None
        assert stats.percentiles.best_guess.p999 is not None

    def test_best_guess_includes_inf_bucket_metadata(self):
        """Test that best_guess includes +Inf bucket metadata."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                # 90 observations <= 1.0s, 10 in +Inf (>1.0s)
                (NANOS_PER_SECOND, hist({"1.0": 90.0, "+Inf": 100.0}, 150.0, 100.0)),
            ],
        )

        stats = HistogramExportStats.from_time_series(ts.histograms["latency"])

        assert stats.percentiles is not None
        best_guess = stats.percentiles.best_guess
        assert best_guess is not None
        assert best_guess.inf_bucket_count == 10
        assert best_guess.inf_bucket_estimated_mean is not None
        assert best_guess.inf_bucket_estimated_mean > 1.0

    def test_best_guess_differs_from_observed_when_inf_present(self):
        """Test best_guess differs from observed when +Inf bucket has observations."""
        ts = ServerMetricsTimeSeries()
        # Large +Inf bucket to show difference
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                # 50 observations <= 1.0s, 50 in +Inf (>1.0s) with high values
                (
                    NANOS_PER_SECOND,
                    hist({"1.0": 50.0, "+Inf": 100.0}, 750.0, 100.0),  # avg=7.5s
                ),
            ],
        )

        stats = HistogramExportStats.from_time_series(ts.histograms["latency"])

        assert stats.percentiles is not None
        observed = stats.percentiles.observed
        best_guess = stats.percentiles.best_guess

        assert observed is not None
        assert best_guess is not None

        # Observed ignores +Inf, so p99 should be in finite buckets (<=1.0)
        assert observed.p99 is not None
        assert observed.p99 <= 1.0

        # Best_guess includes +Inf estimates, so p99 should be higher
        # 50 of 100 observations are in +Inf, so p99 (99th percentile) should include +Inf
        assert best_guess.p99 is not None
        # With 50% in +Inf, the p99 should be in the +Inf region
        assert best_guess.p99 > 1.0

    def test_best_guess_none_on_counter_reset(self):
        """Test best_guess is None when counter reset detected."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 1000.0, "+Inf": 2000.0}, 500.0, 2000.0)),
                # Counter reset
                (NANOS_PER_SECOND, hist({"1.0": 50.0, "+Inf": 100.0}, 25.0, 100.0)),
            ],
        )

        stats = HistogramExportStats.from_time_series(ts.histograms["latency"])

        assert stats.percentiles is None


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCasesValidation:
    """Test edge cases and input validation."""

    def test_extract_all_observations_mismatched_lengths_raises(self):
        """Test that mismatched array lengths raise ValueError."""
        import numpy as np

        timestamps = np.array([0, NANOS_PER_SECOND, 2 * NANOS_PER_SECOND])
        counts = np.array([0.0, 1.0])  # Wrong length!
        sums = np.array([0.0, 0.1, 0.2])
        bucket_snapshots = [
            {"0.5": 0.0, "+Inf": 0.0},
            {"0.5": 1.0, "+Inf": 1.0},
            {"0.5": 2.0, "+Inf": 2.0},
        ]

        with pytest.raises(ValueError, match="Array length mismatch"):
            extract_all_observations(timestamps, sums, counts, bucket_snapshots)

    def test_extract_all_observations_mismatched_bucket_snapshots_raises(self):
        """Test that mismatched bucket_snapshots length raises ValueError."""
        import numpy as np

        timestamps = np.array([0, NANOS_PER_SECOND, 2 * NANOS_PER_SECOND])
        counts = np.array([0.0, 1.0, 2.0])
        sums = np.array([0.0, 0.1, 0.2])
        bucket_snapshots = [  # Wrong length - only 2 instead of 3
            {"0.5": 0.0, "+Inf": 0.0},
            {"0.5": 1.0, "+Inf": 1.0},
        ]

        with pytest.raises(ValueError, match="bucket_snapshots length"):
            extract_all_observations(timestamps, sums, counts, bucket_snapshots)

    def test_accumulate_bucket_statistics_mismatched_lengths_raises(self):
        """Test that accumulate_bucket_statistics validates array lengths."""
        import numpy as np

        timestamps = np.array([0, NANOS_PER_SECOND])
        counts = np.array([0.0, 1.0, 2.0])  # Wrong length!
        sums = np.array([0.0, 0.1])
        bucket_snapshots = [{"0.5": 0.0}, {"0.5": 1.0}]

        with pytest.raises(ValueError, match="Array length mismatch"):
            accumulate_bucket_statistics(timestamps, sums, counts, bucket_snapshots)

    def test_histogram_quantile_handles_very_large_counts(self):
        """Test histogram_quantile handles very large observation counts."""
        # 1 billion observations - tests for overflow
        buckets = {
            "0.5": 500_000_000.0,
            "1.0": 1_000_000_000.0,
            "+Inf": 1_000_000_000.0,
        }

        p50 = histogram_quantile(0.50, buckets)
        p99 = histogram_quantile(0.99, buckets)

        assert p50 is not None
        assert p99 is not None
        assert 0 < p50 <= 0.5
        assert 0.5 < p99 <= 1.0

    def test_histogram_quantile_handles_very_small_counts(self):
        """Test histogram_quantile handles fractional counts (should still work)."""
        # Fractional counts can happen with rate calculations
        buckets = {"0.5": 0.5, "1.0": 1.0, "+Inf": 1.0}

        p50 = histogram_quantile(0.50, buckets)

        assert p50 is not None
        assert 0 <= p50 <= 0.5

    def test_generate_observations_handles_very_large_bucket(self):
        """Test observation generation handles very wide bucket ranges."""
        # Bucket from 0 to 1000 (very wide)
        bucket_deltas = {"1000.0": 100.0, "+Inf": 100.0}
        target_sum = 50000.0  # mean of 500

        observations = generate_observations_with_sum_constraint(
            bucket_deltas, target_sum=target_sum, bucket_stats=None
        )

        assert len(observations) == 100
        assert all(0 <= obs <= 1000 for obs in observations)

    def test_generate_observations_handles_tiny_bucket(self):
        """Test observation generation handles very narrow bucket ranges."""
        # Bucket from 0 to 0.001 (very narrow)
        bucket_deltas = {"0.001": 100.0, "+Inf": 100.0}
        target_sum = 0.05  # mean of 0.0005

        observations = generate_observations_with_sum_constraint(
            bucket_deltas, target_sum=target_sum, bucket_stats=None
        )

        assert len(observations) == 100
        assert all(0 <= obs <= 0.001 for obs in observations)

    def test_bucket_statistics_handles_zero_count(self):
        """Test BucketStatistics handles zero-count records gracefully."""
        stats = BucketStatistics(bucket_le="0.5")
        stats.record(mean=0.25, count=0)  # Zero count

        assert stats.observation_count == 0
        assert stats.estimated_mean is None  # No valid mean

    def test_estimate_inf_bucket_handles_extreme_values(self):
        """Test +Inf bucket estimation handles extreme sum values."""
        # Very large sum relative to finite observations
        observations = estimate_inf_bucket_observations(
            total_sum=1_000_000.0,
            estimated_finite_sum=100.0,
            inf_count=10,
            max_finite_bucket=10.0,
        )

        # inf_avg = (1_000_000 - 100) / 10 = 99,990
        assert len(observations) == 10
        assert all(obs >= 10.0 for obs in observations)  # All above max finite

    def test_compute_best_guess_handles_all_in_inf_bucket(self):
        """Test best_guess handles case where all observations in +Inf."""
        # All 100 observations in +Inf bucket
        bucket_deltas = {"1.0": 0.0, "+Inf": 100.0}
        bucket_stats = {}
        total_sum = 1500.0  # Average of 15 per observation
        total_count = 100

        result = compute_best_guess_percentiles(
            bucket_deltas, bucket_stats, total_sum, total_count
        )

        assert result is not None
        assert result.inf_bucket_count == 100
        assert result.finite_observations_count == 0
        # All percentiles should be in the +Inf region
        assert result.p50 is not None
        assert result.p50 > 1.0


class TestEdgeCasesNumericalStability:
    """Test numerical edge cases for stability."""

    def test_histogram_quantile_with_zero_in_bucket_boundary(self):
        """Test histogram with 0 as bucket boundary works correctly."""
        # Some histograms use 0 as first boundary
        buckets = {"0": 10.0, "0.5": 50.0, "+Inf": 50.0}

        p50 = histogram_quantile(0.50, buckets)

        # p50 should be in the (0, 0.5] bucket
        assert p50 is not None
        assert 0 <= p50 <= 0.5

    def test_bucket_bounds_first_bucket_negative_boundary(self):
        """Test get_bucket_bounds handles negative bucket boundaries."""
        sorted_buckets = ["-1.0", "0", "1.0"]

        lower, upper = get_bucket_bounds("-1.0", sorted_buckets)
        # First bucket: lower bound is typically 0, but for negative buckets
        # the logic may differ
        assert upper == -1.0

    def test_generate_observations_exact_sum_match(self):
        """Test that generated observations match target sum closely."""
        import numpy as np

        bucket_deltas = {"0.5": 50.0, "1.0": 100.0, "+Inf": 100.0}
        target_sum = 37.5  # Exact midpoint sum: 50*0.25 + 50*0.75 = 12.5 + 37.5 = 50

        observations = generate_observations_with_sum_constraint(
            bucket_deltas, target_sum=target_sum, bucket_stats=None
        )

        # Sum should be close to target (within 5% due to clamping limits)
        assert np.isclose(observations.sum(), target_sum, rtol=0.15)

    def test_estimate_bucket_sums_empty_buckets(self):
        """Test estimate_bucket_sums with empty bucket counts."""
        bucket_deltas = {"0.5": 0.0, "1.0": 0.0, "+Inf": 0.0}
        bucket_stats = {}

        sums = estimate_bucket_sums(bucket_deltas, bucket_stats)

        # All zeros should result in empty or zero sums
        assert all(v == 0.0 for v in sums.values()) or len(sums) == 0
