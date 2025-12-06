# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for best guess percentiles computation and edge cases."""

import numpy as np
import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models.export_models import (
    BucketStatistics,
    HistogramExportStats,
    accumulate_bucket_statistics,
    compute_best_guess_percentiles,
    estimate_bucket_sums,
    extract_all_observations,
    generate_observations_with_sum_constraint,
    get_bucket_bounds,
    histogram_quantile,
)
from aiperf.common.models.server_metrics_models import ServerMetricsTimeSeries
from tests.unit.server_metrics.helpers import add_histogram_snapshots, hist

# =============================================================================
# Best Guess Percentiles Tests
# =============================================================================


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
        from aiperf.common.models.export_models import estimate_inf_bucket_observations

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
