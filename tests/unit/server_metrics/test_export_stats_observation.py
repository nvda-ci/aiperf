# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for observation extraction from histogram time series."""

import numpy as np
import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models.histogram_analysis import (
    extract_all_observations,
    extract_observations_from_scrape,
)

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
