# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Histogram analysis functions and bucket statistics for Prometheus histograms.

This module provides:
- histogram_quantile: Standard Prometheus quantile estimation via bucket interpolation
- Observation extraction: Recover individual observations from histogram deltas
- BucketStatistics: Per-bucket mean tracking (polynomial histogram approach)
- Bucket utility functions: Bounds calculation, cumulative-to-per-bucket conversion

The polynomial histogram approach (based on HistogramTools research) improves
percentile estimation accuracy by learning per-bucket means from single-bucket
scrape intervals, rather than assuming uniform distribution within buckets.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel

# =============================================================================
# Histogram Quantile Functions
# =============================================================================


def histogram_quantile(q: float, buckets: dict[str, float]) -> float | None:
    """Estimate quantile from Prometheus histogram buckets using linear interpolation.

    Implements Prometheus's histogram_quantile() algorithm. Assumes observations
    are uniformly distributed within each bucket (linear interpolation).

    Note: Accuracy depends on bucket granularity. Finer bucket boundaries around
    percentiles of interest yield more accurate estimates.

    Args:
        q: Quantile to compute (0.0 to 1.0, e.g., 0.95 for p95)
        buckets: Dict mapping bucket upper bounds to cumulative counts
                 e.g., {"0.01": 50, "0.1": 200, "1.0": 950, "+Inf": 1000}

    Returns:
        Estimated quantile value, or None if insufficient data
    """
    if not buckets or q < 0 or q > 1:
        return None

    # Parse and sort buckets by upper bound
    parsed: list[tuple[float, float]] = []
    for le, count in buckets.items():
        upper = float("inf") if le == "+Inf" else float(le)
        parsed.append((upper, count))
    parsed.sort(key=lambda x: x[0])

    # Need at least 2 buckets (one real + +Inf)
    if len(parsed) < 2:
        return None

    # Total observations (from +Inf bucket)
    total = parsed[-1][1]
    if total == 0:
        return None

    # Target rank
    rank = q * total

    # Find the bucket containing our target rank
    prev_upper = 0.0
    prev_count = 0.0

    for i, (upper, cumulative_count) in enumerate(parsed):
        if cumulative_count >= rank:
            # Handle +Inf bucket: return upper bound of second-highest bucket
            if upper == float("inf"):
                if i > 0:
                    return parsed[i - 1][0]
                return None

            # Count in this bucket
            bucket_count = cumulative_count - prev_count
            if bucket_count == 0:
                prev_upper = upper
                prev_count = cumulative_count
                continue

            # Rank within this bucket
            rank_in_bucket = rank - prev_count

            # Linear interpolation
            # Handle first bucket: assume lower bound of 0 if upper > 0
            bucket_start = prev_upper if prev_upper > 0 or i > 0 else 0.0

            return bucket_start + (upper - bucket_start) * (
                rank_in_bucket / bucket_count
            )

        prev_upper = upper
        prev_count = cumulative_count

    # Quantile beyond all buckets (shouldn't happen with +Inf)
    return parsed[-2][0] if len(parsed) >= 2 else None


# =============================================================================
# Observation Extraction (Per-scrape observation recovery)
# =============================================================================


def extract_observations_from_scrape(
    count_delta: int,
    sum_delta: float,
    bucket_deltas: dict[str, float],
) -> tuple[list[float], int, int]:
    """Extract observations from a single scrape delta using bucket information.

    When count_delta == 1, we know the exact observation value (sum_delta).
    When count_delta > 1, we use bucket deltas to place observations within
    their respective buckets using linear interpolation.

    Args:
        count_delta: Number of observations in this scrape interval
        sum_delta: Sum of observation values in this scrape interval
        bucket_deltas: Per-bucket count deltas for this scrape interval

    Returns:
        Tuple of (observations, exact_count, bucket_placed_count)
        - observations: List of observation values (exact or bucket-interpolated)
        - exact_count: Number of exact observations (count_delta == 1)
        - bucket_placed_count: Number of bucket-interpolated observations
    """
    if count_delta <= 0:
        return [], 0, 0

    if count_delta == 1:
        # Exact observation value!
        return [sum_delta], 1, 0

    # Multiple observations - use bucket deltas to place them
    observations: list[float] = []

    # Parse and sort buckets by upper bound
    parsed: list[tuple[float, float]] = []
    for le, count in bucket_deltas.items():
        upper = float("inf") if le == "+Inf" else float(le)
        parsed.append((upper, count))
    parsed.sort(key=lambda x: x[0])

    if not parsed:
        return [], 0, 0

    # Convert cumulative bucket counts to per-bucket counts and generate observations
    prev_upper = 0.0
    prev_cumulative = 0.0

    for upper, cumulative in parsed:
        bucket_obs_count = int(cumulative - prev_cumulative)
        if bucket_obs_count > 0 and upper != float("inf"):
            # Distribute observations within bucket using linear interpolation
            for i in range(bucket_obs_count):
                # Place at center of each "slot" within the bucket
                frac = (i + 0.5) / bucket_obs_count
                bucket_start = prev_upper if prev_upper > 0 else 0.0
                value = bucket_start + (upper - bucket_start) * frac
                observations.append(value)
        prev_cumulative = cumulative
        if upper != float("inf"):
            prev_upper = upper

    return observations, 0, len(observations)


def extract_all_observations(
    timestamps: np.ndarray,
    sums: np.ndarray,
    counts: np.ndarray,
    bucket_snapshots: list[dict[str, float]],
    start_idx: int = 0,
) -> tuple[np.ndarray, int, int]:
    """Extract all observations from histogram time series using per-scrape deltas.

    Processes each scrape interval to extract exact observations (when count_delta == 1)
    or bucket-placed observations (when count_delta > 1).

    Args:
        timestamps: Array of scrape timestamps (nanoseconds)
        sums: Array of cumulative sum values per scrape
        counts: Array of cumulative count values per scrape
        bucket_snapshots: List of bucket snapshots per scrape
        start_idx: Starting index (for time filtering, e.g., skip warmup)

    Returns:
        Tuple of (observations_array, exact_count, bucket_placed_count)

    Raises:
        ValueError: If array lengths don't match
    """
    # Validate input array lengths match
    n_timestamps = len(timestamps)
    if len(sums) != n_timestamps or len(counts) != n_timestamps:
        raise ValueError(
            f"Array length mismatch: timestamps={n_timestamps}, "
            f"sums={len(sums)}, counts={len(counts)}"
        )
    if len(bucket_snapshots) != n_timestamps:
        raise ValueError(
            f"bucket_snapshots length ({len(bucket_snapshots)}) must match "
            f"timestamps length ({n_timestamps})"
        )

    all_observations: list[float] = []
    total_exact = 0
    total_bucket_placed = 0

    # Process each scrape interval
    for i in range(start_idx + 1, len(timestamps)):
        count_delta = int(counts[i] - counts[i - 1])
        sum_delta = float(sums[i] - sums[i - 1])

        if count_delta <= 0:
            continue

        # Compute bucket deltas for this interval
        curr_buckets = bucket_snapshots[i]
        prev_buckets = bucket_snapshots[i - 1]

        bucket_deltas: dict[str, float] = {}
        for le, curr_val in curr_buckets.items():
            prev_val = prev_buckets.get(le, 0.0)
            delta = curr_val - prev_val
            if delta >= 0:  # Skip negative deltas (counter reset)
                bucket_deltas[le] = delta

        # Extract observations from this scrape
        obs, exact, bucket_placed = extract_observations_from_scrape(
            count_delta, sum_delta, bucket_deltas
        )
        all_observations.extend(obs)
        total_exact += exact
        total_bucket_placed += bucket_placed

    return (
        np.array(all_observations, dtype=np.float64),
        total_exact,
        total_bucket_placed,
    )


# =============================================================================
# Polynomial Histogram Statistics (Per-Bucket Mean Tracking)
# =============================================================================
# Based on HistogramTools research (arXiv 2504.00001) showing 2.5x accuracy
# improvement by storing per-bucket means instead of just counts.


class BucketStatistics(AIPerfBaseModel):
    """Statistics for a single histogram bucket learned from single-bucket scrape intervals.

    When all observations in a scrape interval land in ONE bucket, we can compute the
    exact mean for that bucket: mean = sum_delta / count_delta. Over many such intervals,
    we learn the typical position of observations within each bucket.

    This is a core component of the "polynomial histogram" approach which improves
    percentile estimation accuracy by 2.5x compared to simple linear interpolation.
    """

    bucket_le: str = Field(description="Bucket upper bound (le value)")
    observation_count: int = Field(
        default=0, description="Total observations used to learn this bucket's mean"
    )
    weighted_mean_sum: float = Field(
        default=0.0,
        description="Sum of (mean * count) for weighted average calculation",
    )
    sample_count: int = Field(
        default=0, description="Number of single-bucket intervals observed"
    )

    @property
    def estimated_mean(self) -> float | None:
        """Compute the weighted average position within this bucket.

        Returns None if no single-bucket intervals have been observed.
        """
        if self.observation_count == 0:
            return None
        return self.weighted_mean_sum / self.observation_count

    def record(self, mean: float, count: int) -> None:
        """Record statistics from a single-bucket scrape interval.

        Args:
            mean: Exact mean value for observations in this interval (sum_delta/count_delta)
            count: Number of observations in this interval
        """
        self.observation_count += count
        self.weighted_mean_sum += mean * count
        self.sample_count += 1


def accumulate_bucket_statistics(
    timestamps: np.ndarray,
    sums: np.ndarray,
    counts: np.ndarray,
    bucket_snapshots: list[dict[str, float]],
    start_idx: int = 0,
) -> dict[str, BucketStatistics]:
    """Learn per-bucket mean positions from single-bucket scrape intervals.

    This implements the polynomial histogram approach: when all observations
    in a scrape interval land in a single bucket, we can compute the exact mean
    for that bucket (sum_delta / count_delta).

    Over time, this learns the typical position of observations within each bucket,
    which is more accurate than assuming uniform distribution (midpoint).

    Args:
        timestamps: Array of scrape timestamps (nanoseconds)
        sums: Array of cumulative sum values per scrape
        counts: Array of cumulative count values per scrape
        bucket_snapshots: List of bucket snapshots per scrape
        start_idx: Starting index for analysis

    Returns:
        Dict mapping bucket le values to BucketStatistics with learned means

    Raises:
        ValueError: If array lengths don't match
    """
    # Validate input array lengths match
    n_timestamps = len(timestamps)
    if len(sums) != n_timestamps or len(counts) != n_timestamps:
        raise ValueError(
            f"Array length mismatch: timestamps={n_timestamps}, "
            f"sums={len(sums)}, counts={len(counts)}"
        )
    if len(bucket_snapshots) != n_timestamps:
        raise ValueError(
            f"bucket_snapshots length ({len(bucket_snapshots)}) must match "
            f"timestamps length ({n_timestamps})"
        )

    bucket_stats: dict[str, BucketStatistics] = {}

    for i in range(start_idx + 1, len(timestamps)):
        count_delta = int(counts[i] - counts[i - 1])
        sum_delta = float(sums[i] - sums[i - 1])

        if count_delta <= 0:
            continue

        # Compute bucket deltas for this interval
        curr_buckets = bucket_snapshots[i]
        prev_buckets = bucket_snapshots[i - 1]

        # Compute cumulative bucket deltas for this interval
        cumulative_deltas: dict[str, float] = {}
        for le, curr_val in curr_buckets.items():
            prev_val = prev_buckets.get(le, 0.0)
            delta = curr_val - prev_val
            if delta > 0:
                cumulative_deltas[le] = delta

        # Convert cumulative deltas to per-bucket deltas
        per_bucket_deltas = cumulative_to_per_bucket(cumulative_deltas)

        # Find active buckets (those with observations in this interval)
        active_buckets: list[tuple[str, float]] = [
            (le, delta) for le, delta in per_bucket_deltas.items() if delta > 0
        ]

        # If all observations landed in ONE bucket, we know the exact mean for that bucket
        if len(active_buckets) == 1:
            le, delta = active_buckets[0]
            bucket_mean = sum_delta / count_delta  # Exact mean for this bucket

            if le not in bucket_stats:
                bucket_stats[le] = BucketStatistics(bucket_le=le)
            bucket_stats[le].record(bucket_mean, count_delta)

    return bucket_stats


# =============================================================================
# Bucket Utility Functions
# =============================================================================


def get_bucket_bounds(le: str, sorted_buckets: list[str]) -> tuple[float, float]:
    """Get the lower and upper bounds for a bucket.

    Args:
        le: The bucket's upper bound (le value)
        sorted_buckets: List of all bucket le values sorted by numeric value

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    upper = float("inf") if le == "+Inf" else float(le)

    # Find previous bucket for lower bound
    idx = sorted_buckets.index(le)
    if idx == 0:
        lower = 0.0
    else:
        prev_le = sorted_buckets[idx - 1]
        lower = float(prev_le) if prev_le != "+Inf" else 0.0

    return lower, upper


def cumulative_to_per_bucket(
    bucket_deltas: dict[str, float],
) -> dict[str, float]:
    """Convert cumulative bucket counts to per-bucket counts.

    Prometheus histograms use cumulative counts (le="less than or equal").
    This converts to per-bucket counts (observations within each bucket range).

    Args:
        bucket_deltas: Cumulative bucket counts (Prometheus style)

    Returns:
        Dict mapping bucket le values to per-bucket counts
    """
    # Sort buckets by numeric value
    finite_buckets = [le for le in bucket_deltas if le != "+Inf"]
    sorted_buckets = sorted(finite_buckets, key=lambda x: float(x))

    per_bucket: dict[str, float] = {}
    prev_cumulative = 0.0

    for le in sorted_buckets:
        cumulative = bucket_deltas[le]
        per_bucket[le] = cumulative - prev_cumulative
        prev_cumulative = cumulative

    # Handle +Inf bucket if present
    if "+Inf" in bucket_deltas:
        inf_cumulative = bucket_deltas["+Inf"]
        per_bucket["+Inf"] = inf_cumulative - prev_cumulative

    return per_bucket


def estimate_bucket_sums(
    bucket_deltas: dict[str, float],
    bucket_stats: dict[str, BucketStatistics],
) -> dict[str, float]:
    """Estimate the sum of observations in each finite bucket.

    Uses learned per-bucket means when available, falls back to midpoint interpolation.

    Args:
        bucket_deltas: Cumulative bucket counts (Prometheus style)
        bucket_stats: Learned per-bucket statistics from polynomial histogram approach

    Returns:
        Dict mapping bucket le values to estimated sums (excludes +Inf bucket)
    """
    # Convert cumulative to per-bucket counts
    per_bucket_counts = cumulative_to_per_bucket(bucket_deltas)

    # Sort buckets by numeric value for bound calculation
    finite_buckets = [le for le in bucket_deltas if le != "+Inf"]
    sorted_buckets = sorted(finite_buckets, key=lambda x: float(x))

    sums: dict[str, float] = {}
    for le, count in per_bucket_counts.items():
        if le == "+Inf" or count <= 0:
            continue

        # Try to use learned mean first
        if le in bucket_stats and bucket_stats[le].estimated_mean is not None:
            mean = bucket_stats[le].estimated_mean
        else:
            # Fall back to midpoint interpolation
            lower, upper = get_bucket_bounds(le, sorted_buckets)
            mean = (lower + upper) / 2

        sums[le] = count * mean

    return sums


def estimate_inf_bucket_observations(
    total_sum: float,
    estimated_finite_sum: float,
    inf_count: int,
    max_finite_bucket: float,
) -> list[float]:
    """Estimate observation values for the +Inf bucket using back-calculation.

    Key insight: We have the exact total sum from Prometheus. By estimating the
    sum of finite bucket observations, we can back-calculate what the +Inf bucket
    observations must be on average.

    Args:
        total_sum: Exact total sum from histogram (sum_delta)
        estimated_finite_sum: Estimated sum of observations in finite buckets
        inf_count: Number of observations in the +Inf bucket
        max_finite_bucket: Upper bound of the highest finite bucket

    Returns:
        List of estimated observation values for +Inf bucket (all > max_finite_bucket)
    """
    if inf_count <= 0:
        return []

    # Back-calculate +Inf bucket sum
    inf_sum = total_sum - estimated_finite_sum

    # Validate: +Inf sum must be positive and mean must be > max finite bucket
    if inf_sum <= 0:
        # Estimation error - fall back to placing at 1.5x max bucket
        inf_avg = max_finite_bucket * 1.5
    else:
        inf_avg = inf_sum / inf_count
        # Validate: average must be > max_finite_bucket (by definition of +Inf bucket)
        if inf_avg <= max_finite_bucket:
            # Estimation error - use minimum valid value
            inf_avg = max_finite_bucket * 1.5

    # Generate observations spread around the estimated mean
    # Using uniform distribution: [lower_bound, upper_bound] where mean = (lower + upper) / 2
    # lower = max_finite_bucket, upper = 2 * inf_avg - max_finite_bucket
    upper_estimate = 2 * inf_avg - max_finite_bucket

    # Safety check: upper must be > lower
    if upper_estimate <= max_finite_bucket:
        upper_estimate = max_finite_bucket * 2

    # Generate observations uniformly distributed to match estimated mean
    observations = np.linspace(max_finite_bucket, upper_estimate, int(inf_count))

    return observations.tolist()
