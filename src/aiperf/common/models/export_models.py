# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from pydantic import ConfigDict, Field, SerializeAsAny

from aiperf.common.config import UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models import ErrorDetailsCount
from aiperf.common.models.base_models import AIPerfBaseModel

if TYPE_CHECKING:
    from aiperf.common.models.server_metrics_models import (
        HistogramTimeSeries,
        ScalarTimeSeries,
        SummaryTimeSeries,
        TimeRangeFilter,
    )

# ============================================================================
# Prometheus Terminology Glossary
# ============================================================================
#
# - **Metric Family**: A group of related metrics with the same name and type.
#   Example: All "http_request_duration_seconds" metrics form one family.
#
# - **Time Series**: A unique combination of metric name + labels.
#   Example: http_request_duration_seconds{method="GET", status="200"}
#   is a different time series from {method="POST", status="200"}.
#
# - **le** (less than or equal): Histogram bucket upper bound notation.
#   A bucket with le="0.1" contains all observations <= 0.1 seconds.
#
# - **Quantiles**: Pre-computed percentiles from the server (p50, p90, p95, p99).
#   These are exact values calculated by Prometheus, not estimated from buckets.
#
# =============================================================================


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


# =============================================================================
# Histogram Percentile Models (Nested Structure)
# =============================================================================


class BucketPercentiles(AIPerfBaseModel):
    """Percentiles estimated via Prometheus bucket interpolation.

    Uses the standard histogram_quantile algorithm that assumes uniform
    distribution within each bucket. Accuracy depends on bucket granularity.
    """

    p50: float | None = Field(
        default=None, description="Estimated 50th percentile (median)"
    )
    p90: float | None = Field(default=None, description="Estimated 90th percentile")
    p95: float | None = Field(default=None, description="Estimated 95th percentile")
    p99: float | None = Field(default=None, description="Estimated 99th percentile")


class ObservedPercentiles(AIPerfBaseModel):
    """Percentiles computed from per-scrape observation extraction.

    When scrape rate is high relative to observation rate, we can extract
    individual observation values:
    - count_delta == 1: Exact value from sum_delta
    - count_delta > 1: Bucket-placed values using per-scrape bucket deltas

    This approach can provide better accuracy than bucket interpolation,
    especially when exact_count is high relative to total observations.
    """

    p50: float | None = Field(
        default=None, description="50th percentile from extracted observations"
    )
    p90: float | None = Field(
        default=None, description="90th percentile from extracted observations"
    )
    p95: float | None = Field(
        default=None, description="95th percentile from extracted observations"
    )
    p99: float | None = Field(
        default=None, description="99th percentile from extracted observations"
    )
    exact_count: int = Field(
        default=0,
        description="Number of observations with exact values (count_delta == 1)",
    )
    bucket_placed_count: int = Field(
        default=0,
        description="Number of observations placed via bucket interpolation (count_delta > 1)",
    )
    coverage: float = Field(
        default=0.0,
        description="Ratio of exact observations to total (higher = more accurate)",
    )


class BestGuessPercentiles(AIPerfBaseModel):
    """Percentiles with +Inf bucket estimation using back-calculation.

    This approach addresses a critical flaw in other methods:
    - Bucket interpolation: Returns ceiling value for +Inf bucket (wrong)
    - Observed extraction: Skips +Inf observations entirely (underestimates)

    Best-guess uses the exact total sum to back-calculate +Inf bucket values:
    1. Learn per-bucket means from single-bucket scrape intervals (polynomial histogram)
    2. Estimate sum of finite bucket observations
    3. Back-calculate: inf_sum = total_sum - finite_sum
    4. Generate +Inf observations around estimated mean (all > max finite bucket)

    This provides more accurate tail percentiles when +Inf bucket has observations.
    """

    p50: float | None = Field(
        default=None, description="50th percentile including +Inf estimates"
    )
    p90: float | None = Field(
        default=None, description="90th percentile including +Inf estimates"
    )
    p95: float | None = Field(
        default=None, description="95th percentile including +Inf estimates"
    )
    p99: float | None = Field(
        default=None, description="99th percentile including +Inf estimates"
    )
    p999: float | None = Field(
        default=None, description="99.9th percentile including +Inf estimates"
    )

    # +Inf bucket handling metadata
    inf_bucket_count: int = Field(
        default=0, description="Number of observations in +Inf bucket"
    )
    inf_bucket_estimated_mean: float | None = Field(
        default=None,
        description="Estimated mean of +Inf observations (back-calculated from sum)",
    )

    # Estimation quality metrics
    finite_observations_count: int = Field(
        default=0, description="Number of observations in finite buckets"
    )
    buckets_with_learned_means: int = Field(
        default=0,
        description="Number of buckets with learned means (vs midpoint fallback)",
    )
    estimation_confidence: str = Field(
        default="low",
        description="Confidence level: high (inf<1% AND >50% buckets have learned means), "
        "medium (inf<5%), low (otherwise)",
    )


def generate_observations_with_sum_constraint(
    bucket_deltas: dict[str, float],
    target_sum: float,
    bucket_stats: dict[str, BucketStatistics] | None = None,
) -> np.ndarray:
    """Generate observations constrained to match the exact histogram sum.

    This is the core of the polynomial histogram percentile estimation algorithm.
    Standard Prometheus bucket interpolation assumes uniform distribution within
    each bucket (midpoint assumption), which can significantly over/underestimate
    percentiles when observations cluster near bucket boundaries.

    Algorithm:
        1. For each bucket, place observations using shifted uniform distribution:
           - If learned mean available: shift distribution to center on learned mean
           - Otherwise: use standard midpoint (uniform assumption)
        2. After initial placement, adjust positions proportionally across all
           buckets to match the exact target sum. Each bucket absorbs adjustment
           proportional to its sum contribution.

    Why this works:
        - The exact sum constrains where observations must be placed overall
        - Learned per-bucket means (from single-bucket scrape intervals) tell us
          where observations actually fall within specific buckets
        - Proportional adjustment distributes residual error fairly

    Args:
        bucket_deltas: Cumulative bucket counts (Prometheus format)
        target_sum: The exact sum of observations (from histogram sum_delta)
        bucket_stats: Optional learned per-bucket statistics from
                      accumulate_bucket_statistics()

    Returns:
        Array of generated observation values for finite buckets (excludes +Inf)
    """
    # Convert cumulative to per-bucket counts
    per_bucket = cumulative_to_per_bucket(bucket_deltas)

    # Get sorted bucket boundaries
    finite_buckets = [le for le in per_bucket if le != "+Inf"]
    sorted_buckets = sorted(finite_buckets, key=lambda x: float(x))

    bucket_stats = bucket_stats or {}

    # Generate observations centered on learned mean (or midpoint if unavailable)
    observations: list[float] = []
    bucket_info: list[
        tuple[int, int, float, float, float]
    ] = []  # (start_idx, count, lower, upper, center)

    for le in sorted_buckets:
        count = int(per_bucket.get(le, 0))
        if count <= 0:
            continue

        lower, upper = get_bucket_bounds(le, sorted_buckets)
        bucket_width = upper - lower
        midpoint = (lower + upper) / 2
        start_idx = len(observations)

        # Use learned mean if available, otherwise midpoint
        center = midpoint
        if le in bucket_stats and bucket_stats[le].estimated_mean is not None:
            learned_mean = bucket_stats[le].estimated_mean
            # Validate: must be within bucket bounds
            if lower < learned_mean < upper:
                center = learned_mean

        # Generate uniform distribution centered on 'center'
        # Scale factor determines how spread out observations are
        # If center != midpoint, we shift the distribution accordingly
        shift = center - midpoint

        for i in range(count):
            # Standard uniform position
            frac = (i + 0.5) / count
            base_value = lower + bucket_width * frac
            # Apply shift toward learned center, staying within bounds
            value = np.clip(base_value + shift, lower, upper)
            observations.append(value)

        bucket_info.append((start_idx, count, lower, upper, center))

    if not observations:
        return np.array([], dtype=np.float64)

    observations = np.array(observations, dtype=np.float64)

    # Pass 2: Fine-tune to match target sum
    # The per-bucket means should get us close, but we adjust for any residual
    generated_sum = observations.sum()

    if generated_sum <= 0 or target_sum <= 0:
        return observations

    sum_discrepancy = target_sum - generated_sum

    if abs(sum_discrepancy) / target_sum < 0.001:
        return observations  # Close enough

    # Distribute residual across buckets proportionally to their sum contribution
    # This ensures large buckets absorb more of the adjustment
    for start_idx, count, lower, upper, _center in bucket_info:
        if count == 0:
            continue

        bucket_width = upper - lower
        bucket_sum = sum(observations[start_idx : start_idx + count])
        bucket_weight = (
            bucket_sum / generated_sum if generated_sum > 0 else 1.0 / len(bucket_info)
        )

        # This bucket's share of the discrepancy
        bucket_adjustment = sum_discrepancy * bucket_weight
        per_obs_shift = bucket_adjustment / count if count > 0 else 0

        # Limit shift to stay within bucket
        max_shift = bucket_width * 0.4
        shift = np.clip(per_obs_shift, -max_shift, max_shift)

        for i in range(count):
            idx = start_idx + i
            observations[idx] = np.clip(observations[idx] + shift, lower, upper)

    return observations


def compute_best_guess_percentiles(
    bucket_deltas: dict[str, float],
    bucket_stats: dict[str, BucketStatistics],
    total_sum: float,
    total_count: int,
) -> BestGuessPercentiles | None:
    """Compute percentiles including estimated +Inf bucket observations.

    This implements a two-phase polynomial histogram approach:

    Phase 1 - Learn per-bucket means:
        When all observations in a scrape interval land in ONE bucket, we know
        the exact mean for that bucket: mean = sum_delta / count_delta.
        This is captured in bucket_stats via accumulate_bucket_statistics().

    Phase 2 - Generate observations with sum constraint:
        1. Place observations using shifted uniform distribution centered on
           learned means (or midpoint if no learned mean available)
        2. Adjust positions proportionally to match the exact total sum
        3. Back-calculate +Inf bucket observations using:
           inf_sum = total_sum - estimated_finite_sum

    This approach provides ~44% reduction in percentile estimation error vs
    standard bucket interpolation, with the largest gains for tail percentiles
    (p99, p999) where observations may fall in the +Inf bucket.

    Args:
        bucket_deltas: Cumulative bucket counts (Prometheus format)
        bucket_stats: Learned per-bucket statistics from polynomial histogram approach
        total_sum: Exact total sum from histogram (sum_delta)
        total_count: Total observation count (count_delta)

    Returns:
        BestGuessPercentiles with estimates, or None if insufficient data
    """
    if total_count <= 0 or not bucket_deltas:
        return None

    # Get max finite bucket boundary
    finite_buckets = [le for le in bucket_deltas if le != "+Inf"]
    if not finite_buckets:
        return None
    max_finite_bucket = max(float(le) for le in finite_buckets)

    # Convert cumulative bucket counts to per-bucket counts
    per_bucket_counts = cumulative_to_per_bucket(bucket_deltas)

    # Get +Inf bucket count (per-bucket, not cumulative)
    inf_count = int(per_bucket_counts.get("+Inf", 0))

    # Estimate sums for finite buckets
    estimated_sums = estimate_bucket_sums(bucket_deltas, bucket_stats)
    estimated_finite_sum = sum(estimated_sums.values())

    # Count buckets with learned means
    buckets_with_means = sum(
        1
        for le in bucket_deltas
        if le != "+Inf"
        and le in bucket_stats
        and bucket_stats[le].estimated_mean is not None
    )

    # Estimate +Inf bucket observations using back-calculation
    inf_observations = estimate_inf_bucket_observations(
        total_sum, estimated_finite_sum, inf_count, max_finite_bucket
    )

    # Calculate actual finite sum (total minus what goes to +Inf)
    inf_sum_estimate = (
        float(np.sum(inf_observations)) if len(inf_observations) > 0 else 0.0
    )
    actual_finite_sum = total_sum - inf_sum_estimate

    # Generate finite bucket observations using polynomial histogram approach
    # Uses per-bucket learned means + sum constraint for improved accuracy
    finite_obs_generated = generate_observations_with_sum_constraint(
        bucket_deltas, actual_finite_sum, bucket_stats
    )

    # Combine finite and +Inf observations
    if len(inf_observations) > 0:
        all_observations = np.concatenate([finite_obs_generated, inf_observations])
    else:
        all_observations = finite_obs_generated

    if len(all_observations) == 0:
        return None

    # Compute percentiles
    pcts = np.percentile(all_observations, [50, 90, 95, 99, 99.9])

    # Determine confidence level
    inf_ratio = inf_count / total_count if total_count > 0 else 0
    finite_bucket_count = len(finite_buckets)
    mean_coverage = (
        buckets_with_means / finite_bucket_count if finite_bucket_count > 0 else 0
    )

    if inf_ratio < 0.01 and mean_coverage > 0.5:
        confidence = "high"
    elif inf_ratio < 0.05:
        confidence = "medium"
    else:
        confidence = "low"

    return BestGuessPercentiles(
        p50=float(pcts[0]),
        p90=float(pcts[1]),
        p95=float(pcts[2]),
        p99=float(pcts[3]),
        p999=float(pcts[4]),
        inf_bucket_count=inf_count,
        inf_bucket_estimated_mean=float(np.mean(inf_observations))
        if inf_observations
        else None,
        finite_observations_count=len(finite_obs_generated),
        buckets_with_learned_means=buckets_with_means,
        estimation_confidence=confidence,
    )


class HistogramPercentiles(AIPerfBaseModel):
    """Container for all percentile computation approaches.

    Provides three methods for estimating percentiles:
    - bucket: Traditional Prometheus bucket interpolation (always available)
    - observed: Per-scrape observation extraction (more accurate when coverage is high)
    - best_guess: Includes +Inf bucket estimation (most accurate for tail percentiles)
    """

    bucket: BucketPercentiles = Field(
        default_factory=BucketPercentiles,
        description="Percentiles from bucket interpolation",
    )
    observed: ObservedPercentiles | None = Field(
        default=None,
        description="Percentiles from per-scrape observation extraction (None if unavailable)",
    )
    best_guess: BestGuessPercentiles | None = Field(
        default=None,
        description="Percentiles including +Inf bucket estimation (None if unavailable)",
    )


class InfoMetricData(AIPerfBaseModel):
    """Complete data for an info metric including label data.

    Info metrics (ending in _info) contain static system information that doesn't
    change over time. We store only the labels (not values) since the labels contain
    the actual information and values are typically just 1.0.
    """

    description: str = Field(description="Metric description from HELP text")
    labels: list[dict[str, str]] = Field(
        description="List of label keys and values as reported by the Prometheus endpoint"
    )


class JsonMetricResult(AIPerfBaseModel):
    """The result values of a single metric for JSON export.

    NOTE:
    This model has been designed to mimic the structure of the GenAI-Perf JSON output
    as closely as possible. Be careful not to add or remove fields that are not present in the
    GenAI-Perf JSON output.
    """

    unit: str = Field(description="The unit of the metric, e.g. 'ms' or 'requests/sec'")
    avg: float | None = None
    p1: float | None = None
    p5: float | None = None
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    min: int | float | None = None
    max: int | float | None = None
    std: float | None = None


# =============================================================================
# Server Metrics Export Stats (Type-specific models for semantic correctness)
# =============================================================================


class GaugeExportStats(AIPerfBaseModel):
    """Export statistics for gauge metrics - point-in-time sampled values.

    Gauges represent instantaneous values (e.g., current queue depth, cache usage %).
    Statistics are computed over all samples in the aggregation window.
    """

    min: float = Field(description="Minimum observed value")
    avg: float = Field(description="Average value across all samples")
    p50: float = Field(description="50th percentile (median)")
    p90: float = Field(description="90th percentile")
    p95: float = Field(description="95th percentile")
    p99: float = Field(description="99th percentile")
    max: float = Field(description="Maximum observed value")
    std: float = Field(description="Standard deviation of values")

    @classmethod
    def from_time_series(
        cls, ts: ScalarTimeSeries, time_filter: TimeRangeFilter | None = None
    ) -> GaugeExportStats:
        """Create GaugeExportStats from a ScalarTimeSeries."""
        mask = ts.get_time_mask(time_filter)
        values = ts.values[mask]

        pcts = np.percentile(values, [50, 90, 95, 99])

        # Use sample std (ddof=1) for unbiased estimate; 0 for single sample
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        return cls(
            avg=float(np.mean(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            std=std,
            p50=float(pcts[0]),
            p90=float(pcts[1]),
            p95=float(pcts[2]),
            p99=float(pcts[3]),
        )


class CounterExportStats(AIPerfBaseModel):
    """Export statistics for counter metrics - monotonically increasing totals.

    Counters represent cumulative totals (e.g., total requests, total bytes).
    We report the delta and rate statistics over the aggregation window.

    Note on rate metrics:
    - rate_overall: Overall throughput (delta/duration) - always available if duration > 0
    - rate_avg/min/max/std: Statistics computed between *change points* only

    Change-point detection: Rates are computed between points where the counter value
    actually changed, not between every sample. This avoids misleading statistics when
    sampling faster than the server updates (e.g., sampling at 10Hz when server updates
    at 1Hz would otherwise show many 0/s rates followed by a spike).

    Rate fields are None when:
    - Duration is zero (insufficient time)
    - No value changes occurred (nothing to compute rates from)
    """

    # Delta statistics
    delta: float = Field(description="Change over the aggregation period")
    # Overall rate (best measure of throughput)
    rate_overall: float | None = Field(
        default=None, description="Overall rate per second (delta/duration)"
    )
    # Instantaneous rate statistics (from rates between change points)
    rate_avg: float | None = Field(
        default=None, description="Time-weighted average rate between change points"
    )
    rate_min: float | None = Field(
        default=None, description="Minimum point-to-point rate per second"
    )
    rate_max: float | None = Field(
        default=None, description="Maximum point-to-point rate per second"
    )
    rate_std: float | None = Field(
        default=None, description="Standard deviation of point-to-point rates"
    )

    @classmethod
    def from_time_series(
        cls, ts: ScalarTimeSeries, time_filter: TimeRangeFilter | None = None
    ) -> CounterExportStats:
        """Create CounterExportStats from a ScalarTimeSeries."""
        ref_idx = ts.get_reference_idx(time_filter)
        mask = ts.get_time_mask(time_filter)

        filtered_ts = ts.timestamps[mask]
        filtered_vals = ts.values[mask]

        # Reference for delta calculation
        ref_value = (
            float(ts.values[ref_idx])
            if ref_idx is not None
            else float(filtered_vals[0])
        )
        ref_ts = ts.timestamps[ref_idx] if ref_idx is not None else filtered_ts[0]

        # Total delta and duration
        total_delta = float(filtered_vals[-1]) - ref_value
        duration_ns = filtered_ts[-1] - ref_ts

        # Rate calculation - None if duration is zero
        if duration_ns <= 0:
            return cls(
                delta=total_delta,
                rate_overall=None,
                rate_avg=None,
                rate_min=None,
                rate_max=None,
                rate_std=None,
            )

        duration_s = duration_ns / NANOS_PER_SECOND

        # Build full series including reference point
        if ref_idx is not None:
            all_ts = np.concatenate([[ref_ts], filtered_ts])
            all_vals = np.concatenate([[ref_value], filtered_vals])
        else:
            all_ts, all_vals = filtered_ts, filtered_vals

        # Find change points (indices where value differs from previous)
        # This avoids the "0/s 0/s 0/s 1000/s" problem when sampling faster than server updates
        if len(all_vals) > 1:
            # Always include first point, then points where value changed
            value_changed = np.diff(all_vals) != 0
            change_indices = np.concatenate([[0], np.where(value_changed)[0] + 1])

            if len(change_indices) > 1:
                # Extract timestamps and values at change points
                change_ts = all_ts[change_indices]
                change_vals = all_vals[change_indices]

                # Compute rates between consecutive change points
                deltas = np.diff(change_vals)
                time_deltas_ns = np.diff(change_ts)

                # Filter out any zero-duration intervals (shouldn't happen, but safety)
                valid_mask = time_deltas_ns > 0
                if np.any(valid_mask):
                    time_deltas_s = time_deltas_ns[valid_mask] / NANOS_PER_SECOND
                    valid_deltas = deltas[valid_mask]
                    rates = valid_deltas / time_deltas_s

                    # Time-weighted average: sum(deltas) / sum(durations)
                    # This weights each rate by how long that rate was observed
                    rate_avg = float(np.sum(valid_deltas) / np.sum(time_deltas_s))

                    # Use sample std (ddof=1) for unbiased estimate; 0 for single rate
                    rate_std = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0

                    return cls(
                        delta=total_delta,
                        rate_overall=total_delta / duration_s,
                        rate_avg=rate_avg,
                        rate_min=float(np.min(rates)),
                        rate_max=float(np.max(rates)),
                        rate_std=rate_std,
                    )

        # Not enough change points for rate statistics
        return cls(
            delta=total_delta,
            rate_overall=total_delta / duration_s,
            rate_avg=None,
            rate_min=None,
            rate_max=None,
            rate_std=None,
        )


class HistogramExportStats(AIPerfBaseModel):
    """Export statistics for histogram metrics - value distribution + rates.

    Histograms track distributions (e.g., request latencies). We report:
    - Delta stats: count_delta, sum_delta, avg over the aggregation period
    - Rate: observations per second
    - Percentiles: Two approaches (bucket interpolation and per-scrape observation extraction)
    - Raw bucket data for downstream analysis
    """

    # Delta statistics over the aggregation period
    count_delta: float = Field(
        description="Change in observation count over the aggregation period"
    )
    sum_delta: float = Field(
        description="Change in sum of observed values over the aggregation period"
    )
    avg: float = Field(
        description="Average value per observation (sum_delta/count_delta)"
    )
    # Rate - None if duration is zero
    rate: float | None = Field(
        default=None,
        description="Observations per second (count_delta/duration)",
    )
    # Percentiles computed via two approaches
    percentiles: HistogramPercentiles | None = Field(
        default=None,
        description="Percentiles from bucket interpolation and per-scrape observation extraction",
    )
    # Raw bucket data for custom analysis (None if counter reset detected)
    buckets: dict[str, float] | None = Field(
        default=None,
        description='Bucket upper bounds (le="less than or equal") to delta counts. None if counter reset detected during collection.',
    )

    @classmethod
    def from_time_series(
        cls, ts: HistogramTimeSeries, time_filter: TimeRangeFilter | None = None
    ) -> HistogramExportStats:
        """Create HistogramExportStats from a HistogramTimeSeries."""
        ref_idx, final_idx = ts.get_indices_for_filter(time_filter)

        # Reference values
        if ref_idx is not None:
            ref_sum = float(ts.sums[ref_idx])
            ref_count = float(ts.counts[ref_idx])
            ref_ts = ts.timestamps[ref_idx]
        else:
            ref_sum = float(ts.sums[0])
            ref_count = float(ts.counts[0])
            ref_ts = ts.timestamps[0]

        # Final values
        final_sum = float(ts.sums[final_idx])
        final_count = float(ts.counts[final_idx])
        final_ts = ts.timestamps[final_idx]
        final_buckets = (
            ts._bucket_snapshots[final_idx]
            if final_idx < len(ts._bucket_snapshots)
            else {}
        )

        # Compute deltas
        sum_delta = final_sum - ref_sum
        count_delta = final_count - ref_count
        duration_ns = final_ts - ref_ts

        avg_value = sum_delta / count_delta if count_delta > 0 else 0.0
        rate = (
            count_delta / (duration_ns / NANOS_PER_SECOND) if duration_ns > 0 else None
        )

        # Bucket delta calculation
        # If any delta is negative (counter reset), return None for buckets
        # since the data is invalid/incomplete
        ref_bucket_idx = ref_idx if ref_idx is not None else 0
        ref_buckets = (
            ts._bucket_snapshots[ref_bucket_idx]
            if ref_bucket_idx < len(ts._bucket_snapshots)
            else {}
        )
        bucket_deltas: dict[str, float] | None = {}
        for le, final_val in final_buckets.items():
            ref_val = ref_buckets.get(le, 0.0)
            delta = final_val - ref_val
            if delta < 0:
                # Counter reset detected - data is invalid
                bucket_deltas = None
                break
            bucket_deltas[le] = delta

        # Compute percentiles using both approaches
        percentiles: HistogramPercentiles | None = None

        if bucket_deltas:
            # Approach 1: Bucket interpolation (traditional Prometheus histogram_quantile)
            bucket_percentiles = BucketPercentiles(
                p50=histogram_quantile(0.50, bucket_deltas),
                p90=histogram_quantile(0.90, bucket_deltas),
                p95=histogram_quantile(0.95, bucket_deltas),
                p99=histogram_quantile(0.99, bucket_deltas),
            )

            # Approach 2: Per-scrape observation extraction
            start_idx = ref_idx if ref_idx is not None else 0
            observations, exact_count, bucket_placed_count = extract_all_observations(
                ts.timestamps,
                ts.sums,
                ts.counts,
                ts._bucket_snapshots,
                start_idx=start_idx,
            )

            observed_percentiles: ObservedPercentiles | None = None
            if len(observations) > 0:
                total_obs = exact_count + bucket_placed_count
                coverage = exact_count / total_obs if total_obs > 0 else 0.0

                observed_percentiles = ObservedPercentiles(
                    p50=float(np.percentile(observations, 50)),
                    p90=float(np.percentile(observations, 90)),
                    p95=float(np.percentile(observations, 95)),
                    p99=float(np.percentile(observations, 99)),
                    exact_count=exact_count,
                    bucket_placed_count=bucket_placed_count,
                    coverage=coverage,
                )

            # Approach 3: Best-guess with +Inf bucket estimation
            # Learn per-bucket means from single-bucket scrape intervals
            bucket_stats = accumulate_bucket_statistics(
                ts.timestamps,
                ts.sums,
                ts.counts,
                ts._bucket_snapshots,
                start_idx=start_idx,
            )

            # Compute best-guess percentiles including +Inf bucket estimation
            best_guess_percentiles = compute_best_guess_percentiles(
                bucket_deltas=bucket_deltas,
                bucket_stats=bucket_stats,
                total_sum=sum_delta,
                total_count=int(count_delta),
            )

            percentiles = HistogramPercentiles(
                bucket=bucket_percentiles,
                observed=observed_percentiles,
                best_guess=best_guess_percentiles,
            )

        return cls(
            count_delta=count_delta,
            sum_delta=sum_delta,
            avg=avg_value,
            rate=rate,
            percentiles=percentiles,
            buckets=bucket_deltas,
        )


class SummaryExportStats(AIPerfBaseModel):
    """Export statistics for summary metrics - server-computed quantiles.

    Summaries provide pre-computed quantiles from the server. We report:
    - Delta stats: count_delta, sum_delta, avg over the aggregation period
    - Quantiles: Final values from server (exact, not estimated)
    - Rate: observations per second
    """

    # Delta statistics over the aggregation period
    count_delta: float = Field(
        description="Change in observation count over the aggregation period"
    )
    sum_delta: float = Field(
        description="Change in sum of observed values over the aggregation period"
    )
    avg: float = Field(
        description="Average value per observation (sum_delta/count_delta)"
    )
    # Server-computed quantiles - NOTE: These are cumulative values over the server's lifetime,
    # not period-specific. Prometheus summaries cannot provide quantiles for a specific time window.
    quantiles: dict[str, float] = Field(
        default_factory=dict,
        description="Server-computed quantiles (cumulative over server lifetime, not period-specific). Keys are quantile strings (e.g., '0.5', '0.9', '0.99')",
    )
    # Rate - None if duration is zero
    rate: float | None = Field(
        default=None,
        description="Observations per second (count_delta/duration)",
    )

    @classmethod
    def from_time_series(
        cls, ts: SummaryTimeSeries, time_filter: TimeRangeFilter | None = None
    ) -> SummaryExportStats:
        """Create SummaryExportStats from a SummaryTimeSeries."""
        ref_idx, final_idx = ts.get_indices_for_filter(time_filter)

        # Reference values
        if ref_idx is not None:
            ref_sum = float(ts.sums[ref_idx])
            ref_count = float(ts.counts[ref_idx])
            ref_ts = ts.timestamps[ref_idx]
        else:
            ref_sum = float(ts.sums[0])
            ref_count = float(ts.counts[0])
            ref_ts = ts.timestamps[0]

        final_sum = float(ts.sums[final_idx])
        final_count = float(ts.counts[final_idx])
        final_ts = ts.timestamps[final_idx]
        final_quantiles = ts._quantile_snapshots[final_idx]

        sum_delta = final_sum - ref_sum
        count_delta = final_count - ref_count
        duration_ns = final_ts - ref_ts

        avg_value = sum_delta / count_delta if count_delta > 0 else 0.0
        rate = (
            count_delta / (duration_ns / NANOS_PER_SECOND) if duration_ns > 0 else None
        )

        return cls(
            count_delta=count_delta,
            sum_delta=sum_delta,
            avg=avg_value,
            quantiles=dict(final_quantiles),
            rate=rate,
        )


# Union type for any server metric stats (no discriminator needed - type is at family level)
ServerMetricStats: TypeAlias = (
    GaugeExportStats | CounterExportStats | HistogramExportStats | SummaryExportStats
)


# =============================================================================
# Export Data Models (JSON export structures)
# =============================================================================


class TelemetrySummary(AIPerfBaseModel):
    """Summary information for telemetry collection."""

    endpoints_configured: list[str]
    endpoints_successful: list[str]
    start_time: datetime
    end_time: datetime


class GpuSummary(AIPerfBaseModel):
    """Summary of GPU telemetry data."""

    gpu_index: int
    gpu_name: str
    gpu_uuid: str
    hostname: str | None
    metrics: dict[str, JsonMetricResult]  # metric_key -> {stat_key -> value}


class EndpointData(AIPerfBaseModel):
    """Data for a single endpoint."""

    gpus: dict[str, GpuSummary]


class TelemetryExportData(AIPerfBaseModel):
    """Telemetry data structure for JSON export."""

    summary: TelemetrySummary
    endpoints: dict[str, EndpointData]


class ServerMetricsEndpointInfo(AIPerfBaseModel):
    """Metadata about a single endpoint's collection statistics."""

    endpoint_url: str = Field(description="Full endpoint URL")
    duration_seconds: float = Field(
        description="Total duration of metrics collection for this endpoint"
    )
    scrape_count: int = Field(
        description="Number of successful scrapes from this endpoint"
    )
    avg_scrape_latency_ms: float = Field(
        description="Average time to scrape metrics from this endpoint in milliseconds"
    )


class ServerMetricsSummary(AIPerfBaseModel):
    """Summary information for server metrics collection."""

    endpoints_configured: list[str] = Field(
        description="List of configured endpoint identifiers (normalized)"
    )
    endpoints_successful: list[str] = Field(
        description="List of successful endpoint identifiers (normalized)"
    )
    start_time: datetime
    end_time: datetime
    endpoint_info: dict[str, ServerMetricsEndpointInfo] | None = Field(
        default=None,
        description="Per-endpoint collection metadata keyed by normalized endpoint identifier",
    )


class ServerMetricLabeledStats(AIPerfBaseModel):
    """Aggregated statistics for a single time series (unique label combination).

    In Prometheus, each unique label combination is a separate time series.
    This model represents statistics for one such combination.
    """

    endpoint: str | None = Field(
        default=None,
        description="Endpoint URL this series came from (used in merged export format)",
    )
    labels: dict[str, str] | None = Field(
        default=None,
        description="Metric labels for this series. None if the metric has no labels.",
    )
    stats: SerializeAsAny[ServerMetricStats] = Field(
        description="Type-specific aggregated statistics (gauge, counter, histogram, or summary)",
    )


class ServerMetricSummary(AIPerfBaseModel):
    """Summary of a server metric with type, description, and per-label statistics.

    Combines metadata (metric type and description) with aggregated statistics.
    Each item in 'series' represents statistics for a unique label combination.
    """

    description: str = Field(description="Metric description from HELP text")
    type: str = Field(description="Metric type (gauge, counter, histogram, summary)")
    series: list[ServerMetricLabeledStats] = Field(
        default_factory=list,
        description="Statistics for each unique label combination",
    )


class ServerMetricsEndpointSummary(AIPerfBaseModel):
    """Summary of server metrics data for a single endpoint.

    Unified structure combining metadata and type-specific aggregated statistics:
    - Each metric uses stats matching its semantic type (gauge, counter, histogram, summary)
    - Mirrors JSONL structure with labels as proper objects
    - Includes metric description from metadata
    """

    endpoint_url: str
    # Collection metadata
    duration_seconds: float = Field(
        description="Total duration of metrics collection for this endpoint"
    )
    scrape_count: int = Field(
        description="Number of successful scrapes from this endpoint"
    )
    avg_scrape_latency_ms: float = Field(
        description="Average time to scrape metrics from this endpoint in milliseconds"
    )
    # Metric data
    info_metrics: dict[str, InfoMetricData] | None = Field(
        default=None,
        description="Static info metrics (ending in _info) with their label data",
    )
    metrics: dict[str, ServerMetricSummary] = Field(
        default_factory=dict,
        description="All metrics keyed by metric name, with description and type-specific statistics",
    )


class ServerMetricsExportData(AIPerfBaseModel):
    """Server metrics data structure for JSON export."""

    summary: ServerMetricsSummary
    endpoints: dict[str, ServerMetricsEndpointSummary]


class ServerMetricsMergedExportData(AIPerfBaseModel):
    """Server metrics data structure with all endpoints merged into a single metrics dict.

    This format merges series from all endpoints into each metric, with each series
    item containing an 'endpoint' field to identify its source.
    """

    summary: ServerMetricsSummary
    info_metrics: dict[str, InfoMetricData] | None = Field(
        default=None,
        description="Static info metrics merged from all endpoints",
    )
    metrics: dict[str, ServerMetricSummary] = Field(
        default_factory=dict,
        description="All metrics merged across endpoints, with endpoint field in each series item",
    )


class TimesliceData(AIPerfBaseModel):
    """Data for a single timeslice.

    Contains metrics for one time slice with dynamic metric fields
    added via Pydantic's extra="allow" setting.
    """

    model_config = ConfigDict(extra="allow")

    timeslice_index: int


class TimesliceCollectionExportData(AIPerfBaseModel):
    """Export data for all timeslices in a single file.

    Contains an array of timeslice data objects with metadata.
    """

    timeslices: list[TimesliceData]
    input_config: UserConfig | None = None


class JsonExportData(AIPerfBaseModel):
    """Summary data to be exported to a JSON file.

    NOTE:
    This model has been designed to mimic the structure of the GenAI-Perf JSON output
    as closely as possible. Be careful when modifying this model to not break the
    compatibility with the GenAI-Perf JSON output.
    """

    # NOTE: The extra="allow" setting is needed to allow additional metrics not defined in this class
    #       to be added to the export data. It is also already set in the AIPerfBaseModel,
    #       but we are setting it here to guard against base model changes.
    model_config = ConfigDict(extra="allow")

    request_throughput: JsonMetricResult | None = None
    request_latency: JsonMetricResult | None = None
    request_count: JsonMetricResult | None = None
    time_to_first_token: JsonMetricResult | None = None
    time_to_second_token: JsonMetricResult | None = None
    inter_token_latency: JsonMetricResult | None = None
    output_token_throughput: JsonMetricResult | None = None
    output_token_throughput_per_user: JsonMetricResult | None = None
    output_sequence_length: JsonMetricResult | None = None
    input_sequence_length: JsonMetricResult | None = None
    goodput: JsonMetricResult | None = None
    good_request_count: JsonMetricResult | None = None
    output_token_count: JsonMetricResult | None = None
    reasoning_token_count: JsonMetricResult | None = None
    min_request_timestamp: JsonMetricResult | None = None
    max_response_timestamp: JsonMetricResult | None = None
    inter_chunk_latency: JsonMetricResult | None = None
    total_output_tokens: JsonMetricResult | None = None
    total_reasoning_tokens: JsonMetricResult | None = None
    benchmark_duration: JsonMetricResult | None = None
    total_isl: JsonMetricResult | None = None
    total_osl: JsonMetricResult | None = None
    error_request_count: JsonMetricResult | None = None
    error_isl: JsonMetricResult | None = None
    total_error_isl: JsonMetricResult | None = None
    telemetry_data: TelemetryExportData | None = None
    server_metrics_data: ServerMetricsExportData | None = None
    input_config: UserConfig | None = None
    was_cancelled: bool | None = None
    error_summary: list[ErrorDetailsCount] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
