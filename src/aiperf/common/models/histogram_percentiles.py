# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Histogram percentile models and computation functions.

This module provides percentile estimation for Prometheus histograms using
multiple approaches:

1. BucketPercentiles: Traditional Prometheus bucket interpolation
2. ObservedPercentiles: Per-scrape observation extraction
3. BestGuessPercentiles: Polynomial histogram with +Inf bucket estimation

The "best guess" approach implements a polynomial histogram algorithm that:
- Learns per-bucket mean positions from single-bucket scrape intervals
- Uses exact sum constraint to improve observation placement
- Back-calculates +Inf bucket observations for accurate tail percentiles
"""

from __future__ import annotations

import numpy as np
from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.histogram_analysis import (
    BucketStatistics,
    cumulative_to_per_bucket,
    estimate_bucket_sums,
    estimate_inf_bucket_observations,
    get_bucket_bounds,
)

# =============================================================================
# Percentile Models
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


# =============================================================================
# Observation Generation with Sum Constraint
# =============================================================================


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


# =============================================================================
# Best Guess Percentile Computation
# =============================================================================


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
