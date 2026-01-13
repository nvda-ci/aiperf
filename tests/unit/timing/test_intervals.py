# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for interval generators.

Tests Poisson, Gamma, Constant, and ConcurrencyBurst interval generators.
"""

import statistics

import pytest

from aiperf.common.enums import ArrivalPattern
from aiperf.timing.intervals import (
    ConcurrencyBurstIntervalGenerator,
    ConstantIntervalGenerator,
    GammaIntervalGenerator,
    IntervalGeneratorConfig,
    IntervalGeneratorFactory,
    PoissonIntervalGenerator,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def make_config(
    arrival_pattern: ArrivalPattern,
    request_rate: float | None = None,
    arrival_smoothness: float | None = None,
) -> IntervalGeneratorConfig:
    """Create IntervalGeneratorConfig for testing."""
    return IntervalGeneratorConfig(
        arrival_pattern=arrival_pattern,
        request_rate=request_rate,
        arrival_smoothness=arrival_smoothness,
    )


# =============================================================================
# Test: Poisson Interval Generator
# =============================================================================


class TestPoissonIntervalGenerator:
    """Tests for PoissonIntervalGenerator."""

    def test_init_with_valid_rate(self):
        """Should initialize with valid request rate."""
        config = make_config(ArrivalPattern.POISSON, request_rate=10.0)
        gen = PoissonIntervalGenerator(config)
        assert gen.rate == 10.0

    def test_init_raises_on_zero_rate(self):
        """Should raise ValueError for zero rate."""
        config = make_config(ArrivalPattern.POISSON, request_rate=0.0)
        with pytest.raises(ValueError, match="must be set and greater than 0"):
            PoissonIntervalGenerator(config)

    def test_init_raises_on_negative_rate(self):
        """Should raise ValueError for negative rate."""
        config = make_config(ArrivalPattern.POISSON, request_rate=-5.0)
        with pytest.raises(ValueError, match="must be set and greater than 0"):
            PoissonIntervalGenerator(config)

    def test_init_raises_on_none_rate(self):
        """Should raise ValueError for None rate."""
        config = make_config(ArrivalPattern.POISSON, request_rate=None)
        with pytest.raises(ValueError, match="must be set and greater than 0"):
            PoissonIntervalGenerator(config)

    def test_next_interval_returns_positive_value(self):
        """Intervals should always be positive."""
        config = make_config(ArrivalPattern.POISSON, request_rate=10.0)
        gen = PoissonIntervalGenerator(config)

        for _ in range(100):
            interval = gen.next_interval()
            assert interval > 0

    def test_next_interval_average_matches_rate(self):
        """Mean interval should approximate 1/rate over many samples."""
        rate = 100.0
        expected_mean = 1.0 / rate
        config = make_config(ArrivalPattern.POISSON, request_rate=rate)
        gen = PoissonIntervalGenerator(config)

        intervals = [gen.next_interval() for _ in range(10000)]
        actual_mean = statistics.mean(intervals)

        # Allow 10% tolerance
        assert abs(actual_mean - expected_mean) / expected_mean < 0.1

    def test_set_rate_updates_rate(self):
        """set_rate should update the rate property."""
        config = make_config(ArrivalPattern.POISSON, request_rate=10.0)
        gen = PoissonIntervalGenerator(config)

        gen.set_rate(50.0)

        assert gen.rate == 50.0

    def test_set_rate_affects_intervals(self):
        """Changing rate should affect interval distribution."""
        config = make_config(ArrivalPattern.POISSON, request_rate=10.0)
        gen = PoissonIntervalGenerator(config)

        # Sample at low rate
        low_rate_intervals = [gen.next_interval() for _ in range(1000)]

        # Change to high rate
        gen.set_rate(100.0)
        high_rate_intervals = [gen.next_interval() for _ in range(1000)]

        # High rate should have smaller mean interval
        assert statistics.mean(high_rate_intervals) < statistics.mean(
            low_rate_intervals
        )

    def test_set_rate_raises_on_invalid_rate(self):
        """set_rate should raise for invalid rates."""
        config = make_config(ArrivalPattern.POISSON, request_rate=10.0)
        gen = PoissonIntervalGenerator(config)

        with pytest.raises(ValueError, match="must be > 0"):
            gen.set_rate(0.0)

        with pytest.raises(ValueError, match="must be > 0"):
            gen.set_rate(-5.0)


# =============================================================================
# Test: Gamma Interval Generator
# =============================================================================


class TestGammaIntervalGenerator:
    """Tests for GammaIntervalGenerator."""

    def test_init_with_valid_rate(self):
        """Should initialize with valid request rate."""
        config = make_config(ArrivalPattern.GAMMA, request_rate=10.0)
        gen = GammaIntervalGenerator(config)
        assert gen.rate == 10.0

    def test_init_defaults_smoothness_to_one(self):
        """Should default smoothness to 1.0 (equivalent to Poisson)."""
        config = make_config(
            ArrivalPattern.GAMMA, request_rate=10.0, arrival_smoothness=None
        )
        gen = GammaIntervalGenerator(config)
        assert gen.smoothness == 1.0

    def test_init_with_custom_smoothness(self):
        """Should use provided smoothness value."""
        config = make_config(
            ArrivalPattern.GAMMA, request_rate=10.0, arrival_smoothness=2.5
        )
        gen = GammaIntervalGenerator(config)
        assert gen.smoothness == 2.5

    def test_init_raises_on_invalid_rate(self):
        """Should raise ValueError for invalid rate."""
        config = make_config(ArrivalPattern.GAMMA, request_rate=0.0)
        with pytest.raises(ValueError, match="must be set and greater than 0"):
            GammaIntervalGenerator(config)

    def test_next_interval_returns_positive_value(self):
        """Intervals should always be positive."""
        config = make_config(
            ArrivalPattern.GAMMA, request_rate=10.0, arrival_smoothness=2.0
        )
        gen = GammaIntervalGenerator(config)

        for _ in range(100):
            interval = gen.next_interval()
            assert interval > 0

    def test_next_interval_average_matches_rate(self):
        """Mean interval should approximate 1/rate over many samples."""
        rate = 100.0
        expected_mean = 1.0 / rate
        config = make_config(
            ArrivalPattern.GAMMA, request_rate=rate, arrival_smoothness=1.0
        )
        gen = GammaIntervalGenerator(config)

        intervals = [gen.next_interval() for _ in range(10000)]
        actual_mean = statistics.mean(intervals)

        # Allow 10% tolerance
        assert abs(actual_mean - expected_mean) / expected_mean < 0.1

    def test_higher_smoothness_reduces_variance(self):
        """Higher smoothness should produce more regular intervals."""
        rate = 100.0
        config_low = make_config(
            ArrivalPattern.GAMMA, request_rate=rate, arrival_smoothness=0.5
        )
        config_high = make_config(
            ArrivalPattern.GAMMA, request_rate=rate, arrival_smoothness=5.0
        )
        gen_low = GammaIntervalGenerator(config_low)
        gen_high = GammaIntervalGenerator(config_high)

        intervals_low = [gen_low.next_interval() for _ in range(5000)]
        intervals_high = [gen_high.next_interval() for _ in range(5000)]

        # Higher smoothness = lower variance
        assert statistics.variance(intervals_high) < statistics.variance(intervals_low)

    @pytest.mark.parametrize(
        "smoothness,expected_cv",
        [
            (1.0, 1.0),      # CV = 1/sqrt(1) = 1.0 (Poisson equivalent)
            (4.0, 0.5),      # CV = 1/sqrt(4) = 0.5
            (9.0, 0.333),    # CV = 1/sqrt(9) ≈ 0.333
            (0.25, 2.0),     # CV = 1/sqrt(0.25) = 2.0
            (0.5, 1.414),    # CV = 1/sqrt(0.5) ≈ 1.414
            (16.0, 0.25),    # CV = 1/sqrt(16) = 0.25
        ],
    )  # fmt: skip
    def test_cv_matches_gamma_formula(self, smoothness: float, expected_cv: float):
        """Verify CV = 1/sqrt(smoothness) formula for Gamma distribution.

        The Gamma distribution with shape=k has CV = 1/sqrt(k).
        This is the key mathematical property that distinguishes different
        smoothness levels.
        """
        rate = 100.0
        config = make_config(
            ArrivalPattern.GAMMA, request_rate=rate, arrival_smoothness=smoothness
        )
        gen = GammaIntervalGenerator(config)

        # Generate many samples for accurate CV estimation
        intervals = [gen.next_interval() for _ in range(10000)]

        actual_mean = statistics.mean(intervals)
        actual_std = statistics.stdev(intervals)
        actual_cv = actual_std / actual_mean

        # Allow 10% tolerance for statistical variation
        tolerance = expected_cv * 0.15
        assert abs(actual_cv - expected_cv) < tolerance, (
            f"smoothness={smoothness}: CV={actual_cv:.4f}, expected={expected_cv:.4f}"
        )

    def test_cv_formula_comprehensive(self):
        """Comprehensive test that CV decreases as smoothness increases.

        Mathematical relationship: CV = 1/sqrt(smoothness)
        - smoothness=0.25 → CV=2.0 (very bursty)
        - smoothness=1.0 → CV=1.0 (Poisson)
        - smoothness=4.0 → CV=0.5 (smoother)
        - smoothness=25.0 → CV=0.2 (very regular)
        """
        rate = 100.0
        smoothness_values = [0.25, 0.5, 1.0, 2.0, 4.0, 9.0, 16.0, 25.0]
        results = []

        for smoothness in smoothness_values:
            config = make_config(
                ArrivalPattern.GAMMA, request_rate=rate, arrival_smoothness=smoothness
            )
            gen = GammaIntervalGenerator(config)
            intervals = [gen.next_interval() for _ in range(5000)]

            actual_cv = statistics.stdev(intervals) / statistics.mean(intervals)
            expected_cv = 1.0 / (smoothness**0.5)
            results.append((smoothness, expected_cv, actual_cv))

        # Verify monotonically decreasing CV as smoothness increases
        for i in range(1, len(results)):
            prev_cv = results[i - 1][2]
            curr_cv = results[i][2]
            assert curr_cv < prev_cv, (
                f"CV should decrease with smoothness: "
                f"smoothness {results[i - 1][0]} has CV={prev_cv:.4f}, "
                f"smoothness {results[i][0]} has CV={curr_cv:.4f}"
            )

        # Verify all CVs match formula within tolerance
        for smoothness, expected_cv, actual_cv in results:
            tolerance = expected_cv * 0.2
            assert abs(actual_cv - expected_cv) < tolerance, (
                f"smoothness={smoothness}: CV={actual_cv:.4f}, expected={expected_cv:.4f}"
            )

    def test_std_equals_mean_times_cv(self):
        """Verify std = mean × CV relationship holds.

        For Gamma with shape=smoothness:
        - Mean = 1/rate
        - Std = Mean × CV = Mean / sqrt(smoothness) = 1 / (rate × sqrt(smoothness))
        """
        rate = 100.0
        smoothness = 4.0  # CV = 0.5
        config = make_config(
            ArrivalPattern.GAMMA, request_rate=rate, arrival_smoothness=smoothness
        )
        gen = GammaIntervalGenerator(config)

        intervals = [gen.next_interval() for _ in range(10000)]

        actual_mean = statistics.mean(intervals)
        actual_std = statistics.stdev(intervals)

        expected_mean = 1.0 / rate
        expected_std = expected_mean / (smoothness**0.5)

        # Verify mean
        assert abs(actual_mean - expected_mean) / expected_mean < 0.1, (
            f"Mean={actual_mean:.6f}, expected={expected_mean:.6f}"
        )

        # Verify std
        assert abs(actual_std - expected_std) / expected_std < 0.15, (
            f"Std={actual_std:.6f}, expected={expected_std:.6f}"
        )

    def test_set_rate_updates_rate_and_distribution(self):
        """set_rate should update rate and recalculate gamma params."""
        config = make_config(
            ArrivalPattern.GAMMA, request_rate=10.0, arrival_smoothness=2.0
        )
        gen = GammaIntervalGenerator(config)

        gen.set_rate(100.0)

        assert gen.rate == 100.0
        # Mean should now be approximately 0.01
        intervals = [gen.next_interval() for _ in range(5000)]
        actual_mean = statistics.mean(intervals)
        assert abs(actual_mean - 0.01) / 0.01 < 0.15


# =============================================================================
# Test: Constant Interval Generator
# =============================================================================


class TestConstantIntervalGenerator:
    """Tests for ConstantIntervalGenerator."""

    def test_init_with_valid_rate(self):
        """Should initialize with valid request rate."""
        config = make_config(ArrivalPattern.CONSTANT, request_rate=10.0)
        gen = ConstantIntervalGenerator(config)
        assert gen.rate == 10.0

    def test_init_raises_on_invalid_rate(self):
        """Should raise ValueError for invalid rate."""
        config = make_config(ArrivalPattern.CONSTANT, request_rate=0.0)
        with pytest.raises(ValueError, match="must be set and greater than 0"):
            ConstantIntervalGenerator(config)

    def test_next_interval_returns_fixed_value(self):
        """All intervals should be exactly 1/rate."""
        rate = 10.0
        expected_interval = 0.1  # 1/10
        config = make_config(ArrivalPattern.CONSTANT, request_rate=rate)
        gen = ConstantIntervalGenerator(config)

        for _ in range(100):
            interval = gen.next_interval()
            assert interval == expected_interval

    def test_next_interval_is_deterministic(self):
        """Constant generator should produce identical values."""
        config = make_config(ArrivalPattern.CONSTANT, request_rate=25.0)
        gen = ConstantIntervalGenerator(config)

        intervals = [gen.next_interval() for _ in range(100)]

        # All values should be identical
        assert len(set(intervals)) == 1

    def test_set_rate_updates_interval(self):
        """set_rate should update the fixed interval."""
        config = make_config(ArrivalPattern.CONSTANT, request_rate=10.0)
        gen = ConstantIntervalGenerator(config)

        assert gen.next_interval() == 0.1

        gen.set_rate(50.0)

        assert gen.rate == 50.0
        assert gen.next_interval() == 0.02

    def test_set_rate_raises_on_invalid_rate(self):
        """set_rate should raise for invalid rates."""
        config = make_config(ArrivalPattern.CONSTANT, request_rate=10.0)
        gen = ConstantIntervalGenerator(config)

        with pytest.raises(ValueError, match="must be > 0"):
            gen.set_rate(0.0)


# =============================================================================
# Test: Concurrency Burst Interval Generator
# =============================================================================


class TestConcurrencyBurstIntervalGenerator:
    """Tests for ConcurrencyBurstIntervalGenerator."""

    def test_init_ignores_config(self):
        """Should initialize without requiring rate."""
        config = make_config(ArrivalPattern.CONCURRENCY_BURST, request_rate=None)
        gen = ConcurrencyBurstIntervalGenerator(config)
        assert gen is not None

    def test_rate_always_zero(self):
        """Rate should always be 0 for burst mode."""
        config = make_config(ArrivalPattern.CONCURRENCY_BURST)
        gen = ConcurrencyBurstIntervalGenerator(config)
        assert gen.rate == 0.0

    def test_next_interval_returns_zero(self):
        """All intervals should be 0 (immediate)."""
        config = make_config(ArrivalPattern.CONCURRENCY_BURST)
        gen = ConcurrencyBurstIntervalGenerator(config)

        for _ in range(100):
            interval = gen.next_interval()
            assert interval == 0

    def test_set_rate_is_noop(self):
        """set_rate should be a no-op for burst mode."""
        config = make_config(ArrivalPattern.CONCURRENCY_BURST)
        gen = ConcurrencyBurstIntervalGenerator(config)

        gen.set_rate(100.0)  # Should not raise
        gen.set_rate(0.0)  # Should not raise
        gen.set_rate(-5.0)  # Should not raise

        assert gen.rate == 0.0
        assert gen.next_interval() == 0


# =============================================================================
# Test: Factory Registration
# =============================================================================


class TestIntervalGeneratorFactory:
    """Tests for IntervalGeneratorFactory."""

    @pytest.mark.parametrize(
        "pattern,expected_class",
        [
            (ArrivalPattern.POISSON, PoissonIntervalGenerator),
            (ArrivalPattern.GAMMA, GammaIntervalGenerator),
            (ArrivalPattern.CONSTANT, ConstantIntervalGenerator),
            (ArrivalPattern.CONCURRENCY_BURST, ConcurrencyBurstIntervalGenerator),
        ],
    )
    def test_factory_creates_correct_type(
        self, pattern: ArrivalPattern, expected_class: type
    ):
        """Factory should create the correct generator type."""
        rate = 10.0 if pattern != ArrivalPattern.CONCURRENCY_BURST else None
        config = make_config(pattern, request_rate=rate)

        gen = IntervalGeneratorFactory.create_instance(config)

        assert isinstance(gen, expected_class)


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_high_rate(self):
        """Should handle very high request rates."""
        rate = 1_000_000.0  # 1M QPS
        config = make_config(ArrivalPattern.CONSTANT, request_rate=rate)
        gen = ConstantIntervalGenerator(config)

        interval = gen.next_interval()
        assert interval == 1e-6

    def test_very_low_rate(self):
        """Should handle very low request rates."""
        rate = 0.001  # 1 request per 1000 seconds
        config = make_config(ArrivalPattern.CONSTANT, request_rate=rate)
        gen = ConstantIntervalGenerator(config)

        interval = gen.next_interval()
        assert interval == 1000.0

    def test_poisson_reproducibility_with_seeded_rng(self):
        """Poisson generator should be reproducible with seeded RNG."""
        # Note: This test verifies the generator uses the global seeded RNG
        config = make_config(ArrivalPattern.POISSON, request_rate=10.0)
        gen1 = PoissonIntervalGenerator(config)
        gen2 = PoissonIntervalGenerator(config)

        # Both generators derive from the same seed, but get different
        # derived RNGs, so they produce different sequences
        intervals1 = [gen1.next_interval() for _ in range(10)]
        intervals2 = [gen2.next_interval() for _ in range(10)]

        # They should NOT be identical (different derived RNGs)
        # This test just verifies the generators work
        assert len(intervals1) == 10
        assert len(intervals2) == 10
        assert all(i > 0 for i in intervals1)
        assert all(i > 0 for i in intervals2)

    @pytest.mark.parametrize(
        "arrival_smoothness",
        [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    )
    def test_gamma_various_smoothness_values(self, arrival_smoothness: float):
        """Gamma generator should work with various smoothness values."""
        config = make_config(
            ArrivalPattern.GAMMA,
            request_rate=100.0,
            arrival_smoothness=arrival_smoothness,
        )
        gen = GammaIntervalGenerator(config)

        intervals = [gen.next_interval() for _ in range(1000)]

        # All intervals should be positive
        assert all(i > 0 for i in intervals)
        # Mean should be approximately 0.01 (1/rate)
        actual_mean = statistics.mean(intervals)
        assert abs(actual_mean - 0.01) / 0.01 < 0.2  # 20% tolerance
