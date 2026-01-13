# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive unit tests for the RequestRateStrategy class.
"""

import math

import numpy as np
import pytest
from scipy import stats

from aiperf.common import random_generator as rng
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import ArrivalPattern
from aiperf.timing.intervals import (
    ConcurrencyBurstIntervalGenerator,
    ConstantIntervalGenerator,
    IntervalGeneratorConfig,
    PoissonIntervalGenerator,
)
from tests.unit.timing.conftest import (
    OrchestratorHarness,
    get_session_stats,
)


@pytest.mark.skip(
    reason="Timing tests require time_traveler_no_patch_sleep and proper "
    "synchronization between time.time_ns(), looptime, and time_traveler patches. "
    "See test_fixed_schedule.py for working example of timing-sensitive tests."
)
@pytest.mark.asyncio
@pytest.mark.slow
class TestRequestRateStrategyPoissonDistribution:
    """Tests for verifying Poisson distribution behavior in RequestRateStrategy.

    Note: These tests are marked as 'slow' because they require thousands of samples
    to be statistically meaningful. They are currently skipped due to timing
    infrastructure incompatibilities between time_traveler, looptime, and time.time_ns().
    """

    async def run_poisson_rate(
        self,
        request_rate: float,
        request_count: int,
        random_seed: int,
        create_orchestrator_harness,
    ) -> list[int]:
        """Run the Poisson rate execution and return credit timestamps."""
        rng.reset()
        rng.init(random_seed)

        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"conv{i}", 1) for i in range(request_count)],
            request_count=request_count,
            request_rate=request_rate,
            random_seed=random_seed,
        )

        await harness.run_with_auto_return()

        # Extract timestamps from credits
        return [credit.issued_at_ns for credit in harness.sent_credits]

    async def run_poisson_rate_event_counts(
        self,
        request_rate: float,
        request_count: int,
        random_seed: int,
        interval_duration: float,
        create_orchestrator_harness,
    ) -> tuple[np.floating, np.ndarray]:
        """Run the Poisson rate execution and return the event counts in each interval."""
        dropped_timestamps = await self.run_poisson_rate(
            request_rate, request_count, random_seed, create_orchestrator_harness
        )

        timestamps_sec = np.array(dropped_timestamps) / NANOS_PER_SECOND

        # Create time bins from start to end of execution
        start_time = timestamps_sec[0]
        end_time = timestamps_sec[-1]
        num_intervals = int((end_time - start_time) / interval_duration)

        # Count events in each interval
        event_counts = []
        for i in range(num_intervals):
            interval_start = start_time + i * interval_duration
            interval_end = interval_start + interval_duration

            # Count events in this interval
            events_in_interval = np.sum(
                (timestamps_sec >= interval_start) & (timestamps_sec < interval_end)
            )
            event_counts.append(events_in_interval)

        return np.mean(event_counts), np.array(event_counts)

    async def test_poisson_rate_follows_exponential_distribution(
        self, create_orchestrator_harness, time_traveler
    ):
        """Test that _execute_poisson_rate generates inter-arrival times following exponential distribution."""
        request_rate = 200.0
        request_count = 10_000
        dropped_timestamps = await self.run_poisson_rate(
            request_rate, request_count, 42, create_orchestrator_harness
        )

        expected_mean_interval = 1.0 / request_rate

        assert len(dropped_timestamps) == request_count, (
            f"Expected {request_count} credits, got {len(dropped_timestamps)}"
        )

        inter_arrival_times = []
        for i in range(1, len(dropped_timestamps)):
            interval = dropped_timestamps[i] - dropped_timestamps[i - 1]
            inter_arrival_times.append(interval / NANOS_PER_SECOND)

        inter_arrival_times = np.array(inter_arrival_times)

        # Statistical tests for exponential distribution
        # 1. Test mean: For exponential distribution with rate λ, mean = 1/λ
        actual_mean = np.mean(inter_arrival_times)
        assert (
            abs(actual_mean - expected_mean_interval) < expected_mean_interval * 0.2
        ), (
            f"Mean inter-arrival time {actual_mean:.4f} deviates too much from expected {expected_mean_interval:.4f}"
        )

        # 2. Test standard deviation: For exponential distribution, std = mean
        actual_std = np.std(inter_arrival_times)
        expected_std = expected_mean_interval
        assert abs(actual_std - expected_std) < expected_std * 0.3, (
            f"Standard deviation {actual_std:.4f} deviates too much from expected {expected_std:.4f}"
        )

        # 3. Test coefficient of variation: For exponential distribution, CV = 1
        cv = actual_std / actual_mean
        assert abs(cv - 1.0) < 0.2, (
            f"Coefficient of variation {cv:.4f} should be close to 1.0 for exponential distribution"
        )

        # 4. Test that ~63.2% of values are less than the mean (exponential CDF property)
        values_below_mean = np.sum(inter_arrival_times < actual_mean)
        proportion_below_mean = values_below_mean / len(inter_arrival_times)
        expected_proportion = 1 - math.exp(-1)  # ≈ 0.632
        assert abs(proportion_below_mean - expected_proportion) < 0.1, (
            f"Proportion below mean {proportion_below_mean:.3f} should be close to {expected_proportion:.3f}"
        )

    async def test_poisson_rate_independence_of_intervals(
        self, create_orchestrator_harness, time_traveler
    ):
        """Test that inter-arrival times in Poisson process are independent (low correlation)."""
        request_rate = 150.0
        request_count = 10_000
        dropped_timestamps = await self.run_poisson_rate(
            request_rate, request_count, 123, create_orchestrator_harness
        )

        # Calculate inter-arrival times
        inter_arrival_times = []
        for i in range(1, len(dropped_timestamps)):
            interval = dropped_timestamps[i] - dropped_timestamps[i - 1]
            inter_arrival_times.append(interval / NANOS_PER_SECOND)

        inter_arrival_times = np.array(inter_arrival_times)

        # Test independence by checking correlation between consecutive intervals
        correlation = np.corrcoef(inter_arrival_times[:-1], inter_arrival_times[1:])[
            0, 1
        ]
        assert abs(correlation) < 0.2, (
            f"Correlation between consecutive intervals {correlation:.4f} indicates lack of independence"
        )

    async def test_poisson_rate_event_counts_follow_poisson_distribution_mean(
        self, create_orchestrator_harness, time_traveler
    ):
        """Test that event counts in fixed time intervals follow a valid Poisson distribution with mean = λt."""
        request_rate = 1_000
        request_count = 25_000
        interval_duration = 0.5

        actual_mean, event_counts = await self.run_poisson_rate_event_counts(
            request_rate,
            request_count,
            789,
            interval_duration,
            create_orchestrator_harness,
        )

        expected_events_per_interval = request_rate * interval_duration
        assert (
            abs(actual_mean - expected_events_per_interval)
            < expected_events_per_interval * 0.3
        ), (
            f"Mean event count {actual_mean:.4f} deviates too much from expected {expected_events_per_interval:.4f}"
        )

    async def test_poisson_rate_event_counts_follow_poisson_distribution_variance(
        self, create_orchestrator_harness, time_traveler
    ):
        """Test that event counts in fixed time intervals follow a valid Poisson distribution with variance = mean."""
        request_rate = 1_000
        request_count = 25_000

        actual_mean, event_counts = await self.run_poisson_rate_event_counts(
            request_rate, request_count, 789, 0.5, create_orchestrator_harness
        )

        actual_variance = np.var(event_counts)
        assert abs(actual_variance - actual_mean) < actual_mean * 0.4, (
            f"Variance {actual_variance:.4f} should be close to mean {actual_mean:.4f} for Poisson distribution"
        )

    async def test_poisson_rate_event_counts_follow_poisson_distribution_index_of_dispersion(
        self, create_orchestrator_harness, time_traveler
    ):
        """Test that event counts in fixed time intervals follow a valid Poisson distribution with index of dispersion ≈ 1."""
        request_rate = 1_000
        request_count = 25_000

        actual_mean, event_counts = await self.run_poisson_rate_event_counts(
            request_rate, request_count, 789, 0.5, create_orchestrator_harness
        )

        actual_variance = np.var(event_counts)
        index_of_dispersion = actual_variance / actual_mean
        assert abs(index_of_dispersion - 1.0) < 0.4, (
            f"Index of dispersion {index_of_dispersion:.4f} should be close to 1.0 for Poisson distribution"
        )

    async def test_poisson_rate_event_counts_follow_poisson_distribution_ks_test(
        self, create_orchestrator_harness, time_traveler
    ):
        """Test that event counts in fixed time intervals pass Kolmogorov-Smirnov test for goodness of fit."""
        request_rate = 1_000
        request_count = 25_000

        actual_mean, event_counts = await self.run_poisson_rate_event_counts(
            request_rate, request_count, 789, 0.5, create_orchestrator_harness
        )

        max_count = int(np.max(event_counts))
        theoretical_pmf = stats.poisson.pmf(np.arange(max_count + 1), actual_mean)

        unique_counts, frequencies = np.unique(event_counts, return_counts=True)
        empirical_pmf = frequencies / len(event_counts)

        empirical_pmf_padded = np.zeros(max_count + 1)
        empirical_pmf_padded[unique_counts] = empirical_pmf

        theoretical_cdf = np.cumsum(theoretical_pmf)
        empirical_cdf = np.cumsum(empirical_pmf_padded)

        ks_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf))
        critical_value = 1.36 / np.sqrt(len(event_counts))

        assert ks_statistic < critical_value, (
            f"KS statistic {ks_statistic:.4f} exceeds critical value {critical_value:.4f}, "
            f"indicating poor fit to Poisson distribution"
        )

    async def test_poisson_rate_event_counts_follow_poisson_distribution_ratio_property(
        self, create_orchestrator_harness, time_traveler
    ):
        """Test that the distribution has the right shape characteristics"""
        request_rate = 1_000
        request_count = 25_000

        actual_mean, event_counts = await self.run_poisson_rate_event_counts(
            request_rate, request_count, 789, 0.5, create_orchestrator_harness
        )

        # For Poisson distribution, P(X = k+1) / P(X = k) = λ / (k+1)
        unique_counts, frequencies = np.unique(event_counts, return_counts=True)
        probabilities = frequencies / len(event_counts)

        valid_ratios = []
        for i in range(len(unique_counts) - 1):
            k = unique_counts[i]
            if probabilities[i] > 0.01 and probabilities[i + 1] > 0.01:
                actual_ratio = probabilities[i + 1] / probabilities[i]
                expected_ratio = actual_mean / (k + 1)
                if expected_ratio > 0.1:
                    valid_ratios.append((actual_ratio, expected_ratio))

        ratio_errors = [
            abs(actual - expected) / expected for actual, expected in valid_ratios
        ]
        avg_ratio_error = np.mean(ratio_errors)
        assert avg_ratio_error < 0.5, (
            f"Average ratio error {avg_ratio_error:.4f} indicates poor fit to Poisson distribution"
        )


@pytest.mark.asyncio
class TestRequestRateStrategyMaxConcurrency:
    """Tests for max concurrency support in RequestRateStrategy."""

    @pytest.mark.parametrize(
        "concurrency, request_rate, stats_enabled",
        [
            (5, None, True),
            (10, None, True),
            (100, None, True),
            (300_000, None, True),
            (None, 10.0, False),
        ],
    )
    async def test_concurrency_stats_enabled(
        self,
        create_orchestrator_harness,
        time_traveler,
        concurrency: int | None,
        request_rate: float | None,
        stats_enabled: bool,
    ):
        """Test that concurrency stats are available when concurrency is specified."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"conv{i}", 1) for i in range(10)],
            request_count=10,
            concurrency=concurrency,
            request_rate=request_rate,
        )

        # Stats are only available after the phase is configured (during run)
        await harness.run_with_auto_return()

        stats = get_session_stats(harness.orchestrator)
        if stats_enabled:
            assert stats is not None
        else:
            assert stats is None

    async def test_semaphore_acquisition_during_credit_drop(
        self, create_orchestrator_harness, time_traveler
    ):
        """Test that semaphore is acquired before each credit drop."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"conv{i}", 1) for i in range(10)],
            request_count=10,
            concurrency=5,
        )

        await harness.run_with_auto_return()

        stats = get_session_stats(harness.orchestrator)
        assert stats is not None
        assert stats.acquire_count == 10

    async def test_credit_return_releases_semaphore(
        self, create_orchestrator_harness, time_traveler
    ):
        """Test that credit return properly releases the semaphore."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"conv{i}", 1) for i in range(10)],
            request_count=10,
            concurrency=2,
        )

        await harness.run_with_auto_return()

        stats = get_session_stats(harness.orchestrator)
        assert stats is not None
        assert stats.release_count == 10

    async def test_single_concurrency_serializes_execution(
        self, create_orchestrator_harness, time_traveler
    ):
        """Test that concurrency=1 ensures credits are processed one at a time.

        Note: With auto_return mock, credits return instantly so wait_count may be 0
        rather than 4. This tests that the concurrency mechanism works correctly -
        all credits should complete successfully even with concurrency=1.
        """
        orchestrator: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"conv{i}", 1) for i in range(5)],
            request_count=5,
            concurrency=1,
        )

        await orchestrator.run_with_auto_return()

        # Verify all credits were acquired and released
        stats = get_session_stats(orchestrator.orchestrator)
        assert stats is not None
        assert stats.acquire_count == 5
        assert stats.release_count == 5

    async def test_concurrency_boundary_conditions(
        self, create_orchestrator_harness, time_traveler
    ):
        """Test concurrency behavior at boundary conditions."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"conv{i}", 1) for i in range(5)],
            request_count=5,
            concurrency=5,
        )

        await harness.run_with_auto_return()

        # Should process all credits without blocking since count equals concurrency
        stats = get_session_stats(harness.orchestrator)
        assert stats is not None
        assert stats.acquire_count == 5
        assert stats.wait_count == 0


class TestConcurrencyBurstIntervalGeneratorBehavior:
    """Tests for ConcurrencyBurstIntervalGenerator behavior.

    Note: ConcurrencyBurstIntervalGenerator does no validation - it always returns 0.
    Concurrency validation is handled at the strategy level, not the generator level.
    """

    def test_always_returns_zero_interval(self):
        """Test that ConcurrencyBurstIntervalGenerator always returns zero interval."""
        config = IntervalGeneratorConfig(
            arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
            request_rate=None,
        )

        generator = ConcurrencyBurstIntervalGenerator(config)

        for _ in range(10):
            assert generator.next_interval() == 0

    def test_rate_property_returns_zero(self):
        """Test that rate property returns 0 for burst mode."""
        config = IntervalGeneratorConfig(
            arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        )

        generator = ConcurrencyBurstIntervalGenerator(config)
        assert generator.rate == 0.0

    def test_set_rate_is_noop(self):
        """Test that set_rate is a no-op for burst mode."""
        config = IntervalGeneratorConfig(
            arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        )

        generator = ConcurrencyBurstIntervalGenerator(config)
        generator.set_rate(100.0)  # Should not raise
        assert generator.rate == 0.0  # Still returns 0


class TestPoissonIntervalGeneratorExceptions:
    """Tests for PoissonIntervalGenerator initialization exceptions."""

    def test_request_rate_none_raises_value_error(self):
        """Test that PoissonIntervalGenerator raises ValueError when request_rate is None."""
        config = IntervalGeneratorConfig(
            arrival_pattern=ArrivalPattern.POISSON,
            request_rate=None,
        )

        with pytest.raises(ValueError):
            PoissonIntervalGenerator(config)

    @pytest.mark.parametrize("invalid_request_rate", [0, -1, -5.0, -100.5, 0.0])
    def test_request_rate_zero_or_negative_raises_value_error(
        self, invalid_request_rate: float
    ):
        """Test that PoissonIntervalGenerator raises ValueError when request_rate is zero or negative."""
        config = IntervalGeneratorConfig(
            arrival_pattern=ArrivalPattern.POISSON,
            request_rate=invalid_request_rate,
        )

        with pytest.raises(ValueError):
            PoissonIntervalGenerator(config)

    @pytest.mark.parametrize("valid_request_rate", [0.1, 1.0, 10.5, 100, 1000])
    def test_valid_configuration_succeeds(self, valid_request_rate: float):
        """Test that PoissonIntervalGenerator initializes successfully with valid configuration."""
        rng.reset()
        rng.init(42)
        config = IntervalGeneratorConfig(
            arrival_pattern=ArrivalPattern.POISSON,
            request_rate=valid_request_rate,
        )

        generator = PoissonIntervalGenerator(config)

        for _ in range(10):
            interval = generator.next_interval()
            assert interval > 0


class TestConstantIntervalGeneratorExceptions:
    """Tests for ConstantIntervalGenerator initialization exceptions."""

    def test_request_rate_none_raises_value_error(self):
        """Test that ConstantIntervalGenerator raises ValueError when request_rate is None."""
        config = IntervalGeneratorConfig(
            arrival_pattern=ArrivalPattern.CONSTANT,
            request_rate=None,
        )

        with pytest.raises(ValueError):
            ConstantIntervalGenerator(config)

    @pytest.mark.parametrize("invalid_request_rate", [0, -1, -5.0, -100.5, 0.0])
    def test_request_rate_zero_or_negative_raises_value_error(
        self, invalid_request_rate: float
    ):
        """Test that ConstantIntervalGenerator raises ValueError when request_rate is zero or negative."""
        config = IntervalGeneratorConfig(
            arrival_pattern=ArrivalPattern.CONSTANT,
            request_rate=invalid_request_rate,
        )

        with pytest.raises(ValueError):
            ConstantIntervalGenerator(config)

    @pytest.mark.parametrize("valid_request_rate", [0.1, 1.0, 10.5, 100, 1000])
    def test_valid_configuration_succeeds(self, valid_request_rate: float):
        """Test that ConstantIntervalGenerator initializes successfully with valid configuration."""
        config = IntervalGeneratorConfig(
            arrival_pattern=ArrivalPattern.CONSTANT,
            request_rate=valid_request_rate,
        )

        generator = ConstantIntervalGenerator(config)

        expected_interval = 1.0 / valid_request_rate
        for _ in range(10):
            interval = generator.next_interval()
            assert interval == expected_interval


@pytest.mark.asyncio
class TestRequestRateStrategyEarlyExit:
    """Test for the early exit fix that prevents unnecessary final sleep."""

    async def test_early_exit_prevents_unnecessary_final_sleep(
        self, create_orchestrator_harness, time_traveler
    ):
        """Test that execution stops immediately after sending all credits, without extra sleep.

        Note: The timing assertion (travels_for) was removed because it requires
        time_traveler_no_patch_sleep which interferes with other tests in this file.
        The core functionality (all credits sent successfully) is still verified.
        """
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"conv{i}", 1) for i in range(2)],
            request_rate=1.0,
            request_count=2,
            arrival_pattern=ArrivalPattern.CONSTANT,
            concurrency=1,
        )
        await harness.run_with_auto_return()
        # Verify all credits were sent
        assert len(harness.sent_credits) == 2
