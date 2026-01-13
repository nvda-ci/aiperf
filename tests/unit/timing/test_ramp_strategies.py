# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ramp strategies."""

import pytest

from aiperf.timing.ramping import (
    BaseRampStrategy as RampStrategy,
)
from aiperf.timing.ramping import (
    ExponentialStrategy,
    LinearStrategy,
    PoissonStrategy,
    RampConfig,
    RampStrategyFactory,
    RampType,
)


def linear_config(
    start: float, target: float, duration_sec: float, step_size: float | None = None
) -> RampConfig:
    """Helper to create a LinearStrategy config."""
    return RampConfig(
        ramp_type=RampType.LINEAR,
        start=start,
        target=target,
        duration_sec=duration_sec,
        step_size=step_size,
    )


def exponential_config(
    start: float, target: float, duration_sec: float, exponent: float = 2.0
) -> RampConfig:
    """Helper to create an ExponentialStrategy config."""
    return RampConfig(
        ramp_type=RampType.EXPONENTIAL,
        start=start,
        target=target,
        duration_sec=duration_sec,
        exponent=exponent,
    )


def poisson_config(start: float, target: float, duration_sec: float) -> RampConfig:
    """Helper to create a PoissonStrategy config."""
    return RampConfig(
        ramp_type=RampType.POISSON,
        start=start,
        target=target,
        duration_sec=duration_sec,
    )


class TestLinearStrategy:
    """Test LinearStrategy behavior."""

    def test_protocol_compliance(self):
        """LinearStrategy should implement RampStrategy protocol."""
        strategy = LinearStrategy(linear_config(start=1, target=100, duration_sec=10.0))
        assert isinstance(strategy, RampStrategy)

    def test_start_target_properties(self):
        """Should expose start and target as properties."""
        strategy = LinearStrategy(linear_config(start=5, target=50, duration_sec=10.0))
        assert strategy.start == 5
        assert strategy.target == 50

    def test_returns_none_at_target(self):
        """Should return None when current equals target."""
        strategy = LinearStrategy(linear_config(start=1, target=100, duration_sec=10.0))
        result = strategy.next_step(100, elapsed_sec=0.0)
        assert result is None

    def test_ramp_up_increments_by_one(self):
        """Should increment by exactly 1 for ramp up."""
        strategy = LinearStrategy(linear_config(start=1, target=100, duration_sec=10.0))
        result = strategy.next_step(1, elapsed_sec=0.0)
        assert result is not None
        delay, next_val = result
        assert next_val == 2

    def test_ramp_down_decrements_by_one(self):
        """Should decrement by exactly 1 for ramp down."""
        strategy = LinearStrategy(linear_config(start=100, target=1, duration_sec=10.0))
        result = strategy.next_step(100, elapsed_sec=0.0)
        assert result is not None
        delay, next_val = result
        assert next_val == 99

    def test_interval_calculation_ramp_up(self):
        """Should calculate interval as duration / steps."""
        # 1→100 in 9.9s = 99 steps = 0.1s interval
        strategy = LinearStrategy(linear_config(start=1, target=100, duration_sec=9.9))
        result = strategy.next_step(1, elapsed_sec=0.0)
        assert result is not None
        delay, _ = result
        expected_interval = 9.9 / 99  # 0.1s
        assert abs(delay - expected_interval) < 0.0001

    def test_interval_calculation_ramp_down(self):
        """Should calculate interval correctly for ramp down."""
        # 100→1 in 9.9s = 99 steps = 0.1s interval
        strategy = LinearStrategy(linear_config(start=100, target=1, duration_sec=9.9))
        result = strategy.next_step(100, elapsed_sec=0.0)
        assert result is not None
        delay, _ = result
        expected_interval = 9.9 / 99
        assert abs(delay - expected_interval) < 0.0001

    def test_precise_timing_self_corrects(self):
        """Should compute delays based on elapsed time for self-correction."""
        strategy = LinearStrategy(linear_config(start=1, target=100, duration_sec=10.0))

        # First call: delay to reach value 2 at progress 1/99
        result1 = strategy.next_step(1, elapsed_sec=0.0)
        assert result1 is not None
        delay1, _ = result1
        expected1 = 10.0 * (1 / 99)  # time_at_next for value 2
        assert abs(delay1 - expected1) < 0.0001

        # Simulate elapsed time passing, then ask for next step
        # If we're at value 50, time should be at 50% progress = 5.0s
        result2 = strategy.next_step(50, elapsed_sec=5.0)
        assert result2 is not None
        delay2, _ = result2
        # Value 51 should be at progress 50/99, time = 10 * (50/99) = 5.05...
        expected2 = 10.0 * (50 / 99) - 5.0
        assert abs(delay2 - expected2) < 0.0001

    def test_precise_timing_example(self):
        """Should calculate precise intervals for documentation example."""
        # 1→500 in 1s = 499 steps = ~2.004ms each
        strategy = LinearStrategy(linear_config(start=1, target=500, duration_sec=1.0))
        result = strategy.next_step(1, elapsed_sec=0.0)
        assert result is not None
        delay, next_val = result
        expected_interval = 1.0 / 499
        assert abs(delay - expected_interval) < 0.000001
        assert abs(delay - 0.002004) < 0.000001
        assert next_val == 2

    def test_small_ramp_single_step(self):
        """Should handle single step ramp (start=1, target=2)."""
        strategy = LinearStrategy(linear_config(start=1, target=2, duration_sec=1.0))
        result = strategy.next_step(1, elapsed_sec=0.0)
        assert result is not None
        delay, next_val = result
        assert delay == 1.0  # 1 step in 1 second
        assert next_val == 2

    def test_already_at_target(self):
        """Should return None immediately if start equals target."""
        strategy = LinearStrategy(linear_config(start=50, target=50, duration_sec=10.0))
        result = strategy.next_step(50, elapsed_sec=0.0)
        assert result is None

    def test_full_ramp_simulation(self):
        """Should correctly ramp through all values."""
        strategy = LinearStrategy(linear_config(start=1, target=10, duration_sec=9.0))

        current = 1
        values = [current]
        while True:
            result = strategy.next_step(current, elapsed_sec=0.0)
            if result is None:
                break
            _, current = result
            values.append(current)

        assert values == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class TestLinearStrategyWithStepSize:
    """Test LinearStrategy behavior with custom step_size."""

    def test_step_size_ramp_up(self):
        """Should jump by step_size for ramp up."""
        strategy = LinearStrategy(
            linear_config(start=1, target=100, duration_sec=10.0, step_size=10)
        )
        result = strategy.next_step(1, elapsed_sec=0.0)
        assert result is not None
        _, next_val = result
        assert next_val == 11

    def test_step_size_ramp_down(self):
        """Should jump by step_size for ramp down."""
        strategy = LinearStrategy(
            linear_config(start=100, target=1, duration_sec=10.0, step_size=10)
        )
        result = strategy.next_step(100, elapsed_sec=0.0)
        assert result is not None
        _, next_val = result
        assert next_val == 90

    def test_clamps_to_target_ramp_up(self):
        """Should clamp to target when step would overshoot."""
        strategy = LinearStrategy(
            linear_config(start=1, target=100, duration_sec=10.0, step_size=10)
        )
        result = strategy.next_step(95, elapsed_sec=0.0)
        assert result is not None
        _, next_val = result
        assert next_val == 100  # Clamped, not 105

    def test_clamps_to_target_ramp_down(self):
        """Should clamp to target when step would undershoot."""
        strategy = LinearStrategy(
            linear_config(start=100, target=1, duration_sec=10.0, step_size=10)
        )
        result = strategy.next_step(5, elapsed_sec=0.0)
        assert result is not None
        _, next_val = result
        assert next_val == 1  # Clamped, not -5

    def test_precise_timing_calculation(self):
        """Should compute delay based on progress of next value."""
        # 1→100 in 10s with step_size=10: first step goes to 11
        # progress at 11 = (11-1)/(100-1) = 10/99 ≈ 0.101
        # time_at_11 = 10.0 * (10/99) ≈ 1.01s
        strategy = LinearStrategy(
            linear_config(start=1, target=100, duration_sec=10.0, step_size=10)
        )
        result = strategy.next_step(1, elapsed_sec=0.0)
        assert result is not None
        delay, next_val = result
        assert next_val == 11
        expected_delay = 10.0 * (10 / 99)  # ~1.0101s
        assert abs(delay - expected_delay) < 0.0001

    def test_precise_timing_self_corrects(self):
        """Should compute delays based on elapsed time for self-correction."""
        strategy = LinearStrategy(
            linear_config(start=1, target=100, duration_sec=10.0, step_size=10)
        )

        # First step: 1 → 11
        result1 = strategy.next_step(1, elapsed_sec=0.0)
        assert result1 is not None
        delay1, _ = result1
        expected1 = 10.0 * (10 / 99)  # time_at_next for value 11
        assert abs(delay1 - expected1) < 0.0001

        # Simulate elapsed time passing, then ask for next step
        # If we're at value 51, progress is 50/99, time should be ~5.05s
        result2 = strategy.next_step(51, elapsed_sec=5.0)
        assert result2 is not None
        delay2, next_val = result2
        assert next_val == 61
        # Value 61 at progress 60/99, time = 10 * (60/99) = 6.06...
        expected2 = 10.0 * (60 / 99) - 5.0
        assert abs(delay2 - expected2) < 0.0001

    def test_full_ramp_simulation_with_step_size(self):
        """Should correctly step through values with custom step_size."""
        strategy = LinearStrategy(
            linear_config(start=1, target=100, duration_sec=4.0, step_size=25)
        )

        current = 1
        values = [current]
        while True:
            result = strategy.next_step(current, elapsed_sec=0.0)
            if result is None:
                break
            _, current = result
            values.append(current)

        assert values == [1, 26, 51, 76, 100]


class TestExponentialStrategy:
    """Test ExponentialStrategy behavior (ease-in curve with precise timing)."""

    def test_protocol_compliance(self):
        """ExponentialStrategy should implement RampStrategy protocol."""
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )
        assert isinstance(strategy, RampStrategy)

    def test_invalid_exponent_raises(self):
        """Should raise ValidationError for exponent <= 1.0."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="greater than 1"):
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=1.0)

        with pytest.raises(ValidationError, match="greater than 1"):
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=0.5)

    def test_returns_none_at_target(self):
        """Should return None when current == target."""
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )
        result = strategy.next_step(100, elapsed_sec=0.0)
        assert result is None

    def test_returns_none_above_target(self):
        """Should return None when current > target (overshoot)."""
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )
        result = strategy.next_step(150, elapsed_sec=0.0)
        assert result is None

    def test_always_increments_by_one(self):
        """Each step should be exactly +1 for ramp-up."""
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )

        result = strategy.next_step(1, elapsed_sec=0.0)
        assert result is not None
        _, next_val = result
        assert next_val == 2  # Exactly +1

        # Also at a later point
        result2 = strategy.next_step(50, elapsed_sec=0.5)
        assert result2 is not None
        _, next_val2 = result2
        assert next_val2 == 51  # Exactly +1

    def test_delays_decrease_over_time(self):
        """Delays should get shorter as we approach target (ease-in)."""
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )
        delays = []

        current = 1
        elapsed = 0.0

        # Collect delays for first 10 steps
        for _ in range(10):
            result = strategy.next_step(current, elapsed_sec=elapsed)
            if result is None:
                break
            delay, current = result
            delays.append(delay)
            elapsed += delay

        # Each delay should be less than or equal to the previous (accelerating)
        for i in range(1, len(delays)):
            assert delays[i] <= delays[i - 1] + 0.001, (
                f"Delay {i} ({delays[i]:.4f}) should be <= delay {i - 1} ({delays[i - 1]:.4f})"
            )

    def test_first_delay_is_longest(self):
        """First step should have the longest delay (slow start)."""
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )

        result = strategy.next_step(1, elapsed_sec=0.0)
        assert result is not None
        first_delay, _ = result

        # For 1→100 in 1s with exp=2: first step to value=2
        # progress = 1/99 ≈ 0.0101, time = 1.0 * (0.0101)^0.5 ≈ 0.1005s
        assert first_delay > 0.09  # Should be around 100ms

    def test_last_delay_is_shortest(self):
        """Last step should have the shortest delay (fast finish)."""
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )
        # Near the end: current=99, next=100
        # progress = 99/99 = 1.0, time = 1.0 * 1.0^0.5 = 1.0s
        # At elapsed=0.99, delay = 1.0 - 0.99 = 0.01s

        result = strategy.next_step(99, elapsed_sec=0.99)
        assert result is not None
        last_delay, next_val = result
        assert next_val == 100
        assert last_delay < 0.02  # Should be around 10ms or less

    def test_higher_exponent_slower_start(self):
        """Higher exponent should mean longer initial delays."""
        strategy_low = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )
        strategy_high = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=3.0)
        )

        result_low = strategy_low.next_step(1, elapsed_sec=0.0)
        result_high = strategy_high.next_step(1, elapsed_sec=0.0)

        assert result_low is not None
        assert result_high is not None
        delay_low, _ = result_low
        delay_high, _ = result_high

        # Higher exponent = longer first delay (slower start)
        assert delay_high > delay_low

    def test_full_ramp_simulation(self):
        """Should complete full ramp with all values from start to target."""
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )

        current = 1
        elapsed = 0.0
        values = [current]

        while current < 100:
            result = strategy.next_step(current, elapsed_sec=elapsed)
            if result is None:
                break
            delay, current = result
            elapsed += delay
            values.append(current)

        # Should have all values from 1 to 100
        assert values == list(range(1, 101))

        # Total elapsed should be close to duration
        assert abs(elapsed - 1.0) < 0.01

    def test_total_time_matches_duration(self):
        """Sum of all delays should equal duration."""
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )

        current = 1
        elapsed = 0.0
        total_delay = 0.0

        while current < 100:
            result = strategy.next_step(current, elapsed_sec=elapsed)
            if result is None:
                break
            delay, current = result
            total_delay += delay
            elapsed += delay

        assert abs(total_delay - 1.0) < 0.001

    def test_ramp_down_decrements_by_one(self):
        """Each step should be exactly -1 for ramp-down."""
        strategy = ExponentialStrategy(
            exponential_config(start=100, target=1, duration_sec=1.0, exponent=2.0)
        )

        result = strategy.next_step(100, elapsed_sec=0.0)
        assert result is not None
        _, next_val = result
        assert next_val == 99  # Exactly -1

    def test_ramp_down_delays_decrease(self):
        """Ramp-down delays should also decrease (ease-in behavior)."""
        strategy = ExponentialStrategy(
            exponential_config(start=100, target=1, duration_sec=1.0, exponent=2.0)
        )
        delays = []

        current = 100
        elapsed = 0.0

        for _ in range(10):
            result = strategy.next_step(current, elapsed_sec=elapsed)
            if result is None:
                break
            delay, current = result
            delays.append(delay)
            elapsed += delay

        # Each delay should decrease (accelerating toward target)
        for i in range(1, len(delays)):
            assert delays[i] <= delays[i - 1] + 0.001

    def test_ramp_down_full_simulation(self):
        """Should complete full ramp-down with all values from start to target."""
        strategy = ExponentialStrategy(
            exponential_config(start=100, target=1, duration_sec=1.0, exponent=2.0)
        )

        current = 100
        elapsed = 0.0
        values = [current]

        while current > 1:
            result = strategy.next_step(current, elapsed_sec=elapsed)
            if result is None:
                break
            delay, current = result
            elapsed += delay
            values.append(current)

        # Should have all values from 100 down to 1
        assert values == list(range(100, 0, -1))

        # Total elapsed should be close to duration
        assert abs(elapsed - 1.0) < 0.01

    def test_returns_none_below_target_ramp_down(self):
        """Should return None when current < target for ramp-down (overshoot)."""
        strategy = ExponentialStrategy(
            exponential_config(start=100, target=1, duration_sec=1.0, exponent=2.0)
        )

        # Test overshoot
        result = strategy.next_step(0, elapsed_sec=0.5)  # Below target
        assert result is None


class TestStrategyEdgeCases:
    """Test edge cases across all strategies."""

    @pytest.mark.parametrize(
        "strategy",
        [
            LinearStrategy(
                linear_config(start=1, target=1_000_000, duration_sec=100.0)
            ),
            LinearStrategy(
                linear_config(
                    start=1, target=1_000_000, duration_sec=100.0, step_size=10
                )
            ),
            ExponentialStrategy(
                exponential_config(
                    start=1, target=1_000_000, duration_sec=100.0, exponent=2.0
                )
            ),
            PoissonStrategy(poisson_config(start=1, target=1_000, duration_sec=100.0)),
        ],
    )
    def test_handles_large_values(self, strategy: RampStrategy):
        """All strategies should handle large values."""
        result = strategy.next_step(1, elapsed_sec=0.0)
        assert result is not None
        delay, next_val = result
        assert next_val > 1
        assert delay > 0

    @pytest.mark.parametrize(
        "strategy",
        [
            LinearStrategy(linear_config(start=1, target=100, duration_sec=0.001)),
            LinearStrategy(
                linear_config(start=1, target=100, duration_sec=0.001, step_size=10)
            ),
        ],
    )
    def test_handles_very_small_duration(self, strategy: RampStrategy):
        """Static strategies should handle very small durations gracefully."""
        result = strategy.next_step(1, elapsed_sec=0.0)
        assert result is not None
        delay, next_val = result
        # Very small duration means very small interval
        assert delay <= 0.001
        assert next_val > 1

    def test_poisson_very_small_duration_returns_few_events(self):
        """Poisson with very small duration should have few events."""
        strategy = PoissonStrategy(
            poisson_config(start=1, target=100, duration_sec=0.001)
        )
        # With very small duration, there are very few events
        result = strategy.next_step(1, elapsed_sec=0.0)
        # Either no events or very quick progression
        if result is None:
            assert len(strategy._event_times) == 0
        else:
            delay, next_val = result
            assert delay >= 0


class TestRampStrategyFactory:
    """Test RampStrategyFactory integration."""

    def test_factory_creates_linear_strategy(self):
        """Factory should create LinearStrategy for LINEAR type."""
        config = linear_config(start=1, target=100, duration_sec=10.0)
        strategy = RampStrategyFactory.create_instance(config)
        assert isinstance(strategy, LinearStrategy)
        assert strategy.start == 1
        assert strategy.target == 100

    def test_factory_creates_linear_strategy_with_step_size(self):
        """Factory should create LinearStrategy with custom step_size."""
        config = linear_config(start=1, target=100, duration_sec=10.0, step_size=10)
        strategy = RampStrategyFactory.create_instance(config)
        assert isinstance(strategy, LinearStrategy)

    def test_factory_creates_exponential_strategy(self):
        """Factory should create ExponentialStrategy for EXPONENTIAL type."""
        config = exponential_config(
            start=1, target=100, duration_sec=10.0, exponent=2.0
        )
        strategy = RampStrategyFactory.create_instance(config)
        assert isinstance(strategy, ExponentialStrategy)


class TestValueAt:
    """Test value_at() continuous sampling method."""

    def test_linear_value_at_start(self):
        """Linear strategy should return start value at elapsed=0."""
        strategy = LinearStrategy(
            linear_config(start=10, target=100, duration_sec=10.0)
        )
        # At t=0, value should be at start (or very close)
        value = strategy.value_at(0.0)
        assert value is not None
        assert value == 10.0

    def test_linear_value_at_midpoint(self):
        """Linear strategy should return midpoint value at half duration."""
        strategy = LinearStrategy(linear_config(start=1, target=101, duration_sec=10.0))
        # At t=5s (half of 10s), value should be 51 (start + 0.5 * range)
        value = strategy.value_at(5.0)
        assert value is not None
        assert abs(value - 51.0) < 0.01

    def test_linear_value_at_returns_none_at_completion(self):
        """Linear strategy should return None when elapsed >= duration."""
        strategy = LinearStrategy(linear_config(start=1, target=100, duration_sec=10.0))
        # At or after duration, should return None
        assert strategy.value_at(10.0) is None
        assert strategy.value_at(15.0) is None

    def test_linear_value_at_ramp_down(self):
        """Linear strategy should handle ramp-down correctly."""
        strategy = LinearStrategy(linear_config(start=100, target=1, duration_sec=10.0))
        # At t=5s, value should be ~50.5 (start - 0.5 * range = 100 - 0.5 * 99)
        value = strategy.value_at(5.0)
        assert value is not None
        assert abs(value - 50.5) < 0.01

    def test_exponential_value_at_slow_start(self):
        """Exponential strategy should have slower progress early (ease-in)."""
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=101, duration_sec=10.0, exponent=2.0)
        )
        # At t=5s (half time), value should be less than 51 due to ease-in
        # time_progress = 0.5, value_progress = 0.5^2 = 0.25, value = 1 + 25 = 26
        value = strategy.value_at(5.0)
        assert value is not None
        assert value < 51.0  # Slower than linear
        assert abs(value - 26.0) < 0.1  # Should be around 26

    def test_exponential_value_at_accelerates(self):
        """Exponential strategy should accelerate toward end."""
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=101, duration_sec=10.0, exponent=2.0)
        )
        # At t=8s (80% time), value_progress = 0.8^2 = 0.64, value = 1 + 64 = 65
        value = strategy.value_at(8.0)
        assert value is not None
        assert abs(value - 65.0) < 0.1

    def test_linear_with_step_size_value_at_interpolates(self):
        """LinearStrategy with step_size should interpolate linearly in value_at mode."""
        strategy = LinearStrategy(
            linear_config(start=1, target=101, duration_sec=10.0, step_size=25)
        )
        # value_at uses linear interpolation, not steps
        value = strategy.value_at(5.0)
        assert value is not None
        assert abs(value - 51.0) < 0.01

    def test_value_at_returns_none_for_zero_range(self):
        """Should return None if start == target."""
        strategy = LinearStrategy(linear_config(start=50, target=50, duration_sec=10.0))
        assert strategy.value_at(0.0) is None
        assert strategy.value_at(5.0) is None

    def test_value_at_handles_very_small_duration(self):
        """Should handle very small duration gracefully."""
        strategy = LinearStrategy(
            linear_config(start=1, target=100, duration_sec=0.001)
        )
        # With very small duration, elapsed time quickly exceeds it
        assert strategy.value_at(0.001) is None
        assert strategy.value_at(0.01) is None

    def test_higher_exponent_slower_value_progress(self):
        """Higher exponent should mean slower value progress early."""
        strategy_exp2 = ExponentialStrategy(
            exponential_config(start=1, target=101, duration_sec=10.0, exponent=2.0)
        )
        strategy_exp3 = ExponentialStrategy(
            exponential_config(start=1, target=101, duration_sec=10.0, exponent=3.0)
        )

        # At halfway time, exp3 should have less progress
        value_exp2 = strategy_exp2.value_at(5.0)
        value_exp3 = strategy_exp3.value_at(5.0)

        assert value_exp2 is not None
        assert value_exp3 is not None
        assert value_exp3 < value_exp2  # Higher exponent = slower start


class TestPoissonStrategy:
    """Test PoissonStrategy behavior (normalized exponential intervals)."""

    def test_protocol_compliance(self):
        """PoissonStrategy should implement RampStrategy protocol."""
        strategy = PoissonStrategy(
            poisson_config(start=1, target=100, duration_sec=10.0)
        )
        assert isinstance(strategy, RampStrategy)

    def test_start_target_properties(self):
        """Should expose start and target as properties."""
        strategy = PoissonStrategy(
            poisson_config(start=5, target=50, duration_sec=10.0)
        )
        assert strategy.start == 5
        assert strategy.target == 50

    def test_returns_none_when_complete(self):
        """Should return None after all steps consumed."""
        strategy = PoissonStrategy(poisson_config(start=1, target=3, duration_sec=1.0))

        # Consume all steps (2 steps: 1→2→3)
        result1 = strategy.next_step(1, elapsed_sec=0.0)
        assert result1 is not None
        result2 = strategy.next_step(2, elapsed_sec=0.5)
        assert result2 is not None
        result3 = strategy.next_step(3, elapsed_sec=1.0)
        assert result3 is None

    def test_ramp_up_increases_toward_target(self):
        """Should increase values toward target for ramp up."""
        strategy = PoissonStrategy(
            poisson_config(start=1, target=100, duration_sec=10.0)
        )
        result = strategy.next_step(1, elapsed_sec=0.0)
        assert result is not None
        _, next_val = result
        assert next_val > 1  # Moving toward target
        assert next_val <= 100  # Not exceeding target

    def test_ramp_down_decreases_toward_target(self):
        """Should decrease values toward target for ramp down."""
        strategy = PoissonStrategy(
            poisson_config(start=100, target=1, duration_sec=10.0)
        )
        result = strategy.next_step(100, elapsed_sec=0.0)
        assert result is not None
        _, next_val = result
        assert next_val < 100  # Moving toward target
        assert next_val >= 1  # Not going below target

    def test_full_ramp_simulation(self):
        """Should ramp with monotonically increasing values, ending at target."""
        strategy = PoissonStrategy(poisson_config(start=1, target=10, duration_sec=9.0))

        current = 1
        values = [current]
        while True:
            result = strategy.next_step(current, elapsed_sec=0.0)
            if result is None:
                break
            _, current = result
            values.append(current)

        # Values should be monotonically increasing
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]

        # Should end exactly at target (guaranteed by scaling)
        assert values[-1] == 10

    def test_total_time_matches_duration(self):
        """Sum of all delays should equal duration (normalized)."""
        strategy = PoissonStrategy(
            poisson_config(start=1, target=100, duration_sec=10.0)
        )

        current = 1
        elapsed = 0.0
        total_delay = 0.0

        while current < 100:
            result = strategy.next_step(current, elapsed_sec=elapsed)
            if result is None:
                break
            delay, current = result
            total_delay += delay
            elapsed += delay

        # Normalized intervals should sum exactly to duration
        assert abs(total_delay - 10.0) < 0.001

    def test_intervals_are_variable(self):
        """Intervals should vary (not constant like linear)."""
        strategy = PoissonStrategy(
            poisson_config(start=1, target=20, duration_sec=10.0)
        )

        delays = []
        current = 1
        elapsed = 0.0

        for _ in range(10):
            result = strategy.next_step(current, elapsed_sec=elapsed)
            if result is None:
                break
            delay, current = result
            delays.append(delay)
            elapsed += delay

        # Check that delays are not all the same (Poisson characteristic)
        unique_delays = set(round(d, 6) for d in delays)
        assert len(unique_delays) > 1, "Poisson intervals should vary"

    def test_deterministic_with_same_seed(self):
        """Same global seed should produce identical trajectories."""
        # Both strategies use rng.derive() with same identifier
        strategy1 = PoissonStrategy(
            poisson_config(start=1, target=10, duration_sec=5.0)
        )
        strategy2 = PoissonStrategy(
            poisson_config(start=1, target=10, duration_sec=5.0)
        )

        # Extract all event times
        times1 = strategy1._event_times
        times2 = strategy2._event_times

        assert len(times1) == len(times2)
        for t1, t2 in zip(times1, times2, strict=True):
            assert abs(t1 - t2) < 1e-10

    def test_already_at_target(self):
        """Should return None immediately if start equals target."""
        strategy = PoissonStrategy(
            poisson_config(start=50, target=50, duration_sec=10.0)
        )
        result = strategy.next_step(50, elapsed_sec=0.0)
        assert result is None

    def test_fractional_range_ramp_up(self):
        """Should handle non-integer ranges with monotonic values."""
        strategy = PoissonStrategy(
            poisson_config(start=1.0, target=10.7, duration_sec=5.0)
        )

        # Should have at least one step
        assert len(strategy._event_times) >= 1

        # Values should be monotonically increasing
        for i in range(1, len(strategy._values)):
            assert strategy._values[i] >= strategy._values[i - 1]

        # Should start at start
        assert strategy._values[0] == 1.0

        # Final value should be clamped at or below target
        assert strategy._values[-1] <= 10.7

    def test_fractional_range_ramp_down(self):
        """Should handle non-integer ranges for ramp down."""
        strategy = PoissonStrategy(
            poisson_config(start=10.7, target=1.0, duration_sec=5.0)
        )

        # Values should be monotonically decreasing
        for i in range(1, len(strategy._values)):
            assert strategy._values[i] <= strategy._values[i - 1]

        # Should start at start
        assert strategy._values[0] == 10.7

        # Final value should be clamped at or above target
        assert strategy._values[-1] >= 1.0


class TestPoissonStrategyValueAt:
    """Test PoissonStrategy value_at() continuous sampling."""

    def test_value_at_start(self):
        """Should return start value at elapsed=0."""
        strategy = PoissonStrategy(
            poisson_config(start=10, target=100, duration_sec=10.0)
        )
        value = strategy.value_at(0.0)
        assert value is not None
        assert value == 10.0

    def test_value_at_returns_none_at_completion(self):
        """Should return None when elapsed >= duration."""
        strategy = PoissonStrategy(
            poisson_config(start=1, target=100, duration_sec=10.0)
        )
        assert strategy.value_at(10.0) is None
        assert strategy.value_at(15.0) is None

    def test_value_at_returns_none_for_zero_range(self):
        """Should return None if start == target."""
        strategy = PoissonStrategy(
            poisson_config(start=50, target=50, duration_sec=10.0)
        )
        assert strategy.value_at(0.0) is None
        assert strategy.value_at(5.0) is None

    def test_value_at_is_step_function(self):
        """value_at should return step values (not interpolated)."""
        strategy = PoissonStrategy(poisson_config(start=1, target=10, duration_sec=9.0))

        # Get the event times and values
        event_times = strategy._event_times
        values = strategy._values

        # Just before first event, should still be at start
        if event_times:
            just_before = event_times[0] - 0.001
            if just_before > 0:
                value = strategy.value_at(just_before)
                assert value == values[0]  # Still at start

            # Just after first event, should be at second value
            just_after = event_times[0] + 0.001
            if just_after < 9.0:
                value = strategy.value_at(just_after)
                assert value == values[1]  # Jumped to next value

    def test_value_at_returns_valid_value_near_end(self):
        """value_at near end should return a value within range."""
        strategy = PoissonStrategy(
            poisson_config(start=1, target=100, duration_sec=10.0)
        )

        # Just before duration ends
        value = strategy.value_at(9.999)
        assert value is not None
        # Should be somewhere in the valid range
        assert 1 <= value <= 100

    def test_value_at_handles_ramp_down(self):
        """value_at should work correctly for ramp-down."""
        strategy = PoissonStrategy(
            poisson_config(start=100, target=1, duration_sec=10.0)
        )

        # At start
        value_start = strategy.value_at(0.0)
        assert value_start == 100

        # At some midpoint (value should be less than start)
        value_mid = strategy.value_at(5.0)
        assert value_mid is not None
        assert value_mid < 100

    def test_value_at_consistent_with_next_step(self):
        """value_at should return same values as next_step trajectory."""
        strategy = PoissonStrategy(poisson_config(start=1, target=10, duration_sec=5.0))

        # Get trajectory from next_step
        trajectory: list[tuple[float, float]] = [(0.0, 1.0)]  # (time, value)
        elapsed = 0.0
        current = 1

        # Make a copy of event_times before consuming next_step
        event_times = list(strategy._event_times)

        while True:
            result = strategy.next_step(current, elapsed_sec=elapsed)
            if result is None:
                break
            delay, current = result
            elapsed += delay
            trajectory.append((elapsed, float(current)))

        # Create fresh strategy for value_at testing
        strategy2 = PoissonStrategy(
            poisson_config(start=1, target=10, duration_sec=5.0)
        )

        # Check that value_at matches at each event time
        for i, event_time in enumerate(event_times):
            # Just after the event, value_at should return the new value
            value = strategy2.value_at(event_time + 0.0001)
            if value is not None:
                expected = trajectory[i + 1][1]  # Value after this event
                assert value == expected


class TestPoissonStrategyFactory:
    """Test factory integration for PoissonStrategy."""

    def test_factory_creates_poisson_strategy(self):
        """Factory should create PoissonStrategy for POISSON type."""
        config = poisson_config(start=1, target=100, duration_sec=10.0)
        strategy = RampStrategyFactory.create_instance(config)
        assert isinstance(strategy, PoissonStrategy)
        assert strategy.start == 1
        assert strategy.target == 100
