# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Ramper class."""

import asyncio
import contextlib
from unittest.mock import MagicMock

import pytest

from aiperf.timing.ramping import RampConfig, Ramper, RampType


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


class TestRamperBasics:
    """Test basic Ramper functionality."""

    @pytest.mark.asyncio
    async def test_calls_setter_with_start_value(self, time_traveler):
        """Should call setter with start value immediately."""
        values: list[float] = []
        setter = values.append

        config = linear_config(start=10, target=10, duration_sec=1.0)
        ramper = Ramper(setter=setter, config=config)
        await ramper.start()

        assert 10 in values

    @pytest.mark.asyncio
    async def test_calls_setter_with_target_value(self, time_traveler):
        """Should call setter with target value at completion."""
        values: list[float] = []
        setter = values.append

        config = linear_config(start=1, target=5, duration_sec=0.1)
        ramper = Ramper(setter=setter, config=config)
        await ramper.start()

        assert values[-1] == 5

    @pytest.mark.asyncio
    async def test_ramps_through_all_values_linear(self, time_traveler):
        """Should ramp through all intermediate values with LinearStrategy."""
        values: list[float] = []
        setter = values.append

        config = linear_config(start=1, target=5, duration_sec=0.1)
        ramper = Ramper(setter=setter, config=config)
        await ramper.start()

        assert values == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_ramps_down_correctly(self, time_traveler):
        """Should ramp down through values."""
        values: list[float] = []
        setter = values.append

        config = linear_config(start=5, target=1, duration_sec=0.1)
        ramper = Ramper(setter=setter, config=config)
        await ramper.start()

        assert values == [5, 4, 3, 2, 1]

    @pytest.mark.asyncio
    async def test_start_equals_target(self, time_traveler):
        """Should handle start == target (no ramping needed)."""
        values: list[float] = []
        setter = values.append

        config = linear_config(start=50, target=50, duration_sec=1.0)
        ramper = Ramper(setter=setter, config=config)
        await ramper.start()

        # Should set initial value, strategy returns None, done
        assert values == [50]


class TestRamperStop:
    """Test Ramper stop behavior."""

    @pytest.mark.asyncio
    async def test_stop_stays_at_current_value(self, time_traveler):
        """Should stay at current value when stop() is called."""
        values: list[float] = []
        setter = values.append

        config = linear_config(start=1, target=100, duration_sec=10.0)
        ramper = Ramper(setter=setter, config=config)

        # Start in background (don't await - it would block)
        task = ramper.start()

        # Let it start (virtual time)
        await time_traveler.sleep(0.01)
        assert ramper.is_running

        # Stop early (cancels task)
        ramper.stop()

        # Wait for task to complete
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Should NOT have jumped to target - stays at current value
        assert values[-1] != 100
        assert values[-1] >= 1  # At least the start value was set

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, time_traveler):
        """Should be safe to call stop() multiple times."""
        setter = MagicMock()
        config = linear_config(start=1, target=10, duration_sec=0.1)
        ramper = Ramper(setter=setter, config=config)
        await ramper.start()

        # Should not raise
        ramper.stop()
        ramper.stop()
        ramper.stop()

    @pytest.mark.asyncio
    async def test_stop_before_start(self):
        """Should be safe to call stop() before start()."""
        setter = MagicMock()
        config = linear_config(start=1, target=10, duration_sec=0.1)
        ramper = Ramper(setter=setter, config=config)

        # Should not raise
        ramper.stop()


class TestRamperWithRampTypes:
    """Test Ramper with different ramp types."""

    @pytest.mark.asyncio
    async def test_with_step_size(self, time_traveler):
        """Should work with custom step_size."""
        values: list[float] = []
        setter = values.append

        config = linear_config(start=1, target=100, duration_sec=0.1, step_size=25)
        ramper = Ramper(setter=setter, config=config)
        await ramper.start()

        # Should step by 25: 1, 26, 51, 76, 100
        assert values == [1, 26, 51, 76, 100]

    @pytest.mark.asyncio
    async def test_with_exponential_config(self, time_traveler):
        """Should work with EXPONENTIAL ramp type (ease-in curve)."""
        values: list[float] = []
        setter = values.append

        config = exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        ramper = Ramper(setter=setter, config=config)
        await ramper.start()

        # Should follow ease-in curve with precise timing
        # Each step is exactly +1, covering all values from 1 to 100
        assert values[0] == 1  # Starts at 1
        assert values[-1] == 100  # Ends at target
        assert len(values) == 100  # Should have exactly 100 values (1 through 100)

        # Verify values are consecutive (incrementing by 1.0)
        for i, val in enumerate(values):
            assert val == i + 1, f"Expected {i + 1}, got {val} at index {i}"


class TestRamperIsRunning:
    """Test Ramper is_running property."""

    @pytest.mark.asyncio
    async def test_not_running_before_start(self):
        """Should not be running before start()."""
        setter = MagicMock()
        config = linear_config(start=1, target=10, duration_sec=0.1)
        ramper = Ramper(setter=setter, config=config)
        assert not ramper.is_running

    @pytest.mark.asyncio
    async def test_running_during_ramp(self, time_traveler):
        """Should be running during ramping."""
        setter = MagicMock()
        config = linear_config(start=1, target=100, duration_sec=10.0)
        ramper = Ramper(setter=setter, config=config)

        # Start in background
        task = ramper.start()

        await time_traveler.sleep(0.01)
        assert ramper.is_running

        ramper.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_not_running_after_completion(self, time_traveler):
        """Should not be running after completion."""
        setter = MagicMock()
        config = linear_config(start=1, target=5, duration_sec=0.05)
        ramper = Ramper(setter=setter, config=config)
        await ramper.start()

        assert not ramper.is_running


class TestRamperTiming:
    """Test Ramper timing behavior."""

    @pytest.mark.asyncio
    async def test_ramp_completes_all_steps(self, time_traveler):
        """Should complete all ramp steps from start to target."""
        values: list[float] = []
        setter = values.append

        config = linear_config(start=1, target=10, duration_sec=0.2)
        ramper = Ramper(setter=setter, config=config)
        await ramper.start()

        # Should have called setter for all values from 1 to 10
        assert values == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class TestRamperEdgeCases:
    """Test Ramper edge cases."""

    @pytest.mark.asyncio
    async def test_large_ramp(self, time_traveler):
        """Should handle large ramps efficiently."""
        call_count = 0

        def counting_setter(value: float) -> None:
            nonlocal call_count
            call_count += 1

        # Use step_size=100 to avoid 999 individual calls
        config = linear_config(start=1, target=1000, duration_sec=0.1, step_size=100)
        ramper = Ramper(setter=counting_setter, config=config)
        await ramper.start()

        # Should be reasonable number of calls (10 steps + initial)
        assert call_count == 11

    @pytest.mark.asyncio
    async def test_very_short_duration(self, time_traveler):
        """Should handle very short duration by completing quickly."""
        values: list[float] = []
        setter = values.append

        config = linear_config(start=1, target=5, duration_sec=0.001)
        ramper = Ramper(setter=setter, config=config)
        await ramper.start()

        # With very short duration, it should still ramp through all values
        assert values == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_setter_exception_handling(self, time_traveler):
        """Should propagate setter exceptions."""

        def failing_setter(value: float) -> None:
            if value > 2:
                raise ValueError("Test error")

        config = linear_config(start=1, target=5, duration_sec=0.1)
        ramper = Ramper(setter=failing_setter, config=config)

        with pytest.raises(ValueError, match="Test error"):
            await ramper.start()

    @pytest.mark.asyncio
    async def test_restart_with_new_ramper(self, time_traveler):
        """Should be able to create a new ramper after completion."""
        values: list[float] = []
        setter = values.append

        config1 = linear_config(start=1, target=3, duration_sec=0.05)
        ramper = Ramper(setter=setter, config=config1)

        # First ramp
        await ramper.start()
        assert values == [1, 2, 3]

        # Clear values and create new ramper with new config
        values.clear()
        config2 = linear_config(start=10, target=12, duration_sec=0.05)
        ramper2 = Ramper(setter=setter, config=config2)
        await ramper2.start()
        assert values == [10, 11, 12]


def continuous_config(
    start: float, target: float, duration_sec: float, update_interval: float
) -> RampConfig:
    """Helper to create a continuous (update_interval-based) config."""
    return RampConfig(
        ramp_type=RampType.LINEAR,
        start=start,
        target=target,
        duration_sec=duration_sec,
        update_interval=update_interval,
    )


class TestRamperContinuousMode:
    """Test Ramper continuous mode using value_at() and update_interval."""

    @pytest.mark.asyncio
    async def test_sets_start_value_immediately(self, time_traveler):
        """Should set start value before any sleeps."""
        values: list[float] = []
        setter = values.append

        config = continuous_config(
            start=10, target=100, duration_sec=1.0, update_interval=0.2
        )
        ramper = Ramper(setter=setter, config=config)
        await ramper.start()

        # First value should be the start
        assert values[0] == 10.0

    @pytest.mark.asyncio
    async def test_sets_target_value_at_completion(self, time_traveler):
        """Should set target value when ramp completes."""
        values: list[float] = []
        setter = values.append

        config = continuous_config(
            start=1, target=100, duration_sec=1.0, update_interval=0.2
        )
        ramper = Ramper(setter=setter, config=config)
        await ramper.start()

        # Last value should be the target
        assert values[-1] == 100.0

    @pytest.mark.asyncio
    async def test_interpolates_values(self, time_traveler):
        """Should set interpolated values at each update interval."""
        values: list[float] = []
        setter = values.append

        # 10 second duration, update every 2 seconds = 5 updates + start + target
        config = continuous_config(
            start=1, target=100, duration_sec=10.0, update_interval=2.0
        )
        ramper = Ramper(setter=setter, config=config)
        await ramper.start()

        # Should have: start (1), then intermediate values, then target (100)
        assert values[0] == 1.0
        assert values[-1] == 100.0
        # Intermediate values should be roughly linear
        assert len(values) >= 5

    @pytest.mark.asyncio
    async def test_uses_update_interval_for_sleep(self, time_traveler):
        """Should sleep for update_interval duration between updates."""
        values: list[float] = []
        setter = values.append

        # Short duration for quick test
        config = continuous_config(
            start=1, target=10, duration_sec=0.5, update_interval=0.1
        )
        ramper = Ramper(setter=setter, config=config)
        await ramper.start()

        # With 0.5s duration and 0.1s interval, should have ~5 intermediate values
        # Plus start and target
        assert len(values) >= 5

    @pytest.mark.asyncio
    async def test_stop_stays_at_current_value(self, time_traveler):
        """Should stay at current value when stop() is called in continuous mode."""
        values: list[float] = []
        setter = values.append

        config = continuous_config(
            start=1, target=100, duration_sec=10.0, update_interval=0.5
        )
        ramper = Ramper(setter=setter, config=config)

        # Start in background
        task = ramper.start()
        await time_traveler.sleep(0.01)
        assert ramper.is_running

        # Stop early
        ramper.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Should NOT have jumped to target
        assert values[-1] != 100.0
        assert values[-1] >= 1.0

    @pytest.mark.asyncio
    async def test_handles_float_values(self, time_traveler):
        """Should handle float start/target values correctly."""
        values: list[float] = []
        setter = values.append

        config = continuous_config(
            start=1.5, target=5.5, duration_sec=1.0, update_interval=0.2
        )
        ramper = Ramper(setter=setter, config=config)
        await ramper.start()

        assert values[0] == 1.5
        assert values[-1] == 5.5
        # All values should be between start and target
        for v in values:
            assert 1.5 <= v <= 5.5

    @pytest.mark.asyncio
    async def test_ramp_down_continuous(self, time_traveler):
        """Should handle ramp-down in continuous mode."""
        values: list[float] = []
        setter = values.append

        config = continuous_config(
            start=100, target=1, duration_sec=1.0, update_interval=0.2
        )
        ramper = Ramper(setter=setter, config=config)
        await ramper.start()

        assert values[0] == 100.0
        assert values[-1] == 1.0
        # Values should be decreasing
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1]
