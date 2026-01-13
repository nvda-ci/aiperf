# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for timing factories - IntervalGeneratorFactory, RampStrategyFactory, TimingStrategyFactory.

These tests verify factory behavior: instance creation, error handling, and protocol conformance
of created instances. Tests are organized by behavior rather than implementation.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from aiperf.common.enums import ArrivalPattern, TimingMode
from aiperf.common.exceptions import FactoryCreationError
from aiperf.timing.intervals import (
    IntervalGeneratorConfig,
    IntervalGeneratorFactory,
    IntervalGeneratorProtocol,
)
from aiperf.timing.ramping import (
    RampConfig,
    RampStrategyFactory,
    RampStrategyProtocol,
    RampType,
)
from aiperf.timing.strategies.core import (
    TimingStrategyFactory,
    TimingStrategyProtocol,
)
from tests.unit.timing.conftest import make_phase_config

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def interval_config_constant():
    """Configuration for constant interval generator."""
    return IntervalGeneratorConfig(
        arrival_pattern=ArrivalPattern.CONSTANT,
        request_rate=10.0,
    )


@pytest.fixture
def interval_config_poisson():
    """Configuration for Poisson interval generator."""
    return IntervalGeneratorConfig(
        arrival_pattern=ArrivalPattern.POISSON,
        request_rate=10.0,
    )


@pytest.fixture
def interval_config_gamma():
    """Configuration for Gamma interval generator."""
    return IntervalGeneratorConfig(
        arrival_pattern=ArrivalPattern.GAMMA,
        request_rate=10.0,
        arrival_smoothness=2.0,
    )


@pytest.fixture
def interval_config_burst():
    """Configuration for concurrency burst interval generator."""
    return IntervalGeneratorConfig(
        arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
    )


@pytest.fixture
def ramp_config_linear():
    """Configuration for linear ramp strategy."""
    return RampConfig(
        ramp_type=RampType.LINEAR,
        start=1.0,
        target=10.0,
        duration_sec=5.0,
    )


@pytest.fixture
def ramp_config_exponential():
    """Configuration for exponential ramp strategy."""
    return RampConfig(
        ramp_type=RampType.EXPONENTIAL,
        start=1.0,
        target=10.0,
        duration_sec=5.0,
        exponent=2.0,
    )


@dataclass
class MockConversationSource:
    """Mock conversation source for testing."""

    conversations: list[Any] = field(default_factory=list)

    def next_conversation(self):
        """Return next conversation (mock implementation)."""
        return None if not self.conversations else self.conversations.pop(0)


@dataclass
class MockStopChecker:
    """Mock stop condition checker for testing."""

    can_send: bool = True
    can_start: bool = True

    def can_send_any_turn(self) -> bool:
        return self.can_send

    def can_start_new_session(self) -> bool:
        return self.can_start


@dataclass
class MockCreditIssuer:
    """Mock credit issuer for testing."""

    issued_credits: list = field(default_factory=list)

    async def issue_credit(self, **kwargs) -> None:
        self.issued_credits.append(kwargs)


@dataclass
class MockLifecycle:
    """Mock phase lifecycle for testing."""

    is_complete: bool = False
    is_sending_complete: bool = False
    started_at_perf_ns: int = 0

    def start(self) -> None:
        pass

    def mark_sending_complete(self) -> None:
        self.is_sending_complete = True

    def mark_complete(self) -> None:
        self.is_complete = True

    def cancel(self) -> None:
        pass


@dataclass
class MockScheduler:
    """Mock scheduler for testing - doesn't require a real event loop."""

    scheduled_tasks: list = field(default_factory=list)

    def schedule_later(self, delay: float, coro) -> None:
        self.scheduled_tasks.append((delay, coro))

    def cancel_all(self) -> None:
        self.scheduled_tasks.clear()


@pytest.fixture
def timing_strategy_deps():
    """Common dependencies for timing strategy creation (sync-compatible)."""
    return {
        "conversation_source": MockConversationSource(),
        "scheduler": MockScheduler(),
        "stop_checker": MockStopChecker(),
        "credit_issuer": MockCreditIssuer(),
        "lifecycle": MockLifecycle(),
    }


# =============================================================================
# IntervalGeneratorFactory Tests
# =============================================================================


class TestIntervalGeneratorFactoryCreation:
    """Test IntervalGeneratorFactory instance creation."""

    @pytest.mark.parametrize(
        "arrival_pattern,request_rate,expected_interval",
        [
            (ArrivalPattern.CONSTANT, 10.0, 0.1),  # 1/rate = 0.1s
            (ArrivalPattern.CONSTANT, 100.0, 0.01),  # 1/rate = 0.01s
            (ArrivalPattern.CONSTANT, 1.0, 1.0),  # 1/rate = 1s
        ],  # fmt: skip
    )
    def test_creates_constant_generator_with_correct_interval(
        self, arrival_pattern, request_rate, expected_interval
    ):
        """Factory creates constant generator with period = 1/rate."""
        config = IntervalGeneratorConfig(
            arrival_pattern=arrival_pattern,
            request_rate=request_rate,
        )
        generator = IntervalGeneratorFactory.create_instance(config)

        assert generator.next_interval() == pytest.approx(expected_interval)
        assert generator.rate == request_rate

    def test_creates_poisson_generator(self, interval_config_poisson):
        """Factory creates Poisson generator that produces variable intervals."""
        generator = IntervalGeneratorFactory.create_instance(interval_config_poisson)

        # Poisson generator should produce varying intervals (exponential distribution)
        intervals = [generator.next_interval() for _ in range(10)]
        assert generator.rate == 10.0
        # All intervals should be positive
        assert all(i > 0 for i in intervals)
        # Intervals should vary (not all identical like constant)
        assert len(set(intervals)) > 1

    def test_creates_gamma_generator_with_smoothness(self, interval_config_gamma):
        """Factory creates Gamma generator with configurable smoothness."""
        generator = IntervalGeneratorFactory.create_instance(interval_config_gamma)

        intervals = [generator.next_interval() for _ in range(10)]
        assert generator.rate == 10.0
        assert all(i > 0 for i in intervals)
        # Should have a smoothness property
        assert hasattr(generator, "smoothness")
        assert generator.smoothness == 2.0

    def test_creates_burst_generator_with_zero_interval(self, interval_config_burst):
        """Factory creates burst generator that always returns zero."""
        generator = IntervalGeneratorFactory.create_instance(interval_config_burst)

        # Burst mode: immediate issuance, no delay
        assert generator.next_interval() == 0
        assert generator.rate == 0.0  # Rate not applicable in burst mode

    @pytest.mark.parametrize(
        "arrival_pattern",
        [ArrivalPattern.CONSTANT, ArrivalPattern.POISSON, ArrivalPattern.GAMMA],
    )
    def test_validates_request_rate_required(self, arrival_pattern):
        """Factory raises FactoryCreationError (wrapping ValueError) when request_rate is missing."""
        config = IntervalGeneratorConfig(
            arrival_pattern=arrival_pattern,
            request_rate=None,  # Missing rate
        )

        with pytest.raises(FactoryCreationError) as exc_info:
            IntervalGeneratorFactory.create_instance(config)
        # Verify the underlying cause is ValueError with correct message
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "must be set and greater than 0" in str(exc_info.value.__cause__)

    @pytest.mark.parametrize(
        "arrival_pattern",
        [ArrivalPattern.CONSTANT, ArrivalPattern.POISSON, ArrivalPattern.GAMMA],
    )
    def test_validates_request_rate_positive(self, arrival_pattern):
        """Factory raises FactoryCreationError (wrapping ValueError) when request_rate is zero or negative."""
        for invalid_rate in [0.0, -1.0]:
            config = IntervalGeneratorConfig(
                arrival_pattern=arrival_pattern,
                request_rate=invalid_rate,
            )

            with pytest.raises(FactoryCreationError) as exc_info:
                IntervalGeneratorFactory.create_instance(config)
            assert isinstance(exc_info.value.__cause__, ValueError)
            assert "must be set and greater than 0" in str(exc_info.value.__cause__)

    def test_burst_generator_ignores_request_rate(self):
        """Burst generator accepts any request_rate (it's ignored)."""
        config = IntervalGeneratorConfig(
            arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
            request_rate=None,
        )
        generator = IntervalGeneratorFactory.create_instance(config)
        assert generator.next_interval() == 0


class TestIntervalGeneratorFactoryProtocolConformance:
    """Test that created generators conform to IntervalGeneratorProtocol."""

    @pytest.mark.parametrize(
        "config_fixture",
        [
            "interval_config_constant",
            "interval_config_poisson",
            "interval_config_gamma",
            "interval_config_burst",
        ],
    )
    def test_generator_implements_protocol(self, config_fixture, request):
        """All generators implement IntervalGeneratorProtocol."""
        config = request.getfixturevalue(config_fixture)
        generator = IntervalGeneratorFactory.create_instance(config)

        # Protocol requires: next_interval() -> float, rate property
        assert isinstance(generator, IntervalGeneratorProtocol)
        assert hasattr(generator, "next_interval")
        assert hasattr(generator, "rate")
        assert callable(generator.next_interval)

    @pytest.mark.parametrize(
        "config_fixture",
        [
            "interval_config_constant",
            "interval_config_poisson",
            "interval_config_gamma",
        ],
    )
    def test_rate_settable_generators_have_set_rate(self, config_fixture, request):
        """Rate-based generators have callable set_rate method (part of IntervalGeneratorProtocol)."""
        config = request.getfixturevalue(config_fixture)
        generator = IntervalGeneratorFactory.create_instance(config)

        # All generators should have set_rate as part of IntervalGeneratorProtocol
        assert hasattr(generator, "set_rate")
        assert callable(generator.set_rate)


class TestIntervalGeneratorFactoryDynamicRate:
    """Test dynamic rate adjustment via set_rate()."""

    def test_constant_generator_rate_update(self, interval_config_constant):
        """Constant generator updates interval when rate changes."""
        generator = IntervalGeneratorFactory.create_instance(interval_config_constant)

        # Initial rate 10.0 -> interval 0.1s
        assert generator.next_interval() == pytest.approx(0.1)

        # Update to 20.0 -> interval 0.05s
        generator.set_rate(20.0)
        assert generator.rate == 20.0
        assert generator.next_interval() == pytest.approx(0.05)

    def test_poisson_generator_rate_update(self, interval_config_poisson):
        """Poisson generator updates distribution when rate changes."""
        generator = IntervalGeneratorFactory.create_instance(interval_config_poisson)

        # Change rate and verify it's reflected
        generator.set_rate(100.0)
        assert generator.rate == 100.0
        # Higher rate should generally produce smaller intervals
        intervals = [generator.next_interval() for _ in range(100)]
        avg_interval = sum(intervals) / len(intervals)
        assert avg_interval < 0.1  # Should be ~0.01 on average for rate=100

    def test_gamma_generator_rate_update(self, interval_config_gamma):
        """Gamma generator updates distribution when rate changes."""
        generator = IntervalGeneratorFactory.create_instance(interval_config_gamma)

        generator.set_rate(50.0)
        assert generator.rate == 50.0

    def test_burst_generator_ignores_set_rate(self, interval_config_burst):
        """Burst generator's set_rate is a no-op."""
        generator = IntervalGeneratorFactory.create_instance(interval_config_burst)

        # Should not raise, but also should not change behavior
        generator.set_rate(100.0)
        assert generator.next_interval() == 0
        assert generator.rate == 0.0

    @pytest.mark.parametrize(
        "config_fixture",
        [
            "interval_config_constant",
            "interval_config_poisson",
            "interval_config_gamma",
        ],
    )
    def test_set_rate_validates_positive(self, config_fixture, request):
        """set_rate raises error for non-positive rates."""
        config = request.getfixturevalue(config_fixture)
        generator = IntervalGeneratorFactory.create_instance(config)

        with pytest.raises(ValueError, match="must be > 0"):
            generator.set_rate(0.0)

        with pytest.raises(ValueError, match="must be > 0"):
            generator.set_rate(-1.0)


class TestIntervalGeneratorFactoryErrors:
    """Test error handling for IntervalGeneratorFactory."""

    def test_none_config_raises_error(self):
        """Factory raises error when config is None."""
        with pytest.raises((AttributeError, TypeError)):
            IntervalGeneratorFactory.create_instance(None)  # type: ignore


# =============================================================================
# RampStrategyFactory Tests
# =============================================================================


class TestRampStrategyFactoryCreation:
    """Test RampStrategyFactory instance creation."""

    def test_creates_linear_strategy(self, ramp_config_linear):
        """Factory creates linear ramp strategy."""
        strategy = RampStrategyFactory.create_instance(ramp_config_linear)

        assert strategy.start == 1.0
        assert strategy.target == 10.0

    def test_creates_exponential_strategy(self, ramp_config_exponential):
        """Factory creates exponential ramp strategy."""
        strategy = RampStrategyFactory.create_instance(ramp_config_exponential)

        assert strategy.start == 1.0
        assert strategy.target == 10.0

    @pytest.mark.parametrize(
        "start,target,direction",
        [
            (1.0, 10.0, "increasing"),
            (10.0, 1.0, "decreasing"),
            (5.0, 5.0, "constant"),
        ],  # fmt: skip
    )
    def test_linear_handles_all_directions(self, start, target, direction):
        """Linear strategy handles increasing, decreasing, and constant ramps."""
        config = RampConfig(
            ramp_type=RampType.LINEAR,
            start=start,
            target=target,
            duration_sec=5.0,
        )
        strategy = RampStrategyFactory.create_instance(config)

        assert strategy.start == start
        assert strategy.target == target

        # Test next_step behavior
        result = strategy.next_step(start, 0.0)
        if direction == "constant":
            assert result is None  # No steps needed
        else:
            assert result is not None
            delay, next_val = result
            assert delay >= 0
            if direction == "increasing":
                assert next_val > start
            else:
                assert next_val < start


class TestRampStrategyFactoryProtocolConformance:
    """Test that created strategies conform to RampStrategyProtocol."""

    @pytest.mark.parametrize(
        "config_fixture",
        ["ramp_config_linear", "ramp_config_exponential"],
    )
    def test_strategy_implements_protocol(self, config_fixture, request):
        """All strategies implement RampStrategyProtocol."""
        config = request.getfixturevalue(config_fixture)
        strategy = RampStrategyFactory.create_instance(config)

        assert isinstance(strategy, RampStrategyProtocol)
        # Check protocol methods
        assert hasattr(strategy, "start")
        assert hasattr(strategy, "target")
        assert hasattr(strategy, "next_step")
        assert hasattr(strategy, "value_at")


class TestRampStrategyFactoryBehavior:
    """Test ramp strategy behavior."""

    def test_linear_discrete_stepping(self, ramp_config_linear):
        """Linear strategy steps discretely by 1.0."""
        strategy = RampStrategyFactory.create_instance(ramp_config_linear)

        # First step from 1.0 should go to 2.0
        result = strategy.next_step(1.0, 0.0)
        assert result is not None
        delay, next_val = result
        assert next_val == 2.0
        assert delay >= 0

    def test_linear_with_custom_step_size(self):
        """Linear strategy respects custom step size."""
        config = RampConfig(
            ramp_type=RampType.LINEAR,
            start=1.0,
            target=10.0,
            duration_sec=5.0,
            step_size=3.0,
        )
        strategy = RampStrategyFactory.create_instance(config)

        result = strategy.next_step(1.0, 0.0)
        assert result is not None
        _, next_val = result
        assert next_val == 4.0  # 1.0 + 3.0

    def test_linear_continuous_value_at(self, ramp_config_linear):
        """Linear value_at returns interpolated value at time T."""
        strategy = RampStrategyFactory.create_instance(ramp_config_linear)

        # At t=0, should be start
        val = strategy.value_at(0.0)
        assert val is not None
        assert val == pytest.approx(1.0)

        # At t=2.5 (halfway), should be ~5.5
        val = strategy.value_at(2.5)
        assert val is not None
        assert val == pytest.approx(5.5)

        # At t >= duration, returns None (complete)
        assert strategy.value_at(5.0) is None

    def test_exponential_ease_in_curve(self, ramp_config_exponential):
        """Exponential strategy applies ease-in curve."""
        strategy = RampStrategyFactory.create_instance(ramp_config_exponential)

        # Ease-in: slow start, fast end
        # At 50% time, value should be < 50% progress (< 5.5)
        val = strategy.value_at(2.5)
        assert val is not None
        # With exponent=2, at t=0.5*duration, value_progress = 0.5^2 = 0.25
        # value = 1 + (9 * 0.25) = 3.25
        assert val == pytest.approx(3.25)

    def test_exponential_validates_exponent(self):
        """Exponential strategy requires exponent > 1.0 via Pydantic validation."""
        from pydantic import ValidationError

        for invalid_exp in [1.0, 0.5, 0.0, -1.0]:
            with pytest.raises(ValidationError, match="greater than 1"):
                RampConfig(
                    ramp_type=RampType.EXPONENTIAL,
                    start=1.0,
                    target=10.0,
                    duration_sec=5.0,
                    exponent=invalid_exp,
                )

    def test_ramp_completion_detection(self, ramp_config_linear):
        """Strategy returns None when ramp is complete."""
        strategy = RampStrategyFactory.create_instance(ramp_config_linear)

        # At target value, next_step returns None
        assert strategy.next_step(10.0, 0.0) is None

        # After duration, value_at returns None
        assert strategy.value_at(10.0) is None


class TestRampStrategyFactoryErrors:
    """Test error handling for RampStrategyFactory."""

    def test_none_config_raises_error(self):
        """Factory raises error when config is None."""
        with pytest.raises((AttributeError, TypeError)):
            RampStrategyFactory.create_instance(None)  # type: ignore


# =============================================================================
# TimingStrategyFactory Tests
# =============================================================================


class TestTimingStrategyFactoryCreation:
    """Test TimingStrategyFactory instance creation."""

    def test_creates_request_rate_strategy(self, timing_strategy_deps):
        """Factory creates REQUEST_RATE timing strategy."""
        config = make_phase_config(
            timing_mode=TimingMode.REQUEST_RATE,
            request_rate=10.0,
            arrival_pattern=ArrivalPattern.POISSON,
            request_count=100,
        )

        strategy = TimingStrategyFactory.create_instance(
            timing_mode=TimingMode.REQUEST_RATE,
            config=config,
            **timing_strategy_deps,
        )

        assert isinstance(strategy, TimingStrategyProtocol)

    def test_creates_fixed_schedule_strategy(self, timing_strategy_deps):
        """Factory creates FIXED_SCHEDULE timing strategy."""
        config = make_phase_config(
            timing_mode=TimingMode.FIXED_SCHEDULE,
            request_count=100,
        )

        strategy = TimingStrategyFactory.create_instance(
            timing_mode=TimingMode.FIXED_SCHEDULE,
            config=config,
            **timing_strategy_deps,
        )

        assert isinstance(strategy, TimingStrategyProtocol)

    def test_creates_user_centric_rate_strategy(self, timing_strategy_deps):
        """Factory creates USER_CENTRIC_RATE timing strategy."""
        config = make_phase_config(
            timing_mode=TimingMode.USER_CENTRIC_RATE,
            request_rate=10.0,
            num_users=5,
            request_count=100,
        )

        strategy = TimingStrategyFactory.create_instance(
            timing_mode=TimingMode.USER_CENTRIC_RATE,
            config=config,
            **timing_strategy_deps,
        )

        assert isinstance(strategy, TimingStrategyProtocol)


class TestTimingStrategyFactoryProtocolConformance:
    """Test that created timing strategies conform to TimingStrategyProtocol."""

    @pytest.mark.parametrize(
        "timing_mode,extra_config",
        [
            (
                TimingMode.REQUEST_RATE,
                {"request_rate": 10.0, "arrival_pattern": ArrivalPattern.POISSON},
            ),
            (TimingMode.FIXED_SCHEDULE, {}),
            (TimingMode.USER_CENTRIC_RATE, {"request_rate": 10.0, "num_users": 5}),
        ],  # fmt: skip
    )
    def test_strategy_implements_protocol(
        self, timing_mode, extra_config, timing_strategy_deps
    ):
        """All timing strategies implement TimingStrategyProtocol."""
        config = make_phase_config(
            timing_mode=timing_mode,
            request_count=100,
            **extra_config,
        )

        strategy = TimingStrategyFactory.create_instance(
            timing_mode=timing_mode,
            config=config,
            **timing_strategy_deps,
        )

        assert isinstance(strategy, TimingStrategyProtocol)
        # Protocol requires setup_phase() and execute_phase() async methods
        assert hasattr(strategy, "setup_phase")
        assert hasattr(strategy, "execute_phase")
        assert asyncio.iscoroutinefunction(strategy.setup_phase)
        assert asyncio.iscoroutinefunction(strategy.execute_phase)


class TestTimingStrategyFactoryErrors:
    """Test error handling for TimingStrategyFactory."""

    def test_missing_dependencies_raises_error(self):
        """Factory raises error when required dependencies are missing."""
        config = make_phase_config(
            timing_mode=TimingMode.REQUEST_RATE,
            request_rate=10.0,
            arrival_pattern=ArrivalPattern.POISSON,
        )

        with pytest.raises((TypeError, FactoryCreationError)):
            # Missing all required dependencies
            TimingStrategyFactory.create_instance(
                timing_mode=TimingMode.REQUEST_RATE,
                config=config,
            )

    def test_unregistered_type_raises_error(self, timing_strategy_deps):
        """Factory raises FactoryCreationError for unregistered types."""
        config = make_phase_config(timing_mode=TimingMode.REQUEST_RATE)

        with pytest.raises(FactoryCreationError, match="No implementation registered"):
            TimingStrategyFactory.create_instance(
                timing_mode="not_a_real_timing_mode",  # type: ignore
                config=config,
                **timing_strategy_deps,
            )


# =============================================================================
# Factory Registry Tests
# =============================================================================


class TestFactoryRegistry:
    """Test factory registration and introspection."""

    def test_interval_generator_factory_has_all_patterns(self):
        """IntervalGeneratorFactory has all ArrivalPattern types registered."""
        registered_types = IntervalGeneratorFactory.get_all_class_types()

        for pattern in ArrivalPattern:
            assert pattern in registered_types, f"{pattern} not registered"

    def test_ramp_strategy_factory_has_all_types(self):
        """RampStrategyFactory has all RampType types registered."""
        registered_types = RampStrategyFactory.get_all_class_types()

        for ramp_type in RampType:
            assert ramp_type in registered_types, f"{ramp_type} not registered"

    def test_timing_strategy_factory_has_all_modes(self):
        """TimingStrategyFactory has all TimingMode types registered."""
        registered_types = TimingStrategyFactory.get_all_class_types()

        for mode in TimingMode:
            assert mode in registered_types, f"{mode} not registered"

    def test_get_class_from_type(self):
        """Factories can retrieve registered class types."""
        # Test each factory
        poisson_class = IntervalGeneratorFactory.get_class_from_type(
            ArrivalPattern.POISSON
        )
        assert poisson_class is not None

        linear_class = RampStrategyFactory.get_class_from_type(RampType.LINEAR)
        assert linear_class is not None

        request_rate_class = TimingStrategyFactory.get_class_from_type(
            TimingMode.REQUEST_RATE
        )
        assert request_rate_class is not None

    def test_get_all_classes(self):
        """Factories return all registered classes."""
        interval_classes = IntervalGeneratorFactory.get_all_classes()
        assert len(interval_classes) == len(ArrivalPattern)

        ramp_classes = RampStrategyFactory.get_all_classes()
        assert len(ramp_classes) == len(RampType)

        timing_classes = TimingStrategyFactory.get_all_classes()
        assert len(timing_classes) == len(TimingMode)


# =============================================================================
# Integration Tests - Factory + Protocol Chain
# =============================================================================


class TestFactoryProtocolIntegration:
    """Integration tests verifying factory-created instances work end-to-end."""

    @pytest.mark.parametrize(
        "arrival_pattern",
        [ArrivalPattern.CONSTANT, ArrivalPattern.POISSON, ArrivalPattern.GAMMA],
    )
    def test_interval_generator_produces_valid_intervals(self, arrival_pattern):
        """Factory-created generators produce valid intervals for timing loops."""
        config = IntervalGeneratorConfig(
            arrival_pattern=arrival_pattern,
            request_rate=10.0,
            arrival_smoothness=1.0 if arrival_pattern == ArrivalPattern.GAMMA else None,
        )
        generator = IntervalGeneratorFactory.create_instance(config)

        # Simulate a timing loop that would use these intervals
        total_time = 0.0
        for _ in range(100):
            interval = generator.next_interval()
            assert interval >= 0, f"Interval must be non-negative, got {interval}"
            total_time += interval

        # Average interval should be roughly 1/rate
        avg = total_time / 100
        # Poisson and Gamma will vary, but should be in reasonable range
        assert 0.05 <= avg <= 0.5, f"Average interval {avg} out of expected range"

    @pytest.mark.parametrize(
        "ramp_type",
        [RampType.LINEAR, RampType.EXPONENTIAL],
    )
    def test_ramp_strategy_reaches_target(self, ramp_type):
        """Factory-created ramp strategies reach target value."""
        config = RampConfig(
            ramp_type=ramp_type,
            start=1.0,
            target=10.0,
            duration_sec=5.0,
            exponent=2.0 if ramp_type == RampType.EXPONENTIAL else None,
        )
        strategy = RampStrategyFactory.create_instance(config)

        # Simulate ramping with discrete steps
        current = strategy.start
        elapsed = 0.0
        steps = 0
        max_steps = 1000  # Safety limit

        while steps < max_steps:
            result = strategy.next_step(current, elapsed)
            if result is None:
                break
            delay, next_val = result
            elapsed += delay
            current = next_val
            steps += 1

        assert current == strategy.target, f"Ramp did not reach target: {current}"

    def test_gamma_smoothness_affects_variance(self):
        """Gamma generator smoothness parameter affects interval variance."""
        # Lower smoothness = more bursty = higher variance
        bursty_config = IntervalGeneratorConfig(
            arrival_pattern=ArrivalPattern.GAMMA,
            request_rate=10.0,
            arrival_smoothness=0.5,
        )
        # Higher smoothness = more regular = lower variance
        smooth_config = IntervalGeneratorConfig(
            arrival_pattern=ArrivalPattern.GAMMA,
            request_rate=10.0,
            arrival_smoothness=5.0,
        )

        bursty_gen = IntervalGeneratorFactory.create_instance(bursty_config)
        smooth_gen = IntervalGeneratorFactory.create_instance(smooth_config)

        bursty_intervals = [bursty_gen.next_interval() for _ in range(1000)]
        smooth_intervals = [smooth_gen.next_interval() for _ in range(1000)]

        # Calculate variance
        def variance(data):
            mean = sum(data) / len(data)
            return sum((x - mean) ** 2 for x in data) / len(data)

        bursty_var = variance(bursty_intervals)
        smooth_var = variance(smooth_intervals)

        # Bursty should have higher variance
        assert bursty_var > smooth_var, (
            f"Bursty variance ({bursty_var}) should exceed smooth variance ({smooth_var})"
        )
