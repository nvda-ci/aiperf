# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for timing protocol conformance.

Validates that all implementations correctly conform to their protocols.
Uses runtime_checkable protocols for structural type checking.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import (
    ArrivalPattern,
    CreditPhase,
    TimingMode,
)
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.credit.structs import TurnToSend
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.intervals import (
    ConcurrencyBurstIntervalGenerator,
    ConstantIntervalGenerator,
    GammaIntervalGenerator,
    IntervalGeneratorConfig,
    IntervalGeneratorFactory,
    IntervalGeneratorProtocol,
    PoissonIntervalGenerator,
)
from aiperf.timing.phase.credit_counter import CreditCounter
from aiperf.timing.phase.lifecycle import PhaseLifecycle
from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
from aiperf.timing.phase.stop_conditions import StopConditionChecker
from aiperf.timing.ramping import (
    ExponentialStrategy,
    LinearStrategy,
    RampConfig,
    RampStrategyFactory,
    RampStrategyProtocol,
    RampType,
)
from aiperf.timing.strategies.core import (
    RateSettableProtocol,
    TimingStrategyFactory,
    TimingStrategyProtocol,
)
from aiperf.timing.strategies.request_rate import RequestRateStrategy

pytestmark = pytest.mark.looptime


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def phase_config():
    """Default phase config for testing."""
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        total_expected_requests=100,
        request_rate=10.0,
    )


@pytest.fixture
def interval_config():
    """Default interval generator config."""
    return IntervalGeneratorConfig(
        arrival_pattern=ArrivalPattern.POISSON,
        request_rate=10.0,
    )


@pytest.fixture
def ramp_config():
    """Default ramp config."""
    return RampConfig(
        ramp_type=RampType.LINEAR,
        start=1,
        target=100,
        duration_sec=10.0,
    )


@pytest.fixture
def mock_conversation_source():
    """Mock conversation source."""
    mock = MagicMock()
    mock.next = MagicMock()
    mock.get_metadata = MagicMock()
    mock.get_next_turn_metadata = MagicMock()
    return mock


@pytest.fixture
def mock_stop_checker():
    """Mock stop checker."""
    mock = MagicMock()
    mock.can_send_any_turn = MagicMock(return_value=True)
    mock.can_start_new_session = MagicMock(return_value=True)
    return mock


@pytest.fixture
def mock_credit_issuer():
    """Mock credit issuer."""
    from unittest.mock import AsyncMock

    mock = MagicMock()
    mock.issue_credit = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_lifecycle():
    """Mock lifecycle."""
    mock = MagicMock()
    mock.is_started = False
    mock.is_sending_complete = False
    mock.is_complete = False
    # Return a real float, not MagicMock (needed for comparisons in setup_phase)
    mock.time_left_in_seconds.return_value = 60.0
    return mock


# =============================================================================
# Test: IntervalGeneratorProtocol Conformance
# =============================================================================


class TestIntervalGeneratorProtocolConformance:
    """Verify all interval generators conform to IntervalGeneratorProtocol."""

    @pytest.mark.parametrize(
        "generator_class,arrival_pattern",
        [
            (PoissonIntervalGenerator, ArrivalPattern.POISSON),
            (GammaIntervalGenerator, ArrivalPattern.GAMMA),
            (ConstantIntervalGenerator, ArrivalPattern.CONSTANT),
            (ConcurrencyBurstIntervalGenerator, ArrivalPattern.CONCURRENCY_BURST),
        ],
    )
    def test_generator_isinstance_check(
        self, generator_class, arrival_pattern: ArrivalPattern
    ):
        """All generators should pass isinstance check for protocol."""
        config = IntervalGeneratorConfig(
            arrival_pattern=arrival_pattern,
            request_rate=10.0,
            arrival_smoothness=1.0,
        )
        generator = generator_class(config)
        assert isinstance(generator, IntervalGeneratorProtocol)

    @pytest.mark.parametrize(
        "arrival_pattern",
        [
            ArrivalPattern.POISSON,
            ArrivalPattern.GAMMA,
            ArrivalPattern.CONSTANT,
            ArrivalPattern.CONCURRENCY_BURST,
        ],
    )
    def test_generator_has_next_interval_method(self, arrival_pattern: ArrivalPattern):
        """All generators should have next_interval() method."""
        config = IntervalGeneratorConfig(
            arrival_pattern=arrival_pattern,
            request_rate=10.0,
        )
        generator = IntervalGeneratorFactory.create_instance(config)

        result = generator.next_interval()
        assert isinstance(result, int | float)  # ConcurrencyBurst returns int
        assert result >= 0.0

    @pytest.mark.parametrize(
        "arrival_pattern",
        [
            ArrivalPattern.POISSON,
            ArrivalPattern.GAMMA,
            ArrivalPattern.CONSTANT,
            ArrivalPattern.CONCURRENCY_BURST,
        ],
    )
    def test_generator_has_set_rate_method(self, arrival_pattern: ArrivalPattern):
        """All generators should have set_rate() method."""
        config = IntervalGeneratorConfig(
            arrival_pattern=arrival_pattern,
            request_rate=10.0,
        )
        generator = IntervalGeneratorFactory.create_instance(config)

        # Should not raise
        generator.set_rate(20.0)


# =============================================================================
# Test: RampStrategyProtocol Conformance
# =============================================================================


class TestRampStrategyProtocolConformance:
    """Verify all ramp strategies conform to RampStrategyProtocol."""

    @pytest.mark.parametrize(
        "strategy_class,ramp_type",
        [
            (LinearStrategy, RampType.LINEAR),
            (ExponentialStrategy, RampType.EXPONENTIAL),
        ],
    )
    def test_strategy_isinstance_check(self, strategy_class, ramp_type: RampType):
        """All ramp strategies should pass isinstance check for protocol."""
        config = RampConfig(
            ramp_type=ramp_type,
            start=1,
            target=100,
            duration_sec=10.0,
        )
        strategy = strategy_class(config)
        assert isinstance(strategy, RampStrategyProtocol)

    @pytest.mark.parametrize("ramp_type", [RampType.LINEAR, RampType.EXPONENTIAL])
    def test_strategy_has_start_property(self, ramp_type: RampType):
        """All ramp strategies should have start property."""
        config = RampConfig(
            ramp_type=ramp_type,
            start=5,
            target=100,
            duration_sec=10.0,
        )
        strategy = RampStrategyFactory.create_instance(config)
        assert strategy.start == 5

    @pytest.mark.parametrize("ramp_type", [RampType.LINEAR, RampType.EXPONENTIAL])
    def test_strategy_has_target_property(self, ramp_type: RampType):
        """All ramp strategies should have target property."""
        config = RampConfig(
            ramp_type=ramp_type,
            start=1,
            target=50,
            duration_sec=10.0,
        )
        strategy = RampStrategyFactory.create_instance(config)
        assert strategy.target == 50

    @pytest.mark.parametrize("ramp_type", [RampType.LINEAR, RampType.EXPONENTIAL])
    def test_strategy_has_next_step_method(self, ramp_type: RampType):
        """All ramp strategies should have next_step() method."""
        config = RampConfig(
            ramp_type=ramp_type,
            start=1,
            target=10,
            duration_sec=10.0,
        )
        strategy = RampStrategyFactory.create_instance(config)

        result = strategy.next_step(1, 0.0)
        assert result is None or (isinstance(result, tuple) and len(result) == 2)

    @pytest.mark.parametrize("ramp_type", [RampType.LINEAR, RampType.EXPONENTIAL])
    def test_strategy_has_value_at_method(self, ramp_type: RampType):
        """All ramp strategies should have value_at() method."""
        config = RampConfig(
            ramp_type=ramp_type,
            start=1,
            target=100,
            duration_sec=10.0,
        )
        strategy = RampStrategyFactory.create_instance(config)

        result = strategy.value_at(5.0)
        assert result is None or isinstance(result, int | float)


# =============================================================================
# Test: RateSettableProtocol Conformance
# =============================================================================


class TestRateSettableProtocolConformance:
    """Verify RequestRateStrategy conforms to RateSettableProtocol."""

    async def test_rate_strategy_isinstance_check(
        self,
        phase_config,
        mock_conversation_source,
        mock_stop_checker,
        mock_credit_issuer,
        mock_lifecycle,
    ):
        """RequestRateStrategy should pass isinstance check for RateSettableProtocol."""
        scheduler = LoopScheduler()
        strategy = RequestRateStrategy(
            config=phase_config,
            conversation_source=mock_conversation_source,
            scheduler=scheduler,
            stop_checker=mock_stop_checker,
            credit_issuer=mock_credit_issuer,
            lifecycle=mock_lifecycle,
        )
        assert isinstance(strategy, RateSettableProtocol)

    async def test_rate_strategy_has_set_request_rate_method(
        self,
        phase_config,
        mock_conversation_source,
        mock_stop_checker,
        mock_credit_issuer,
        mock_lifecycle,
    ):
        """RequestRateStrategy should have set_request_rate method."""
        scheduler = LoopScheduler()
        strategy = RequestRateStrategy(
            config=phase_config,
            conversation_source=mock_conversation_source,
            scheduler=scheduler,
            stop_checker=mock_stop_checker,
            credit_issuer=mock_credit_issuer,
            lifecycle=mock_lifecycle,
        )

        # Must call setup_phase first to initialize the rate generator
        await strategy.setup_phase()

        # Should not raise
        strategy.set_request_rate(20.0)


# =============================================================================
# Test: TimingStrategyProtocol Conformance
# =============================================================================


class TestTimingStrategyProtocolConformance:
    """Verify all timing modes conform to TimingStrategyProtocol."""

    @pytest.mark.parametrize(
        "timing_mode",
        [
            TimingMode.REQUEST_RATE,
            TimingMode.FIXED_SCHEDULE,
            TimingMode.USER_CENTRIC_RATE,
        ],
    )
    async def test_strategy_isinstance_check(
        self,
        mock_conversation_source,
        mock_stop_checker,
        mock_credit_issuer,
        mock_lifecycle,
        timing_mode: TimingMode,
    ):
        """All timing modes should pass isinstance check for protocol."""
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=timing_mode,
            total_expected_requests=100,
            request_rate=10.0,
            num_users=10,
        )
        scheduler = LoopScheduler()

        strategy = TimingStrategyFactory.create_instance(
            timing_mode,
            config=config,
            conversation_source=mock_conversation_source,
            scheduler=scheduler,
            stop_checker=mock_stop_checker,
            credit_issuer=mock_credit_issuer,
            lifecycle=mock_lifecycle,
        )

        assert isinstance(strategy, TimingStrategyProtocol)

    @pytest.mark.parametrize(
        "timing_mode",
        [
            TimingMode.REQUEST_RATE,
            TimingMode.FIXED_SCHEDULE,
            TimingMode.USER_CENTRIC_RATE,
        ],
    )
    async def test_strategy_has_setup_phase_method(
        self,
        mock_conversation_source,
        mock_stop_checker,
        mock_credit_issuer,
        mock_lifecycle,
        timing_mode: TimingMode,
    ):
        """All timing modes should have setup_phase method."""
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=timing_mode,
            total_expected_requests=100,
            request_rate=10.0,
            num_users=10,
        )
        scheduler = LoopScheduler()

        strategy = TimingStrategyFactory.create_instance(
            timing_mode,
            config=config,
            conversation_source=mock_conversation_source,
            scheduler=scheduler,
            stop_checker=mock_stop_checker,
            credit_issuer=mock_credit_issuer,
            lifecycle=mock_lifecycle,
        )

        # Verify protocol conformance: method exists and is async
        # (Not calling setup_phase() since strategies require properly configured deps)
        assert hasattr(strategy, "setup_phase")
        assert asyncio.iscoroutinefunction(strategy.setup_phase)

    @pytest.mark.parametrize(
        "timing_mode",
        [
            TimingMode.REQUEST_RATE,
            TimingMode.FIXED_SCHEDULE,
            TimingMode.USER_CENTRIC_RATE,
        ],
    )
    async def test_strategy_has_execute_phase_method(
        self,
        mock_conversation_source,
        mock_stop_checker,
        mock_credit_issuer,
        mock_lifecycle,
        timing_mode: TimingMode,
    ):
        """All timing modes should have execute_phase method."""
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=timing_mode,
            total_expected_requests=100,
            request_rate=10.0,
            num_users=10,
        )
        scheduler = LoopScheduler()

        strategy = TimingStrategyFactory.create_instance(
            timing_mode,
            config=config,
            conversation_source=mock_conversation_source,
            scheduler=scheduler,
            stop_checker=mock_stop_checker,
            credit_issuer=mock_credit_issuer,
            lifecycle=mock_lifecycle,
        )

        # Should have the method
        assert hasattr(strategy, "execute_phase")
        assert asyncio.iscoroutinefunction(strategy.execute_phase)

    @pytest.mark.parametrize(
        "timing_mode",
        [
            TimingMode.REQUEST_RATE,
            TimingMode.FIXED_SCHEDULE,
            TimingMode.USER_CENTRIC_RATE,
        ],
    )
    async def test_strategy_has_handle_credit_return_method(
        self,
        mock_conversation_source,
        mock_stop_checker,
        mock_credit_issuer,
        mock_lifecycle,
        timing_mode: TimingMode,
    ):
        """All timing modes should have handle_credit_return method."""
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=timing_mode,
            total_expected_requests=100,
            request_rate=10.0,
            num_users=10,
        )
        scheduler = LoopScheduler()

        strategy = TimingStrategyFactory.create_instance(
            timing_mode,
            config=config,
            conversation_source=mock_conversation_source,
            scheduler=scheduler,
            stop_checker=mock_stop_checker,
            credit_issuer=mock_credit_issuer,
            lifecycle=mock_lifecycle,
        )

        # Should have the method
        assert hasattr(strategy, "handle_credit_return")
        assert asyncio.iscoroutinefunction(strategy.handle_credit_return)


# =============================================================================
# Test: Protocol Contract Validation
# =============================================================================


class TestProtocolContracts:
    """Tests validating protocol contracts are upheld."""

    def test_increment_sent_is_atomic(self, phase_config):
        """increment_sent should not await (preserves atomicity)."""
        tracker = PhaseProgressTracker(phase_config)

        turn = TurnToSend(
            conversation_id="conv1",
            x_correlation_id="corr1",
            turn_index=0,
            num_turns=1,
        )

        # The method should return synchronously, not be a coroutine
        result = tracker.increment_sent(turn)
        assert not asyncio.iscoroutine(result)
        assert isinstance(result, tuple)

    def test_increment_returned_is_atomic(self, phase_config):
        """increment_returned should not await (preserves atomicity)."""
        tracker = PhaseProgressTracker(phase_config)

        turn = TurnToSend(
            conversation_id="conv1",
            x_correlation_id="corr1",
            turn_index=0,
            num_turns=1,
        )
        tracker.increment_sent(turn)
        tracker.freeze_sent_counts()

        # The method should return synchronously, not be a coroutine
        result = tracker.increment_returned(is_final_turn=True, cancelled=False)
        assert not asyncio.iscoroutine(result)
        assert isinstance(result, bool)

    def test_stop_checker_is_read_only(self, phase_config):
        """StopConditionChecker should not mutate lifecycle or counter."""
        lifecycle = PhaseLifecycle(phase_config)
        counter = CreditCounter(phase_config)
        checker = StopConditionChecker(
            config=phase_config, lifecycle=lifecycle, counter=counter
        )

        # Record initial state
        initial_is_started = lifecycle.is_started
        initial_requests_sent = counter.requests_sent

        # Call checker methods
        checker.can_send_any_turn()
        checker.can_start_new_session()

        # State should not have changed
        assert lifecycle.is_started == initial_is_started
        assert counter.requests_sent == initial_requests_sent


# =============================================================================
# Test: Negative Protocol Checks
# =============================================================================


class TestNegativeProtocolChecks:
    """Tests verifying non-conforming objects fail isinstance checks."""

    def test_plain_object_fails_protocol_check(self):
        """Plain object should not pass protocol isinstance check."""

        class NotAGenerator:
            pass

        obj = NotAGenerator()
        assert not isinstance(obj, IntervalGeneratorProtocol)
        assert not isinstance(obj, RampStrategyProtocol)
        assert not isinstance(obj, TimingStrategyProtocol)

    def test_partial_implementation_fails_check(self):
        """Partial implementation should fail protocol check."""

        class PartialGenerator:
            def next_interval(self) -> float:
                return 0.1

            # Missing set_rate

        obj = PartialGenerator()
        assert not isinstance(obj, IntervalGeneratorProtocol)
