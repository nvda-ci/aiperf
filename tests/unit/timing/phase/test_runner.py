# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PhaseRunner.

Tests phase lifecycle management, ramper creation, timeout handling,
and coordination between phase components.
"""

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import ArrivalPattern, CreditPhase, TimingMode
from aiperf.common.models import CreditPhaseStats
from aiperf.credit.structs import Credit
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.runner import PhaseRunner

pytestmark = pytest.mark.looptime


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockStrategy:
    """Mock timing strategy for testing PhaseRunner."""

    setup_called: bool = False
    execute_called: bool = False
    handle_credit_return_calls: list[Credit] = field(default_factory=list)
    execute_delay: float = 0.0
    _execute_event: asyncio.Event = field(default_factory=asyncio.Event)

    async def setup_phase(self) -> None:
        self.setup_called = True

    async def execute_phase(self) -> None:
        self.execute_called = True
        if self.execute_delay > 0:
            await asyncio.sleep(self.execute_delay)
        self._execute_event.set()

    async def handle_credit_return(self, credit: Credit) -> None:
        self.handle_credit_return_calls.append(credit)


@pytest.fixture
def mock_conversation_source():
    """Mock conversation source."""
    mock = MagicMock()
    mock.next = MagicMock()
    return mock


@pytest.fixture
def mock_phase_publisher():
    """Mock phase publisher with async methods."""
    mock = MagicMock()
    mock.publish_phase_start = AsyncMock()
    mock.publish_phase_sending_complete = AsyncMock()
    mock.publish_phase_complete = AsyncMock()
    mock.publish_progress = AsyncMock()
    mock.publish_credits_complete = AsyncMock()
    return mock


@pytest.fixture
def mock_credit_router():
    """Mock credit router."""
    mock = MagicMock()
    mock.send_credit = AsyncMock()
    mock.cancel_all_credits = AsyncMock()
    mock.mark_credits_complete = MagicMock()
    return mock


@pytest.fixture
def mock_concurrency_manager():
    """Mock concurrency manager."""
    mock = MagicMock()
    mock.configure_for_phase = MagicMock()
    mock.acquire_session_slot = AsyncMock(return_value=True)
    mock.acquire_prefill_slot = AsyncMock(return_value=True)
    mock.release_session_slot = MagicMock()
    mock.release_prefill_slot = MagicMock()
    mock.set_session_limit = MagicMock()
    mock.set_prefill_limit = MagicMock()
    mock.release_stuck_slots = MagicMock(return_value=(0, 0))
    return mock


@pytest.fixture
def mock_cancellation_policy():
    """Mock cancellation policy."""
    mock = MagicMock()
    mock.next_cancellation_delay_ns = MagicMock(return_value=None)
    return mock


@pytest.fixture
def mock_callback_handler():
    """Mock credit callback handler."""
    mock = MagicMock()
    mock.register_phase = MagicMock()
    mock.unregister_phase = MagicMock()
    mock.on_credit_return = AsyncMock()
    mock.on_first_token = AsyncMock()
    return mock


def make_phase_config(
    phase: CreditPhase = CreditPhase.PROFILING,
    timing_mode: TimingMode = TimingMode.REQUEST_RATE,
    request_count: int | None = 10,
    duration_sec: float | None = None,
    concurrency: int | None = None,
    prefill_concurrency: int | None = None,
    request_rate: float | None = 10.0,
    grace_period_sec: float | None = 1.0,
    seamless: bool = False,
    concurrency_ramp_duration_sec: float | None = None,
    prefill_concurrency_ramp_duration_sec: float | None = None,
    request_rate_ramp_duration_sec: float | None = None,
) -> CreditPhaseConfig:
    """Create a CreditPhaseConfig for testing."""
    return CreditPhaseConfig(
        phase=phase,
        timing_mode=timing_mode,
        total_expected_requests=request_count,
        expected_duration_sec=duration_sec,
        concurrency=concurrency,
        prefill_concurrency=prefill_concurrency,
        request_rate=request_rate,
        arrival_pattern=ArrivalPattern.POISSON,
        grace_period_sec=grace_period_sec,
        seamless=seamless,
        concurrency_ramp_duration_sec=concurrency_ramp_duration_sec,
        prefill_concurrency_ramp_duration_sec=prefill_concurrency_ramp_duration_sec,
        request_rate_ramp_duration_sec=request_rate_ramp_duration_sec,
    )


@pytest.fixture
async def phase_runner(
    mock_conversation_source,
    mock_phase_publisher,
    mock_credit_router,
    mock_concurrency_manager,
    mock_cancellation_policy,
    mock_callback_handler,
):
    """Create PhaseRunner with mocked dependencies (async for LoopScheduler)."""
    config = make_phase_config()
    return PhaseRunner(
        config=config,
        conversation_source=mock_conversation_source,
        phase_publisher=mock_phase_publisher,
        credit_router=mock_credit_router,
        concurrency_manager=mock_concurrency_manager,
        cancellation_policy=mock_cancellation_policy,
        callback_handler=mock_callback_handler,
    )


def create_phase_runner(
    config: CreditPhaseConfig,
    mock_conversation_source,
    mock_phase_publisher,
    mock_credit_router,
    mock_concurrency_manager,
    mock_cancellation_policy,
    mock_callback_handler,
) -> PhaseRunner:
    """Factory to create PhaseRunner with specific config."""
    return PhaseRunner(
        config=config,
        conversation_source=mock_conversation_source,
        phase_publisher=mock_phase_publisher,
        credit_router=mock_credit_router,
        concurrency_manager=mock_concurrency_manager,
        cancellation_policy=mock_cancellation_policy,
        callback_handler=mock_callback_handler,
    )


# =============================================================================
# Test: Basic Lifecycle
# =============================================================================


class TestPhaseRunnerLifecycle:
    """Tests for basic PhaseRunner lifecycle."""

    async def test_run_creates_strategy_via_factory(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
    ):
        """PhaseRunner should create timing strategy via factory."""
        config = make_phase_config()
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )

        # Mock the factory to capture creation
        mock_strategy = MockStrategy()
        with patch(
            "aiperf.timing.phase.runner.TimingStrategyFactory.create_instance",
            return_value=mock_strategy,
        ) as mock_factory:
            # Need to trigger completion to avoid hanging
            runner._progress.all_credits_sent_event.set()
            runner._progress.all_credits_returned_event.set()

            await runner.run(is_final_phase=True)

            mock_factory.assert_called_once()
            call_kwargs = mock_factory.call_args.kwargs
            assert call_kwargs["config"] == config
            assert "scheduler" in call_kwargs
            assert "stop_checker" in call_kwargs
            assert "credit_issuer" in call_kwargs
            assert "lifecycle" in call_kwargs

    async def test_run_registers_phase_with_callback_handler(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
    ):
        """PhaseRunner should register phase with callback handler before execution."""
        config = make_phase_config()
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )

        mock_strategy = MockStrategy()
        with patch(
            "aiperf.timing.phase.runner.TimingStrategyFactory.create_instance",
            return_value=mock_strategy,
        ):
            runner._progress.all_credits_sent_event.set()
            runner._progress.all_credits_returned_event.set()

            await runner.run(is_final_phase=True)

            mock_callback_handler.register_phase.assert_called_once()
            call_kwargs = mock_callback_handler.register_phase.call_args.kwargs
            assert call_kwargs["phase"] == CreditPhase.PROFILING
            assert call_kwargs["strategy"] == mock_strategy

    async def test_run_configures_concurrency_manager(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
    ):
        """PhaseRunner should configure concurrency manager for the phase."""
        config = make_phase_config(concurrency=10)
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )

        mock_strategy = MockStrategy()
        with patch(
            "aiperf.timing.phase.runner.TimingStrategyFactory.create_instance",
            return_value=mock_strategy,
        ):
            runner._progress.all_credits_sent_event.set()
            runner._progress.all_credits_returned_event.set()

            await runner.run(is_final_phase=True)

            mock_concurrency_manager.configure_for_phase.assert_called_once_with(
                config.phase, config.concurrency, config.prefill_concurrency
            )

    async def test_run_publishes_phase_start(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
    ):
        """PhaseRunner should publish phase start event."""
        config = make_phase_config()
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )

        mock_strategy = MockStrategy()
        with patch(
            "aiperf.timing.phase.runner.TimingStrategyFactory.create_instance",
            return_value=mock_strategy,
        ):
            runner._progress.all_credits_sent_event.set()
            runner._progress.all_credits_returned_event.set()

            await runner.run(is_final_phase=True)

            mock_phase_publisher.publish_phase_start.assert_called_once()

    async def test_run_publishes_phase_complete(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
    ):
        """PhaseRunner should publish phase complete event."""
        config = make_phase_config()
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )

        mock_strategy = MockStrategy()
        with patch(
            "aiperf.timing.phase.runner.TimingStrategyFactory.create_instance",
            return_value=mock_strategy,
        ):
            runner._progress.all_credits_sent_event.set()
            runner._progress.all_credits_returned_event.set()

            await runner.run(is_final_phase=True)

            mock_phase_publisher.publish_phase_complete.assert_called_once()

    async def test_run_returns_stats(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
    ):
        """PhaseRunner.run() should return CreditPhaseStats."""
        config = make_phase_config()
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )

        mock_strategy = MockStrategy()
        with patch(
            "aiperf.timing.phase.runner.TimingStrategyFactory.create_instance",
            return_value=mock_strategy,
        ):
            runner._progress.all_credits_sent_event.set()
            runner._progress.all_credits_returned_event.set()

            result = await runner.run(is_final_phase=True)

            assert isinstance(result, CreditPhaseStats)
            assert result.phase == CreditPhase.PROFILING


# =============================================================================
# Test: Ramper Creation
# =============================================================================


class TestRamperCreation:
    """Tests for ramper creation based on configuration."""

    async def test_no_rampers_created_without_ramp_duration(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
    ):
        """No rampers should be created when ramp durations are not set."""
        config = make_phase_config(
            concurrency=10,
            request_rate=100.0,
            concurrency_ramp_duration_sec=None,
            request_rate_ramp_duration_sec=None,
        )
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )

        mock_strategy = MockStrategy()
        with patch(
            "aiperf.timing.phase.runner.TimingStrategyFactory.create_instance",
            return_value=mock_strategy,
        ):
            runner._progress.all_credits_sent_event.set()
            runner._progress.all_credits_returned_event.set()

            await runner.run(is_final_phase=True)

            assert len(runner._rampers) == 0

    async def test_session_concurrency_ramper_created(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
    ):
        """Session concurrency ramper should be created when configured."""
        config = make_phase_config(
            concurrency=10,
            concurrency_ramp_duration_sec=5.0,
        )
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )

        mock_strategy = MockStrategy()
        with patch(
            "aiperf.timing.phase.runner.TimingStrategyFactory.create_instance",
            return_value=mock_strategy,
        ):
            runner._progress.all_credits_sent_event.set()
            runner._progress.all_credits_returned_event.set()

            await runner.run(is_final_phase=True)

            # Should have created a ramper
            assert len(runner._rampers) >= 1

    async def test_prefill_concurrency_ramper_created(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
    ):
        """Prefill concurrency ramper should be created when configured."""
        config = make_phase_config(
            prefill_concurrency=5,
            prefill_concurrency_ramp_duration_sec=3.0,
        )
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )
        # Manually set the prefill_concurrency on the config
        runner._config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=10,
            prefill_concurrency=5,
            prefill_concurrency_ramp_duration_sec=3.0,
            request_rate=10.0,
        )

        mock_strategy = MockStrategy()
        with patch(
            "aiperf.timing.phase.runner.TimingStrategyFactory.create_instance",
            return_value=mock_strategy,
        ):
            runner._progress.all_credits_sent_event.set()
            runner._progress.all_credits_returned_event.set()

            await runner.run(is_final_phase=True)

            # Should have created a ramper
            assert len(runner._rampers) >= 1

    async def test_rate_ramper_requires_rate_settable_strategy(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
    ):
        """Rate ramper should only be created for RateSettableProtocol strategies."""
        config = make_phase_config(
            request_rate=100.0,
            request_rate_ramp_duration_sec=10.0,
        )
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )

        # Strategy without RateSettableProtocol
        mock_strategy = MockStrategy()
        with patch(
            "aiperf.timing.phase.runner.TimingStrategyFactory.create_instance",
            return_value=mock_strategy,
        ):
            runner._progress.all_credits_sent_event.set()
            runner._progress.all_credits_returned_event.set()

            await runner.run(is_final_phase=True)

            # No ramper for rate since strategy doesn't implement RateSettableProtocol
            assert len(runner._rampers) == 0


# =============================================================================
# Test: Cancellation
# =============================================================================


class TestPhaseRunnerCancellation:
    """Tests for PhaseRunner cancellation behavior."""

    async def test_cancel_sets_was_cancelled_flag(self, phase_runner):
        """Cancellation should set the internal flag."""
        assert phase_runner._was_cancelled is False

        phase_runner.cancel()

        assert phase_runner._was_cancelled is True

    async def test_cancel_cancels_lifecycle(self, phase_runner):
        """Cancellation should cancel the lifecycle."""
        phase_runner.cancel()

        assert phase_runner._lifecycle.was_cancelled is True

    async def test_cancel_stops_rampers(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
    ):
        """Cancellation should stop all rampers."""
        config = make_phase_config(
            concurrency=10,
            concurrency_ramp_duration_sec=5.0,
        )
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )

        # Add mock rampers
        mock_ramper = MagicMock()
        runner._rampers = [mock_ramper]

        runner.cancel()

        mock_ramper.stop.assert_called_once()

    async def test_cancel_cancels_scheduler(self, phase_runner):
        """Cancellation should cancel all scheduled tasks."""

        # Schedule a coroutine to run later
        async def dummy_coro():
            await asyncio.sleep(10)  # Long enough that it won't fire before cancel

        phase_runner._scheduler.schedule_later(10.0, dummy_coro())

        # Verify something was scheduled
        assert phase_runner._scheduler.pending_count > 0

        phase_runner.cancel()

        # Scheduler pending timers should have been cancelled
        assert phase_runner._scheduler.pending_count == 0


# =============================================================================
# Test: Timeout Handling
# =============================================================================


class TestTimeoutHandling:
    """Tests for timeout behavior during phase execution."""

    async def test_wait_for_event_returns_false_when_event_set(
        self, phase_runner, time_traveler
    ):
        """Should return False when event is set before timeout."""
        event = asyncio.Event()
        event.set()

        result = await phase_runner._wait_for_event_with_timeout(
            name="test_event",
            event=event,
            timeout=10.0,
            task_to_cancel=None,
        )

        assert result is False

    async def test_wait_for_event_returns_true_on_timeout(
        self, phase_runner, time_traveler
    ):
        """Should return True when timeout occurs."""
        event = asyncio.Event()

        result = await phase_runner._wait_for_event_with_timeout(
            name="test_event",
            event=event,
            timeout=0.001,
            task_to_cancel=None,
        )

        assert result is True

    async def test_wait_for_event_cancels_task_on_timeout(
        self, phase_runner, time_traveler
    ):
        """Should cancel provided task on timeout."""
        event = asyncio.Event()
        # Use an event that's never set, so the task blocks until cancelled
        # (asyncio.sleep gets mocked by time_traveler and completes instantly)
        never_set = asyncio.Event()
        task = asyncio.create_task(never_set.wait())

        await phase_runner._wait_for_event_with_timeout(
            name="test_event",
            event=event,
            timeout=0.001,
            task_to_cancel=task,
        )

        # Allow the event loop to process the cancellation
        await asyncio.sleep(0)
        assert task.cancelled()

    async def test_wait_for_event_sets_event_on_timeout_when_configured(
        self, phase_runner, time_traveler
    ):
        """Should set event on timeout when set_event_on_timeout=True."""
        event = asyncio.Event()
        assert not event.is_set()

        await phase_runner._wait_for_event_with_timeout(
            name="test_event",
            event=event,
            timeout=0.001,
            task_to_cancel=None,
            set_event_on_timeout=True,
        )

        assert event.is_set()

    async def test_wait_for_event_returns_immediately_when_timeout_zero(
        self, phase_runner, time_traveler
    ):
        """Should return immediately when timeout is 0 or negative."""
        event = asyncio.Event()

        result = await phase_runner._wait_for_event_with_timeout(
            name="test_event",
            event=event,
            timeout=0,
            task_to_cancel=None,
        )

        assert result is True

    async def test_wait_for_event_waits_indefinitely_when_timeout_none(
        self, phase_runner, time_traveler
    ):
        """Should wait indefinitely when timeout is None."""
        event = asyncio.Event()

        async def set_event_later():
            await asyncio.sleep(0.01)
            event.set()

        asyncio.create_task(set_event_later())

        result = await phase_runner._wait_for_event_with_timeout(
            name="test_event",
            event=event,
            timeout=None,
            task_to_cancel=None,
        )

        assert result is False


# =============================================================================
# Test: Stuck Slots Release
# =============================================================================


class TestStuckSlotsRelease:
    """Tests for releasing stuck concurrency slots."""

    async def test_release_stuck_slots_calls_concurrency_manager(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
    ):
        """Should call concurrency manager to release stuck slots."""
        config = make_phase_config()
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )

        runner._release_stuck_slots()

        mock_concurrency_manager.release_stuck_slots.assert_called_once_with(
            CreditPhase.PROFILING
        )


# =============================================================================
# Test: Progress Reporting
# =============================================================================


class TestProgressReporting:
    """Tests for progress reporting loop."""

    async def test_progress_loop_publishes_stats(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
        time_traveler,
    ):
        """Progress loop should publish stats at regular intervals."""
        config = make_phase_config()
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )

        # Start progress loop and let it run briefly
        task = asyncio.create_task(runner._progress_report_loop())

        # Wait for a few progress reports
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Should have published at least once
        assert mock_phase_publisher.publish_progress.call_count >= 1

    async def test_progress_loop_handles_cancellation_gracefully(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
    ):
        """Progress loop should handle cancellation without errors."""
        config = make_phase_config()
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )

        task = asyncio.create_task(runner._progress_report_loop())
        await asyncio.sleep(0.001)
        task.cancel()

        # Should raise CancelledError when awaited
        with pytest.raises(asyncio.CancelledError):
            await task


# =============================================================================
# Test: Seamless Mode
# =============================================================================


class TestSeamlessMode:
    """Tests for seamless phase transitions."""

    async def test_seamless_mode_returns_early_for_non_final_phase(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
    ):
        """Seamless mode should return after sending complete for non-final phase."""
        config = make_phase_config(seamless=True)
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )

        mock_strategy = MockStrategy()
        with patch(
            "aiperf.timing.phase.runner.TimingStrategyFactory.create_instance",
            return_value=mock_strategy,
        ):
            runner._progress.all_credits_sent_event.set()
            # Don't set all_credits_returned_event - seamless mode should return anyway

            # Use asyncio.wait_for to avoid hanging
            result = await asyncio.wait_for(
                runner.run(is_final_phase=False), timeout=1.0
            )

            assert isinstance(result, CreditPhaseStats)

    async def test_seamless_mode_waits_for_returns_on_final_phase(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
    ):
        """Seamless mode should still wait for returns on final phase."""
        config = make_phase_config(seamless=True)
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )

        mock_strategy = MockStrategy()
        with patch(
            "aiperf.timing.phase.runner.TimingStrategyFactory.create_instance",
            return_value=mock_strategy,
        ):
            runner._progress.all_credits_sent_event.set()
            runner._progress.all_credits_returned_event.set()

            _ = await runner.run(is_final_phase=True)

            # Should have published phase complete
            mock_phase_publisher.publish_phase_complete.assert_called_once()


# =============================================================================
# Test: Component Ownership
# =============================================================================


class TestComponentOwnership:
    """Tests verifying correct component ownership."""

    def test_phase_runner_owns_scheduler(self, phase_runner):
        """PhaseRunner should own a LoopScheduler."""
        assert phase_runner._scheduler is not None

    def test_phase_runner_owns_lifecycle(self, phase_runner):
        """PhaseRunner should own a PhaseLifecycle."""
        assert phase_runner._lifecycle is not None

    def test_phase_runner_owns_progress_tracker(self, phase_runner):
        """PhaseRunner should own a PhaseProgressTracker."""
        assert phase_runner._progress is not None

    def test_phase_runner_owns_stop_checker(self, phase_runner):
        """PhaseRunner should own a StopConditionChecker."""
        assert phase_runner._stop_checker is not None

    def test_phase_runner_owns_credit_issuer(self, phase_runner):
        """PhaseRunner should own a CreditIssuer."""
        assert phase_runner._credit_issuer is not None

    def test_phase_property_returns_correct_phase(self, phase_runner):
        """Phase property should return the configured phase."""
        assert phase_runner.phase == CreditPhase.PROFILING


# =============================================================================
# Test: Phase Phases (WARMUP vs PROFILING)
# =============================================================================


class TestPhaseTypes:
    """Tests for different phase types."""

    @pytest.mark.parametrize("phase", [CreditPhase.WARMUP, CreditPhase.PROFILING])
    async def test_runner_works_with_both_phases(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
        phase: CreditPhase,
    ):
        """PhaseRunner should work with both WARMUP and PROFILING phases."""
        config = make_phase_config(phase=phase)
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )

        mock_strategy = MockStrategy()
        with patch(
            "aiperf.timing.phase.runner.TimingStrategyFactory.create_instance",
            return_value=mock_strategy,
        ):
            runner._progress.all_credits_sent_event.set()
            runner._progress.all_credits_returned_event.set()

            result = await runner.run(is_final_phase=True)

            assert result.phase == phase


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_already_complete_credits_returns_immediately(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
    ):
        """Should handle case where all credits already returned."""
        config = make_phase_config()
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )

        mock_strategy = MockStrategy()
        with patch(
            "aiperf.timing.phase.runner.TimingStrategyFactory.create_instance",
            return_value=mock_strategy,
        ):
            # Simulate credits already sent and returned
            runner._progress.all_credits_sent_event.set()
            runner._progress.all_credits_returned_event.set()
            runner._progress._counter._final_requests_sent = 0

            result = await runner.run(is_final_phase=True)

            assert isinstance(result, CreditPhaseStats)

    async def test_cleanup_runs_on_exception(
        self,
        mock_conversation_source,
        mock_phase_publisher,
        mock_credit_router,
        mock_concurrency_manager,
        mock_cancellation_policy,
        mock_callback_handler,
    ):
        """Cleanup should run even if an exception occurs."""
        config = make_phase_config()
        runner = create_phase_runner(
            config,
            mock_conversation_source,
            mock_phase_publisher,
            mock_credit_router,
            mock_concurrency_manager,
            mock_cancellation_policy,
            mock_callback_handler,
        )

        # Add a ramper to verify cleanup
        mock_ramper = MagicMock()
        runner._rampers = [mock_ramper]

        # Make strategy raise exception
        mock_strategy = MagicMock()
        mock_strategy.setup_phase = AsyncMock(side_effect=RuntimeError("Test error"))

        with patch(
            "aiperf.timing.phase.runner.TimingStrategyFactory.create_instance",
            return_value=mock_strategy,
        ):
            with pytest.raises(RuntimeError):
                await runner.run(is_final_phase=True)

            # Cleanup should have been called
            mock_ramper.stop.assert_called_once()
