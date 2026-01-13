# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PhaseLifecycle state machine."""

import pytest

from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.lifecycle import PhaseLifecycle, PhaseState


@pytest.fixture
def minimal_config():
    """Create minimal phase config for lifecycle testing."""
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        request_rate=10.0,
    )


@pytest.fixture
def config_with_duration():
    """Create config with duration and grace period for time_left testing."""
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        request_rate=10.0,
        expected_duration_sec=60.0,
        grace_period_sec=10.0,
    )


class TestPhaseState:
    """Test PhaseState enum."""

    def test_all_states_exist(self, minimal_config):
        """All expected states should be defined."""
        assert PhaseState.CREATED.value == "created"
        assert PhaseState.STARTED.value == "started"
        assert PhaseState.SENDING_COMPLETE.value == "sending_complete"
        assert PhaseState.COMPLETE.value == "complete"

    def test_state_count(self, minimal_config):
        """Should have exactly 4 states."""
        assert len(PhaseState) == 4


class TestPhaseLifecycleInitialization:
    """Test PhaseLifecycle initialization."""

    def test_initial_state_is_created(self, minimal_config):
        """Should start in CREATED state."""
        lifecycle = PhaseLifecycle(minimal_config)
        assert lifecycle.state == PhaseState.CREATED

    def test_initial_timestamps_are_none(self, minimal_config):
        """All timestamps should be None initially."""
        lifecycle = PhaseLifecycle(minimal_config)
        assert lifecycle.started_at_ns is None
        assert lifecycle.sending_complete_at_ns is None
        assert lifecycle.complete_at_ns is None

    def test_initial_flags_are_false(self, minimal_config):
        """All flags should be False initially."""
        lifecycle = PhaseLifecycle(minimal_config)
        assert lifecycle.timeout_triggered is False
        assert lifecycle.grace_period_triggered is False
        assert lifecycle.was_cancelled is False


class TestPhaseLifecycleTransitions:
    """Test valid state transitions."""

    def test_created_to_started(self, minimal_config):
        """Should transition from CREATED to STARTED."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        assert lifecycle.state == PhaseState.STARTED
        assert lifecycle.started_at_ns is not None

    def test_started_to_sending_complete(self, minimal_config):
        """Should transition from STARTED to SENDING_COMPLETE."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.mark_sending_complete()
        assert lifecycle.state == PhaseState.SENDING_COMPLETE
        assert lifecycle.sending_complete_at_ns is not None

    def test_sending_complete_to_complete(self, minimal_config):
        """Should transition from SENDING_COMPLETE to COMPLETE."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.mark_sending_complete()
        lifecycle.mark_complete()
        assert lifecycle.state == PhaseState.COMPLETE
        assert lifecycle.complete_at_ns is not None

    def test_full_lifecycle(self, minimal_config):
        """Should complete full lifecycle."""
        lifecycle = PhaseLifecycle(minimal_config)

        lifecycle.start()
        assert lifecycle.is_started
        assert not lifecycle.is_sending_complete
        assert not lifecycle.is_complete

        lifecycle.mark_sending_complete()
        assert lifecycle.is_started
        assert lifecycle.is_sending_complete
        assert not lifecycle.is_complete

        lifecycle.mark_complete()
        assert lifecycle.is_started
        assert lifecycle.is_sending_complete
        assert lifecycle.is_complete


class TestPhaseLifecycleInvalidTransitions:
    """Test invalid state transitions raise errors."""

    def test_cannot_start_twice(self, minimal_config):
        """Starting twice should raise ValueError."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        with pytest.raises(ValueError, match="Credit phase already started"):
            lifecycle.start()

    def test_cannot_mark_sending_complete_before_start(self, minimal_config):
        """Cannot mark sending complete before starting."""
        lifecycle = PhaseLifecycle(minimal_config)
        with pytest.raises(ValueError, match="Credit phase not started"):
            lifecycle.mark_sending_complete()

    def test_cannot_mark_sending_complete_twice(self, minimal_config):
        """Cannot mark sending complete twice."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.mark_sending_complete()
        with pytest.raises(ValueError, match="already completed sending"):
            lifecycle.mark_sending_complete()

    def test_cannot_mark_complete_before_sending_complete(self, minimal_config):
        """Cannot mark complete before sending complete."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        with pytest.raises(ValueError, match="has not completed sending"):
            lifecycle.mark_complete()

    def test_cannot_mark_complete_from_created(self, minimal_config):
        """Cannot mark complete from CREATED state."""
        lifecycle = PhaseLifecycle(minimal_config)
        with pytest.raises(ValueError, match="has not completed sending"):
            lifecycle.mark_complete()

    def test_cannot_mark_complete_twice(self, minimal_config):
        """Cannot mark complete twice."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.mark_sending_complete()
        lifecycle.mark_complete()
        with pytest.raises(ValueError, match="Credit phase already completed"):
            lifecycle.mark_complete()


class TestPhaseLifecycleFlags:
    """Test timeout and cancellation flags."""

    def test_timeout_triggered_flag(self, minimal_config):
        """timeout_triggered should be set on mark_sending_complete."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.mark_sending_complete(timeout_triggered=True)
        assert lifecycle.timeout_triggered is True

    def test_timeout_triggered_default_false(self, minimal_config):
        """timeout_triggered should default to False."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.mark_sending_complete()
        assert lifecycle.timeout_triggered is False

    def test_grace_period_triggered_flag(self, minimal_config):
        """grace_period_triggered should be set on mark_complete."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.mark_sending_complete()
        lifecycle.mark_complete(grace_period_triggered=True)
        assert lifecycle.grace_period_triggered is True

    def test_grace_period_triggered_default_false(self, minimal_config):
        """grace_period_triggered should default to False."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.mark_sending_complete()
        lifecycle.mark_complete()
        assert lifecycle.grace_period_triggered is False


class TestPhaseLifecycleCancellation:
    """Test cancellation behavior."""

    def test_cancel_sets_flag(self, minimal_config):
        """cancel() should set was_cancelled flag."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.cancel()
        assert lifecycle.was_cancelled is True

    def test_cancel_from_created(self, minimal_config):
        """Can cancel from CREATED state."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.cancel()
        assert lifecycle.was_cancelled is True
        assert lifecycle.state == PhaseState.CREATED

    def test_cancel_from_started(self, minimal_config):
        """Can cancel from STARTED state."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.cancel()
        assert lifecycle.was_cancelled is True
        assert lifecycle.state == PhaseState.STARTED

    def test_cancel_from_sending_complete(self, minimal_config):
        """Can cancel from SENDING_COMPLETE state."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.mark_sending_complete()
        lifecycle.cancel()
        assert lifecycle.was_cancelled is True
        assert lifecycle.state == PhaseState.SENDING_COMPLETE

    def test_cancelled_phase_can_still_complete(self, minimal_config):
        """Cancelled phase should still be able to transition to COMPLETE."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.cancel()
        lifecycle.mark_sending_complete()
        lifecycle.mark_complete()
        assert lifecycle.was_cancelled is True
        assert lifecycle.state == PhaseState.COMPLETE


class TestPhaseLifecycleConvenienceProperties:
    """Test convenience properties."""

    def test_is_started_false_in_created(self, minimal_config):
        """is_started should be False in CREATED state."""
        lifecycle = PhaseLifecycle(minimal_config)
        assert lifecycle.is_started is False

    def test_is_started_true_after_start(self, minimal_config):
        """is_started should be True after start()."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        assert lifecycle.is_started is True

    def test_is_sending_complete_false_before_complete(self, minimal_config):
        """is_sending_complete should be False before marking sending complete."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        assert lifecycle.is_sending_complete is False

    def test_is_sending_complete_true_in_sending_complete_state(self, minimal_config):
        """is_sending_complete should be True in SENDING_COMPLETE state."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.mark_sending_complete()
        assert lifecycle.is_sending_complete is True

    def test_is_sending_complete_true_in_complete_state(self, minimal_config):
        """is_sending_complete should be True in COMPLETE state too."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.mark_sending_complete()
        lifecycle.mark_complete()
        assert lifecycle.is_sending_complete is True

    def test_is_complete_false_before_complete(self, minimal_config):
        """is_complete should be False before marking complete."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.mark_sending_complete()
        assert lifecycle.is_complete is False

    def test_is_complete_true_after_complete(self, minimal_config):
        """is_complete should be True after mark_complete()."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.mark_sending_complete()
        lifecycle.mark_complete()
        assert lifecycle.is_complete is True


class TestPhaseLifecycleTimestampOrdering:
    """Test that timestamps are recorded in order."""

    def test_timestamps_increase(self, minimal_config):
        """Timestamps should be recorded in increasing order."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.mark_sending_complete()
        lifecycle.mark_complete()

        assert lifecycle.started_at_ns is not None
        assert lifecycle.sending_complete_at_ns is not None
        assert lifecycle.complete_at_ns is not None

        assert lifecycle.started_at_ns <= lifecycle.sending_complete_at_ns
        assert lifecycle.sending_complete_at_ns <= lifecycle.complete_at_ns


class TestPhaseLifecycleTimeLeft:
    """Test time_left_in_seconds calculation."""

    def test_time_left_returns_none_without_duration(self, minimal_config):
        """Should return None when no duration configured."""
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        assert lifecycle.time_left_in_seconds() is None

    def test_time_left_returns_none_before_start(self, config_with_duration):
        """Should return None before phase starts."""
        lifecycle = PhaseLifecycle(config_with_duration)
        assert lifecycle.time_left_in_seconds() is None

    def test_time_left_returns_full_duration_at_start(self, config_with_duration):
        """Should return full duration immediately after start."""
        lifecycle = PhaseLifecycle(config_with_duration)
        lifecycle.start()
        time_left = lifecycle.time_left_in_seconds()
        assert time_left is not None
        assert time_left <= 60.0
        assert time_left >= 59.9  # Allow for small timing variance

    def test_time_left_decreases_over_time(self, config_with_duration):
        """Should decrease as time passes."""
        import time

        lifecycle = PhaseLifecycle(config_with_duration)
        lifecycle.start()

        first_check = lifecycle.time_left_in_seconds()
        time.sleep(0.1)
        second_check = lifecycle.time_left_in_seconds()

        assert first_check is not None
        assert second_check is not None
        assert second_check < first_check

    def test_time_left_returns_zero_when_elapsed(self, minimal_config):
        """Should return 0.0 when duration has elapsed."""
        # Create config with very short duration
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            request_rate=10.0,
            expected_duration_sec=0.001,  # 1ms
        )
        import time

        lifecycle = PhaseLifecycle(config)
        lifecycle.start()
        time.sleep(0.01)  # Wait 10ms

        time_left = lifecycle.time_left_in_seconds()
        assert time_left == 0.0

    def test_time_left_without_grace_period(self, config_with_duration):
        """Should not include grace period by default."""
        lifecycle = PhaseLifecycle(config_with_duration)
        lifecycle.start()

        time_left = lifecycle.time_left_in_seconds(include_grace_period=False)
        assert time_left is not None
        assert time_left <= 60.0

    def test_time_left_with_grace_period(self, config_with_duration):
        """Should include grace period when requested."""
        lifecycle = PhaseLifecycle(config_with_duration)
        lifecycle.start()

        without_grace = lifecycle.time_left_in_seconds(include_grace_period=False)
        with_grace = lifecycle.time_left_in_seconds(include_grace_period=True)

        assert without_grace is not None
        assert with_grace is not None
        # With grace should be approximately 10 seconds more
        assert with_grace > without_grace
        assert with_grace - without_grace >= 9.9  # Allow for timing variance

    def test_time_left_with_zero_grace_period(self, minimal_config):
        """Should handle None grace_period correctly."""
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            request_rate=10.0,
            expected_duration_sec=60.0,
            grace_period_sec=None,
        )
        lifecycle = PhaseLifecycle(config)
        lifecycle.start()

        without_grace = lifecycle.time_left_in_seconds(include_grace_period=False)
        with_grace = lifecycle.time_left_in_seconds(include_grace_period=True)

        # Should be approximately the same when grace_period_sec is None
        # (allow for microsecond timing variance between calls)
        assert without_grace == pytest.approx(with_grace, abs=0.001)
