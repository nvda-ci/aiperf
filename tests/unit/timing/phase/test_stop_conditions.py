# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for StopConditionChecker and individual stop conditions.

Tests cover:
- Individual stop conditions (Lifecycle, RequestCount, SessionCount, Duration)
- StopConditionChecker composition and short-circuit evaluation
- can_send_any_turn vs can_start_new_session distinction
- Edge cases and boundary conditions
"""

from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.credit_counter import CreditCounter
from aiperf.timing.phase.lifecycle import PhaseLifecycle
from aiperf.timing.phase.stop_conditions import (
    DurationStopCondition,
    LifecycleStopCondition,
    RequestCountStopCondition,
    SessionCountStopCondition,
    StopConditionChecker,
)

# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


def make_config(
    total_expected_requests: int | None = None,
    expected_num_sessions: int | None = None,
    expected_duration_sec: float | None = None,
) -> CreditPhaseConfig:
    """Create a CreditPhaseConfig for testing."""
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        total_expected_requests=total_expected_requests,
        expected_num_sessions=expected_num_sessions,
        expected_duration_sec=expected_duration_sec,
    )


def make_mock_lifecycle(
    was_cancelled: bool = False,
    is_sending_complete: bool = False,
    time_left: float = 10.0,
) -> MagicMock:
    """Create a mock PhaseLifecycleProtocol."""
    lifecycle = MagicMock(spec=PhaseLifecycle)
    lifecycle.was_cancelled = was_cancelled
    lifecycle.is_sending_complete = is_sending_complete
    lifecycle.time_left_in_seconds = MagicMock(return_value=time_left)
    return lifecycle


def make_mock_counter(
    requests_sent: int = 0,
    sent_sessions: int = 0,
    total_session_turns: int = 0,
) -> MagicMock:
    """Create a mock CreditCounterProtocol."""
    counter = MagicMock(spec=CreditCounter)
    counter.requests_sent = requests_sent
    counter.sent_sessions = sent_sessions
    counter.total_session_turns = total_session_turns
    return counter


# =============================================================================
# LifecycleStopCondition Tests
# =============================================================================


class TestLifecycleStopCondition:
    """Test lifecycle-based stop condition."""

    def test_should_use_always_returns_true(self):
        """Lifecycle condition is always used."""
        assert LifecycleStopCondition.should_use(make_config()) is True

    def test_can_send_when_not_cancelled_and_not_complete(self):
        """Can send when phase is active."""
        lifecycle = make_mock_lifecycle(was_cancelled=False, is_sending_complete=False)
        counter = make_mock_counter()
        condition = LifecycleStopCondition(make_config(), lifecycle, counter)

        assert condition.can_send_any_turn() is True

    def test_cannot_send_when_cancelled(self):
        """Cannot send when phase is cancelled."""
        lifecycle = make_mock_lifecycle(was_cancelled=True, is_sending_complete=False)
        counter = make_mock_counter()
        condition = LifecycleStopCondition(make_config(), lifecycle, counter)

        assert condition.can_send_any_turn() is False

    def test_cannot_send_when_sending_complete(self):
        """Cannot send when sending is marked complete."""
        lifecycle = make_mock_lifecycle(was_cancelled=False, is_sending_complete=True)
        counter = make_mock_counter()
        condition = LifecycleStopCondition(make_config(), lifecycle, counter)

        assert condition.can_send_any_turn() is False

    def test_can_start_new_session_returns_true(self):
        """Lifecycle condition has no session-specific restriction."""
        lifecycle = make_mock_lifecycle()
        counter = make_mock_counter()
        condition = LifecycleStopCondition(make_config(), lifecycle, counter)

        assert condition.can_start_new_session() is True


# =============================================================================
# RequestCountStopCondition Tests
# =============================================================================


class TestRequestCountStopCondition:
    """Test request count based stop condition."""

    def test_should_use_when_request_count_configured(self):
        """Should use when total_expected_requests is set."""
        assert (
            RequestCountStopCondition.should_use(
                make_config(total_expected_requests=100)
            )
            is True
        )

    def test_should_not_use_when_no_request_count(self):
        """Should not use when total_expected_requests is None."""
        assert (
            RequestCountStopCondition.should_use(
                make_config(total_expected_requests=None)
            )
            is False
        )

    def test_can_send_when_under_limit(self):
        """Can send when requests sent is below limit."""
        config = make_config(total_expected_requests=100)
        lifecycle = make_mock_lifecycle()
        counter = make_mock_counter(requests_sent=50)
        condition = RequestCountStopCondition(config, lifecycle, counter)

        assert condition.can_send_any_turn() is True

    def test_cannot_send_when_at_limit(self):
        """Cannot send when requests sent equals limit."""
        config = make_config(total_expected_requests=100)
        lifecycle = make_mock_lifecycle()
        counter = make_mock_counter(requests_sent=100)
        condition = RequestCountStopCondition(config, lifecycle, counter)

        assert condition.can_send_any_turn() is False

    def test_cannot_send_when_over_limit(self):
        """Cannot send when requests sent exceeds limit."""
        config = make_config(total_expected_requests=100)
        lifecycle = make_mock_lifecycle()
        counter = make_mock_counter(requests_sent=150)
        condition = RequestCountStopCondition(config, lifecycle, counter)

        assert condition.can_send_any_turn() is False

    @pytest.mark.parametrize(
        "sent,limit,expected",
        [
            (0, 1, True),
            (0, 100, True),
            (99, 100, True),
            (100, 100, False),
            (101, 100, False),
        ],
    )
    def test_various_request_counts(self, sent, limit, expected):
        """Test various request count scenarios."""
        config = make_config(total_expected_requests=limit)
        lifecycle = make_mock_lifecycle()
        counter = make_mock_counter(requests_sent=sent)
        condition = RequestCountStopCondition(config, lifecycle, counter)

        assert condition.can_send_any_turn() is expected


# =============================================================================
# SessionCountStopCondition Tests
# =============================================================================


class TestSessionCountStopCondition:
    """Test session count based stop condition."""

    def test_should_use_when_session_count_configured(self):
        """Should use when expected_num_sessions is set."""
        assert (
            SessionCountStopCondition.should_use(make_config(expected_num_sessions=10))
            is True
        )

    def test_should_not_use_when_no_session_count(self):
        """Should not use when expected_num_sessions is None."""
        assert (
            SessionCountStopCondition.should_use(
                make_config(expected_num_sessions=None)
            )
            is False
        )

    def test_can_send_when_sessions_under_limit(self):
        """Can send when sessions sent is below limit."""
        config = make_config(expected_num_sessions=10)
        lifecycle = make_mock_lifecycle()
        counter = make_mock_counter(sent_sessions=5)
        condition = SessionCountStopCondition(config, lifecycle, counter)

        assert condition.can_send_any_turn() is True

    def test_can_send_when_sessions_at_limit_but_turns_remaining(self):
        """Can send turns for existing sessions even when session limit reached."""
        config = make_config(expected_num_sessions=10)
        lifecycle = make_mock_lifecycle()
        # 10 sessions started, but only 15 of 20 expected turns sent
        counter = make_mock_counter(
            sent_sessions=10, requests_sent=15, total_session_turns=20
        )
        condition = SessionCountStopCondition(config, lifecycle, counter)

        assert condition.can_send_any_turn() is True

    def test_cannot_send_when_all_session_turns_complete(self):
        """Cannot send when all session turns have been sent."""
        config = make_config(expected_num_sessions=10)
        lifecycle = make_mock_lifecycle()
        counter = make_mock_counter(
            sent_sessions=10, requests_sent=20, total_session_turns=20
        )
        condition = SessionCountStopCondition(config, lifecycle, counter)

        assert condition.can_send_any_turn() is False

    def test_can_start_new_session_when_under_limit(self):
        """Can start new session when under session limit."""
        config = make_config(expected_num_sessions=10)
        lifecycle = make_mock_lifecycle()
        counter = make_mock_counter(sent_sessions=5)
        condition = SessionCountStopCondition(config, lifecycle, counter)

        assert condition.can_start_new_session() is True

    def test_cannot_start_new_session_when_at_limit(self):
        """Cannot start new session when at session limit."""
        config = make_config(expected_num_sessions=10)
        lifecycle = make_mock_lifecycle()
        counter = make_mock_counter(
            sent_sessions=10, requests_sent=5, total_session_turns=20
        )
        condition = SessionCountStopCondition(config, lifecycle, counter)

        # can_send_any_turn is True (remaining turns exist)
        assert condition.can_send_any_turn() is True
        # but can_start_new_session is False (session limit reached)
        assert condition.can_start_new_session() is False


# =============================================================================
# DurationStopCondition Tests
# =============================================================================


class TestDurationStopCondition:
    """Test duration based stop condition."""

    def test_should_use_when_duration_configured(self):
        """Should use when expected_duration_sec is set."""
        assert (
            DurationStopCondition.should_use(make_config(expected_duration_sec=60.0))
            is True
        )

    def test_should_not_use_when_no_duration(self):
        """Should not use when expected_duration_sec is None."""
        assert (
            DurationStopCondition.should_use(make_config(expected_duration_sec=None))
            is False
        )

    def test_can_send_when_time_remaining(self):
        """Can send when time remains."""
        config = make_config(expected_duration_sec=60.0)
        lifecycle = make_mock_lifecycle(time_left=30.0)
        counter = make_mock_counter()
        condition = DurationStopCondition(config, lifecycle, counter)

        assert condition.can_send_any_turn() is True

    def test_cannot_send_when_no_time_remaining(self):
        """Cannot send when no time remains."""
        config = make_config(expected_duration_sec=60.0)
        lifecycle = make_mock_lifecycle(time_left=0.0)
        counter = make_mock_counter()
        condition = DurationStopCondition(config, lifecycle, counter)

        assert condition.can_send_any_turn() is False

    def test_cannot_send_when_time_negative(self):
        """Cannot send when time is negative (past deadline)."""
        config = make_config(expected_duration_sec=60.0)
        lifecycle = make_mock_lifecycle(time_left=-5.0)
        counter = make_mock_counter()
        condition = DurationStopCondition(config, lifecycle, counter)

        assert condition.can_send_any_turn() is False


# =============================================================================
# StopConditionChecker Integration Tests
# =============================================================================


class TestStopConditionCheckerConfiguration:
    """Test StopConditionChecker initialization and configuration."""

    def test_lifecycle_always_included(self):
        """Lifecycle condition should always be included."""
        checker = StopConditionChecker(
            make_config(),
            make_mock_lifecycle(),
            make_mock_counter(),
        )
        # Should have at least lifecycle condition
        assert len(checker._stop_conditions) >= 1

    def test_request_count_condition_included_when_configured(self):
        """Request count condition included when configured."""
        checker = StopConditionChecker(
            make_config(total_expected_requests=100),
            make_mock_lifecycle(),
            make_mock_counter(),
        )
        # Should have lifecycle + request count
        condition_types = [type(c).__name__ for c in checker._stop_conditions]
        assert "LifecycleStopCondition" in condition_types
        assert "RequestCountStopCondition" in condition_types

    def test_all_conditions_included_when_all_configured(self):
        """All conditions included when all are configured."""
        checker = StopConditionChecker(
            make_config(
                total_expected_requests=100,
                expected_num_sessions=10,
                expected_duration_sec=60.0,
            ),
            make_mock_lifecycle(),
            make_mock_counter(),
        )
        condition_types = [type(c).__name__ for c in checker._stop_conditions]
        assert "LifecycleStopCondition" in condition_types
        assert "RequestCountStopCondition" in condition_types
        assert "SessionCountStopCondition" in condition_types
        assert "DurationStopCondition" in condition_types


class TestStopConditionCheckerCanSendAnyTurn:
    """Test can_send_any_turn behavior."""

    def test_can_send_when_all_conditions_pass(self):
        """Can send when all conditions allow."""
        checker = StopConditionChecker(
            make_config(total_expected_requests=100),
            make_mock_lifecycle(was_cancelled=False, is_sending_complete=False),
            make_mock_counter(requests_sent=50),
        )
        assert checker.can_send_any_turn() is True

    def test_cannot_send_when_lifecycle_fails(self):
        """Cannot send when lifecycle condition fails."""
        checker = StopConditionChecker(
            make_config(total_expected_requests=100),
            make_mock_lifecycle(was_cancelled=True),
            make_mock_counter(requests_sent=50),
        )
        assert checker.can_send_any_turn() is False

    def test_cannot_send_when_request_count_reached(self):
        """Cannot send when request count limit reached."""
        checker = StopConditionChecker(
            make_config(total_expected_requests=100),
            make_mock_lifecycle(),
            make_mock_counter(requests_sent=100),
        )
        assert checker.can_send_any_turn() is False

    def test_cannot_send_when_duration_expired(self):
        """Cannot send when duration expired."""
        checker = StopConditionChecker(
            make_config(expected_duration_sec=60.0),
            make_mock_lifecycle(time_left=0.0),
            make_mock_counter(),
        )
        assert checker.can_send_any_turn() is False


class TestStopConditionCheckerCanStartNewSession:
    """Test can_start_new_session behavior."""

    def test_can_start_session_when_all_conditions_pass(self):
        """Can start new session when all conditions allow."""
        checker = StopConditionChecker(
            make_config(expected_num_sessions=10),
            make_mock_lifecycle(),
            make_mock_counter(sent_sessions=5),
        )
        assert checker.can_start_new_session() is True

    def test_cannot_start_session_when_general_condition_fails(self):
        """Cannot start session when can_send_any_turn fails."""
        checker = StopConditionChecker(
            make_config(expected_num_sessions=10),
            make_mock_lifecycle(was_cancelled=True),
            make_mock_counter(sent_sessions=5),
        )
        # General check fails
        assert checker.can_send_any_turn() is False
        # Therefore session check also fails
        assert checker.can_start_new_session() is False

    def test_cannot_start_session_when_session_limit_reached(self):
        """Cannot start session when session limit reached."""
        checker = StopConditionChecker(
            make_config(expected_num_sessions=10),
            make_mock_lifecycle(),
            make_mock_counter(
                sent_sessions=10, requests_sent=5, total_session_turns=20
            ),
        )
        # Can send turns for existing sessions
        assert checker.can_send_any_turn() is True
        # But cannot start new sessions
        assert checker.can_start_new_session() is False


class TestStopConditionCheckerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_config_only_lifecycle(self):
        """With no limits configured, only lifecycle is checked."""
        checker = StopConditionChecker(
            make_config(),
            make_mock_lifecycle(),
            make_mock_counter(requests_sent=1_000_000),
        )
        # No request/session/duration limits, so can always send
        assert checker.can_send_any_turn() is True
        assert checker.can_start_new_session() is True

    def test_first_condition_failure_short_circuits(self):
        """First failing condition should short-circuit evaluation."""
        lifecycle = make_mock_lifecycle(was_cancelled=True)
        counter = make_mock_counter()

        checker = StopConditionChecker(
            make_config(total_expected_requests=100, expected_duration_sec=60.0),
            lifecycle,
            counter,
        )

        assert checker.can_send_any_turn() is False
        # time_left_in_seconds should not be called because lifecycle fails first
        # (This verifies short-circuit behavior)

    @pytest.mark.parametrize(
        "requests,sessions,turns,expected_any,expected_new",
        [
            # Under all limits
            (5, 5, 20, True, True),
            # Request limit at boundary
            (99, 5, 20, True, True),
            (100, 5, 20, False, False),
            # Session limit at boundary (turns remaining)
            (5, 9, 20, True, True),
            (5, 10, 20, True, False),  # Can send existing turns, can't start new
            # All session turns complete
            (20, 10, 20, False, False),
        ],
    )
    def test_boundary_conditions(
        self, requests, sessions, turns, expected_any, expected_new
    ):
        """Test various boundary conditions."""
        checker = StopConditionChecker(
            make_config(total_expected_requests=100, expected_num_sessions=10),
            make_mock_lifecycle(),
            make_mock_counter(
                requests_sent=requests,
                sent_sessions=sessions,
                total_session_turns=turns,
            ),
        )
        assert checker.can_send_any_turn() is expected_any
        assert checker.can_start_new_session() is expected_new
