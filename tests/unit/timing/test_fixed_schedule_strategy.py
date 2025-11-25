# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for fixed schedule credit issuing strategy.

Focused on testing the dual-queue architecture: absolute schedule + pending turns.

Key areas tested:
- Schedule building and data structure initialization
- Validation of timing data
- Credit numbering and completion detection
- Timestamp handling (floats, grouping, offsets)
"""

import time

import pytest

from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.common.exceptions import FactoryCreationError
from aiperf.common.models import CreditPhaseStats
from aiperf.timing import FixedScheduleStrategy, TimingManagerConfig
from tests.unit.timing.conftest import (
    MockCreditManager,
    create_mock_dataset_metadata,
    create_mock_dataset_metadata_with_schedule,
)
from tests.unit.utils.time_traveler import TimeTraveler


def create_strategy(
    mock_credit_manager: MockCreditManager,
    schedule: list[tuple[int | float, str]],
    auto_offset: bool = False,
    manual_offset: int | None = None,
) -> FixedScheduleStrategy:
    """Helper to create a strategy from a schedule."""
    config = TimingManagerConfig.model_construct(
        timing_mode=TimingMode.FIXED_SCHEDULE,
        auto_offset_timestamps=auto_offset,
        fixed_schedule_start_offset=manual_offset,
    )
    dataset_metadata = create_mock_dataset_metadata_with_schedule(schedule)
    return FixedScheduleStrategy(
        config=config,
        credit_manager=mock_credit_manager,
        dataset_metadata=dataset_metadata,
    )


class TestScheduleBuilding:
    """Test absolute schedule and pending queue construction."""

    def test_empty_schedule_raises_error(self, mock_credit_manager: MockCreditManager):
        """Empty dataset should raise FactoryCreationError."""
        with pytest.raises(
            FactoryCreationError, match="conversation_ids cannot be empty"
        ):
            create_strategy(mock_credit_manager, [])

    def test_single_turn_all_in_absolute_schedule(
        self, mock_credit_manager: MockCreditManager
    ):
        """Single-turn conversations go entirely into absolute schedule."""
        schedule = [(0, "conv1"), (100, "conv2"), (200, "conv3")]
        strategy = create_strategy(mock_credit_manager, schedule)

        assert len(strategy._absolute_schedule) == 3
        assert len(strategy._pending_turns) == 0
        assert strategy._total_expected_credits == 3

    def test_multi_turn_split_absolute_and_pending(
        self, mock_credit_manager: MockCreditManager
    ):
        """Multi-turn conversations split: first turn absolute, rest pending."""
        schedule = [(0, "conv1"), (100, "conv1"), (200, "conv1")]
        strategy = create_strategy(mock_credit_manager, schedule)

        # First turn only in absolute schedule
        assert len(strategy._absolute_schedule) == 1
        assert 0 in strategy._absolute_schedule

        # Subsequent turns in pending queue
        assert "conv1" in strategy._pending_turns
        assert len(strategy._pending_turns["conv1"]) == 2

        # Verify properties
        assert strategy._total_expected_credits == 3
        assert strategy._pending_turns["conv1"][0].turn_index == 1
        assert strategy._pending_turns["conv1"][1].turn_index == 2
        assert strategy._pending_turns["conv1"][1].is_final_turn

    def test_mixed_single_multi_turn(self, mock_credit_manager: MockCreditManager):
        """Mix of single and multi-turn conversations."""
        schedule = [
            (0, "single"),
            (50, "multi"),
            (150, "multi"),
        ]
        strategy = create_strategy(mock_credit_manager, schedule)

        assert len(strategy._absolute_schedule) == 2  # 2 first turns
        assert len(strategy._pending_turns) == 1  # Only multi has pending
        assert "multi" in strategy._pending_turns
        assert "single" not in strategy._pending_turns


class TestTimestampGrouping:
    """Test timestamp truncation and grouping behavior."""

    def test_sub_millisecond_grouping(self, mock_credit_manager: MockCreditManager):
        """Timestamps within same millisecond are grouped."""
        schedule = [
            (100.0, "conv1"),
            (100.1, "conv2"),
            (100.9, "conv3"),
        ]
        strategy = create_strategy(mock_credit_manager, schedule)

        # All truncate to 100ms
        assert len(strategy._absolute_schedule) == 1
        assert 100 in strategy._absolute_schedule
        assert len(strategy._absolute_schedule[100]) == 3

    def test_millisecond_boundaries(self, mock_credit_manager: MockCreditManager):
        """Test truncation at millisecond boundaries."""
        schedule = [
            (100.9, "conv1"),
            (101.0, "conv2"),
        ]
        strategy = create_strategy(mock_credit_manager, schedule)

        # Should be in different groups
        assert len(strategy._absolute_schedule) == 2
        assert 100 in strategy._absolute_schedule
        assert 101 in strategy._absolute_schedule

    def test_sorting_within_group(self, mock_credit_manager: MockCreditManager):
        """Turns within a group are sorted by nanosecond precision."""
        schedule = [
            (100.9, "conv1"),
            (100.1, "conv2"),
            (100.5, "conv3"),
        ]
        strategy = create_strategy(mock_credit_manager, schedule)

        group = strategy._absolute_schedule[100]
        assert group[0].conversation_id == "conv2"  # 100.1ms
        assert group[1].conversation_id == "conv3"  # 100.5ms
        assert group[2].conversation_id == "conv1"  # 100.9ms


class TestScheduleOffsets:
    """Test different offset configurations."""

    def test_no_offset_default(self, mock_credit_manager: MockCreditManager):
        """Default: no offset, zero = 0."""
        schedule = [(1000, "conv1")]
        strategy = create_strategy(mock_credit_manager, schedule)
        assert strategy._schedule_zero_ms == 0

    def test_auto_offset(self, mock_credit_manager: MockCreditManager):
        """Auto offset: zero = first timestamp."""
        schedule = [(1000, "conv1"), (1100, "conv2")]
        strategy = create_strategy(mock_credit_manager, schedule, auto_offset=True)
        assert strategy._schedule_zero_ms == 1000

    def test_manual_offset(self, mock_credit_manager: MockCreditManager):
        """Manual offset: zero = specified value."""
        schedule = [(1000, "conv1")]
        strategy = create_strategy(mock_credit_manager, schedule, manual_offset=500)
        assert strategy._schedule_zero_ms == 500


class TestValidation:
    """Test input validation."""

    def test_missing_first_turn_timestamp_raises(
        self, mock_credit_manager: MockCreditManager
    ):
        """First turn must have timestamp_ms."""
        config = TimingManagerConfig.model_construct(
            timing_mode=TimingMode.FIXED_SCHEDULE,
        )
        dataset_metadata = create_mock_dataset_metadata(
            conversation_ids=["conv1"],
            has_timing_data=True,
            first_turn_timestamps=[None],
            turn_counts=[1],
        )

        with pytest.raises(ValueError, match="invalid timing data"):
            FixedScheduleStrategy(
                config=config,
                credit_manager=mock_credit_manager,
                dataset_metadata=dataset_metadata,
            )

    def test_pending_turn_model_validation(self):
        """PendingTurn model validates that timestamp_ms or delay_ms is set."""
        from aiperf.timing.fixed_schedule_strategy import PendingTurn

        # Valid: has timestamp_ms
        turn1 = PendingTurn(
            conversation_id="conv1",
            turn_index=1,
            is_final_turn=False,
            timestamp_ms=100,
            delay_ms=None,
        )
        assert turn1.timestamp_ms == 100

        # Valid: has delay_ms
        turn2 = PendingTurn(
            conversation_id="conv1",
            turn_index=1,
            is_final_turn=False,
            timestamp_ms=None,
            delay_ms=50,
        )
        assert turn2.delay_ms == 50

        # Invalid: has neither
        with pytest.raises(
            ValueError, match="Either timestamp_ms or delay_ms must be set"
        ):
            PendingTurn(
                conversation_id="conv1",
                turn_index=1,
                is_final_turn=False,
                timestamp_ms=None,
                delay_ms=None,
            )


class TestSingleTurnExecution:
    """Test execution for simple single-turn scenarios."""

    @pytest.mark.asyncio
    async def test_simple_execution(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
    ):
        """Test basic single-turn execution."""
        schedule = [(0, "conv1"), (100, "conv2"), (200, "conv3")]
        strategy = create_strategy(mock_credit_manager, schedule)
        phase_stats = CreditPhaseStats(
            type=CreditPhase.PROFILING,
            start_ns=time.time_ns(),
            total_expected_requests=3,
        )
        conversation_provider = strategy.ordered_phase_configs[0][1]

        with time_traveler.sleeps_for(0.2):
            await strategy._execute_single_phase(phase_stats, conversation_provider)
            await strategy.wait_for_tasks()

        # All credits sent
        assert phase_stats.sent == 3
        assert len(mock_credit_manager.dropped_credits) == 3
        assert strategy._all_credits_sent.is_set()

    @pytest.mark.asyncio
    async def test_credit_numbering(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
    ):
        """Test sequential credit numbering."""
        schedule = [(0, "conv1"), (50, "conv2"), (100, "conv3")]
        strategy = create_strategy(mock_credit_manager, schedule)
        phase_stats = CreditPhaseStats(
            type=CreditPhase.PROFILING,
            start_ns=time.time_ns(),
            total_expected_requests=3,
        )
        conversation_provider = strategy.ordered_phase_configs[0][1]

        with time_traveler.sleeps_for(0.1):
            await strategy._execute_single_phase(phase_stats, conversation_provider)
            await strategy.wait_for_tasks()

        nums = [c.num for c in mock_credit_manager.dropped_credits]
        assert nums == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_is_final_turn_single_conversations(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
    ):
        """Test is_final_turn is True for single-turn conversations."""
        schedule = [(0, "conv1"), (100, "conv2")]
        strategy = create_strategy(mock_credit_manager, schedule)
        phase_stats = CreditPhaseStats(
            type=CreditPhase.PROFILING,
            start_ns=time.time_ns(),
            total_expected_requests=2,
        )
        conversation_provider = strategy.ordered_phase_configs[0][1]

        with time_traveler.sleeps_for(0.1):
            await strategy._execute_single_phase(phase_stats, conversation_provider)
            await strategy.wait_for_tasks()

        for credit in mock_credit_manager.dropped_credits:
            assert credit.is_final_turn  # All single-turn


class TestCompletionEvent:
    """Test _all_credits_sent event signaling."""

    @pytest.mark.asyncio
    async def test_event_set_after_all_sent(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
    ):
        """Event is set when last credit sent."""
        schedule = [(0, "conv1"), (100, "conv2")]
        strategy = create_strategy(mock_credit_manager, schedule)
        phase_stats = CreditPhaseStats(
            type=CreditPhase.PROFILING,
            start_ns=time.time_ns(),
            total_expected_requests=2,
        )
        conversation_provider = strategy.ordered_phase_configs[0][1]

        assert not strategy._all_credits_sent.is_set()

        with time_traveler.sleeps_for(0.1):
            await strategy._execute_single_phase(phase_stats, conversation_provider)
            await strategy.wait_for_tasks()

        assert strategy._all_credits_sent.is_set()

    @pytest.mark.asyncio
    async def test_event_set_on_last_credit(
        self,
        mock_credit_manager: MockCreditManager,
    ):
        """Event is set when _send_credit sends the final credit."""
        schedule = [(0, "conv1"), (100, "conv2")]
        strategy = create_strategy(mock_credit_manager, schedule)
        phase_stats = CreditPhaseStats(
            type=CreditPhase.PROFILING,
            start_ns=time.time_ns(),
            total_expected_credits=2,
        )

        # Send first credit
        turn = strategy._absolute_schedule[0][0]
        await strategy._send_credit(turn, phase_stats)
        assert not strategy._all_credits_sent.is_set()

        # Send second (final) credit
        turn = strategy._absolute_schedule[100][0]
        await strategy._send_credit(turn, phase_stats)
        assert strategy._all_credits_sent.is_set()


class TestPendingTurnStructure:
    """Test pending turn queue structure for multi-turn conversations."""

    def test_pending_turns_have_timing_data(
        self, mock_credit_manager: MockCreditManager
    ):
        """Pending turns must have timestamp_ms or delay_ms."""
        schedule = [(0, "conv1"), (100, "conv1")]
        strategy = create_strategy(mock_credit_manager, schedule)

        pending_turn = strategy._pending_turns["conv1"][0]
        # At least one must be set (validated by PendingTurn model)
        assert (
            pending_turn.timestamp_ms is not None or pending_turn.delay_ms is not None
        )

    def test_is_final_turn_flag_in_pending(
        self, mock_credit_manager: MockCreditManager
    ):
        """is_final_turn is set correctly in pending queue."""
        schedule = [(0, "conv1"), (100, "conv1"), (200, "conv1")]
        strategy = create_strategy(mock_credit_manager, schedule)

        assert not strategy._pending_turns["conv1"][0].is_final_turn  # Turn 1
        assert strategy._pending_turns["conv1"][1].is_final_turn  # Turn 2 (last)


# Summary of what's being tested:
# ✅ Schedule structure: absolute + pending separation
# ✅ Validation: missing timestamps, invalid data
# ✅ Completion detection: event signaling
# ✅ Credit numbering: sequential ordering
# ✅ Timestamp handling: floats, grouping, offsets
# ✅ Multi-turn structure: pending queues, is_final_turn
#
# NOT tested (requires complex async orchestration):
# ⏭  Full multi-turn execution flow (needs credit returns)
# ⏭  Delayed turn scheduling (needs background task coordination)
# ⏭  Concurrent scenarios (hard to test reliably without integration tests)
#
# These scenarios are better validated through integration tests or manual testing.
