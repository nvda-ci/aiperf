# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the timing manager fixed schedule strategy.
"""

import time

import pytest

from aiperf.common.constants import MILLIS_PER_SECOND
from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.common.models import CreditPhaseStats
from aiperf.timing import FixedScheduleStrategy, TimingManagerConfig
from tests.unit.timing.conftest import (
    MockCreditManager,
    create_mock_dataset_metadata_with_schedule,
)
from tests.unit.utils.time_traveler import TimeTraveler


class TestFixedScheduleStrategy:
    """Tests for the fixed schedule strategy."""

    @pytest.fixture
    def simple_schedule(self) -> list[tuple[int, str]]:
        """Simple schedule with 3 requests."""
        return [
            (0, "conv1"),
            (100, "conv2"),
            (200, "conv3"),
        ]

    @pytest.fixture
    def schedule_with_offset(self) -> list[tuple[int, str]]:
        """Schedule with auto offset."""
        return [(1000, "conv1"), (1100, "conv2"), (1200, "conv3")]

    def _create_strategy(
        self,
        mock_credit_manager: MockCreditManager,
        schedule: list[tuple[int, str]],
        auto_offset: bool = False,
        manual_offset: int | None = None,
    ) -> tuple[FixedScheduleStrategy, CreditPhaseStats]:
        """Helper to create a strategy with optional config overrides."""
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
        ), CreditPhaseStats(
            type=CreditPhase.PROFILING,
            start_ns=time.time_ns(),
            total_expected_requests=len(schedule),
        )

    def test_initialization_phase_configs(
        self,
        simple_schedule: list[tuple[int, str]],
        mock_credit_manager: MockCreditManager,
    ):
        """Test initialization creates correct phase configurations."""
        strategy, _ = self._create_strategy(mock_credit_manager, simple_schedule)

        assert len(strategy.ordered_phase_configs) == 1
        assert strategy._num_conversations == len(simple_schedule)

        # Check phase types - only profiling phase supported
        assert strategy.ordered_phase_configs[0].type == CreditPhase.PROFILING

    def test_empty_schedule_raises_error(self, mock_credit_manager: MockCreditManager):
        """Test that empty schedule raises ValueError."""
        with pytest.raises(ValueError, match="No schedule loaded"):
            self._create_strategy(mock_credit_manager, [])

    @pytest.mark.parametrize(
        "schedule,expected_groups,expected_keys",
        [
            (
                [(0, "conv1"), (100, "conv2"), (200, "conv3")],
                {
                    0: ["conv1"],
                    100: ["conv2"],
                    200: ["conv3"],
                },
                [0, 100, 200],
            ),
            (
                [(0, "conv1"), (0, "conv2"), (100, "conv3"), (100, "conv4")],
                {
                    0: ["conv1", "conv2"],
                    100: ["conv3", "conv4"],
                },
                [0, 100],
            ),
        ],
    )
    def test_timestamp_grouping(
        self,
        mock_credit_manager: MockCreditManager,
        schedule: list[tuple[int, str]],
        expected_groups: dict[int, list[str]],
        expected_keys: list[int],
    ):
        """Test that timestamps are properly grouped (rounded to nearest ms)."""
        strategy, _ = self._create_strategy(mock_credit_manager, schedule)

        assert strategy._timestamp_groups == expected_groups
        assert strategy._sorted_timestamp_keys == expected_keys

    @pytest.mark.parametrize(
        "auto_offset,manual_offset,expected_zero_ms",
        [
            (True, None, 1000),  # Auto offset to first timestamp
            (False, 500, 500),  # Manual offset
            (False, None, 0),  # No offset
        ],
    )
    def test_schedule_offset_configurations(
        self,
        mock_credit_manager: MockCreditManager,
        schedule_with_offset: list[tuple[int, str]],
        auto_offset: bool,
        manual_offset: int | None,
        expected_zero_ms: int,
    ):
        """Test different schedule offset configurations."""
        strategy, _ = self._create_strategy(
            mock_credit_manager, schedule_with_offset, auto_offset, manual_offset
        )

        assert strategy._schedule_zero_ms == expected_zero_ms

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "schedule,expected_duration",
        [
            ([(0, "conv1"), (100, "conv2"), (200, "conv3")], 0.2),  # 200ms total
            ([(0, "conv1"), (0, "conv2"), (0, "conv3")], 0.0),  # All at once
            ([(-100, "conv1"), (-50, "conv2"), (0, "conv3")], 0.0),  # Past timestamps
        ],
    )
    async def test_execution_timing(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
        schedule: list[tuple[int, str]],
        expected_duration: float,
    ):
        """Test that execution timing is correct for different schedules."""
        strategy, phase_stats = self._create_strategy(mock_credit_manager, schedule)

        with time_traveler.sleeps_for(expected_duration):
            await strategy._execute_single_phase(phase_stats)
            await strategy.wait_for_tasks()

        assert phase_stats.sent == len(schedule)
        assert len(mock_credit_manager.dropped_credits) == len(schedule)

        # Verify all conversation IDs were processed
        sent_conversations = {
            credit.conversation_id for credit in mock_credit_manager.dropped_credits
        }
        assert sent_conversations == {conv_id for _, conv_id in schedule}

    @pytest.mark.parametrize("auto_offset", [True, False])
    @pytest.mark.parametrize(
        "schedule",
        [
            [(1000, "conv1"), (1100, "conv2"), (1200, "conv3")],
            [(600, "conv1"), (700, "conv2"), (800, "conv3")],
            [(0, "conv1"), (100, "conv2"), (200, "conv3")],
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_execution_with_auto_offset(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
        auto_offset: bool,
        schedule: list[tuple[int, str]],
    ):
        """Test execution timing with auto offset timestamps."""
        strategy, phase_stats = self._create_strategy(
            mock_credit_manager, schedule, auto_offset
        )

        first_timestamp_ms = schedule[0][0]
        last_timestamp_ms = schedule[-1][0]

        sleep_duration_ms = (
            last_timestamp_ms - first_timestamp_ms if auto_offset else last_timestamp_ms
        )
        with time_traveler.sleeps_for(sleep_duration_ms / MILLIS_PER_SECOND):
            await strategy._execute_single_phase(phase_stats)
            await strategy.wait_for_tasks()

        assert phase_stats.sent == 3
        expected_zero_ms = first_timestamp_ms if auto_offset else 0
        assert strategy._schedule_zero_ms == expected_zero_ms

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_schedule(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
    ):
        """Test that multi-turn conversations only use first turn timestamp."""
        # Create a schedule with multi-turn conversations
        # conv1 has 3 turns: 0, 150, 200 -> only first turn (0) is used
        # conv2 has 2 turns: 100, 250 -> only first turn (100) is used
        schedule = [
            (0, "conv1"),
            (100, "conv2"),
            (
                150,
                "conv1",
            ),  # Second turn for conv1 (will be in metadata but not in schedule)
            (
                200,
                "conv1",
            ),  # Third turn for conv1 (will be in metadata but not in schedule)
            (
                250,
                "conv2",
            ),  # Second turn for conv2 (will be in metadata but not in schedule)
        ]

        strategy, phase_stats = self._create_strategy(mock_credit_manager, schedule)

        # Implementation only uses first turn of each conversation
        # So we expect 2 conversations (conv1 and conv2), not 5 turns
        assert strategy._num_conversations == 2
        # Check timestamp groups instead of _schedule
        assert len(strategy._timestamp_groups) == 2
        assert 0 in strategy._timestamp_groups and strategy._timestamp_groups[0] == [
            "conv1"
        ]
        assert 100 in strategy._timestamp_groups and strategy._timestamp_groups[
            100
        ] == ["conv2"]

        # Verify execution timing (should take 100ms total - only to second conversation)
        with time_traveler.sleeps_for(0.1):
            await strategy._execute_single_phase(phase_stats)
            await strategy.wait_for_tasks()

        assert phase_stats.sent == 2
        assert len(mock_credit_manager.dropped_credits) == 2

        # Verify only first turns were sent (one per conversation)
        sent_conv_ids = [
            credit.conversation_id for credit in mock_credit_manager.dropped_credits
        ]
        assert sent_conv_ids == ["conv1", "conv2"]

    @pytest.mark.asyncio
    async def test_floating_point_timestamps(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
    ):
        """Test that the system supports floating point timestamps (only first turn used)."""
        # Create a schedule with floating point timestamps (milliseconds with decimal places)
        # Simulating high-precision timing from real trace data
        schedule = [
            (0.0, "conv1"),
            (100.5, "conv2"),
            (150.75, "conv1"),  # Second turn for conv1 (metadata only)
            (200.123, "conv1"),  # Third turn for conv1 (metadata only)
            (250.456, "conv2"),  # Second turn for conv2 (metadata only)
        ]

        strategy, phase_stats = self._create_strategy(mock_credit_manager, schedule)

        # Implementation only uses first turn of each conversation
        assert strategy._num_conversations == 2
        # Check timestamp groups - 0.0 rounds to 0, 100.5 rounds to 100 (banker's rounding) or 101
        assert len(strategy._timestamp_groups) == 2
        # Verify conversations are grouped correctly (rounded to nearest ms)
        assert 0 in strategy._timestamp_groups and strategy._timestamp_groups[0] == [
            "conv1"
        ]
        # 100.5 rounds to 100 using banker's rounding (round half to even)
        assert 100 in strategy._timestamp_groups and strategy._timestamp_groups[
            100
        ] == ["conv2"]

        # Verify execution timing
        # Only waits for second conversation at 100.5ms (rounds to 100ms using banker's rounding)
        with time_traveler.sleeps_for(0.1):
            await strategy._execute_single_phase(phase_stats)
            await strategy.wait_for_tasks()

        assert phase_stats.sent == 2
        assert len(mock_credit_manager.dropped_credits) == 2

    @pytest.mark.asyncio
    async def test_mixed_int_and_float_timestamps(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
    ):
        """Test that the system supports both int and float timestamps (only first turn used)."""
        # Mixed int and float timestamps
        schedule = [
            (0, "conv1"),  # int - first turn of conv1
            (100.5, "conv2"),  # float - first turn of conv2
            (150, "conv1"),  # int - second turn of conv1 (metadata only)
            (200.75, "conv3"),  # float - first turn of conv3
        ]

        strategy, phase_stats = self._create_strategy(mock_credit_manager, schedule)

        # Implementation only uses first turn of each conversation
        # We have 3 unique conversations: conv1, conv2, conv3
        assert strategy._num_conversations == 3
        # Check timestamp groups - rounded to nearest ms
        # 0 -> 0, 100.5 -> 100, 200.75 -> 201
        assert len(strategy._timestamp_groups) == 3
        assert 0 in strategy._timestamp_groups and strategy._timestamp_groups[0] == [
            "conv1"
        ]
        assert 100 in strategy._timestamp_groups and strategy._timestamp_groups[
            100
        ] == ["conv2"]
        assert 201 in strategy._timestamp_groups and strategy._timestamp_groups[
            201
        ] == ["conv3"]

        # Verify execution completes successfully (201ms - last conversation at 200.75ms rounds to 201ms)
        with time_traveler.sleeps_for(0.201):
            await strategy._execute_single_phase(phase_stats)
            await strategy.wait_for_tasks()

        assert phase_stats.sent == 3

    @pytest.mark.asyncio
    async def test_timestamp_grouping_with_sub_millisecond_precision(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
    ):
        """Test that timestamps within 1ms are grouped together to avoid excessive sleep operations."""
        # Create a schedule with sub-millisecond timestamps (all should round to 100ms)
        schedule = [
            (100.0, "conv1"),
            (100.1, "conv2"),
            (100.2, "conv3"),
            (100.4, "conv4"),
            (100.499, "conv5"),  # Still rounds to 100
        ]

        strategy, phase_stats = self._create_strategy(mock_credit_manager, schedule)

        # All timestamps should be grouped into a single timestamp group
        assert len(strategy._timestamp_groups) == 1
        assert 100 in strategy._timestamp_groups
        assert len(strategy._timestamp_groups[100]) == 5

        # Should only have one sorted timestamp key
        assert len(strategy._sorted_timestamp_keys) == 1
        assert strategy._sorted_timestamp_keys[0] == 100

        # Execute and verify all credits dropped
        # With rounding, all conversations are at 100ms, so we sleep to 100ms then drop all
        with time_traveler.sleeps_for(0.1):
            await strategy._execute_single_phase(phase_stats)
            await strategy.wait_for_tasks()

        assert phase_stats.sent == 5
        assert len(mock_credit_manager.dropped_credits) == 5

        # Verify that all 5 conversations were dropped in a single batch
        # (no intermediate sleep operations between them)
        # This is the key benefit of grouping - they're all sent at the same timestamp

    @pytest.mark.asyncio
    async def test_floating_point_timestamps_rounding(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
    ):
        """Test that floating point timestamps are rounded to nearest millisecond for grouping."""
        # Use high-precision timestamps to verify rounding behavior
        schedule = [
            (0.123456789, "conv1"),  # rounds to 0
            (100.987654321, "conv2"),  # rounds to 101
            (200.111222333, "conv3"),  # rounds to 200
        ]

        strategy, phase_stats = self._create_strategy(mock_credit_manager, schedule)

        # Verify timestamp groups use rounded values
        assert len(strategy._timestamp_groups) == 3
        assert 0 in strategy._timestamp_groups and strategy._timestamp_groups[0] == [
            "conv1"
        ]
        assert 101 in strategy._timestamp_groups and strategy._timestamp_groups[
            101
        ] == ["conv2"]
        assert 200 in strategy._timestamp_groups and strategy._timestamp_groups[
            200
        ] == ["conv3"]

    @pytest.mark.asyncio
    async def test_timestamp_grouping_boundary_cases(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
    ):
        """Test timestamp grouping with rounding (not truncation)."""
        # Timestamps with rounding:
        # 100.4 -> rounds to 100
        # 100.5 -> rounds to 100 (banker's rounding - round half to even)
        # 100.6 -> rounds to 101
        # 101.1 -> rounds to 101
        schedule = [
            (100.4, "conv1"),
            (100.5, "conv2"),
            (100.6, "conv3"),
            (101.1, "conv4"),
        ]

        strategy, phase_stats = self._create_strategy(mock_credit_manager, schedule)

        # Should have 2 groups: 100 and 101
        assert len(strategy._timestamp_groups) == 2
        assert 100 in strategy._timestamp_groups
        assert 101 in strategy._timestamp_groups

        # Verify grouping with rounding
        assert (
            len(strategy._timestamp_groups[100]) == 2
        )  # conv1 (100.4->100), conv2 (100.5->100 banker's rounding)
        assert (
            len(strategy._timestamp_groups[101]) == 2
        )  # conv3 (100.6->101), conv4 (101.1->101)

    @pytest.mark.asyncio
    async def test_floating_point_precision_grouping(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
    ):
        """Test that floating point precision issues are handled by grouping."""
        # Timestamps that are effectively the same but differ due to floating point precision
        schedule = [
            (100.0, "conv1"),
            (100.0 + 1e-10, "conv2"),  # Essentially 100.0
            (100.0000000001, "conv3"),  # Essentially 100.0
        ]

        strategy, phase_stats = self._create_strategy(mock_credit_manager, schedule)

        # All should be grouped into a single timestamp
        assert len(strategy._timestamp_groups) == 1
        assert 100 in strategy._timestamp_groups
        assert len(strategy._timestamp_groups[100]) == 3
