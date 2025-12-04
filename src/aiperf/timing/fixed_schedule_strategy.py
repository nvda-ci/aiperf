# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from collections import defaultdict

from aiperf.common.constants import MILLIS_PER_SECOND
from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.common.models import CreditPhaseConfig, CreditPhaseStats, DatasetMetadata
from aiperf.timing.config import TimingManagerConfig
from aiperf.timing.credit_issuing_strategy import (
    CreditIssuingStrategy,
    CreditIssuingStrategyFactory,
)
from aiperf.timing.credit_manager import CreditManagerProtocol


def _perf_counter_ms() -> float:
    return time.perf_counter() * MILLIS_PER_SECOND


@CreditIssuingStrategyFactory.register(TimingMode.FIXED_SCHEDULE)
class FixedScheduleStrategy(CreditIssuingStrategy):
    """
    Class for fixed schedule credit issuing strategy.

    This strategy replays requests according to timestamps from a trace dataset,
    maintaining the original timing characteristics of the recorded traffic.
    """

    def __init__(
        self,
        config: TimingManagerConfig,
        credit_manager: CreditManagerProtocol,
        dataset_metadata: DatasetMetadata,
    ):
        # NOTE: This all needs to be set before the super call, because the base class will call
        # _setup_profiling_phase_config() which uses it to set the total expected requests.

        # Store dataset metadata for schedule creation
        self._dataset_metadata = dataset_metadata
        self._num_conversations = len(dataset_metadata.conversations)

        # Store config values
        self._auto_offset_timestamps = config.auto_offset_timestamps
        self._start_offset = config.fixed_schedule_start_offset
        self._schedule_zero_ms = 0

        super().__init__(
            config=config,
            credit_manager=credit_manager,
            dataset_metadata=dataset_metadata,
        )

    def _create_timestamp_groups(self) -> None:
        """
        Create a dictionary of timestamp groups from the dataset.

        Groups conversations by their rounded millisecond timestamp for efficient execution.
        All conversations within the same millisecond are batched together.

        Timestamps are rounded (not truncated) to the nearest millisecond using Python's
        built-in round() function, which uses banker's rounding (round half to even).
        This provides better accuracy than truncation for timestamp grouping.

        Performance: Grouping reduces time.perf_counter() syscalls from O(N conversations)
        to O(M unique milliseconds), significantly improving efficiency when many requests
        occur at similar timestamps (e.g., 1000 requests at ~100ms → 1 syscall instead of 1000).

        Raises:
            ValueError: If no conversations are found or if any conversation lacks timing data.
        """
        if self._num_conversations == 0:
            raise ValueError(
                "No schedule loaded, unable to setup fixed schedule strategy"
            )

        # Group conversations by rounded millisecond timestamp
        self._timestamp_groups = defaultdict(list[str])
        for conversation in self._dataset_metadata.conversations:
            if conversation.turns[0].timestamp_ms is not None:
                # Round to nearest millisecond to group similar timestamps together
                rounded_timestamp = round(conversation.turns[0].timestamp_ms)
                self._timestamp_groups[rounded_timestamp].append(
                    conversation.conversation_id
                )
            else:
                raise ValueError(
                    f"Conversation {conversation.conversation_id} has no timing data"
                )

        # Sort the timestamps for ordered execution
        self._sorted_timestamp_keys = sorted(self._timestamp_groups.keys())

        # Define the zero reference point for the schedule
        if self._auto_offset_timestamps:
            self._schedule_zero_ms = self._sorted_timestamp_keys[0]
        elif self._start_offset is not None:
            self._schedule_zero_ms = self._start_offset
        else:
            self._schedule_zero_ms = 0

    def _setup_profiling_phase_config(self) -> None:
        """
        Setup the profiling phase.

        Overrides the base implementation to set the total expected requests based on the number of requests in the schedule.
        """
        self._create_timestamp_groups()

        self.ordered_phase_configs.append(
            CreditPhaseConfig(
                type=CreditPhase.PROFILING,
                total_expected_requests=self._num_conversations,
            )
        )

    async def _execute_single_phase(self, phase_stats: CreditPhaseStats) -> None:
        """
        Execute the fixed schedule by issuing credits at their scheduled timestamps.

        Iterates through timestamp groups in chronological order, sleeping until each
        timestamp is reached, then issuing all credits in that group without additional delays.

        Timing precision is at the millisecond level. Sub-millisecond differences in
        the original trace are lost due to rounding, but this is acceptable for most
        replay scenarios where millisecond accuracy is sufficient.

        Args:
            phase_stats: Statistics tracking object for this execution phase.
        """
        # This is used as a reference point for the wait duration calculation
        start_time_ms = _perf_counter_ms()

        # Drop credits in order of the schedule
        for timestamp_ms in self._sorted_timestamp_keys:
            # Get the conversations at this timestamp group
            conversation_ids = self._timestamp_groups[timestamp_ms]

            # Calculate the wait duration for this timestamp
            # (timestamp_ms - schedule_zero_ms) is the offset from the start of the schedule
            # (_perf_counter_ms() - start_time_ms) is how much time has passed since we started
            wait_duration_ms = (timestamp_ms - self._schedule_zero_ms) - (
                _perf_counter_ms() - start_time_ms
            )
            wait_duration_sec = wait_duration_ms / MILLIS_PER_SECOND

            if wait_duration_sec > 0:
                await asyncio.sleep(wait_duration_sec)

            # Drop credits for all conversations at this timestamp
            for conversation_id in conversation_ids:
                should_cancel = self.cancellation_strategy.should_cancel_request()
                cancel_after_ns = self.cancellation_strategy.get_cancellation_delay_ns()

                await self.credit_manager.drop_credit(
                    credit_phase=CreditPhase.PROFILING,
                    credit_num=phase_stats.sent,
                    conversation_id=conversation_id,
                    # We already waited, so it can be sent ASAP
                    credit_drop_ns=None,
                    should_cancel=should_cancel,
                    cancel_after_ns=cancel_after_ns,
                )
                # NOTE: This is incremented here, as the credit_num is used up above, and needs the current value.
                phase_stats.sent += 1

        duration_sec = (_perf_counter_ms() - start_time_ms) / MILLIS_PER_SECOND
        self.info(
            f"Sent all {self._num_conversations:,} fixed schedule requests in {duration_sec:.2f}s. Waiting for responses..."
        )
