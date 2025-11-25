# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from collections import defaultdict

from aiperf.common.constants import (
    MILLIS_PER_SECOND,
    NANOS_PER_MILLIS,
    NANOS_PER_SECOND,
)
from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.common.models import CreditPhaseConfig, CreditPhaseStats, DatasetMetadata
from aiperf.common.utils import yield_to_event_loop
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
    """

    def __init__(
        self,
        config: TimingManagerConfig,
        credit_manager: CreditManagerProtocol,
        dataset_metadata: DatasetMetadata,
    ):
        # NOTE: This all needs to be set before the super call, because the base class will call
        # _setup_profiling_phase_config() which uses it to set the total expected requests.

        # Reconstruct the full schedule from first_turn_timestamp and turn_delays
        self._schedule: list[tuple[int | float, str]] = []
        for conversation in dataset_metadata.conversations:
            if conversation.turns[0].timestamp_ms is not None:
                # Add first turn
                self._schedule.append(
                    (conversation.turns[0].timestamp_ms, conversation.conversation_id)
                )
            else:
                raise ValueError(
                    f"Conversation {conversation.conversation_id} has no timing data"
                )

        self._num_conversations = len(self._schedule)
        self._auto_offset_timestamps = config.auto_offset_timestamps
        self._start_offset = config.fixed_schedule_start_offset
        self._end_offset = config.fixed_schedule_end_offset
        self._schedule_zero_ms = 0
        self._schedule_zero_ns = 0
        super().__init__(
            config=config,
            credit_manager=credit_manager,
            dataset_metadata=dataset_metadata,
        )

    def _create_timestamp_groups(self) -> None:
        """
        Create a dictionary of timestamp groups, and filter the schedule to only include the requested subset.

        Note: Timestamps are rounded to the nearest millisecond to group conversations that are very close together.
        This prevents creating excessive sleep operations for timestamps that differ by fractions of a millisecond.
        """
        if not self._schedule or self._num_conversations == 0:
            raise ValueError(
                "No schedule loaded, unable to setup fixed schedule strategy"
            )
        # Group the schedule by timestamp, rounding to nearest millisecond to avoid
        # excessive grouping with floating point timestamps
        self._timestamp_groups = defaultdict(list[tuple[int, str]])
        for timestamp, conversation_id in self._schedule:
            # Truncate to nearest millisecond to group similar timestamps together
            truncated_timestamp = int(timestamp)
            self._timestamp_groups[truncated_timestamp].append(
                (int(timestamp * NANOS_PER_MILLIS), conversation_id)
            )

        # Sort the conversations by timestamp
        for _, conversation_tuples in self._timestamp_groups.items():
            conversation_tuples.sort(key=lambda x: x[0])

        # Sort the timestamps, so we can drop credits in order
        self._sorted_timestamp_keys = sorted(self._timestamp_groups.keys())

        # Define the zero reference point for the schedule
        if self._auto_offset_timestamps:
            self._schedule_zero_ms = self._sorted_timestamp_keys[0]
        elif self._start_offset is not None:
            self._schedule_zero_ms = self._start_offset
        else:
            self._schedule_zero_ms = 0
        self._schedule_zero_ns = int(self._schedule_zero_ms * NANOS_PER_MILLIS)

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
        # This is used as a reference point for the wait duration calculation
        start_time_ns = time.perf_counter_ns()
        start_time_ms = start_time_ns / NANOS_PER_MILLIS

        # Drop credits in order of the schedule
        for timestamp in self._sorted_timestamp_keys:
            # Get the conversations at this timestamp group now, so we are ready
            conversation_tuples = self._timestamp_groups[timestamp]
            # Calculate the wait duration for this timestamp
            # (timestamp - schedule_zero_ms) is the offset of the conversation(s) from the start of the schedule
            # (_perf_counter_ms() - start_time_ms) is how much time has passed since we started dropping credits
            wait_duration_ms = (timestamp - self._schedule_zero_ms) - (
                _perf_counter_ms() - start_time_ms
            )
            wait_duration_sec = wait_duration_ms / MILLIS_PER_SECOND

            if wait_duration_sec > 0:
                await asyncio.sleep(wait_duration_sec)

            # Drop credits asynchronously for all conversations at this timestamp
            for conversation_ns, conversation_id in conversation_tuples:
                # Use tight loop to wait for precise timestamp (sub-millisecond precision)
                while (
                    time.perf_counter_ns() - start_time_ns
                    < conversation_ns - self._schedule_zero_ns
                ):
                    await yield_to_event_loop()

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

        duration_sec = (time.perf_counter_ns() - start_time_ns) / NANOS_PER_SECOND
        self.info(
            f"Sent all {self._num_conversations:,} fixed schedule requests in {duration_sec:,.2f}s. Waiting for responses..."
        )
