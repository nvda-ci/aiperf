# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed schedule credit issuing strategy for trace replay.

This module implements a credit issuing strategy that replays multi-turn conversations with
precise timing, supporting both absolute timestamps and inter-turn delays. It is designed
for replaying real-world workload traces with high accuracy.

Key Concepts
------------
**Credit**: Permission to make a single request (turn) to an LLM endpoint.

**Turn**: A single interaction in a conversation (e.g., user sends message, receives response).

**Conversation**: Sequence of turns between a user and an LLM (can be single or multi-turn).

**Absolute Timestamp**: Specific time when a turn should be sent (e.g., 1500ms from schedule start).

**Inter-turn Delay**: Relative delay between consecutive turns (e.g., 100ms after previous turn completes).

Architecture
------------
The strategy uses a **dual-queue architecture**:

1. **Absolute Schedule** (dict): First turns with absolute timestamps
   - Sent sequentially by main loop at precise times
   - Example: {0: [conv1_turn0], 100: [conv2_turn0, conv3_turn0]}

2. **Pending Queue** (dict of deque): Subsequent turns per conversation
   - Sent dynamically when previous turn completes
   - Example: {"conv1": deque([turn1, turn2]), "conv2": deque([turn1])}

This separation allows:
- Precise timing for initial workload distribution
- Natural inter-turn delays based on actual completion times
- Efficient handling of conversations with varying turn counts

Timing Modes
------------
1. **Absolute Timestamps** (timestamp_ms):
   - Turn scheduled at specific time relative to schedule start
   - Use case: Replaying exact trace timing
   - Example: Turn at 1500ms means "send 1500ms after schedule starts"

2. **Inter-turn Delays** (delay_ms):
   - Turn scheduled X milliseconds after previous turn completes
   - Use case: Modeling think time / user interaction patterns
   - Example: delay=100ms means "send 100ms after previous turn returns"

3. **Mixed Mode**:
   - First turn: absolute timestamp (controls workload distribution)
   - Subsequent turns: delays or absolute timestamps (models user behavior)
   - Most common pattern in real traces

Thread Safety
-------------
No locks required because:
- asyncio is single-threaded (cooperative multitasking)
- Critical sections (credit numbering) have no await points
- Background tasks interleave safely via event loop

Performance Considerations
--------------------------
- Timestamp truncation: Groups nearby turns to reduce sleep operations
- Inline vs background: Zero-delay turns sent inline to avoid task overhead
- Early termination: Skips wait if no pending turns (optimization for single-turn workloads)
"""

import asyncio
import time
import uuid
from collections import defaultdict, deque

from pydantic import Field, model_validator
from typing_extensions import Self

from aiperf.common.constants import (
    MILLIS_PER_SECOND,
    NANOS_PER_MILLIS,
    NANOS_PER_SECOND,
)
from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.common.messages import CreditReturnMessage
from aiperf.common.messages.credit_messages import Credit
from aiperf.common.models import (
    AIPerfBaseModel,
    CreditPhaseConfig,
    CreditPhaseStats,
    DatasetMetadata,
)
from aiperf.common.utils import yield_to_event_loop
from aiperf.timing.config import TimingManagerConfig
from aiperf.timing.conversation_provider import (
    BaseConversationProvider,
    LiveConversationProvider,
)
from aiperf.timing.credit_issuing_strategy import (
    CreditIssuingStrategy,
    CreditIssuingStrategyFactory,
)
from aiperf.timing.credit_manager import CreditManagerProtocol


def _perf_counter_ms() -> float:
    return time.perf_counter() * MILLIS_PER_SECOND


class BaseTurn(AIPerfBaseModel):
    """Base turn model containing common fields for all turn types.

    This is the base class for AbsoluteTurn (first turns in absolute schedule)
    and PendingTurn (subsequent turns in pending queue).

    Attributes:
        conversation_id: Unique identifier for the conversation this turn belongs to.
        turn_index: Zero-based index of this turn within the conversation.
        is_final_turn: True if this is the last turn in the conversation (used for cleanup).
    """

    conversation_id: str = Field(
        description="The ID of the conversation that this turn belongs to."
    )
    turn_index: int = Field(
        ge=0, description="The index of the turn in the conversation (0-based)."
    )
    is_final_turn: bool = Field(
        description="True if this is the last turn of the conversation."
    )


class AbsoluteTurn(BaseTurn):
    """Turn scheduled at an absolute timestamp (used in absolute schedule).

    These are first turns (turn_index=0) that are sent by the main loop at specific times.
    Organized in groups by truncated timestamp for efficient scheduling.

    Attributes:
        timestamp_ns: Absolute timestamp in nanoseconds (high precision for sub-ms timing).
    """

    timestamp_ns: int = Field(
        description="The absolute timestamp of the turn in nanoseconds."
    )


class PendingTurn(BaseTurn):
    """Turn scheduled dynamically after previous turn completes (used in pending queue).

    These are subsequent turns (turn_index > 0) that are sent when the previous turn returns.
    Can have either an absolute timestamp or a relative delay.

    Attributes:
        timestamp_ms: Absolute timestamp in milliseconds (if using absolute scheduling).
        delay_ms: Delay in milliseconds from previous turn completion (if using relative scheduling).

    Validation:
        At least one of timestamp_ms or delay_ms must be set (enforced by model_validator).
    """

    timestamp_ms: int | float | None = Field(
        default=None,
        ge=0,
        description="The absolute timestamp of the turn in milliseconds.",
    )
    delay_ms: int | float | None = Field(
        default=None,
        ge=0,
        description="The delay of the turn in the conversation (in milliseconds).",
    )

    @model_validator(mode="after")
    def validate_timestamp_or_delay(self) -> Self:
        """Ensure either timestamp_ms or delay_ms is set."""
        if self.timestamp_ms is None and self.delay_ms is None:
            raise ValueError("Either timestamp_ms or delay_ms must be set")
        return self


@CreditIssuingStrategyFactory.register(TimingMode.FIXED_SCHEDULE)
class FixedScheduleStrategy(CreditIssuingStrategy):
    """Fixed schedule credit issuing strategy for replaying traced workloads.

    This strategy replays multi-turn conversations with precise timing, supporting both absolute
    timestamps and inter-turn delays. It is designed to accurately reproduce real-world workload
    patterns captured in trace data.

    Architecture Overview
    ---------------------
    The strategy uses a two-phase approach to handle both absolute and relative timing:

    1. **Absolute Schedule** (_absolute_schedule):
       - Contains all first turns with absolute timestamps
       - Organized as: {timestamp_ms: [turn1, turn2, ...]}
       - Main loop sends these sequentially at precise times

    2. **Pending Turn Queue** (_pending_turns):
       - Contains subsequent turns (turn 1, 2, 3...) for each conversation
       - Organized as: {conversation_id: deque([turn1, turn2, ...])}
       - Turns are sent dynamically as previous turns complete

    ASCII Architecture Diagram
    --------------------------
    ::

        INITIALIZATION
        ==============
        Dataset Metadata (conversations with turns)
                 |
                 v
        ┌────────────────────────────────────────────────────────────────┐
        │ _create_absolute_schedule()                                    │
        │                                                                │
        │  For each conversation:                                        │
        │    • Turn 0 → Absolute Schedule (by timestamp_ms)              │
        │    • Turn 1+ → Pending Queue (by conversation_id)              │
        └────────────────────────────────────────────────────────────────┘
                 |
                 v
        ┌─────────────────────┐     ┌──────────────────────────────┐
        │ Absolute Schedule   │     │ Pending Queue                │
        │ =================== │     │ =========================    │
        │ {                   │     │ {                            │
        │   0ms:  [C1_T0]     │     │   "conv1": [C1_T1, C1_T2]    │
        │   50ms: [C2_T0]     │     │   "conv2": [C2_T1]           │
        │   100ms:[C3_T0]     │     │   "conv3": [C3_T1]           │
        │ }                   │     │ }                            │
        └─────────────────────┘     └──────────────────────────────┘


        EXECUTION FLOW
        ==============

        Main Loop (_execute_single_phase)          Credit Returns (_on_credit_return)
        =====================================      ====================================

        t=0ms:  Send C1_T0 ───────────────┐
                                          │ (LLM processes)
        t=50ms: Send C2_T0 ───────────┐   │
                                      │   │
        t=100ms:Send C3_T0 ───────┐   │   │
                                  │   │   │
        Wait for _all_credits_sent│   │   └──> t=20ms: C1_T0 returns
                   Event...       │   │               ↓
                                  │   │        Pop C1_T1 from pending["conv1"]
                                  │   │               ↓
                                  │   │        Schedule: delay_ms=100ms
                                  │   │               ↓
                                  │   └──────> t=80ms: C2_T0 returns
                                  │                   ↓
                                  │            Pop C2_T1 from pending["conv2"]
                                  │                   ↓
                                  │            Schedule: delay_ms=200ms
                                  │                   ↓
                                  │            t=120ms: Send C1_T1 (20+100)
                                  │                   ↓
                                  └──────────> t=150ms: C3_T0 returns
                                                      ↓
                                               Pop C3_T1 from pending["conv3"]
                                                      ↓
                                               Schedule: timestamp_ms=500ms
                                                      ↓
                                               t=280ms: Send C2_T1 (80+200)
                                                      ↓
                                               t=500ms: Send C3_T1 (absolute)
                                                      ↓
                                               Set _all_credits_sent ─────┐
                                                                          │
        ◄─────────────────────────────────────────────────────────────────┘
        Main loop continues (all credits sent)


        KEY COMPONENTS
        ==============

        ┌─────────────────────────────────────────────────────────────────┐
        │ Credit Scheduling Decision Tree                                 │
        │                                                                 │
        │  When credit returns:                                           │
        │    │                                                            │
        │    ├──[Has pending turns?]──No──> Done (conversation complete)  │
        │    │                                                            │
        │    └──Yes──> Pop next turn                                      │
        │               │                                                 │
        │               ├──[delay_ms = 0?]──Yes──> Send inline            │
        │               │                          (no task overhead)     │
        │               │                                                 │
        │               └──No──> Schedule background task                 │
        │                        │                                        │
        │                        ├──[delay_ms > 0?]──Yes──> Sleep delay   │
        │                        │                          then send     │
        │                        │                                        │
        │                        └──[timestamp_ms?]──Yes──> Calculate     │
        │                                                   wait, sleep,  │
        │                                                   then send     │
        └─────────────────────────────────────────────────────────────────┘

    Execution Flow
    --------------
    1. Initialization (_create_absolute_schedule):
       - First turn of each conversation → absolute schedule
       - Subsequent turns → pending queue for that conversation

    2. Main Loop (_execute_single_phase):
       - Iterates through absolute schedule in timestamp order
       - Sends all first turns at their scheduled times
       - Waits for all credits to be sent (including delayed ones)

    3. Dynamic Scheduling (_on_credit_return):
       - When a credit returns, check if conversation has more turns
       - If yes, pop next turn from pending queue and schedule it:
         * delay_ms = 0: Send immediately (inline)
         * delay_ms > 0: Sleep then send (background task)
         * timestamp_ms: Calculate wait time, sleep then send (background task)
       - If no more turns, clean up the conversation from pending queue

    4. Completion Detection:
       - Track total credits sent via phase_stats.sent
       - When sent == total_expected_credits, set _all_credits_sent event
       - Main loop waits for this event before returning

    Example Scenario
    ----------------
    Dataset: 3 conversations with 2 turns each
    - Conv1: turn0 at 0ms, turn1 with delay 100ms
    - Conv2: turn0 at 50ms, turn1 with delay 200ms
    - Conv3: turn0 at 100ms, turn1 at absolute 500ms

    Timeline:
    - t=0ms:   Send conv1_turn0 (from absolute schedule)
    - t=50ms:  Send conv2_turn0 (from absolute schedule)
    - t=100ms: Send conv3_turn0 (from absolute schedule)
    - Main loop waits for _all_credits_sent...

    Meanwhile (async credit returns):
    - t=20ms:  Conv1_turn0 returns → schedule conv1_turn1 (100ms delay)
    - t=80ms:  Conv2_turn0 returns → schedule conv2_turn1 (200ms delay)
    - t=120ms: Conv1_turn1 sent (20ms + 100ms delay)
    - t=150ms: Conv3_turn0 returns → schedule conv3_turn1 (at 500ms absolute)
    - t=280ms: Conv2_turn1 sent (80ms + 200ms delay)
    - t=500ms: Conv3_turn1 sent (absolute timestamp)
    - All 6 credits sent → _all_credits_sent set → main loop continues

    Key Design Decisions
    --------------------
    - No locks needed: asyncio is single-threaded, lines 220-221 are atomic
    - Inline vs background: delay=0 sends inline to avoid task overhead
    - Cleanup: Empty pending queues are removed to prevent memory leaks
    - Optimization: Skip wait if no pending turns (all single-turn conversations)

    Attributes
    ----------
    _absolute_schedule : dict[int, list[AbsoluteTurn]]
        First turns grouped by timestamp (milliseconds, truncated for grouping).

    _pending_turns : dict[str, deque[PendingTurn]]
        Subsequent turns queued per conversation, processed as turns complete.

    _total_expected_credits : int
        Total number of credits that will be sent (all turns across all conversations).

    _all_credits_sent : asyncio.Event
        Set when all credits have been sent (used to signal main loop completion).

    _schedule_zero_ms : float
        Reference point for schedule timestamps (auto-offset or manual offset).

    _start_time_ms : float
        Timestamp when phase execution started (used for wait calculations).
    """

    def __init__(
        self,
        config: TimingManagerConfig,
        credit_manager: CreditManagerProtocol,
        dataset_metadata: DatasetMetadata,
    ):
        # NOTE: This all needs to be set before the super call, because the base class will call
        # _setup_profiling_phase_config() which uses it to set the total expected requests.

        self._dataset_metadata = dataset_metadata

        self._absolute_schedule: dict[int, list[AbsoluteTurn]] = defaultdict(list)
        self._pending_turns: dict[str, deque[PendingTurn]] = defaultdict(deque)

        self._total_expected_credits = dataset_metadata.total_turn_count
        self._all_credits_sent = asyncio.Event()

        self._schedule_zero_ms = 0

        self._start_time_ms = 0

        self._auto_offset = config.auto_offset_timestamps
        self._start_offset_ms = config.fixed_schedule_start_offset
        self._end_offset_ms = config.fixed_schedule_end_offset

        super().__init__(
            config=config,
            credit_manager=credit_manager,
            dataset_metadata=dataset_metadata,
        )

    def _setup_profiling_phase_config(self) -> None:
        """
        Setup the profiling phase.

        Overrides the base implementation to set the total expected requests based on the number of requests in the schedule.
        """
        self._create_absolute_schedule()

        profiling_conversation_provider = LiveConversationProvider(
            self._dataset_metadata, self._dataset_sampler
        )
        self.ordered_phase_configs.append(
            (
                CreditPhaseConfig(
                    type=CreditPhase.PROFILING,
                    total_expected_requests=self._total_expected_credits,
                ),
                profiling_conversation_provider,
            )
        )

    def _create_absolute_schedule(self) -> None:
        """Build absolute schedule and pending turn queues from dataset metadata.

        This method separates first turns (absolute schedule) from subsequent turns (pending queue)
        and performs validation to ensure all turns have valid timing data.

        Process:
        1. Iterate through all conversations and turns in dataset
        2. First turn (turn_index=0):
           - Must have timestamp_ms (absolute timestamp)
           - Added to _absolute_schedule grouped by truncated timestamp
           - Truncation to integer milliseconds groups nearby turns to reduce sleep operations
        3. Subsequent turns (turn_index > 0):
           - Must have either timestamp_ms OR delay_ms (validated)
           - Added to _pending_turns queue for that conversation
           - Will be scheduled dynamically when previous turn completes
        4. Calculate schedule zero reference point (for offset support)

        The absolute schedule is organized as: {timestamp_ms: [turn1, turn2, ...]}
        where timestamp_ms is truncated to int to group turns within the same millisecond.

        The pending queue is organized as: {conversation_id: deque([turn1, turn2, ...])}
        where each conversation's turns are queued in order.

        Raises:
            ValueError: If dataset is empty or turns have invalid timing data.

        Note:
            Timestamps are truncated to the nearest millisecond to group conversations that are
            very close together. This prevents creating excessive sleep operations for timestamps
            that differ by fractions of a millisecond.
        """
        if not self._dataset_metadata or self._total_expected_credits == 0:
            raise ValueError(
                "No schedule loaded, unable to setup fixed schedule strategy"
            )

        for conversation in self._dataset_metadata.conversations:
            for turn_index, turn in enumerate(conversation.turns):
                if turn_index == 0:
                    if turn.timestamp_ms is None:
                        raise ValueError(
                            f"Conversation {conversation.conversation_id} has invalid timing data"
                        )
                    # Truncate to nearest millisecond to avoid excessive grouping with floating point timestamps
                    self._absolute_schedule[int(turn.timestamp_ms)].append(
                        AbsoluteTurn(
                            timestamp_ns=int(turn.timestamp_ms * NANOS_PER_MILLIS),
                            conversation_id=conversation.conversation_id,
                            turn_index=turn_index,
                            is_final_turn=turn_index == len(conversation.turns) - 1,
                        )
                    )
                else:
                    if turn.timestamp_ms is None and turn.delay_ms is None:
                        raise ValueError(
                            f"Conversation {conversation.conversation_id} turn {turn_index} "
                            f"must have either timestamp_ms or delay_ms"
                        )
                    pending_turn = PendingTurn(
                        conversation_id=conversation.conversation_id,
                        turn_index=turn_index,
                        timestamp_ms=turn.timestamp_ms,
                        delay_ms=turn.delay_ms,
                        is_final_turn=turn_index == len(conversation.turns) - 1,
                    )
                    self._pending_turns[conversation.conversation_id].append(
                        pending_turn
                    )

        # Sort the turns within each timestamp group by timestamp
        for _, absolute_turns in self._absolute_schedule.items():
            absolute_turns.sort(key=lambda x: x.timestamp_ns)

        # Define the zero reference point for the schedule
        if self._auto_offset:
            self._schedule_zero_ms = sorted(self._absolute_schedule.keys())[0]
        elif self._start_offset_ms is not None:
            self._schedule_zero_ms = self._start_offset_ms
        else:
            self._schedule_zero_ms = 0
        self._schedule_zero_ns = int(self._schedule_zero_ms * NANOS_PER_MILLIS)

        self._schedule_first_timestamp_ms = (
            sorted(self._absolute_schedule.keys())[0] - self._schedule_zero_ms
        )
        self.notice(
            f"Fixed schedule configuration: first_timestamp={self._schedule_first_timestamp_ms}ms, "
            f"total_conversations={len(self._dataset_metadata.conversations)}, "
            f"total_turns={self._total_expected_credits}"
        )

    def _get_wait_duration_ms(self, timestamp_ms: int | float) -> float:
        """Get the wait duration in milliseconds for a given timestamp.

        Calculate the wait duration for this timestamp

        (timestamp_ms - schedule_zero_ms) is the offset of the timestamp from the start of the schedule
        (_perf_counter_ms() - start_time_ms) is how much time has passed since we started dropping credits
        """
        return (timestamp_ms - self._schedule_zero_ms) - (
            _perf_counter_ms() - self._start_time_ms
        )

    async def _execute_single_phase(
        self,
        phase_stats: CreditPhaseStats,
        conversation_provider: BaseConversationProvider,
    ) -> None:
        """Execute the main credit sending loop for the absolute schedule.

        This method sends all first turns from the absolute schedule at their precise timestamps,
        then waits for all delayed/subsequent turns to be sent by background tasks.

        Execution Flow:
        1. Record start time (reference point for all timing calculations)
        2. Iterate through absolute schedule in timestamp order:
           - Calculate wait time until timestamp
           - Sleep (millisecond precision)
           - For each turn at this timestamp:
             * Tight loop wait for sub-millisecond precision
             * Send credit
        3. If there are pending turns:
           - Wait for _all_credits_sent event
           - This event is set when the last credit is sent (including delayed ones)
        4. If no pending turns (all single-turn conversations):
           - Skip wait (optimization)
           - All credits already sent in step 2

        The method returns when ALL credits have been sent (not completed/returned),
        which matches the base class contract for _execute_single_phase.

        Args:
            phase_stats: Statistics tracker for this phase (tracks sent/completed counts).

        Note:
            While this method is running, credit returns are processed asynchronously by
            _on_credit_return, which schedules subsequent turns as background tasks.
        """
        # This is used as a reference point for the wait duration calculation
        start_time_ns = time.perf_counter_ns()
        self._start_time_ms = start_time_ns / NANOS_PER_MILLIS

        # Drop credits in order of the schedule
        for timestamp_ms in sorted(self._absolute_schedule.keys()):
            # Get the conversations at this timestamp group now, so we are ready
            absolute_turns = self._absolute_schedule[timestamp_ms]

            wait_duration_sec = (
                self._get_wait_duration_ms(timestamp_ms) / MILLIS_PER_SECOND
            )
            if wait_duration_sec > 0:
                await asyncio.sleep(wait_duration_sec)

            # Drop credits for all conversations at this timestamp
            for absolute_turn in absolute_turns:
                # Use tight loop to wait for precise timestamp (sub-millisecond precision)
                while (
                    time.perf_counter_ns() - start_time_ns
                    < absolute_turn.timestamp_ns - self._schedule_zero_ns
                ):
                    await yield_to_event_loop()

                await self._send_credit(absolute_turn, phase_stats)

        if self._pending_turns:
            await self._all_credits_sent.wait()
        duration_sec = (time.perf_counter_ns() - start_time_ns) / NANOS_PER_SECOND
        self.info(
            f"Sent {phase_stats.sent:,} fixed schedule turns in {duration_sec:,.2f}s. Waiting for responses..."
        )

    async def _send_credit(
        self, base_turn: BaseTurn, phase_stats: CreditPhaseStats
    ) -> None:
        """Send a credit immediately and track completion.

        This method creates and sends a Credit message to the credit manager, increments the
        sent counter, and sets the completion event if this is the final credit.

        Thread Safety:
        The read-modify-write of phase_stats.sent (lines ~220-221) is atomic within asyncio's
        single-threaded event loop because there is no await between the read and write.
        Multiple concurrent calls will execute sequentially.

        Completion Detection:
        When this is the final credit (sent == total_expected_credits), sets _all_credits_sent
        event to signal the main loop that all credits have been sent.

        Args:
            base_turn: Turn information (conversation_id, turn_index, is_final_turn).
            phase_stats: Phase statistics tracker (increments sent counter).

        Note:
            Credit numbers (credit.num) maintain global ordering across all credits, even when
            delayed turns are sent by background tasks, because the increment is atomic.
        """
        # NOTE: Get and increment the sent count here, to avoid race conditions from interleaving
        # credit drops and returns.
        credit_num = phase_stats.sent
        phase_stats.sent += 1
        is_final_credit = phase_stats.sent == self._total_expected_credits

        should_cancel = self.cancellation_strategy.should_cancel_request()
        cancel_after_ns = (
            self.cancellation_strategy.get_cancellation_delay_ns()
            if should_cancel
            else None
        )

        credit = Credit(
            id=str(uuid.uuid4()),
            phase=phase_stats.type,
            num=credit_num,
            conversation_id=base_turn.conversation_id,
            x_correlation_id=base_turn.conversation_id,
            turn_index=base_turn.turn_index,
            is_final_turn=base_turn.is_final_turn,
            cancel_after_ns=cancel_after_ns,
        )
        await self.credit_manager.send_credit(credit=credit)

        if is_final_credit:
            self._all_credits_sent.set()

    async def _schedule_next_credit(
        self, pending_turn: PendingTurn, phase_stats: CreditPhaseStats
    ) -> None:
        """Wait for the appropriate delay/timestamp, then send the credit.

        This method handles the timing logic for subsequent turns, supporting both:
        - Relative delays (delay_ms): Wait X milliseconds from now
        - Absolute timestamps (timestamp_ms): Wait until specific time in schedule

        This is called as a background task (via execute_async) for turns with non-zero timing,
        allowing multiple delayed turns to be scheduled concurrently.

        Timing Logic:
        1. If delay_ms is set and > 0:
           - Sleep for delay_ms milliseconds from current time
        2. Else if timestamp_ms is set:
           - Calculate wait time relative to schedule start
           - If positive: sleep for that duration
           - If negative: log warning (timestamp in the past), send immediately
        3. Else (delay_ms = 0 or None and timestamp_ms = None):
           - Send immediately (no sleep)

        Args:
            pending_turn: Turn to schedule (contains timing information).
            phase_stats: Phase statistics tracker (passed to _send_credit).

        Note:
            The PendingTurn validator ensures at least one of timestamp_ms or delay_ms is set,
            so the "send immediately" fallback should only occur for delay_ms=0 or late timestamps.
        """
        if pending_turn.delay_ms is not None and pending_turn.delay_ms > 0:
            await asyncio.sleep(pending_turn.delay_ms / MILLIS_PER_SECOND)
        elif pending_turn.timestamp_ms is not None:
            wait_duration_ms = self._get_wait_duration_ms(pending_turn.timestamp_ms)
            if wait_duration_ms > 0:
                await asyncio.sleep(wait_duration_ms / MILLIS_PER_SECOND)
            elif wait_duration_ms < 0:
                self.warning(
                    f"Conversation {pending_turn.conversation_id} absolute timestamp {pending_turn.timestamp_ms} was in the past by {-wait_duration_ms}ms"
                )
        await self._send_credit(pending_turn, phase_stats)

    async def _on_credit_return(
        self, worker_id: str, message: CreditReturnMessage
    ) -> None:
        """Handle credit return and schedule the next turn if one exists.

        This is the key method for dynamic turn scheduling. When a credit returns (turn completes),
        this method checks if the conversation has more turns and schedules the next one.

        Process:
        1. Check if conversation has pending turns in the queue
        2. If yes:
           a. Pop next turn from the front of the queue
           b. Decide scheduling strategy based on timing:
              - delay_ms = 0: Send immediately (inline, no task overhead)
              - delay_ms > 0 or timestamp_ms set: Schedule as background task
           c. Clean up: if queue is now empty, remove conversation from dict
        3. If no more turns: conversation is complete (no action needed)
        4. Call base class handler for standard credit return processing

        Scheduling Strategy:
        - Inline (await _send_credit): Used for zero-delay turns to avoid task creation overhead
        - Background task (execute_async): Used for delayed turns, allowing concurrent scheduling

        Args:
            message: Credit return message containing the completed credit information.

        Note:
            This method is called asynchronously by the message bus when workers return credits,
            allowing it to run concurrently with the main scheduling loop.
        """

        conversation_id = message.credit.conversation_id
        if (
            conversation_id in self._pending_turns
            and len(self._pending_turns[conversation_id]) > 0
        ):
            pending_turn = self._pending_turns[conversation_id].popleft()
            phase_stats = self.phase_stats[message.credit.phase]
            if pending_turn.delay_ms == 0:
                await self._send_credit(pending_turn, phase_stats)
            else:
                self.execute_async(
                    self._schedule_next_credit(pending_turn, phase_stats)
                )
            if not self._pending_turns[conversation_id]:
                del self._pending_turns[conversation_id]

        await super()._on_credit_return(worker_id, message)
