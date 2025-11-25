# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request rate credit issuing strategy with optional concurrency limiting.

This module implements a credit issuing strategy that generates credits at a specified
request rate (constant, Poisson, or burst), with optional concurrency limiting.

Multi-turn Behavior - Two Modes
--------------------------------
The strategy handles multi-turn conversations differently based on concurrency configuration:

**WITH Concurrency Limit (conversation-centric):**
- Semaphore slot held for entire conversation
- Subsequent turns respect their delay_ms from dataset
- Ensures max N conversations in flight simultaneously
- Example: concurrency=10 means max 10 active conversations

**WITHOUT Concurrency Limit (rate-centric):**
- Request rate strictly maintained
- Subsequent turns queued and sent at next rate interval
- Inter-turn delays ignored to preserve rate
- Example: 10 req/s means exactly 10 requests/sec regardless of delays

This design allows users to choose their priority:
- Set concurrency → preserve conversation timing patterns
- No concurrency → maintain precise request rate

Example Timeline
----------------
Dataset: Conversation with 2 turns, delay_ms=100 between them
Request rate: 10 req/s (100ms intervals)

WITH concurrency=10:
- t=0ms:   Send turn0 (acquire semaphore)
- t=20ms:  Turn0 returns, schedule turn1 with 100ms delay
- t=120ms: Send turn1 (semaphore held for 120ms, then released)
Result: Delay respected, conversation treated as atomic unit

WITHOUT concurrency:
- t=0ms:   Send turn0
- t=20ms:  Turn0 returns, queue turn1
- t=100ms: Main loop sends turn1 (at rate interval, delay ignored)
Result: Rate maintained at 10 req/s, delays not respected

This intentional distinction optimizes for different use cases.
"""

import asyncio
import time
import uuid
from collections import deque

from pydantic import Field

from aiperf.common import random_generator as rng
from aiperf.common.constants import MILLIS_PER_SECOND
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import TimingMode
from aiperf.common.enums.timing_enums import RequestRateMode
from aiperf.common.factories import RequestRateGeneratorFactory
from aiperf.common.messages import Credit, CreditReturnMessage
from aiperf.common.models import (
    AIPerfBaseModel,
    ConversationMetadata,
    CreditPhaseStats,
    DatasetMetadata,
)
from aiperf.common.protocols import RequestRateGeneratorProtocol
from aiperf.timing.config import TimingManagerConfig
from aiperf.timing.conversation_provider import BaseConversationProvider
from aiperf.timing.credit_issuing_strategy import (
    CreditIssuingStrategy,
    CreditIssuingStrategyFactory,
    CreditManagerProtocol,
)


class TurnToSend(AIPerfBaseModel):
    """A turn that needs to be sent."""

    conversation_id: str = Field(
        description="The ID of the conversation that this turn belongs to."
    )
    x_correlation_id: str = Field(
        ..., description="The X-Correlation-ID header of the conversation instance."
    )
    turn_index: int = Field(
        ..., ge=0, description="The index of the turn in the conversation (0-based)."
    )
    is_final_turn: bool = Field(
        ..., description="True if this is the last turn of the conversation."
    )


@CreditIssuingStrategyFactory.register(TimingMode.REQUEST_RATE)
class RequestRateStrategy(CreditIssuingStrategy):
    """Request rate credit issuing strategy with two distinct operational modes.

    Generates credits at a specified request rate (constant, Poisson, or burst),
    with optional concurrency limiting. Handles multi-turn conversations differently
    based on whether concurrency is enabled.

    Rate Modes
    ----------
    - CONSTANT: Fixed intervals (1/rate seconds between credits)
    - POISSON: Exponentially distributed intervals (models random arrivals)
    - CONCURRENCY_BURST: No delays, sends as fast as possible up to concurrency limit

    Multi-turn Conversation Handling
    ---------------------------------
    The strategy operates in two distinct modes based on concurrency configuration:

    **Mode 1: WITH Concurrency (Conversation-Centric)**
    - Semaphore slot acquired when conversation starts
    - Slot held until all turns complete (not released between turns)
    - Subsequent turns respect delay_ms from dataset
    - Use case: Model realistic user behavior with think time
    - Ensures: Max N conversations in flight, conversation timing preserved

    **Mode 2: WITHOUT Concurrency (Rate-Centric)**
    - No semaphore, request rate strictly enforced
    - Subsequent turns queued, sent at next rate interval
    - delay_ms ignored to maintain precise request rate
    - Use case: Stress testing at exact rate, ignoring conversation patterns
    - Ensures: Precise request rate (e.g., exactly 100 req/s)

    Architecture
    ------------
    - _queued_turns: FIFO queue for subsequent turns (rate-centric mode)
    - _semaphore: Optional concurrency limiter (conversation-centric mode)
    - _dataset_sampler: Selects conversations for first turns
    - _conversation_metadata_lookup: Fast lookup for conversation data

    Main Loop Flow:
    1. If semaphore exists: acquire it
    2. If queue not empty: send queued turn (subsequent turn from rate mode)
    3. Else: sample new conversation, send first turn
    4. Sleep for next_interval (from rate generator)

    Credit Return Flow:
    - If final turn: release semaphore (if exists)
    - If not final turn:
      * WITH semaphore: send next turn directly (inline or with delay)
      * WITHOUT semaphore: queue next turn (main loop sends at rate)

    Key Design Insight
    ------------------
    The dichotomy is intentional:
    - Concurrency mode optimizes for conversation realism
    - Rate mode optimizes for request rate precision
    Users can choose based on benchmarking goals.
    """

    def __init__(
        self,
        config: TimingManagerConfig,
        credit_manager: CreditManagerProtocol,
        dataset_metadata: DatasetMetadata,
    ):
        super().__init__(
            config=config,
            credit_manager=credit_manager,
            dataset_metadata=dataset_metadata,
        )
        self._request_rate_generator = RequestRateGeneratorFactory.create_instance(
            config
        )
        # If the user has provided a concurrency, use a semaphore to limit the maximum number of concurrent requests
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(value=config.concurrency) if config.concurrency else None
        )
        self._conversation_metadata_lookup: dict[str, ConversationMetadata] = {
            conversation.conversation_id: conversation
            for conversation in dataset_metadata.conversations
        }
        self._all_credits_sent = asyncio.Event()
        self._queued_turns: deque[TurnToSend] = deque()

    async def _execute_single_phase(
        self,
        phase_stats: CreditPhaseStats,
        conversation_provider: BaseConversationProvider,
    ) -> None:
        """Execute credit drops based on the request rate generator, optionally with a max concurrency limit."""
        # Track next scheduled send time to compensate for execution overhead
        next_send_time: float | None = None
        self._all_credits_sent.clear()

        while phase_stats.should_send():
            # Ensure we have an available credit before dropping
            if self._semaphore:
                await self._semaphore.acquire()
                if self.is_trace_enabled:
                    self.trace(f"Acquired credit drop semaphore: {self._semaphore!r}")
                if not phase_stats.should_send():
                    # Check one last time to see if we should still send a credit in case the
                    # time-based phase expired while we were waiting for the semaphore.
                    self._semaphore.release()
                    if self.is_trace_enabled:
                        self.trace(
                            f"Released semaphore after should_send returned False: {self._semaphore!r}"
                        )
                    break

            if not self._queued_turns:
                # Sample the next conversation id from the dataset so the worker knows
                # which conversation data to use.
                try:
                    conversation_id = conversation_provider.next_conversation_id()
                except StopIteration as e:
                    self.debug(f"No more conversation IDs available: {e}")
                    break
                conversation_metadata = self._conversation_metadata_lookup[
                    conversation_id
                ]
                turn_to_send = TurnToSend(
                    conversation_id=conversation_id,
                    x_correlation_id=str(uuid.uuid4()),
                    turn_index=0,
                    is_final_turn=len(conversation_metadata.turns) == 1,
                )
                await self._send_credit(turn_to_send, phase_stats)

            else:
                # If we have queued turns, then we need to send the first one in line.
                turn_to_send = self._queued_turns.popleft()
                await self._send_credit(turn_to_send, phase_stats)

            # Check if we should break out of the loop before we sleep for the next interval.
            # This is to ensure we don't sleep for any unnecessary time, which could cause race conditions.
            if not phase_stats.should_send():
                break

            next_interval = self._request_rate_generator.next_interval()
            if next_interval > 0:
                # Use scheduled send times to compensate for execution overhead
                current_time = time.perf_counter()
                if next_send_time is None:
                    # First iteration: schedule next send relative to now
                    next_send_time = current_time + next_interval
                else:
                    # Subsequent iterations: schedule next send relative to previous scheduled time
                    # This compensates for any drift from execution overhead
                    next_send_time += next_interval

                sleep_duration = next_send_time - current_time
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)
                else:
                    self.warning(
                        f"Sleep duration is negative: {sleep_duration} seconds"
                    )

        if phase_stats.is_request_count_based:
            await self._all_credits_sent.wait()

    async def _send_credit(
        self, turn_to_send: TurnToSend, phase_stats: CreditPhaseStats
    ) -> None:
        # NOTE: Get and increment the sent count here, to avoid race conditions from interleaving
        # credit drops and returns.
        credit_num = phase_stats.sent
        phase_stats.sent += 1
        is_final_credit = phase_stats.sent == phase_stats.total_expected_requests

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
            conversation_id=turn_to_send.conversation_id,
            x_correlation_id=turn_to_send.x_correlation_id,
            turn_index=turn_to_send.turn_index,
            is_final_turn=turn_to_send.is_final_turn,
            cancel_after_ns=cancel_after_ns,
        )
        await self.credit_manager.send_credit(credit=credit)

        if is_final_credit:
            self._all_credits_sent.set()

    async def _schedule_next_credit(
        self,
        turn_to_send: TurnToSend,
        delay_ms: int | float,
        phase_stats: CreditPhaseStats,
    ) -> None:
        """Schedule the next credit."""
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / MILLIS_PER_SECOND)
        await self._send_credit(turn_to_send, phase_stats)

    async def _on_credit_return(
        self, worker_id: str, message: CreditReturnMessage
    ) -> None:
        """Handle credit return and manage multi-turn conversations.

        This method implements the two-mode behavior for multi-turn conversations:

        For final turns (last turn of conversation):
        - Release semaphore if it exists (allows new conversation to start)
        - Call base class handler

        For non-final turns (more turns remaining):
        - WITH concurrency (semaphore exists):
          * Keep semaphore held (maintains conversation slot)
          * If no delay: send next turn immediately (inline)
          * If delay > 0: schedule next turn with delay (background task)
          * Conversation owns semaphore slot until complete

        - WITHOUT concurrency (no semaphore):
          * Queue next turn for main loop
          * Main loop sends at next rate interval
          * delay_ms ignored to maintain strict request rate
          * Request rate takes precedence

        The key insight: semaphore determines whether we're optimizing for
        conversation atomicity (with) or request rate precision (without).

        Args:
            message: Credit return message with completed credit information.

        Note:
            The semaphore is NOT released between turns when concurrency is enabled.
            This is intentional - each conversation holds a slot until fully complete.
        """

        if message.credit.is_final_turn:
            # Release the semaphore to allow another credit to be issued,
            # then call the superclass to handle the credit return like normal
            if self._semaphore:
                self._semaphore.release()
                if self.is_trace_enabled:
                    self.trace(f"Credit return released semaphore: {self._semaphore!r}")

        else:
            # IMPORTANT: DO NOT RELEASE THE SEMAPHORE HERE!
            # In order to maintain the fixed concurrency limit, we need to keep the semaphore acquired until the entire
            # conversation is completed. We will release the semaphore in the _on_credit_return method for the final turn.
            conversation_id = message.credit.conversation_id
            conversation_metadata = self._conversation_metadata_lookup[conversation_id]
            next_turn_index = message.credit.turn_index + 1
            next_turn = conversation_metadata.turns[next_turn_index]
            phase_stats = self.phase_stats[message.credit.phase]
            num_turns = len(conversation_metadata.turns)

            turn_to_send = TurnToSend(
                conversation_id=conversation_id,
                x_correlation_id=message.credit.x_correlation_id,
                turn_index=next_turn_index,
                is_final_turn=next_turn_index == num_turns - 1,
            )

            if self._semaphore:
                # Conversation-centric mode: hold semaphore, respect delays
                # This keeps the conversation atomic and preserves timing patterns
                if not next_turn.delay_ms:
                    await self._send_credit(turn_to_send, phase_stats)
                else:
                    await self._schedule_next_credit(
                        turn_to_send, next_turn.delay_ms, phase_stats
                    )
            else:
                # Rate-centric mode: queue for main loop, delay ignored
                # Main loop sends at next rate interval to maintain precise request rate
                if next_turn.delay_ms and next_turn.delay_ms > 0:
                    self.warning(
                        f"Conv {conversation_id} turn {next_turn_index}: "
                        f"delay_ms={next_turn.delay_ms} ignored (rate-centric mode)"
                    )
                self._queued_turns.append(turn_to_send)

        await super()._on_credit_return(worker_id, message)


@implements_protocol(RequestRateGeneratorProtocol)
@RequestRateGeneratorFactory.register(RequestRateMode.POISSON)
class PoissonRateGenerator:
    """
    Generator for Poisson process (exponential inter-arrival times).

    In a Poisson process with rate λ (requests per second), the inter-arrival times
    are exponentially distributed with parameter λ. This attempts to model more
    realistic traffic patterns where requests arrive randomly but at a consistent
    average rate.

    Uses the global RandomGenerator for reproducibility.
    """

    def __init__(self, config: TimingManagerConfig) -> None:
        if config.request_rate is None or config.request_rate <= 0:
            raise ValueError(
                f"Request rate {config.request_rate} must be set and greater than 0 for {config.request_rate_mode!r}"
            )

        self._rng = rng.derive("timing.request.poisson_interval")
        self._request_rate: float = config.request_rate

    def next_interval(self) -> float:
        """
        Generate the next inter-arrival time for a Poisson process.

        For Poisson process, inter-arrival times are exponentially distributed.
        expovariate(lambd) generates exponentially distributed random numbers
        where lambd is the rate parameter (requests per second).
        """
        return self._rng.expovariate(self._request_rate)


@implements_protocol(RequestRateGeneratorProtocol)
@RequestRateGeneratorFactory.register(RequestRateMode.CONSTANT)
class ConstantRateGenerator:
    """
    Generator for constant rate (fixed inter-arrival times).

    This generates a fixed inter-arrival time for each request.
    """

    def __init__(self, config: TimingManagerConfig) -> None:
        if config.request_rate is None or config.request_rate <= 0:
            raise ValueError(
                f"Request rate {config.request_rate} must be set and greater than 0 for {config.request_rate_mode!r}"
            )
        self._period: float = 1.0 / config.request_rate

    def next_interval(self) -> float:
        """
        Generate the next inter-arrival time for a constant rate.
        """
        return self._period


@implements_protocol(RequestRateGeneratorProtocol)
@RequestRateGeneratorFactory.register(RequestRateMode.CONCURRENCY_BURST)
class ConcurrencyBurstRateGenerator:
    """
    Generator for concurrency-burst rate (no delay between requests).
    """

    def __init__(self, config: TimingManagerConfig) -> None:
        if config.concurrency is None or config.concurrency < 1:
            raise ValueError(
                f"Concurrency {config.concurrency} must be set and greater than 0 for {config.request_rate_mode!r}"
            )
        if config.request_rate is not None:
            raise ValueError(
                f"Request rate {config.request_rate} should be None for {config.request_rate_mode!r}"
            )

    def next_interval(self) -> float:
        """
        Generate the next inter-arrival time for a concurrency-burst rate.

        This will always return 0, as the requests should be issued as soon as possible.
        """
        return 0
