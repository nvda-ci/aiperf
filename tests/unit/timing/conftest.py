# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar
from unittest.mock import MagicMock

import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config import ServiceConfig
from aiperf.common.enums import (
    ArrivalPattern,
    CommAddress,
    CreditPhase,
    DatasetSamplingStrategy,
    TimingMode,
)
from aiperf.common.factories import DatasetSamplingStrategyFactory
from aiperf.common.models import (
    ConversationMetadata,
    CreditPhaseStats,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.common.utils import yield_to_event_loop
from aiperf.credit.messages import CreditReturn, FirstToken
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.timing.concurrency import ConcurrencyStats
from aiperf.timing.config import (
    CreditPhaseConfig,
    RequestCancellationConfig,
    TimingConfig,
)
from aiperf.timing.phase.publisher import PhasePublisher
from aiperf.timing.phase_orchestrator import PhaseOrchestrator
from aiperf.timing.strategies.core import TimingStrategyProtocol
from tests.harness.fake_communication import FakeCommunication, FakeCommunicationBus

T = TypeVar("T", bound=TimingStrategyProtocol)


_logger = AIPerfLogger(__name__)


# =============================================================================
# Common Mock Fixtures
# =============================================================================
@dataclass
class MockCreditRouter:
    """Mock credit router that captures credits and allows injecting returns."""

    sent_credits: list[Credit] = field(default_factory=list)
    auto_return: bool = False
    _return_callback: Callable[[str, CreditReturn], Awaitable[None]] | None = None
    _first_token_callback: Callable[[FirstToken], Awaitable[None]] | None = None
    _pending_returns: list[asyncio.Task] = field(default_factory=list)

    async def send_credit(self, credit: Credit) -> None:
        """Capture sent credit, optionally auto-return it."""
        self.sent_credits.append(credit)
        if self.auto_return and self._return_callback:
            # Create task and track it - allows concurrent credit processing
            task = asyncio.create_task(self._do_return(credit))
            self._pending_returns.append(task)
            await yield_to_event_loop()

    async def _do_return(self, credit: Credit) -> None:
        """Actually perform the credit return after yielding to event loop."""
        # Yield to allow other coroutines to run (looptime advances virtual time)
        await asyncio.sleep(0.001)
        if self._return_callback:
            credit_return = CreditReturn(
                credit=credit, cancelled=False, first_token_sent=True
            )
            await self._return_callback("worker-1", credit_return)

    async def cancel_all_credits(self) -> None:
        """No-op for tests."""
        pass

    def mark_credits_complete(self) -> None:
        """Mark all credits as complete. No-op for tests."""
        pass

    def set_return_callback(
        self, callback: Callable[[str, CreditReturn], Awaitable[None]]
    ) -> None:
        """Capture the return callback for later invocation."""
        self._return_callback = callback

    def set_first_token_callback(
        self, callback: Callable[[FirstToken], Awaitable[None]]
    ) -> None:
        """Capture the first token callback."""
        self._first_token_callback = callback

    async def return_credit(
        self, credit: Credit, cancelled: bool = False, first_token_sent: bool = True
    ) -> None:
        """Simulate a credit return from worker."""
        if self._return_callback:
            credit_return = CreditReturn(
                credit=credit, cancelled=cancelled, first_token_sent=first_token_sent
            )
            await self._return_callback("worker-1", credit_return)


@dataclass
class OrchestratorHarness:
    """Test fixture with real orchestrator and mock router."""

    orchestrator: PhaseOrchestrator
    router: MockCreditRouter

    @property
    def sent_credits(self) -> list[Credit]:
        """Credits sent so far."""
        return self.router.sent_credits

    async def run_with_auto_return(self) -> None:
        """Run phase, auto-returning credits as they're sent."""
        self.router.auto_return = True
        await self.orchestrator.initialize()
        with contextlib.suppress(asyncio.CancelledError):
            await self.orchestrator.start()

        # Cleanup: stop orchestrator and await ALL pending tasks
        with contextlib.suppress(Exception):
            await self.orchestrator.stop()

        # Await any pending auto-return tasks
        if self.router._pending_returns:
            await asyncio.gather(*self.router._pending_returns, return_exceptions=True)
            self.router._pending_returns.clear()

        # Give event loop one final chance to process any remaining callbacks
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def return_credit(self, credit: Credit) -> None:
        """Return a specific credit."""
        await self.router.return_credit(credit)


@pytest.fixture
def create_orchestrator_harness(mock_zmq, time_traveler):
    """Factory for creating OrchestratorHarness with various configurations.

    Depends on time_traveler to ensure time functions are patched during test.
    Each test file can override time_traveler to choose between:
    - time_traveler (patches time + asyncio.sleep) for instant execution
    - time_traveler_no_patch_sleep for looptime-compatible timing tests
    """

    def create(
        conversations: list[tuple[str, int]] | None = None,
        *,
        schedule: list[tuple[int | float, str]] | None = None,
        request_count: int | None = None,
        num_sessions: int | None = None,
        num_users: int | None = None,
        concurrency: int | None = None,
        request_rate: float | None = None,
        user_centric_rate: float | None = None,
        arrival_pattern: ArrivalPattern = ArrivalPattern.POISSON,
        random_seed: int = 42,
        sampling_strategy: DatasetSamplingStrategy = DatasetSamplingStrategy.SHUFFLE,
        timing_mode: TimingMode | None = None,
        auto_offset_timestamps: bool = False,
        fixed_schedule_start_offset: int | None = None,
    ) -> OrchestratorHarness:
        """Create OrchestratorHarness.

        Args:
            conversations: List of (conversation_id, num_turns). Mutually exclusive with schedule.
            schedule: List of (timestamp_ms, conversation_id) for fixed schedule mode.
            request_count: Total requests limit.
            num_sessions: Session limit.
            num_users: Number of concurrent users (for user-centric mode).
            concurrency: Session concurrency.
            request_rate: Requests per second (for REQUEST_RATE mode).
            user_centric_rate: User-centric QPS (triggers USER_CENTRIC_RATE mode).
            arrival_pattern: Rate mode (POISSON, CONSTANT, CONCURRENCY_BURST).
            random_seed: Random seed for reproducibility.
            sampling_strategy: Dataset sampling strategy.
            timing_mode: Override timing mode (auto-detected if not specified).
            auto_offset_timestamps: Auto-offset timestamps in fixed schedule mode.
            fixed_schedule_start_offset: Manual offset for fixed schedule mode.
        """
        # Build dataset metadata
        if schedule is not None:
            # Fixed schedule mode - use schedule to build metadata with timestamps
            dataset = create_mock_dataset_metadata_with_schedule(
                schedule, sampling_strategy=sampling_strategy
            )
            effective_timing_mode = timing_mode or TimingMode.FIXED_SCHEDULE
            effective_request_count = request_count or len(schedule)
        elif conversations is not None:
            # Standard mode - build metadata from conversation list
            metadata_list = [
                ConversationMetadata(
                    conversation_id=cid,
                    turns=[TurnMetadata() for _ in range(num_turns)],
                )
                for cid, num_turns in conversations
            ]
            dataset = DatasetMetadata(
                conversations=metadata_list,
                sampling_strategy=sampling_strategy,
            )
            # Determine timing mode
            if timing_mode is not None:
                effective_timing_mode = timing_mode
            elif user_centric_rate is not None:
                effective_timing_mode = TimingMode.USER_CENTRIC_RATE
            else:
                effective_timing_mode = TimingMode.REQUEST_RATE
            effective_request_count = request_count
        else:
            raise ValueError("Either conversations or schedule must be provided")

        # Use user_centric_rate as the request_rate if in USER_CENTRIC_RATE mode
        effective_request_rate = (
            user_centric_rate if user_centric_rate is not None else request_rate
        )

        if concurrency is not None and effective_request_rate is None:
            arrival_pattern = ArrivalPattern.CONCURRENCY_BURST

        timing_config = make_timing_config(
            timing_mode=effective_timing_mode,
            arrival_pattern=arrival_pattern,
            concurrency=concurrency,
            request_rate=effective_request_rate,
            request_count=effective_request_count,
            num_sessions=num_sessions,
            num_users=num_users,
            random_seed=random_seed,
            auto_offset_timestamps=auto_offset_timestamps,
            fixed_schedule_start_offset=fixed_schedule_start_offset,
        )

        router = MockCreditRouter()
        pub_client = MagicMock()

        # Use simple async function instead of AsyncMock to avoid GC warnings
        async def mock_publish(*args, **kwargs):
            return None

        pub_client.publish = mock_publish
        phase_publisher = PhasePublisher(pub_client=pub_client, service_id="test")

        orchestrator = PhaseOrchestrator(
            config=timing_config,
            phase_publisher=phase_publisher,
            credit_router=router,
            dataset_metadata=dataset,
            service_id="test-orchestrator",
        )

        return OrchestratorHarness(orchestrator=orchestrator, router=router)

    return create


# =============================================================================
# Common Credit Fixtures
# =============================================================================


@pytest.fixture
def credit_turn_0():
    """Credit for first turn (turn_index=0) of 3-turn conversation."""
    return Credit(
        phase=CreditPhase.PROFILING,
        id=1,
        conversation_id="conv-multi",
        x_correlation_id="corr-multi-001",
        turn_index=0,
        num_turns=3,
        issued_at_ns=time.time_ns(),
    )


@pytest.fixture
def credit_turn_1():
    """Credit for second turn (turn_index=1) of 3-turn conversation."""
    return Credit(
        phase=CreditPhase.PROFILING,
        id=2,
        conversation_id="conv-multi",
        x_correlation_id="corr-multi-001",
        turn_index=1,
        num_turns=3,
        issued_at_ns=time.time_ns(),
    )


@pytest.fixture
def credit_final():
    """Credit for final turn (is_final_turn=True)."""
    return Credit(
        phase=CreditPhase.PROFILING,
        id=3,
        conversation_id="conv-multi",
        x_correlation_id="corr-multi-001",
        turn_index=2,
        num_turns=3,
        issued_at_ns=time.time_ns(),
    )


# =============================================================================
# Common Dataset Fixtures
# =============================================================================


@pytest.fixture
def single_turn_dataset():
    """Dataset with single-turn conversations."""
    return DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="conv-1",
                turns=[TurnMetadata()],
            ),
            ConversationMetadata(
                conversation_id="conv-2",
                turns=[TurnMetadata()],
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


@pytest.fixture
def multi_turn_dataset():
    """Dataset with multi-turn conversations."""
    return DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="conv-1",
                turns=[TurnMetadata(), TurnMetadata(delay_ms=50.0)],
            ),
            ConversationMetadata(
                conversation_id="conv-2",
                turns=[
                    TurnMetadata(),
                    TurnMetadata(delay_ms=0),
                    TurnMetadata(delay_ms=100.0),
                ],
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


@pytest.fixture
def three_turn_conversation():
    """Three-turn conversation metadata."""
    return ConversationMetadata(
        conversation_id="conv-3turn",
        turns=[
            TurnMetadata(timestamp_ms=1000.0),
            TurnMetadata(delay_ms=50.0),
            TurnMetadata(delay_ms=100.0),
        ],
    )


@pytest.fixture
def sample_phase_stats():
    """Sample phase stats for testing."""
    return CreditPhaseStats(
        phase=CreditPhase.PROFILING,
        requests_sent=100,
        requests_completed=80,
        requests_cancelled=5,
        final_requests_sent=100,
        start_ns=1000000,
    )


@pytest.fixture
def sample_phase_config():
    """Sample phase config for testing."""
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        total_expected_requests=100,
    )


class TimingHarness:
    """Test harness using REAL everything with fake communication layer.

    Uses ALL real code - the only fake is the transport layer (FakeCommunication
    replaces ZMQ). InstantWorker extends the real Worker class but skips inference.

    Real code used:
    - PhaseOrchestrator (owns long-lived components + phase execution loop)
    - StickyCreditRouter (real routing, load balancing, sticky sessions)
    - ConcurrencyManager, CancellationPolicy, ConversationSource
    - CreditCallbackHandler (handles credit returns + TTFT)
    - PhaseRunner (creates per-phase: LoopScheduler, lifecycle, progress, stop_checker, CreditIssuer)
    - All TimingMode implementations
    - PhasePublisher
    - Worker (InstantWorker extends it, overrides _process_credit)

    Fake (transport only):
    - FakeCommunication (replaces ZMQ with in-memory routing)
    """

    def __init__(self, service_config: ServiceConfig, user_config):
        from aiperf.credit.sticky_router import StickyCreditRouter

        # Fresh bus for this test (isolates from other tests)
        self.bus = FakeCommunicationBus()
        FakeCommunication.set_shared_bus(self.bus)

        # Create REAL components - they'll use fake communication automatically
        self.router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        self.publisher = PhasePublisher(
            pub_client=self.router.comms.create_pub_client(
                CommAddress.EVENT_BUS_PROXY_FRONTEND
            ),
            service_id="test-service",
        )

        # Create InstantWorker - extends real Worker, skips inference
        InstantWorkerClass = create_instant_worker_class()
        self._worker = InstantWorkerClass(
            service_config=service_config,
            user_config=user_config,
            service_id="instant-worker-1",
        )

    @property
    def dropped_credits(self) -> list[Credit]:
        return self._worker.received_credits

    @property
    def dropped_timestamps(self) -> list[int]:
        return self._worker.received_timestamps

    async def create_orchestrator(
        self,
        config: TimingConfig,
        strategy_type: type[TimingStrategyProtocol]
        | None = None,  # Ignored - mode from config
        dataset_metadata: DatasetMetadata | None = None,
        **kwargs,  # Ignore other legacy params like auto_return_delay
    ) -> PhaseOrchestrator:
        if dataset_metadata is None:
            dataset_metadata = create_mock_dataset_metadata(
                conversation_ids=["conv1", "conv2", "conv3"]
            )
        orchestrator = PhaseOrchestrator(
            config=config,
            phase_publisher=self.publisher,
            credit_router=self.router,
            dataset_metadata=dataset_metadata,
        )
        await orchestrator.initialize()
        return orchestrator

    async def run_orchestrator(self, orchestrator: PhaseOrchestrator) -> None:
        # Initialize and start router first
        await self.router.initialize()
        await self.router.start()
        # Initialize and start worker (sends WorkerReady to router)
        await self._worker.initialize()
        await self._worker.start()
        task = asyncio.create_task(orchestrator.start())
        while not task.done():
            await asyncio.sleep(0.01)
        await task


def create_instant_worker_class():
    """Factory to create InstantWorker class that extends real Worker."""
    from aiperf.credit.structs import CreditContext
    from aiperf.workers import Worker

    class InstantWorker(Worker):
        """Worker that instantly returns credits - no inference, just tracking."""

        received_credits: list[Credit]
        received_timestamps: list[int]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.received_credits = []
            self.received_timestamps = []

        async def _process_credit(self, credit_context: CreditContext) -> None:
            """Override to skip inference - just track the credit."""
            self.received_credits.append(credit_context.credit)
            self.received_timestamps.append(time.time_ns())
            credit_context.first_token_sent = False

    return InstantWorker


@pytest.fixture
def timing_harness(
    service_config, user_config, skip_service_registration
) -> TimingHarness:
    """Fixture providing a test harness with real components and fake communication."""
    return TimingHarness(service_config=service_config, user_config=user_config)


def profiling_phase_stats_from_config(config: TimingConfig) -> CreditPhaseStats:
    """Create a phase stats object from a config.

    Finds the profiling phase config and uses its stop conditions.
    """
    # Find the profiling phase config
    profiling_config = next(
        (pc for pc in config.phase_configs if pc.phase == CreditPhase.PROFILING),
        config.phase_configs[0] if config.phase_configs else None,
    )
    return CreditPhaseStats(
        phase=CreditPhase.PROFILING,
        start_ns=time.time_ns(),
        total_expected_requests=profiling_config.total_expected_requests
        if profiling_config
        else None,
    )


def create_mock_dataset_metadata(
    conversation_ids: list[str],
    first_turn_timestamps: list[int | float | None] | None = None,
    turn_delays: list[list[int | float] | None] | None = None,
    turn_counts: list[int] | None = None,
    sampling_strategy: DatasetSamplingStrategy = DatasetSamplingStrategy.SEQUENTIAL,
) -> DatasetMetadata:
    """Create mock dataset metadata for testing.

    Args:
        conversation_ids: List of conversation IDs to include in the metadata.
        first_turn_timestamps: Optional list of first turn timestamps for each conversation.
        turn_delays: Optional list of turn delay lists for each conversation.
        turn_counts: Optional list of turn counts for each conversation.
        sampling_strategy: The sampling strategy for the dataset.

    Returns:
        DatasetMetadata: Mock dataset metadata for testing.
    """
    conversations = []
    for i, conv_id in enumerate(conversation_ids):
        turns = []
        turn_count = turn_counts[i] if turn_counts else 1

        if first_turn_timestamps:
            # Create first turn
            first_timestamp = first_turn_timestamps[i]
            turns.append(TurnMetadata(timestamp_ms=first_timestamp, delay_ms=None))

            # Create subsequent turns with delays
            if turn_count > 1:
                delays = (
                    turn_delays[i] if turn_delays and i < len(turn_delays) else None
                )
                if delays:
                    for j, delay in enumerate(delays):
                        # Calculate absolute timestamp from delay
                        if first_timestamp is not None:
                            timestamp = first_timestamp + sum(delays[: j + 1])
                        else:
                            timestamp = None
                        turns.append(
                            TurnMetadata(timestamp_ms=timestamp, delay_ms=delay)
                        )
                else:
                    # No delays provided, create turns without timing info
                    for _ in range(turn_count - 1):
                        turns.append(TurnMetadata(timestamp_ms=None, delay_ms=None))
        else:
            # No timing data, create empty turns
            for _ in range(turn_count):
                turns.append(TurnMetadata(timestamp_ms=None, delay_ms=None))

        metadata = ConversationMetadata(
            conversation_id=conv_id,
            turns=turns,
        )
        conversations.append(metadata)

    return DatasetMetadata(
        conversations=conversations,
        sampling_strategy=sampling_strategy,
    )


def create_mock_dataset_metadata_with_schedule(
    schedule: list[tuple[int, str]],
    sampling_strategy: DatasetSamplingStrategy = DatasetSamplingStrategy.SEQUENTIAL,
) -> DatasetMetadata:
    """Create mock dataset metadata from a schedule for fixed schedule testing.

    Args:
        schedule: List of tuples (timestamp, conversation_id).
        sampling_strategy: The sampling strategy for the dataset.

    Returns:
        DatasetMetadata: Mock dataset metadata with timing data.
    """
    # Group schedule by conversation_id
    conv_timestamps: dict[str, list[int]] = {}
    for timestamp, conv_id in schedule:
        if conv_id not in conv_timestamps:
            conv_timestamps[conv_id] = []
        conv_timestamps[conv_id].append(timestamp)

    conversations = []
    for conv_id, timestamps_list in conv_timestamps.items():
        # Create TurnMetadata objects for each turn
        turns = []
        for i, timestamp in enumerate(timestamps_list):
            if i == 0:
                # First turn has no delay
                turns.append(TurnMetadata(timestamp_ms=timestamp, delay_ms=None))
            else:
                # Subsequent turns have delay relative to previous turn
                delay = timestamp - timestamps_list[i - 1]
                turns.append(TurnMetadata(timestamp_ms=timestamp, delay_ms=delay))

        metadata = ConversationMetadata(
            conversation_id=conv_id,
            turns=turns,
        )
        conversations.append(metadata)

    return DatasetMetadata(
        conversations=conversations,
        sampling_strategy=sampling_strategy,
    )


# =============================================================================
# Common Helper Functions
# =============================================================================


def make_turn(
    conversation_id: str = "conv1",
    turn_index: int = 0,
    num_turns: int = 1,
    x_correlation_id: str | None = None,
) -> TurnToSend:
    """Create a TurnToSend for testing.

    Args:
        conversation_id: Conversation ID.
        turn_index: Turn index (0-based).
        num_turns: Total number of turns in the conversation.
        x_correlation_id: Correlation ID for sticky routing. Defaults to "corr-{conversation_id}".

    Returns:
        TurnToSend instance.
    """
    return TurnToSend(
        conversation_id=conversation_id,
        x_correlation_id=x_correlation_id or f"corr-{conversation_id}",
        turn_index=turn_index,
        num_turns=num_turns,
    )


def make_credit(
    credit_id: int = 1,
    conversation_id: str = "conv1",
    turn_index: int = 0,
    num_turns: int | None = None,
    is_final_turn: bool | None = None,
    phase: CreditPhase = CreditPhase.PROFILING,
    x_correlation_id: str | None = None,
) -> Credit:
    """Create a Credit for testing.

    Args:
        credit_id: Credit ID.
        conversation_id: Conversation ID.
        turn_index: Turn index (0-based).
        num_turns: Total number of turns. If provided, takes precedence.
        is_final_turn: Whether this is the final turn. Used to calculate num_turns if not provided.
        phase: Credit phase.
        x_correlation_id: Correlation ID for sticky routing. Defaults to "corr-{conversation_id}".

    Returns:
        Credit instance.

    Note:
        num_turns is determined as follows:
        - If num_turns is provided, use it directly
        - If is_final_turn=True, num_turns = turn_index + 1
        - If is_final_turn=False, num_turns = turn_index + 2 (at least one more turn)
        - If neither provided, defaults to is_final_turn=True behavior
    """
    if num_turns is not None:
        effective_num_turns = num_turns
    elif is_final_turn is not None:
        effective_num_turns = turn_index + 1 if is_final_turn else turn_index + 2
    else:
        # Default: treat as final turn (single-turn conversation)
        effective_num_turns = turn_index + 1
    return Credit(
        id=credit_id,
        phase=phase,
        conversation_id=conversation_id,
        x_correlation_id=x_correlation_id or f"corr-{conversation_id}",
        turn_index=turn_index,
        num_turns=effective_num_turns,
        issued_at_ns=time.time_ns(),
    )


def create_mock_dataset_sampler(
    conversation_ids: list[str] | None = None,
    sampling_strategy: DatasetSamplingStrategy = DatasetSamplingStrategy.SEQUENTIAL,
):
    """Create mock dataset sampler for testing.

    Args:
        conversation_ids: List of conversation IDs. Defaults to ["conv1", "conv2", "conv3"].
        sampling_strategy: Sampling strategy. Defaults to SEQUENTIAL.

    Returns:
        Dataset sampler instance.
    """
    if conversation_ids is None:
        conversation_ids = ["conv1", "conv2", "conv3"]
    return DatasetSamplingStrategyFactory.create_instance(
        sampling_strategy,
        conversation_ids=conversation_ids,
    )


# =============================================================================
# TimingConfig Helpers
# =============================================================================


def make_phase_config(
    phase: CreditPhase = CreditPhase.PROFILING,
    timing_mode: TimingMode = TimingMode.REQUEST_RATE,
    request_count: int | None = None,
    num_sessions: int | None = None,
    duration_sec: float | None = None,
    concurrency: int | None = None,
    prefill_concurrency: int | None = None,
    request_rate: float | None = None,
    arrival_pattern: ArrivalPattern = ArrivalPattern.POISSON,
    num_users: int | None = None,
    grace_period_sec: float | None = None,
    seamless: bool = False,
    auto_offset_timestamps: bool = False,
    fixed_schedule_start_offset: int | None = None,
    fixed_schedule_end_offset: int | None = None,
    concurrency_ramp_duration_sec: float | None = None,
    prefill_concurrency_ramp_duration_sec: float | None = None,
    request_rate_ramp_duration_sec: float | None = None,
) -> CreditPhaseConfig:
    """Create CreditPhaseConfig with sensible defaults for testing.

    Args:
        phase: The phase type (WARMUP or PROFILING).
        timing_mode: The timing mode (REQUEST_RATE, FIXED_SCHEDULE, USER_CENTRIC_RATE).
        request_count: Total expected requests (stop condition).
        num_sessions: Expected number of sessions (stop condition).
        duration_sec: Expected duration in seconds (stop condition).
        concurrency: Session concurrency limit.
        prefill_concurrency: Prefill concurrency limit.
        request_rate: Requests per second.
        arrival_pattern: Rate mode (POISSON, CONSTANT, CONCURRENCY_BURST).
        num_users: Number of concurrent users (for USER_CENTRIC_RATE mode).
        grace_period_sec: Grace period after stop condition met.
        seamless: Whether phase transitions without waiting for returns.
        auto_offset_timestamps: Auto-offset timestamps in fixed schedule mode.
        fixed_schedule_start_offset: Manual offset for fixed schedule mode.
        fixed_schedule_end_offset: End offset for fixed schedule mode.
        concurrency_ramp_duration_sec: Duration to ramp session concurrency.
        prefill_concurrency_ramp_duration_sec: Duration to ramp prefill concurrency.
        request_rate_ramp_duration_sec: Duration to ramp request rate.

    Returns:
        CreditPhaseConfig instance.
    """
    kwargs = {
        "phase": phase,
        "timing_mode": timing_mode,
        "total_expected_requests": request_count,
        "expected_num_sessions": num_sessions,
        "expected_duration_sec": duration_sec,
        "concurrency": concurrency,
        "prefill_concurrency": prefill_concurrency,
        "request_rate": request_rate,
        "arrival_pattern": arrival_pattern,
        "num_users": num_users,
        "grace_period_sec": grace_period_sec,
        "seamless": seamless,
        "auto_offset_timestamps": auto_offset_timestamps,
        "fixed_schedule_start_offset": fixed_schedule_start_offset,
        "fixed_schedule_end_offset": fixed_schedule_end_offset,
        "concurrency_ramp_duration_sec": concurrency_ramp_duration_sec,
        "prefill_concurrency_ramp_duration_sec": prefill_concurrency_ramp_duration_sec,
        "request_rate_ramp_duration_sec": request_rate_ramp_duration_sec,
    }
    return CreditPhaseConfig(**kwargs)


def make_timing_config(
    timing_mode: TimingMode = TimingMode.REQUEST_RATE,
    phase: CreditPhase = CreditPhase.PROFILING,
    request_count: int | None = None,
    num_sessions: int | None = None,
    duration_sec: float | None = None,
    concurrency: int | None = None,
    prefill_concurrency: int | None = None,
    request_rate: float | None = None,
    arrival_pattern: ArrivalPattern = ArrivalPattern.POISSON,
    num_users: int | None = None,
    grace_period_sec: float | None = None,
    random_seed: int | None = None,
    request_cancellation_rate: float | None = None,
    request_cancellation_delay: float = 0.0,
    auto_offset_timestamps: bool = False,
    fixed_schedule_start_offset: int | None = None,
    fixed_schedule_end_offset: int | None = None,
    phase_configs: list[CreditPhaseConfig] | None = None,
    concurrency_ramp_duration_sec: float | None = None,
    prefill_concurrency_ramp_duration_sec: float | None = None,
    request_rate_ramp_duration_sec: float | None = None,
) -> TimingConfig:
    """Create TimingConfig with sensible defaults for testing.

    This helper creates a single-phase TimingConfig. For multi-phase configs,
    pass phase_configs directly.

    Args:
        timing_mode: The timing mode enum value.
        phase: The phase type for single-phase config.
        request_count: Total expected requests.
        num_sessions: Expected number of sessions.
        duration_sec: Expected duration in seconds.
        concurrency: Session concurrency limit.
        prefill_concurrency: Prefill concurrency limit.
        request_rate: Requests per second.
        arrival_pattern: Rate mode.
        num_users: Number of concurrent users.
        grace_period_sec: Grace period after stop condition.
        random_seed: Random seed for reproducibility.
        request_cancellation_rate: Cancellation rate percentage.
        request_cancellation_delay: Delay before cancellation.
        auto_offset_timestamps: Auto-offset timestamps in fixed schedule mode.
        fixed_schedule_start_offset: Manual offset for fixed schedule.
        fixed_schedule_end_offset: End offset for fixed schedule.
        phase_configs: Optional pre-built phase configs (overrides other params).
        concurrency_ramp_duration_sec: Duration to ramp session concurrency.
        prefill_concurrency_ramp_duration_sec: Duration to ramp prefill concurrency.
        request_rate_ramp_duration_sec: Duration to ramp request rate.

    Returns:
        TimingConfig instance.
    """
    if phase_configs is None:
        phase_configs = [
            make_phase_config(
                phase=phase,
                timing_mode=timing_mode,
                request_count=request_count,
                num_sessions=num_sessions,
                duration_sec=duration_sec,
                concurrency=concurrency,
                prefill_concurrency=prefill_concurrency,
                request_rate=request_rate,
                arrival_pattern=arrival_pattern,
                num_users=num_users,
                grace_period_sec=grace_period_sec,
                auto_offset_timestamps=auto_offset_timestamps,
                fixed_schedule_start_offset=fixed_schedule_start_offset,
                fixed_schedule_end_offset=fixed_schedule_end_offset,
                concurrency_ramp_duration_sec=concurrency_ramp_duration_sec,
                prefill_concurrency_ramp_duration_sec=prefill_concurrency_ramp_duration_sec,
                request_rate_ramp_duration_sec=request_rate_ramp_duration_sec,
            )
        ]

    return TimingConfig(
        phase_configs=phase_configs,
        random_seed=random_seed,
        request_cancellation=RequestCancellationConfig(
            rate=request_cancellation_rate,
            delay=request_cancellation_delay,
        ),
    )


# =============================================================================
# Concurrency Stats Helpers
# =============================================================================


def get_session_stats(
    orchestrator: PhaseOrchestrator,
    phase: CreditPhase | None = None,
) -> ConcurrencyStats | None:
    """Get session concurrency stats from an orchestrator.

    Args:
        orchestrator: The PhaseOrchestrator to get stats from
        phase: If provided, get phase-specific stats. Otherwise get global stats.

    Returns:
        ConcurrencyStats or None if session concurrency not enabled.
    """
    return orchestrator._concurrency_manager.get_session_stats(phase)


# =============================================================================
# Mock Fixtures for PhaseRunner and Component Tests
# =============================================================================


@pytest.fixture
def mock_phase_publisher():
    """Create a MagicMock phase publisher."""

    async def async_noop(*args, **kwargs):
        return None

    mock = MagicMock()
    mock.publish_phase_start = async_noop
    mock.publish_phase_sending_complete = async_noop
    mock.publish_phase_complete = async_noop
    mock.publish_progress = async_noop
    mock.publish_credits_complete = async_noop
    return mock


@pytest.fixture
def mock_credit_router():
    """Create a MagicMock credit router."""

    async def async_noop(*args, **kwargs):
        return None

    mock = MagicMock()
    mock.send_credit = async_noop
    mock.cancel_all_credits = async_noop
    mock.mark_credits_complete = MagicMock()
    mock.reset = MagicMock()
    mock.set_return_callback = MagicMock()
    mock.set_first_token_callback = MagicMock()
    return mock


@pytest.fixture
def mock_concurrency_manager():
    """Create a MagicMock concurrency manager for testing."""

    async def async_return_true(*args, **kwargs):
        return True

    mock = MagicMock()
    mock.configure_for_phase = MagicMock()
    mock.acquire_session_slot = async_return_true
    mock.acquire_prefill_slot = async_return_true
    mock.release_session_slot = MagicMock()
    mock.release_prefill_slot = MagicMock()
    mock.set_session_limit = MagicMock()
    mock.set_prefill_limit = MagicMock()
    mock.release_stuck_slots = MagicMock(return_value=(0, 0))
    mock.get_session_stats = MagicMock(return_value=None)
    mock.get_prefill_stats = MagicMock(return_value=None)
    return mock


@pytest.fixture
def mock_stop_checker():
    """Create a MagicMock stop condition checker for testing."""

    mock = MagicMock()
    mock.can_send_any_turn = MagicMock(return_value=True)
    mock.can_start_new_session = MagicMock(return_value=True)
    return mock


@pytest.fixture
def mock_progress_tracker():
    """Create a MagicMock progress tracker for testing."""
    import asyncio

    mock = MagicMock()
    mock.increment_sent = MagicMock(return_value=(1, False))  # (credit_index, is_final)
    mock.increment_returned = MagicMock(return_value=False)  # is_final_returned
    mock.increment_prefill_released = MagicMock()
    mock.freeze_sent_counts = MagicMock()
    mock.freeze_completed_counts = MagicMock()
    mock.all_credits_sent_event = asyncio.Event()
    mock.all_credits_returned_event = asyncio.Event()
    mock.create_stats = MagicMock()
    mock.check_all_returned_or_cancelled = MagicMock(return_value=False)
    mock.in_flight_sessions = 0
    mock.counter = MagicMock()
    return mock


@pytest.fixture
def mock_lifecycle():
    """Create a MagicMock phase lifecycle for testing."""

    mock = MagicMock()
    mock.is_complete = False
    mock.is_sending_complete = False
    mock.started_at_perf_ns = 1_000_000_000  # 1 second in perf_counter_ns
    mock.start = MagicMock()
    mock.mark_sending_complete = MagicMock()
    mock.mark_complete = MagicMock()
    mock.cancel = MagicMock()
    mock.time_left_in_seconds = MagicMock(return_value=60.0)
    return mock


@pytest.fixture
def mock_callback_handler():
    """Create a MagicMock credit callback handler for testing."""

    async def async_noop(*args, **kwargs):
        return None

    mock = MagicMock()
    mock.register_phase = MagicMock()
    mock.unregister_phase = MagicMock()
    mock.on_credit_return = async_noop
    mock.on_first_token = async_noop
    return mock


@pytest.fixture
def mock_cancellation_policy():
    """Create a MagicMock cancellation policy for testing."""

    mock = MagicMock()
    mock.next_cancellation_delay_ns = MagicMock(return_value=None)
    return mock


# =============================================================================
# Legacy Fixture Aliases
# =============================================================================


@pytest.fixture
def mock_orchestrator(create_orchestrator_harness):
    """Legacy fixture alias for create_orchestrator_harness.

    Provides the same functionality as create_orchestrator_harness but
    with positional conversations argument for backwards compatibility.
    """

    def create(
        conversations: list[tuple[str, int]],
        *,
        num_sessions: int | None = None,
        request_count: int | None = None,
        concurrency: int | None = None,
        request_rate: float | None = None,
    ) -> OrchestratorHarness:
        return create_orchestrator_harness(
            conversations=conversations,
            num_sessions=num_sessions,
            request_count=request_count,
            concurrency=concurrency,
            request_rate=request_rate,
        )

    return create


# =============================================================================
# Mock Credit Sender (shared across credit-related tests)
# =============================================================================


class MockCreditSender:
    """Mock CreditSender for testing credit issuance without real routing.

    Use this for testing CreditManager in isolation.
    Use MockCreditRouter for tests that need auto-return behavior.
    """

    def __init__(self):
        self.sent_credits: list[Credit] = []
        self.cancelled = False
        self._callback: Callable[[str, CreditReturn], Awaitable[None]] | None = None

    async def send_credit(self, credit: Credit) -> None:
        self.sent_credits.append(credit)

    async def cancel_all_credits(self) -> None:
        self.cancelled = True

    def reset(self) -> None:
        self.sent_credits.clear()
        self.cancelled = False

    def set_return_callback(
        self, callback: Callable[[str, CreditReturn], Awaitable[None]]
    ) -> None:
        self._callback = callback


@pytest.fixture
def mock_credit_sender():
    """Create mock credit sender for testing."""
    return MockCreditSender()


# =============================================================================
# Router with Worker Fixture (shared across sticky router tests)
# =============================================================================


@pytest.fixture
def router_with_worker(service_config):
    """Fixture providing a StickyCreditRouter with one registered worker."""
    from aiperf.credit.sticky_router import StickyCreditRouter, WorkerLoad

    router = StickyCreditRouter(service_config=service_config, service_id="test-router")
    router._workers = {
        "worker-1": WorkerLoad(worker_id="worker-1", in_flight_credits=0)
    }
    return router


# =============================================================================
# Credit Return Helper
# =============================================================================


def make_credit_return(
    credit: Credit,
    cancelled: bool = False,
    first_token_sent: bool = True,
    error: str | None = None,
) -> CreditReturn:
    """Create a CreditReturn for testing."""
    return CreditReturn(
        credit=credit,
        cancelled=cancelled,
        first_token_sent=first_token_sent,
        error=error,
    )
