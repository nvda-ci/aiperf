# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sticky credit router with fair load balancing and sticky session routing."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from aiperf.common import random_generator as rng
from aiperf.common.config import ServiceConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import CommAddress, MessageType
from aiperf.common.messages import (
    CreditDropMessage,
    CreditReturnMessage,
    Message,
)
from aiperf.common.messages.credit_messages import Credit
from aiperf.common.mixins import CommunicationMixin
from aiperf.common.protocols import (
    StickyCreditRouterProtocol,
    StreamingRouterClientProtocol,
)


@dataclass(slots=True)
class StickySession:
    """Tracks conversation to worker assignment for sticky routing.

    Args:
        x_correlation_id: The ID of the conversation.
        worker_id: The ID of the worker.
        turns_processed: The number of turns processed for the conversation.
    """

    x_correlation_id: str
    worker_id: str
    turns_processed: int = 0


@dataclass(slots=True)
class WorkerLoad:
    """Tracks worker load for fair load balancing.

    Args:
        worker_id: The ID of the worker.
        total_sent_credits: The total number of credits sent to the worker.
        total_returned_credits: The total number of credits returned from the worker.
        in_flight_credits: The number of credits currently in flight to the worker.
        last_activity_ns: The timestamp of the last activity of the worker.
    """

    worker_id: str
    total_sent_credits: int = 0
    total_returned_credits: int = 0
    in_flight_credits: int = 0
    last_activity_ns: int = field(default_factory=time.perf_counter_ns)


@implements_protocol(StickyCreditRouterProtocol)
class StickyCreditRouter(CommunicationMixin):
    """Sticky credit router with fair load balancing and sticky session routing.

    Routes credits from TimingManager to Workers using ZMQ ROUTER/DEALER pattern.
    Enables optimal worker utilization and conversation caching via sticky routing.

    Architecture:

                         ┌──────────────────────┐
                         │   TimingManager      │
                         │ (RequestRateStrategy)│
                         └──────────┬───────────┘
                                    │ send_credit()
                                    ▼
                         ┌──────────────────────┐
                         │  StickyCreditRouter  │  ◀─────┐
                         │   (this class)       │         │
                         │                      │         │
                         │  ┌────────────────┐  │         │
                         │  │ Sticky Sessions│  │         │
                         │  │ (sticky cache) │  │         │
                         │  └────────────────┘  │         │
                         │                      │         │
                         │  ┌────────────────┐  │         │
                         │  │  Worker Loads  │  │         │
                         │  │ (in-flight cnt)│  │         │
                         │  └────────────────┘  │         │
                         └──────────┬───────────┘         │
                                    │                     │
                       ┌────────────┼────────────┐        │
                       │            │            │        │
                       ▼            ▼            ▼        │
                 ┌─────────┐  ┌─────────┐  ┌─────────┐    │
                 │Worker-A │  │Worker-B │  │Worker-C │    │
                 │ (DEALER)│  │ (DEALER)│  │ (DEALER)│    │
                 └────┬────┘  └────┬────┘  └────┬────┘    │
                      │            │            │         │
                      └────────────┴────────────┘         │
                             CreditReturn                 │
                      (updates in-flight count)───────────┘

    Routing Strategy:

    Credit:
    ══════════════════════════════════════════════════════════════════════
    First turn (turn_index=0):
      1. Check sticky sessions map → Not found
      2. Load balance: Select worker with minimum in-flight credits
      3. Create sticky session: {x_correlation_id → worker_id}
      4. Route credit to selected worker

    Subsequent turns (turn_index>0):
      1. Check sticky sessions map → Found
      2. Validate worker still registered
      3. Route credit to same worker (sticky!)
      4. If final_turn: Delete sticky session from map

    Why Sticky Routing Matters:
    ─────────────────────────────────────────────────────────────────────
    1. Session state management:
       - Worker maintains conversation state (user + assistant turns)
       - No need to re-fetch conversation from DatasetManager
       - Worker can build full conversation history across turns

    2. Optimal load distribution:
       - First turn: Fair distribution via least in-flight algorithm
       - Subsequent turns: Sticky to same worker
       - Self-balancing over time

    3. Graceful worker failure:
       - Worker dies → sticky session becomes stale
       - Next turn: Reassign to new worker automatically
       - Automatic recovery without manual intervention

    Concurrency Model:
    ─────────────────────────────────────────────────────────────────────
    - Lock-free routing: sticky sessions dict (turns serialized by strategy)
    - Lock-free counters: in_flight_credits (atomic in asyncio event loop)
    - Single lock: worker registration/unregistration only
    - High performance: No contention in hot path
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        **kwargs,
    ) -> None:
        super().__init__(service_config=service_config, **kwargs)

        self._router_rng = rng.derive("timing.smart_credit_router")

        self.router_client: StreamingRouterClientProtocol = (
            self.comms.create_streaming_router_client(
                address=CommAddress.CREDIT_ROUTER,
                bind=True,
            )
        )
        self.router_client.register_receiver(self._handle_router_message)

        self.on_return_callback: (
            Callable[[str, CreditReturnMessage], Awaitable[None]] | None
        ) = None
        self.sticky_sessions: dict[str, StickySession] = {}
        self.workers: dict[str, WorkerLoad] = {}
        self.worker_registration_lock: asyncio.Lock = asyncio.Lock()

    def set_return_callback(
        self, callback: "Callable[[str, Message], Awaitable[None]]"
    ) -> None:
        """Set callback to be invoked when workers return credits.

        Used by TimingManager to receive notifications when credits complete,
        enabling concurrency control (semaphore release) and progress tracking.

        Args:
            callback: Async function called with (worker_id, CreditReturnMessage)
        """
        self.on_return_callback = callback

    async def register_worker(self, worker_id: str) -> None:
        """Register worker for routing (called when WorkerReadyMessage received).

        Creates WorkerLoad entry to track in-flight credits for load balancing.
        Thread-safe via worker_registration_lock.

        Args:
            worker_id: Unique identifier for the worker (service_id)
        """
        async with self.worker_registration_lock:
            if worker_id not in self.workers:
                self.workers[worker_id] = WorkerLoad(worker_id=worker_id)
                if self.is_trace_enabled:
                    self.trace(
                        f"Worker registered: {worker_id} (total={len(self.workers)})"
                    )

    async def unregister_worker(self, worker_id: str) -> None:
        """Unregister worker (called when WorkerShutdownMessage received or worker dies).

        Removes worker from routing pool. Assigned conversations will automatically
        fail to sticky route and be reassigned to a different worker on next turn.
        Thread-safe via worker_registration_lock.

        Args:
            worker_id: Unique identifier for the worker to remove

        Note: Sticky sessions are NOT deleted - they become stale and trigger reassignment
        on next access. This allows graceful handling of worker failures.
        """
        async with self.worker_registration_lock:
            worker_load = self.workers.pop(worker_id, None)
            if worker_load:
                if worker_load.in_flight_credits > 0:
                    self.warning(
                        f"Worker {worker_id} unregistered with {worker_load.in_flight_credits} in-flight credits"
                    )
                if self.is_trace_enabled:
                    self.trace(
                        f"Worker unregistered: {worker_id} (remaining={len(self.workers)})"
                    )

    def determine_credit_route(self, credit: Credit) -> str:
        """Determine optimal worker for credit using sticky routing and load balancing.

        Routing Logic:
        - Credit: Check sticky session cache for sticky routing, else load balance
        """
        if not self.workers:
            raise RuntimeError("No workers available for routing")

        if not credit.x_correlation_id:
            raise RuntimeError("x_correlation_id must be set in Credit")

        return self._route_credit(credit)

    def _route_credit(self, credit: Credit) -> str:
        """Route credit with sticky routing to enable worker session state reuse.

        Algorithm:
        1. Lookup x_correlation_id in sticky sessions map
        2. If found and worker still valid → Route to same worker (sticky!)
        3. If not found or stale → Load balance to least-loaded worker
        4. If not final turn → Create sticky session for future turns
        5. If final turn → Delete sticky session to free memory

        Sticky routing ensures:
        - Same worker processes all turns of a conversation
        - Worker can maintain session state with conversation history
        - No need to re-fetch conversation from DatasetManager

        Args:
            credit: Credit with x_correlation_id for sticky tracking

        Returns:
            worker_id: ID of selected worker (sticky if assignment exists, else load-balanced)

        Note: Lock-free operation (safe because turns are serialized by strategy)
        """
        x_correlation_id = credit.x_correlation_id
        sticky_session = self.sticky_sessions.get(x_correlation_id)

        # Use existing sticky session if worker still valid
        if sticky_session and sticky_session.worker_id in self.workers:
            worker_id = sticky_session.worker_id
            sticky_session.turns_processed += 1
        else:
            # Load balance to least-loaded worker
            worker_id = self._select_least_loaded_worker()

            # Create sticky session for future turns
            if not credit.is_final_turn:
                self.sticky_sessions[x_correlation_id] = StickySession(
                    x_correlation_id=x_correlation_id,
                    worker_id=worker_id,
                    turns_processed=1,
                )

        # Cleanup on final turn
        if credit.is_final_turn and x_correlation_id in self.sticky_sessions:
            del self.sticky_sessions[x_correlation_id]

        return worker_id

    async def send_credit(self, service_id: str, credit: Credit) -> None:
        """Route credit to optimal worker and send via ROUTER socket.

        Flow:
        1. Determine target worker (via sticky routing or load balancing)
        2. Update worker load tracking (increment in_flight_credits)
        3. Send credit via ROUTER socket to specific worker DEALER

        Args:
            credit: Credit to route and send
        """
        worker_id = self.determine_credit_route(credit)
        self.track_credit_sent(worker_id)
        await self.router_client.send_to(
            worker_id, CreditDropMessage(service_id=service_id, credit=credit)
        )

    async def _handle_router_message(self, worker_id: str, message: Message) -> None:
        """Handle messages from workers received via ROUTER socket.

        Message Types:
        - CREDIT_RETURN: Worker finished processing, update load tracking
        - WORKER_READY: Worker started, register for routing
        - WORKER_SHUTDOWN: Worker stopping, unregister from routing

        Args:
            worker_id: ID of worker that sent the message (from ROUTER envelope)
            message: Message from worker (deserialized)

        Note: ROUTER socket automatically provides worker_id from message envelope.
        """
        match message.message_type:
            case MessageType.CREDIT_RETURN:
                self.track_credit_returned(worker_id)
                if self.on_return_callback:
                    await self.on_return_callback(worker_id, message)
            case MessageType.PARTIAL_CREDIT_NOTIFICATION:
                pass  # TODO: Implement partial credit notification handling
            case MessageType.WORKER_READY:
                await self.register_worker(worker_id)
            case MessageType.WORKER_SHUTDOWN:
                await self.unregister_worker(worker_id)
            case _:
                self.warning(f"Unknown message type: {message.message_type}")

    def _select_least_loaded_worker(self) -> str:
        """Select worker with minimum in-flight credits using fair load balancing.

        Algorithm:
        1. Find minimum in-flight credit count across all workers
        2. Collect all workers with that minimum load
        3. Random selection among tied workers (for reproducibility)

        This ensures:
        - Fair distribution of work across workers
        - Deterministic behavior with seeded RNG
        - Self-balancing as workers complete credits

        Returns:
            worker_id: ID of least-loaded worker (random if tie)
        """
        min_load = min(w.in_flight_credits for w in self.workers.values())
        min_load_workers = [
            w for w in self.workers.values() if w.in_flight_credits == min_load
        ]
        return self._router_rng.choice(min_load_workers).worker_id

    def track_credit_sent(self, worker_id: str) -> None:
        """Update worker load tracking when credit sent to worker.

        Increments:
        - total_sent_credits: Lifetime counter
        - in_flight_credits: Current concurrency (used for load balancing)

        Args:
            worker_id: ID of worker that received the credit
        """
        if worker_load := self.workers.get(worker_id):
            worker_load.total_sent_credits += 1
            worker_load.in_flight_credits += 1
            worker_load.last_activity_ns = time.perf_counter_ns()
        else:
            self.error(f"Worker {worker_id} not found when tracking sent credit")

    def track_credit_returned(self, worker_id: str) -> None:
        """Update worker load tracking when credit returned from worker.

        Increments:
        - total_returned_credits: Lifetime counter
        Decrements:
        - in_flight_credits: Current concurrency (frees up capacity)

        Args:
            worker_id: ID of worker that returned the credit
        """
        if worker_load := self.workers.get(worker_id):
            worker_load.total_returned_credits += 1
            worker_load.in_flight_credits -= 1
            worker_load.last_activity_ns = time.perf_counter_ns()
        else:
            self.error(f"Worker {worker_id} not found when tracking returned credit")
