# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for TimingManager and Worker with streaming ROUTER-DEALER communication.

Tests the complete workflow of:
1. Worker connects and sends WorkerReadyMessage
2. TimingManager auto-registers worker via callback
3. TimingManager sends credits via ROUTER
4. Worker processes and returns credits via DEALER
5. Worker sends WorkerShutdownMessage and disconnects
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.messages import (
    CreditDropMessage,
    CreditReturnMessage,
    WorkerReadyMessage,
    WorkerShutdownMessage,
)
from aiperf.timing.sticky_router import StickyCreditRouter
from aiperf.zmq.streaming_dealer_client import ZMQStreamingDealerClient
from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient


@pytest.mark.integration
class TestTimingManagerWorkerStreamingIntegration:
    """Test complete streaming workflow between TimingManager and Worker."""

    @pytest.mark.asyncio
    async def test_worker_registration_on_ready_message(self):
        """Should automatically register worker when WorkerReadyMessage received."""
        router_address = "tcp://127.0.0.1:45700"

        # Track worker registration
        registered_workers = []

        async def on_worker_connected(worker_id: str):
            registered_workers.append(worker_id)

        # Create ROUTER (simulating TimingManager)
        router = ZMQStreamingRouterClient(address=router_address, bind=True)
        router.register_receiver(AsyncMock())
        router.register_peer_connected_callback(on_worker_connected)
        await router.initialize()
        await router.start()

        # Create DEALER (simulating Worker)
        dealer = ZMQStreamingDealerClient(address=router_address, identity="worker-42")
        dealer.register_receiver(AsyncMock())
        await dealer.initialize()
        await dealer.start()

        try:
            # Send WorkerReadyMessage
            ready_msg = WorkerReadyMessage(service_id="worker-42")
            await dealer.send(ready_msg)

            # Wait for registration
            await asyncio.sleep(0.1)

            assert "worker-42" in registered_workers

        finally:
            await dealer.stop()
            await router.stop()

    @pytest.mark.asyncio
    async def test_worker_unregistration_on_shutdown_message(self):
        """Should automatically unregister worker when WorkerShutdownMessage received."""
        router_address = "tcp://127.0.0.1:45701"

        disconnected_workers = []

        async def on_worker_disconnected(worker_id: str):
            disconnected_workers.append(worker_id)

        # Create ROUTER
        router = ZMQStreamingRouterClient(address=router_address, bind=True)
        router.register_receiver(AsyncMock())
        router.register_peer_disconnected_callback(on_worker_disconnected)
        await router.initialize()
        await router.start()

        # Create DEALER
        dealer = ZMQStreamingDealerClient(address=router_address, identity="worker-99")
        dealer.register_receiver(AsyncMock())
        await dealer.initialize()
        await dealer.start()

        try:
            # Announce presence
            await dealer.send(WorkerReadyMessage(service_id="worker-99"))
            await asyncio.sleep(0.1)

            # Send shutdown message
            await dealer.send(WorkerShutdownMessage(service_id="worker-99"))
            await asyncio.sleep(0.1)

            assert "worker-99" in disconnected_workers

        finally:
            await dealer.stop()
            await router.stop()

    @pytest.mark.asyncio
    async def test_credit_drop_and_return_workflow(self):
        """Should handle complete credit drop and return workflow."""
        router_address = "tcp://127.0.0.1:45702"

        # Track received messages
        worker_credits = []
        router_returns = []

        async def router_handler(identity: str, message):
            if message.message_type == MessageType.CREDIT_RETURN:
                router_returns.append((identity, message))

        async def dealer_handler(message):
            if message.message_type == MessageType.CREDIT_DROP:
                worker_credits.append(message)

        # Create ROUTER (simulating TimingManager)
        router = ZMQStreamingRouterClient(address=router_address, bind=True)
        router.register_receiver(router_handler)
        await router.initialize()
        await router.start()

        # Create DEALER (simulating Worker)
        dealer = ZMQStreamingDealerClient(address=router_address, identity="worker-1")
        dealer.register_receiver(dealer_handler)
        await dealer.initialize()
        await dealer.start()

        try:
            # Worker announces presence
            await dealer.send(WorkerReadyMessage(service_id="worker-1"))
            await asyncio.sleep(0.05)

            # TimingManager sends credit
            credit = CreditDropMessage(
                service_id="timing-manager",
                phase=CreditPhase.WARMUP,
                credit_num=1,
                conversation_id="conv-123",
                turn_index=0,
            )
            await router.send_to("worker-1", credit)
            await asyncio.sleep(0.05)

            # Verify worker received credit
            assert len(worker_credits) == 1
            assert worker_credits[0].credit_num == 1
            assert worker_credits[0].conversation_id == "conv-123"

            # Worker sends return
            return_msg = CreditReturnMessage(
                service_id="worker-1",
                phase=CreditPhase.WARMUP,
                credit_drop_id="credit-1",
                requests_sent=1,
            )
            await dealer.send(return_msg)
            await asyncio.sleep(0.05)

            # Verify TimingManager received return
            assert len(router_returns) == 1
            assert router_returns[0][0] == "worker-1"  # Identity
            assert router_returns[0][1].requests_sent == 1

        finally:
            await dealer.stop()
            await router.stop()

    @pytest.mark.asyncio
    async def test_multiple_workers_registration_and_routing(self):
        """Should handle multiple workers with independent registration and routing."""
        router_address = "tcp://127.0.0.1:45703"

        registered_workers = []
        worker1_messages = []
        worker2_messages = []
        worker3_messages = []

        async def on_connected(worker_id: str):
            registered_workers.append(worker_id)

        async def worker1_handler(message):
            worker1_messages.append(message)

        async def worker2_handler(message):
            worker2_messages.append(message)

        async def worker3_handler(message):
            worker3_messages.append(message)

        # Create ROUTER
        router = ZMQStreamingRouterClient(address=router_address, bind=True)
        router.register_receiver(AsyncMock())
        router.register_peer_connected_callback(on_connected)
        await router.initialize()
        await router.start()

        # Create 3 workers
        worker1 = ZMQStreamingDealerClient(address=router_address, identity="worker-1")
        worker1.register_receiver(worker1_handler)
        await worker1.initialize()
        await worker1.start()

        worker2 = ZMQStreamingDealerClient(address=router_address, identity="worker-2")
        worker2.register_receiver(worker2_handler)
        await worker2.initialize()
        await worker2.start()

        worker3 = ZMQStreamingDealerClient(address=router_address, identity="worker-3")
        worker3.register_receiver(worker3_handler)
        await worker3.initialize()
        await worker3.start()

        try:
            # All workers announce
            await worker1.send(WorkerReadyMessage(service_id="worker-1"))
            await worker2.send(WorkerReadyMessage(service_id="worker-2"))
            await worker3.send(WorkerReadyMessage(service_id="worker-3"))
            await asyncio.sleep(0.1)

            # All should be registered
            assert set(registered_workers) == {"worker-1", "worker-2", "worker-3"}

            # Send targeted messages
            msg1 = CreditDropMessage(
                service_id="tm", phase=CreditPhase.WARMUP, credit_num=1
            )
            msg2 = CreditDropMessage(
                service_id="tm", phase=CreditPhase.WARMUP, credit_num=2
            )
            msg3 = CreditDropMessage(
                service_id="tm", phase=CreditPhase.WARMUP, credit_num=3
            )

            await router.send_to("worker-1", msg1)
            await router.send_to("worker-2", msg2)
            await router.send_to("worker-3", msg3)
            await asyncio.sleep(0.1)

            # Verify correct routing
            assert len(worker1_messages) == 1
            assert worker1_messages[0].credit_num == 1

            assert len(worker2_messages) == 1
            assert worker2_messages[0].credit_num == 2

            assert len(worker3_messages) == 1
            assert worker3_messages[0].credit_num == 3

        finally:
            await worker1.stop()
            await worker2.stop()
            await worker3.stop()
            await router.stop()


@pytest.mark.integration
class TestStickyCreditRouterIntegrationWithStreaming:
    """Test StickyCreditRouter integration with streaming clients."""

    @pytest.mark.asyncio
    async def test_sticky_router_fair_load_balancing(self):
        """Should distribute first-turn credits fairly across workers."""
        router_address = "tcp://127.0.0.1:45704"
        sticky_router = StickyCreditRouter()

        # Track which worker receives which credit
        worker_assignments = {}

        async def on_connected(worker_id: str):
            await sticky_router.register_worker(worker_id)

        async def router_handler(identity: str, message):
            pass  # Not needed for this test

        # Create ROUTER with StickyCreditRouter
        router = ZMQStreamingRouterClient(address=router_address, bind=True)
        router.register_receiver(router_handler)
        router.register_peer_connected_callback(on_connected)
        await router.initialize()
        await router.start()

        # Create 3 workers
        workers = []
        for i in range(3):
            worker_id = f"worker-{i}"
            dealer = ZMQStreamingDealerClient(
                address=router_address, identity=worker_id
            )
            dealer.register_receiver(AsyncMock())
            await dealer.initialize()
            await dealer.start()
            workers.append((worker_id, dealer))

            # Announce worker
            await dealer.send(WorkerReadyMessage(service_id=worker_id))

        await asyncio.sleep(0.1)

        try:
            # Verify all workers registered with StickyCreditRouter
            assert len(sticky_router.workers) == 3
            assert "worker-0" in sticky_router.workers
            assert "worker-1" in sticky_router.workers
            assert "worker-2" in sticky_router.workers

            # Send multiple first-turn credits (turn_index=0)
            # StickyCreditRouter should distribute them fairly
            for credit_num in range(9):
                credit = CreditDropMessage(
                    service_id="tm",
                    phase=CreditPhase.WARMUP,
                    credit_num=credit_num,
                    request_id=f"req-{credit_num}",
                    conversation_id=f"conv-{credit_num}",
                    turn_index=0,  # First turn
                    is_final_turn=True,
                )

                # Route using StickyCreditRouter
                worker_id = await sticky_router.determine_credit_route(credit)
                worker_assignments[credit_num] = worker_id

                # Track credit sent
                await sticky_router.track_credit_sent(worker_id)

            # Verify fair distribution (each worker should get ~3 credits)
            worker_counts = {}
            for worker_id in worker_assignments.values():
                worker_counts[worker_id] = worker_counts.get(worker_id, 0) + 1

            # All workers should have been used
            assert len(worker_counts) == 3
            # Each should have gotten roughly equal share (within ±2)
            for count in worker_counts.values():
                assert 1 <= count <= 5  # Fair distribution

        finally:
            for _, dealer in workers:
                await dealer.stop()
            await router.stop()

    @pytest.mark.asyncio
    async def test_sticky_router_sticky_routing(self):
        """Should route subsequent turns to same worker (sticky sessions)."""
        router_address = "tcp://127.0.0.1:45705"
        sticky_router = StickyCreditRouter()

        async def on_connected(worker_id: str):
            await sticky_router.register_worker(worker_id)

        # Create ROUTER
        router = ZMQStreamingRouterClient(address=router_address, bind=True)
        router.register_receiver(AsyncMock())
        router.register_peer_connected_callback(on_connected)
        await router.initialize()
        await router.start()

        # Create 2 workers
        workers = []
        for i in range(2):
            worker_id = f"worker-{i}"
            dealer = ZMQStreamingDealerClient(
                address=router_address, identity=worker_id
            )
            dealer.register_receiver(AsyncMock())
            await dealer.initialize()
            await dealer.start()
            workers.append((worker_id, dealer))
            await dealer.send(WorkerReadyMessage(service_id=worker_id))

        await asyncio.sleep(0.1)

        try:
            # Send first turn (should be routed by fair load balancing)
            credit_turn_0 = CreditDropMessage(
                service_id="tm",
                phase=CreditPhase.WARMUP,
                credit_num=1,
                request_id="req-1",
                conversation_id="conv-multi-turn",
                turn_index=0,
                is_final_turn=False,
            )

            worker_for_turn_0 = await sticky_router.determine_credit_route(
                credit_turn_0
            )
            await sticky_router.track_credit_sent(worker_for_turn_0)

            # Send subsequent turns (should stick to same worker)
            for turn_index in range(1, 5):
                credit = CreditDropMessage(
                    service_id="tm",
                    phase=CreditPhase.WARMUP,
                    credit_num=turn_index + 1,
                    request_id="req-1",  # Same request_id (same conversation instance)
                    conversation_id="conv-multi-turn",
                    turn_index=turn_index,
                    is_final_turn=(turn_index == 4),
                )

                worker_id = await sticky_router.determine_credit_route(credit)
                await sticky_router.track_credit_sent(worker_id)

                # All turns should go to same worker
                assert worker_id == worker_for_turn_0

        finally:
            for _, dealer in workers:
                await dealer.stop()
            await router.stop()

    @pytest.mark.asyncio
    async def test_complete_credit_lifecycle_with_returns(self):
        """Should handle complete credit lifecycle: drop → process → return."""
        router_address = "tcp://127.0.0.1:45706"
        sticky_router = StickyCreditRouter()

        credits_received_by_worker = []
        returns_received_by_router = []

        async def on_connected(worker_id: str):
            await sticky_router.register_worker(worker_id)

        async def router_handler(identity: str, message):
            """Router receives returns from workers."""
            if message.message_type == MessageType.CREDIT_RETURN:
                returns_received_by_router.append((identity, message))
                # Simulate TimingManager tracking
                await sticky_router.track_credit_returned(identity)

        async def worker_handler(message):
            """Worker receives credits and sends returns."""
            if message.message_type == MessageType.CREDIT_DROP:
                credits_received_by_worker.append(message)

                # Simulate worker processing and returning credit
                return_msg = CreditReturnMessage(
                    service_id="worker-1",
                    phase=message.phase,
                    credit_drop_id=f"credit-{message.credit_num}",
                    requests_sent=1,
                )
                await worker.send(return_msg)

        # Create ROUTER and DEALER
        router = ZMQStreamingRouterClient(address=router_address, bind=True)
        router.register_receiver(router_handler)
        router.register_peer_connected_callback(on_connected)
        await router.initialize()
        await router.start()

        worker = ZMQStreamingDealerClient(address=router_address, identity="worker-1")
        worker.register_receiver(worker_handler)
        await worker.initialize()
        await worker.start()

        try:
            # Worker announces
            await worker.send(WorkerReadyMessage(service_id="worker-1"))
            await asyncio.sleep(0.05)

            # Send 5 credits
            for i in range(5):
                credit = CreditDropMessage(
                    service_id="tm",
                    phase=CreditPhase.WARMUP,
                    credit_num=i,
                    request_id=f"req-{i}",
                    conversation_id=f"conv-{i}",
                    turn_index=0,
                    is_final_turn=True,
                )

                await router.send_to("worker-1", credit)
                await sticky_router.track_credit_sent("worker-1")

            # Wait for all to be processed
            await asyncio.sleep(0.2)

            # Verify all credits received by worker
            assert len(credits_received_by_worker) == 5

            # Verify all returns received by router
            assert len(returns_received_by_router) == 5

            # Verify in-flight credits back to 0
            assert sticky_router.workers["worker-1"].in_flight_credits == 0
            assert sticky_router.workers["worker-1"].total_processed == 5

        finally:
            await worker.stop()
            await router.stop()

    @pytest.mark.asyncio
    async def test_worker_reconnection_after_shutdown(self):
        """Should handle worker reconnecting after graceful shutdown."""
        router_address = "tcp://127.0.0.1:45707"

        connected_count = 0
        disconnected_count = 0

        async def on_connected(worker_id: str):
            nonlocal connected_count
            connected_count += 1

        async def on_disconnected(worker_id: str):
            nonlocal disconnected_count
            disconnected_count += 1

        # Create ROUTER
        router = ZMQStreamingRouterClient(address=router_address, bind=True)
        router.register_receiver(AsyncMock())
        router.register_peer_connected_callback(on_connected)
        router.register_peer_disconnected_callback(on_disconnected)
        await router.initialize()
        await router.start()

        try:
            # First connection
            dealer1 = ZMQStreamingDealerClient(
                address=router_address, identity="worker-1"
            )
            dealer1.register_receiver(AsyncMock())
            await dealer1.initialize()
            await dealer1.start()
            await dealer1.send(WorkerReadyMessage(service_id="worker-1"))
            await asyncio.sleep(0.05)

            assert connected_count == 1

            # Graceful shutdown
            await dealer1.send(WorkerShutdownMessage(service_id="worker-1"))
            await asyncio.sleep(0.05)
            await dealer1.stop()

            assert disconnected_count == 1

            # Reconnect with same identity
            dealer2 = ZMQStreamingDealerClient(
                address=router_address, identity="worker-1"
            )
            dealer2.register_receiver(AsyncMock())
            await dealer2.initialize()
            await dealer2.start()
            await dealer2.send(WorkerReadyMessage(service_id="worker-1"))
            await asyncio.sleep(0.05)

            assert connected_count == 2  # Reconnection detected
            await dealer2.stop()

        finally:
            await router.stop()


# ============================================================================
# Message Protocol Tests
# ============================================================================


@pytest.mark.integration
class TestStreamingMessageProtocol:
    """Test the message protocol between ROUTER and DEALER."""

    @pytest.mark.asyncio
    async def test_worker_ready_message_triggers_registration(self):
        """WorkerReadyMessage should trigger peer connected callback."""
        router_address = "tcp://127.0.0.1:45708"
        registration_event = asyncio.Event()
        registered_worker = None

        async def on_connected(worker_id: str):
            nonlocal registered_worker
            registered_worker = worker_id
            registration_event.set()

        router = ZMQStreamingRouterClient(address=router_address, bind=True)
        router.register_receiver(AsyncMock())
        router.register_peer_connected_callback(on_connected)
        await router.initialize()
        await router.start()

        dealer = ZMQStreamingDealerClient(
            address=router_address, identity="worker-test"
        )
        dealer.register_receiver(AsyncMock())
        await dealer.initialize()
        await dealer.start()

        try:
            await dealer.send(WorkerReadyMessage(service_id="worker-test"))
            await asyncio.wait_for(registration_event.wait(), timeout=1.0)

            assert registered_worker == "worker-test"

        finally:
            await dealer.stop()
            await router.stop()

    @pytest.mark.asyncio
    async def test_worker_shutdown_message_triggers_unregistration(self):
        """WorkerShutdownMessage should trigger peer disconnected callback."""
        router_address = "tcp://127.0.0.1:45709"
        unregistration_event = asyncio.Event()
        unregistered_worker = None

        async def on_disconnected(worker_id: str):
            nonlocal unregistered_worker
            unregistered_worker = worker_id
            unregistration_event.set()

        router = ZMQStreamingRouterClient(address=router_address, bind=True)
        router.register_receiver(AsyncMock())
        router.register_peer_disconnected_callback(on_disconnected)
        await router.initialize()
        await router.start()

        dealer = ZMQStreamingDealerClient(
            address=router_address, identity="worker-test"
        )
        dealer.register_receiver(AsyncMock())
        await dealer.initialize()
        await dealer.start()

        try:
            # Connect first
            await dealer.send(WorkerReadyMessage(service_id="worker-test"))
            await asyncio.sleep(0.05)

            # Then disconnect
            await dealer.send(WorkerShutdownMessage(service_id="worker-test"))
            await asyncio.wait_for(unregistration_event.wait(), timeout=1.0)

            assert unregistered_worker == "worker-test"

        finally:
            await dealer.stop()
            await router.stop()

    @pytest.mark.asyncio
    async def test_mixed_message_types_in_stream(self):
        """Should handle mix of WorkerReady, CreditDrop, CreditReturn, WorkerShutdown messages."""
        router_address = "tcp://127.0.0.1:45710"

        router_messages = []
        worker_messages = []

        async def router_handler(identity: str, message):
            router_messages.append((identity, message.message_type))

        async def worker_handler(message):
            worker_messages.append(message.message_type)

        router = ZMQStreamingRouterClient(address=router_address, bind=True)
        router.register_receiver(router_handler)
        router.register_peer_connected_callback(AsyncMock())
        router.register_peer_disconnected_callback(AsyncMock())
        await router.initialize()
        await router.start()

        dealer = ZMQStreamingDealerClient(address=router_address, identity="worker-1")
        dealer.register_receiver(worker_handler)
        await dealer.initialize()
        await dealer.start()

        try:
            # Send sequence of different message types
            await dealer.send(WorkerReadyMessage(service_id="worker-1"))
            await asyncio.sleep(0.05)

            await router.send_to(
                "worker-1",
                CreditDropMessage(
                    service_id="tm", phase=CreditPhase.WARMUP, credit_num=1
                ),
            )
            await asyncio.sleep(0.05)

            await dealer.send(
                CreditReturnMessage(
                    service_id="worker-1",
                    phase=CreditPhase.WARMUP,
                    credit_drop_id="1",
                    requests_sent=1,
                )
            )
            await asyncio.sleep(0.05)

            await dealer.send(WorkerShutdownMessage(service_id="worker-1"))
            await asyncio.sleep(0.05)

            # Verify messages received in correct order
            # Router should see: WorkerReady, CreditReturn (not WorkerShutdown - it's filtered)
            router_types = [msg_type for _, msg_type in router_messages]
            assert MessageType.WORKER_READY in router_types
            assert MessageType.CREDIT_RETURN in router_types
            assert MessageType.WORKER_SHUTDOWN not in router_types  # Filtered out

            # Worker should see: CreditDrop
            assert MessageType.CREDIT_DROP in worker_messages

        finally:
            await dealer.stop()
            await router.stop()
