# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for StickyCreditRouter."""

import pytest

from aiperf.common.config import ServiceConfig
from aiperf.common.enums import CreditPhase
from aiperf.common.messages.credit_messages import Credit, CreditDropMessage
from aiperf.timing.sticky_router import StickyCreditRouter, WorkerLoad


@pytest.fixture
def service_config():
    """Fixture providing a ServiceConfig for router tests."""
    return ServiceConfig()


def create_test_credit(
    credit_num: int = 1,
    conversation_id: str = "session-1",
    turn_index: int = 0,
    x_correlation_id: str = "test-corr-id",
    is_final_turn: bool = False,
    phase: CreditPhase = CreditPhase.PROFILING,
    total_turns: int | None = None,
) -> CreditDropMessage:
    """Helper to create test credit drop messages."""
    credit = Credit(
        phase=phase,
        num=credit_num,
        conversation_id=conversation_id,
        x_correlation_id=x_correlation_id,
        turn_index=turn_index,
        is_final_turn=is_final_turn,
    )
    return CreditDropMessage(service_id="timing-manager", credit=credit)


class TestStickyCreditRouterFairLoadBalancing:
    """Test fair load balancing for first turns."""

    async def test_routes_to_least_loaded_worker(self, service_config):
        """Test that first turn routes to worker with minimum in-flight credits."""
        router = StickyCreditRouter(service_config=service_config)

        # Register workers with different loads
        router.workers = {
            "worker-1": WorkerLoad(worker_id="worker-1", in_flight_credits=5),
            "worker-2": WorkerLoad(worker_id="worker-2", in_flight_credits=2),
            "worker-3": WorkerLoad(worker_id="worker-3", in_flight_credits=8),
        }

        credit_msg = create_test_credit(
            credit_num=1,
            conversation_id="session-1",
            turn_index=0,
            x_correlation_id="test-corr-id-1",
            is_final_turn=False,
        )

        worker_id = router.determine_credit_route(credit_msg.credit)

        assert worker_id == "worker-2"
        assert len(router.sticky_sessions) == 1
        assert list(router.sticky_sessions.values())[0].worker_id == "worker-2"

    async def test_creates_conversation_assignment(self, service_config):
        """Test that first turn creates sticky assignment for future turns."""
        router = StickyCreditRouter(service_config=service_config)
        await router.register_worker("worker-A")

        credit_msg = create_test_credit()

        router.determine_credit_route(credit_msg.credit)

        assert len(router.sticky_sessions) == 1
        assignment = router.sticky_sessions["test-corr-id"]
        assert assignment.worker_id == "worker-A"
        assert assignment.turns_processed == 1

    async def test_error_if_no_workers_available(self, service_config):
        """Test that routing fails if no workers registered."""
        router = StickyCreditRouter(service_config=service_config)

        credit_msg = create_test_credit()

        with pytest.raises(RuntimeError, match="No workers available"):
            router.determine_credit_route(credit_msg.credit)


class TestStickyCreditRouterStickyRouting:
    """Test sticky routing for subsequent turns."""

    async def test_routes_to_assigned_worker(self, service_config):
        """Test that subsequent turns route to assigned worker."""
        router = StickyCreditRouter(service_config=service_config)
        await router.register_worker("worker-A")
        await router.register_worker("worker-B")

        instance_id = "test-instance-123"

        from aiperf.timing.sticky_router import StickySession

        router.sticky_sessions[instance_id] = StickySession(
            x_correlation_id=instance_id,
            worker_id="worker-A",
            turns_processed=1,
        )

        credit_msg = create_test_credit(
            credit_num=2,
            conversation_id="session-123",
            turn_index=1,
            x_correlation_id=instance_id,
        )

        worker_id = router.determine_credit_route(credit_msg.credit)

        assert worker_id == "worker-A"
        assert router.sticky_sessions[instance_id].turns_processed == 2

    async def test_cleans_up_assignment_on_final_turn(self, service_config):
        """Test that assignment is removed after final turn."""
        router = StickyCreditRouter(service_config=service_config)
        await router.register_worker("worker-A")

        instance_id = "test-instance-456"

        from aiperf.timing.sticky_router import StickySession

        router.sticky_sessions[instance_id] = StickySession(
            x_correlation_id=instance_id,
            worker_id="worker-A",
            turns_processed=4,
        )

        credit_msg = create_test_credit(
            credit_num=5,
            conversation_id="session-456",
            turn_index=4,
            x_correlation_id=instance_id,
            is_final_turn=True,
        )

        worker_id = router.determine_credit_route(credit_msg.credit)
        assert worker_id == "worker-A"
        assert instance_id not in router.sticky_sessions

    async def test_fallback_to_fair_load_if_assignment_missing(self, service_config):
        """Test fallback to fair load balancing if assignment not found."""
        router = StickyCreditRouter(service_config=service_config)
        await router.register_worker("worker-A")
        await router.register_worker("worker-B")

        router.workers["worker-A"].in_flight_credits = 5
        router.workers["worker-B"].in_flight_credits = 2

        credit_msg = create_test_credit(
            credit_num=10,
            conversation_id="session-999",
            turn_index=1,
        )

        worker_id = router.determine_credit_route(credit_msg.credit)
        assert worker_id == "worker-B"


class TestStickyCreditRouterLoadTracking:
    """Test worker load tracking."""

    async def test_track_credit_sent(self, service_config):
        """Test load tracking when credit sent."""
        router = StickyCreditRouter(service_config=service_config)
        await router.register_worker("worker-1")

        assert router.workers["worker-1"].in_flight_credits == 0

        router.track_credit_sent("worker-1")
        assert router.workers["worker-1"].in_flight_credits == 1
        assert router.workers["worker-1"].total_sent_credits == 1

        router.track_credit_sent("worker-1")
        assert router.workers["worker-1"].in_flight_credits == 2
        assert router.workers["worker-1"].total_sent_credits == 2

    async def test_track_credit_returned(self, service_config):
        """Test load tracking when credit returned."""
        router = StickyCreditRouter(service_config=service_config)
        await router.register_worker("worker-1")

        router.workers["worker-1"].in_flight_credits = 5

        router.track_credit_returned("worker-1")
        assert router.workers["worker-1"].in_flight_credits == 4
        assert router.workers["worker-1"].total_returned_credits == 1

        router.track_credit_returned("worker-1")
        assert router.workers["worker-1"].in_flight_credits == 3
        assert router.workers["worker-1"].total_returned_credits == 2

    async def test_register_worker(self, service_config):
        """Test worker registration."""
        router = StickyCreditRouter(service_config=service_config)

        await router.register_worker("worker-A")

        assert "worker-A" in router.workers
        assert router.workers["worker-A"].in_flight_credits == 0
        assert router.workers["worker-A"].total_returned_credits == 0

    async def test_unregister_worker(self, service_config):
        """Test worker unregistration."""
        router = StickyCreditRouter(service_config=service_config)
        await router.register_worker("worker-A")

        await router.unregister_worker("worker-A")

        assert "worker-A" not in router.workers


class TestStickyCreditRouterCompleteScenario:
    """Test complete routing scenario with multiple conversations."""

    async def test_five_turn_conversation(self, service_config):
        """Test routing a complete 5-turn conversation."""
        router = StickyCreditRouter(service_config=service_config)

        await router.register_worker("worker-A")
        await router.register_worker("worker-B")
        await router.register_worker("worker-C")

        instance_id = "test-corr-id"

        # Turn 1 (first turn, fair load)
        credit1_msg = create_test_credit(
            credit_num=1,
            conversation_id="session-test",
            turn_index=0,
            x_correlation_id=instance_id,
            is_final_turn=False,
        )

        worker1 = router.determine_credit_route(credit1_msg.credit)
        router.track_credit_sent(worker1)
        assert worker1 in ["worker-A", "worker-B", "worker-C"]

        # Turns 2-5 (sticky)
        for turn_idx in range(1, 5):
            credit_msg = create_test_credit(
                credit_num=turn_idx + 1,
                conversation_id="session-test",
                turn_index=turn_idx,
                x_correlation_id=instance_id,
                is_final_turn=(turn_idx == 4),
            )
            worker = router.determine_credit_route(credit_msg.credit)
            assert worker == worker1

        # Assignment should be cleaned up after final turn
        assert instance_id not in router.sticky_sessions

    async def test_multiple_conversations_balanced(self, service_config):
        """Test that multiple conversations are balanced across workers."""
        router = StickyCreditRouter(service_config=service_config)

        for i in range(3):
            await router.register_worker(f"worker-{i}")

        # Route first turns of 9 conversations
        instance_ids = []
        for i in range(9):
            instance_id = f"instance-{i}"
            credit_msg = create_test_credit(
                credit_num=i,
                conversation_id=f"session-{i}",
                turn_index=0,
                x_correlation_id=instance_id,
            )

            worker_id = router.determine_credit_route(credit_msg.credit)
            router.track_credit_sent(worker_id)
            instance_ids.append(instance_id)

        # Each worker should get 3 conversations (balanced)
        assert all(w.in_flight_credits == 3 for w in router.workers.values())

        # Route second turns (should be sticky)
        for instance_id in instance_ids:
            credit_msg = create_test_credit(
                credit_num=100,
                conversation_id="session-test",
                turn_index=1,
                x_correlation_id=instance_id,
            )

            worker_id = router.determine_credit_route(credit_msg.credit)
            assert worker_id == router.sticky_sessions[instance_id].worker_id


class TestStickyCreditRouterEdgeCases:
    """Test edge cases and error handling."""

    async def test_single_worker(self, service_config):
        """Test with only one worker."""
        router = StickyCreditRouter(service_config=service_config)
        await router.register_worker("only-worker")

        for i in range(10):
            credit_msg = create_test_credit(
                credit_num=i,
                conversation_id=f"session-{i}",
                turn_index=0,
                x_correlation_id=f"test-corr-id-{i}",
                is_final_turn=True,
            )

            worker_id = router.determine_credit_route(credit_msg.credit)
            assert worker_id == "only-worker"

    async def test_unequal_worker_loads(self, service_config):
        """Test fair load balancing with significantly unequal loads."""
        router = StickyCreditRouter(service_config=service_config)

        await router.register_worker("worker-overloaded")
        await router.register_worker("worker-idle")

        router.workers["worker-overloaded"].in_flight_credits = 100
        router.workers["worker-idle"].in_flight_credits = 0

        credit_msg = create_test_credit()

        worker_id = router.determine_credit_route(credit_msg.credit)
        assert worker_id == "worker-idle"

    async def test_worker_registration_idempotent(self, service_config):
        """Test that re-registering worker is safe."""
        router = StickyCreditRouter(service_config=service_config)

        await router.register_worker("worker-1")
        router.workers["worker-1"].in_flight_credits = 5

        await router.register_worker("worker-1")
        assert router.workers["worker-1"].in_flight_credits == 5
