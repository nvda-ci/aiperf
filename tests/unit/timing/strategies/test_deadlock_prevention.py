# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for RequestRateStrategy deadlock prevention.

Uses real PhaseOrchestrator with mock credit router - only the transport
layer is mocked to capture credits and inject returns.

Note: Time is globally mocked via looptime, so no explicit timeouts needed.
"""

import pytest

# =============================================================================
# Tests for single-turn deadlock prevention
# =============================================================================


@pytest.mark.asyncio
class TestSingleTurnExitsCleanly:
    """Single-turn conversations complete without deadlock."""

    async def test_rate_centric_with_session_limit(self, mock_orchestrator):
        """Single-turn rate-centric with session limit completes."""
        orchestrator = mock_orchestrator(
            [("c1", 1), ("c2", 1), ("c3", 1)],
            num_sessions=3,
            concurrency=None,
            request_rate=1000.0,
        )

        await orchestrator.run_with_auto_return()

        assert len(orchestrator.sent_credits) == 3

    async def test_rate_centric_with_request_limit(self, mock_orchestrator):
        """Single-turn rate-centric with request limit completes."""
        orchestrator = mock_orchestrator(
            [("c1", 1), ("c2", 1), ("c3", 1)],
            request_count=3,
            concurrency=None,
            request_rate=1000.0,
        )

        await orchestrator.run_with_auto_return()

        assert len(orchestrator.sent_credits) == 3

    async def test_session_centric_mode(self, mock_orchestrator):
        """Single-turn session-centric completes."""
        orchestrator = mock_orchestrator(
            [("c1", 1), ("c2", 1), ("c3", 1)],
            num_sessions=3,
            concurrency=10,
            request_rate=1000.0,
        )

        await orchestrator.run_with_auto_return()

        assert len(orchestrator.sent_credits) == 3


# =============================================================================
# Tests for multi-turn handling
# =============================================================================


@pytest.mark.asyncio
class TestMultiTurnHandling:
    """Multi-turn conversations process all turns correctly."""

    async def test_rate_centric_processes_all_turns(self, mock_orchestrator):
        """Rate-centric mode processes all turns via queue."""
        orchestrator = mock_orchestrator(
            [("c1", 3), ("c2", 2)],
            num_sessions=2,
            concurrency=None,
            request_rate=1000.0,
        )

        await orchestrator.run_with_auto_return()

        assert len(orchestrator.sent_credits) == 5  # All turns: 3 + 2

    async def test_session_centric_processes_all_turns(self, mock_orchestrator):
        """Session-centric dispatches all turns via callbacks."""
        orchestrator = mock_orchestrator(
            [("c1", 3), ("c2", 2)],
            num_sessions=2,
            concurrency=10,
            request_rate=1000.0,
        )

        await orchestrator.run_with_auto_return()

        assert len(orchestrator.sent_credits) == 5  # All turns: 3 + 2


# =============================================================================
# Tests for limit semantics
# =============================================================================


@pytest.mark.asyncio
class TestLimitSemantics:
    """Verify request_count vs num_sessions semantics."""

    async def test_request_count_limits_total_requests(self, mock_orchestrator):
        """request_count limits total requests, not sessions."""
        orchestrator = mock_orchestrator(
            [("c1", 5), ("c2", 5)],
            request_count=3,
            concurrency=1,
            request_rate=1000.0,
        )

        await orchestrator.run_with_auto_return()

        # 3 total requests
        assert len(orchestrator.sent_credits) == 3

    async def test_num_sessions_allows_all_turns_within(self, mock_orchestrator):
        """num_sessions limits sessions but allows all turns within them."""
        orchestrator = mock_orchestrator(
            [("c1", 3), ("c2", 3)],
            num_sessions=2,
            concurrency=10,
            request_rate=1000.0,
        )

        await orchestrator.run_with_auto_return()

        # All 6 turns (2 sessions × 3 turns)
        assert len(orchestrator.sent_credits) == 6

        # Only 2 unique sessions
        sessions = {c.x_correlation_id for c in orchestrator.sent_credits}
        assert len(sessions) == 2
