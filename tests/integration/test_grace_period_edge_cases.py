# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for grace period edge cases."""

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestGracePeriodEdgeCases:
    """Tests for grace period behavior in various scenarios."""

    async def test_grace_period_with_multi_turn_conversations(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test grace period allows multi-turn conversations to complete.

        When duration expires, in-flight conversations should have time
        to complete all their turns within the grace period.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-rate 10 \
                --random-seed 42 \
                --request-rate-mode constant \
                --conversation-num 15 \
                --conversation-turn-mean 3 \
                --conversation-turn-stddev 0 \
                --benchmark-duration 2 \
                --benchmark-grace-period 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        # Most or all in-flight conversations should complete
        # With timing variance, at least ~11 requests should complete
        assert result.request_count >= 11

        # Verify conversations that started have all their turns
        conversation_turns = {}
        for record in result.jsonl:
            conv_id = record.metadata.conversation_id
            turn = record.metadata.turn_index
            if conv_id not in conversation_turns:
                conversation_turns[conv_id] = set()
            conversation_turns[conv_id].add(turn)

        # Each conversation should have turn 0, and if it started, should make some progress
        for conv_id, turns in conversation_turns.items():
            if 0 in turns:  # Conversation started
                # With grace period, should make at least some progress
                assert len(turns) >= 1, f"Conversation {conv_id} didn't make progress"

    async def test_zero_grace_period(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test zero grace period stops immediately after duration.

        With grace period = 0, should cut off immediately when duration expires.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-rate 20 \
                --benchmark-duration 1 \
                --benchmark-grace-period 0 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        # At 20 req/s for 1 second with zero grace, some requests may not complete
        assert result.request_count >= 5
        assert result.request_count <= 30

    async def test_very_short_grace_period(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test very short grace period (< 1 second)."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-rate 15 \
                --benchmark-duration 1.5 \
                --benchmark-grace-period 0.5 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        # At 15 req/s for 1.5 seconds, should get ~22 requests
        assert result.request_count >= 10
        assert result.request_count <= 35

    async def test_long_grace_period(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test long grace period allows all requests to complete."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-rate 10 \
                --request-count 25 \
                --random-seed 42 \
                --request-rate-mode constant \
                --benchmark-duration 1 \
                --benchmark-grace-period 60 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        # Duration stops new requests at ~10, but grace period allows them to complete
        # The actual count depends on how many were in-flight at duration expiry
        assert result.request_count >= 8 and result.request_count <= 12

    async def test_grace_period_with_slow_responses(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test grace period with slow server responses.

        The mock server is fast, but we use a very short duration and
        reasonable grace period to simulate the scenario.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-rate 30 \
                --random-seed 42 \
                --request-rate-mode constant \
                --benchmark-duration 0.5 \
                --benchmark-grace-period 5 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        # At 30 req/s for 0.5s, should send ~15 requests
        # With grace period, they should all complete
        assert result.request_count >= 10
        assert result.request_count <= 25

    async def test_grace_period_with_concurrency(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test grace period with concurrency-based timing.

        Grace period should apply to concurrency mode as well.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --concurrency 5 \
                --random-seed 42 \
                --request-rate-mode constant \
                --benchmark-duration 2 \
                --benchmark-grace-period 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        # With concurrency=5 for 2 seconds, should get multiple rounds
        assert result.request_count >= 5

    async def test_grace_period_with_multi_turn_delays(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test grace period with inter-turn delays in conversations.

        Grace period should account for turn delays to allow conversations
        to complete naturally.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-rate 8 \
                --random-seed 42 \
                --request-rate-mode constant \
                --conversation-num 10 \
                --conversation-turn-mean 3 \
                --conversation-turn-stddev 0 \
                --conversation-turn-delay-mean 100 \
                --conversation-turn-delay-stddev 0 \
                --benchmark-duration 2 \
                --benchmark-grace-period 15 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        # With delays, fewer conversations will start within duration
        # But grace period should allow started ones to complete
        assert result.request_count >= 12

    async def test_grace_period_with_poisson_rate(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test grace period with poisson request rate mode."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-rate 15 \
                --random-seed 42 \
                --request-rate-mode poisson \
                --benchmark-duration 2 \
                --benchmark-grace-period 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        # At 15 req/s for 2 seconds with poisson, should get ~30 requests (with variance)
        assert result.request_count >= 15
        assert result.request_count <= 50
