# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the global request rate configuration."""

import pytest
from pytest import approx

from tests.integration.conftest import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.models import AIPerfMockServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestGlobalRequestRate:
    """Test the global request rate configuration."""

    async def test_global_request_rate(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test the global request rate configuration."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-rate 5 \
                --request-rate-mode constant \
                --request-count 25 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.average_request_send_rate == approx(5.0, abs=0.5)
        assert result.average_inter_send_time == approx(0.2, abs=0.1)
        assert result.request_count == 25

    async def test_multi_turn_request_rate(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test the multi-turn request rate configuration."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-rate 5 \
                --request-rate-mode constant \
                --conversation-turn-mean 5 \
                --conversation-turn-stddev 0 \
                --request-count 25 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.average_request_send_rate == approx(5.0, abs=0.5)
        assert result.average_inter_send_time == approx(0.2, abs=0.1)
        assert result.request_count == 25
        turn_indices = set(record.metadata.turn_index for record in result.jsonl)
        # Assert that there were 5 turns in the conversation
        # The fast response from the server should ensure that we keep sending more turns
        # instead of starting a new conversation
        assert len(turn_indices) == 5

    async def test_multi_turn_concurrency(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test the multi-turn concurrency configuration."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --concurrency 3 \
                --conversation-num 5 \
                --conversation-turn-mean 5 \
                --conversation-turn-stddev 0 \
                --conversation-turn-delay-mean 100 \
                --conversation-turn-delay-stddev 0 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        turn_indices = set(record.metadata.turn_index for record in result.jsonl)
        # Assert that there were 5 turns in the conversation
        # The fast response from the server should ensure that we keep sending more turns
        # instead of starting a new conversation
        assert len(turn_indices) == 5
        assert result.request_count == 25
