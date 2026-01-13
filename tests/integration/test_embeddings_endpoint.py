# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/embeddings endpoint."""

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestEmbeddingsEndpoint:
    """Tests for /v1/embeddings endpoint."""

    async def test_basic_embeddings(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Basic embeddings request."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nomic-ai/nomic-embed-text-v1.5 \
                --tokenizer gpt2 \
                --url {aiperf_mock_server.url} \
                --endpoint-type embeddings \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        assert (
            not hasattr(result.json, "time_to_first_token")
            or result.json.time_to_first_token is None
        )
