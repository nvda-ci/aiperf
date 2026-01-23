# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/nim/embeddings endpoint (NIM Image Embeddings)."""

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestNIMImageEmbeddingsEndpoint:
    """Tests for /v1/nim/embeddings endpoint (NIM Image Embeddings like C-RADIO)."""

    async def test_basic_nim_image_embeddings(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Basic NIM image embeddings request completes with expected request count."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/c-radio-v3-h \
                --tokenizer gpt2 \
                --url {aiperf_mock_server.url} \
                --endpoint-type nim_image_embeddings \
                --endpoint /v1/nim/embeddings \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        # NIM Image Embeddings are non-streaming, so streaming metrics should not be present
        assert result.json.time_to_first_token is None
