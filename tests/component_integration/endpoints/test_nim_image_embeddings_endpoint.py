# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for NIM Image Embeddings endpoint."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI

# Mock server uses /v1/nim_image_embeddings, real API uses /v1/embeddings
MOCK_SERVER_ENDPOINT = "/v1/nim_image_embeddings"


@pytest.mark.component_integration
class TestNIMImageEmbeddingsEndpoint:
    """Tests for NIM Image Embeddings endpoint (C-RADIO)."""

    def test_basic_nim_image_embeddings(self, cli: AIPerfCLI):
        """Basic NIM image embeddings request."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model nvidia/c-radio-v3-h \
                --tokenizer gpt2 \
                --endpoint-type nim_image_embeddings \
                --custom-endpoint {MOCK_SERVER_ENDPOINT} \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        # NIM Image Embeddings don't stream, so streaming metrics should be absent
        assert result.json.time_to_first_token is None

    def test_nim_image_embeddings_with_max_pixels(self, cli: AIPerfCLI):
        """NIM image embeddings request with max_pixels configuration."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model nvidia/c-radio-v3-h \
                --tokenizer gpt2 \
                --endpoint-type nim_image_embeddings \
                --custom-endpoint {MOCK_SERVER_ENDPOINT} \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui} \
                --extra-inputs max_pixels:50176
            """
        )
        assert result.request_count == defaults.request_count
        assert result.json.time_to_first_token is None

    def test_nim_image_embeddings_all_extra_params(self, cli: AIPerfCLI):
        """NIM image embeddings with pyramid, max_pixels, and encoding_format via JSON."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model nvidia/c-radio-v3-h \
                --tokenizer gpt2 \
                --endpoint-type nim_image_embeddings \
                --custom-endpoint {MOCK_SERVER_ENDPOINT} \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui} \
                --extra-inputs '{{"pyramid": [[1,1],[3,3],[5,5]], "max_pixels": 50176, "encoding_format": "float"}}'
            """
        )
        assert result.request_count == defaults.request_count
        assert result.json.time_to_first_token is None
