# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/embeddings endpoint."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestEmbeddingsEndpoint:
    """Tests for /v1/embeddings endpoint."""

    def test_basic_embeddings(self, cli: AIPerfCLI):
        """Basic embeddings request."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model nomic-ai/nomic-embed-text-v1.5 \
                --tokenizer gpt2 \
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
