# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /rag/api/prompt endpoint."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestSolidoRAGEndpoint:
    """Tests for /rag/api/prompt endpoint."""

    def test_basic_solido_rag(self, cli: AIPerfCLI):
        """Basic SOLIDO RAG request."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model rag-model \
                --tokenizer gpt2 \
                --endpoint-type solido_rag \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count

    def test_solido_rag_multiple_requests(self, cli: AIPerfCLI):
        """SOLIDO RAG with multiple concurrent requests."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model rag-model \
                --tokenizer gpt2 \
                --endpoint-type solido_rag \
                --request-count 20 \
                --concurrency 4 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == 20
