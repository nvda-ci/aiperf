# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/chat/completions endpoint."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestChatEndpoint:
    """Tests for /v1/chat/completions endpoint."""

    def test_basic_chat(self, cli: AIPerfCLI):
        """Basic non-streaming chat completion."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model microsoft/phi-4 \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
