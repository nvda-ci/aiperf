# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/completions endpoint."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestCompletionsEndpoint:
    """Tests for /v1/completions endpoint."""

    def test_basic_completions(self, cli: AIPerfCLI):
        """Basic non-streaming completions."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type completions \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
