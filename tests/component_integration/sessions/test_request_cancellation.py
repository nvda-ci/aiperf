# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for request cancellation functionality."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestRequestCancellation:
    """Tests for request cancellation functionality."""

    def test_request_cancellation(self, cli: AIPerfCLI):
        """Request cancellation doesn't break pipeline."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 50 \
                --concurrency 5 \
                --random-seed 42 \
                --osl 10 \
                --request-cancellation-rate 30 \
                --request-cancellation-delay 0 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=120.0,
        )

        cancelled_requests = 0
        for request in result.jsonl:
            if request.metadata.was_cancelled:
                cancelled_requests += 1
                assert request.error is not None
                assert request.error.code == 499
                assert request.error.type == "RequestCancellationError"

        assert cancelled_requests > 5
        # This only applies when the actual benchmark is cancelled, not individually cancelled requests.
        assert result.json.was_cancelled is False
        assert result.json.error_summary is not None
        assert result.json.error_summary
        assert result.json.error_summary[0].count > 0
        assert result.json.error_summary[0].error_details.code == 499
        assert (
            result.json.error_summary[0].error_details.type
            == "RequestCancellationError"
        )
