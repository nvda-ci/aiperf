# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for timing mode conflict validation."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestTimingModeConflicts:
    """Tests for mutual exclusivity and conflicts between timing modes."""

    @pytest.mark.slow
    def test_request_rate_with_num_sessions(self, cli: AIPerfCLI):
        """Test request rate with num-sessions specified.

        Both should work together - request rate controls timing,
        num-sessions limits total sessions.
        """
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-rate 200 \
                --random-seed 42 \
                --num-sessions 8 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        # 8 sessions × 3 turns = 24 requests
        assert result.request_count == 24

    @pytest.mark.slow
    def test_workers_max_does_not_limit_concurrency(self, cli: AIPerfCLI):
        """Test that workers-max is independent of concurrency setting.

        workers-max is about worker pool size, concurrency is about
        in-flight requests. They should work independently.
        """
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --concurrency 8 \
                --workers-max 3 \
                --num-sessions 24 \
                --ui {defaults.ui}
            """
        )
        # Should still handle concurrency=8 even with fewer workers
        assert result.request_count == 24
