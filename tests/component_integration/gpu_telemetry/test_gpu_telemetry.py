# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration tests for GPU telemetry collection with DCGM faker."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as Defaults,
)


@pytest.mark.component_integration
class TestGPUTelemetryBasic:
    """Basic GPU telemetry collection tests."""

    def test_dcgm_endpoints(self, cli, mock_dcgm_endpoints):
        """Test profile with multiple DCGM endpoints."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {Defaults.model} \
                --url http://localhost:8000 \
                --gpu-telemetry http://localhost:9401/metrics \
                --request-count 10 \
                --concurrency 2 \
                --workers-max {Defaults.workers_max} \
                --ui {Defaults.ui}
            """
        )

        assert result.exit_code == 0
        assert result.request_count == 10
        assert result.has_gpu_telemetry
        assert result.json.telemetry_data.endpoints is not None
        assert len(result.json.telemetry_data.endpoints) == 2
        for dcgm_url in result.json.telemetry_data.endpoints:
            assert result.json.telemetry_data.endpoints[dcgm_url].gpus is not None
            assert len(result.json.telemetry_data.endpoints[dcgm_url].gpus) == 2
            for gpu_data in result.json.telemetry_data.endpoints[
                dcgm_url
            ].gpus.values():
                assert gpu_data.metrics is not None
                assert gpu_data.metrics
                for _, metric_value in gpu_data.metrics.items():
                    assert metric_value is not None
                    assert metric_value.avg is not None
                    assert metric_value.min is not None
                    assert metric_value.max is not None
                    assert metric_value.unit is not None
