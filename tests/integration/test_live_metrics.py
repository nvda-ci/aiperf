# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for live metrics endpoint functionality."""

import asyncio
import socket

import aiohttp
import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.models import AIPerfMockServer


@pytest.fixture
def live_metrics_port() -> int:
    """Get an available port for the live metrics server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.integration
@pytest.mark.asyncio
class TestLiveMetrics:
    """Tests for live metrics endpoint functionality."""

    async def test_live_metrics_endpoint(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        live_metrics_port: int,
    ):
        """Live metrics endpoint returns valid Prometheus format with correct labels."""
        metrics_url = f"http://127.0.0.1:{live_metrics_port}/metrics"
        health_url = f"http://127.0.0.1:{live_metrics_port}/health"
        captured_metrics: list[str] = []
        health_responses: list[int] = []

        async def poll_endpoints() -> None:
            """Poll the live metrics and health endpoints while benchmark runs."""
            async with aiohttp.ClientSession() as session:
                # Wait for server to be ready
                for _ in range(50):
                    try:
                        async with session.get(
                            health_url, timeout=aiohttp.ClientTimeout(total=1)
                        ) as resp:
                            if resp.status == 200:
                                health_responses.append(resp.status)
                                break
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        pass
                    await asyncio.sleep(0.1)

                # Poll both endpoints during benchmark
                for _ in range(20):
                    try:
                        async with session.get(
                            health_url, timeout=aiohttp.ClientTimeout(total=1)
                        ) as resp:
                            health_responses.append(resp.status)
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        pass

                    try:
                        async with session.get(
                            metrics_url, timeout=aiohttp.ClientTimeout(total=2)
                        ) as resp:
                            if resp.status == 200:
                                content = await resp.text()
                                if content.strip():
                                    captured_metrics.append(content)
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        pass
                    await asyncio.sleep(0.5)

        # Run benchmark and poll concurrently
        benchmark_task = asyncio.create_task(
            cli.run(
                f"""
                aiperf profile \
                    --model {defaults.model} \
                    --url {aiperf_mock_server.url} \
                    --endpoint-type chat \
                    --request-count 20 \
                    --request-rate 10 \
                    --workers-max {defaults.workers_max} \
                    --live-metrics-port {live_metrics_port} \
                    --ui {defaults.ui}
                """
            )
        )
        poll_task = asyncio.create_task(poll_endpoints())

        result = await benchmark_task
        poll_task.cancel()

        # Verify benchmark succeeded
        assert result.exit_code == 0
        assert result.request_count == 20

        # Verify health endpoint
        assert len(health_responses) > 0, "Should have received health responses"
        assert all(s == 200 for s in health_responses), (
            "All health checks should return 200"
        )

        # Verify metrics captured
        assert len(captured_metrics) > 0, "Should have captured metrics responses"

        # Find metrics with actual performance data
        metrics_with_data = [
            m
            for m in captured_metrics
            if "aiperf_request_" in m or "aiperf_output_" in m
        ]
        metrics_content = (
            metrics_with_data[-1] if metrics_with_data else captured_metrics[-1]
        )

        # Verify Prometheus format
        assert "# HELP" in metrics_content
        assert "# TYPE" in metrics_content

        # Verify info metric has version and config
        info_lines = [
            line for line in metrics_content.split("\n") if "aiperf_info{" in line
        ]
        assert len(info_lines) > 0, "Should have aiperf_info metric"
        info_line = info_lines[0]
        assert 'version="' in info_line, "Info metric should have version label"
        assert 'config="' in info_line, "Info metric should have config label"

        # Verify regular metrics have key labels but not config
        metric_lines = [
            line
            for line in metrics_content.split("\n")
            if line.startswith("aiperf_") and "{" in line and "aiperf_info" not in line
        ]
        if metric_lines:
            sample_line = metric_lines[0]
            assert 'model="' in sample_line, f"Should have model label: {sample_line}"
            assert 'endpoint_type="' in sample_line, (
                f"Should have endpoint_type: {sample_line}"
            )
            assert 'config="' not in sample_line, (
                f"Should NOT have config: {sample_line}"
            )
