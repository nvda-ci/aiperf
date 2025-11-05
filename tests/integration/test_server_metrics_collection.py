# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for server metrics collection and reporting."""

import platform

import orjson
import pytest

from aiperf.common.models import ServerMetricRecord
from tests.integration.conftest import AIPerfCLI
from tests.integration.models import AIPerfMockServer


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="This test is flaky on macOS in Github Actions.",
)
@pytest.mark.integration
@pytest.mark.asyncio
class TestServerMetricsCollection:
    """Tests for server metrics collection and reporting."""

    async def test_server_metrics_collection(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Server metrics collection with server metrics endpoint."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --server-metrics {" ".join(aiperf_mock_server.server_metrics_urls)} \
                --streaming \
                --request-count 100 \
                --concurrency 2 \
                --workers-max 2 \
                --ui dashboard
            """
        )
        assert result.request_count == 100
        assert result.has_server_metrics
        assert result.json.server_metrics_data.summary.endpoints_successful is not None
        assert len(result.json.server_metrics_data.summary.endpoints_successful) > 0

        # Verify endpoints hierarchy exists
        assert result.json.server_metrics_data.endpoints is not None
        assert len(result.json.server_metrics_data.endpoints) > 0

        # Verify each server endpoint has server data
        for _, endpoint_data in result.json.server_metrics_data.endpoints.items():
            assert endpoint_data.servers is not None
            assert len(endpoint_data.servers) > 0

            for server_summary in endpoint_data.servers.values():
                assert server_summary.server_id is not None
                assert server_summary.metrics is not None
                assert len(server_summary.metrics) > 0

                # Verify metrics have statistics
                for metric_result in server_summary.metrics.values():
                    assert (
                        metric_result.avg is not None or metric_result.min is not None
                    )

    async def test_server_metrics_export(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test server metrics export to JSONL file with validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --server-metrics {" ".join(aiperf_mock_server.server_metrics_urls)} \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2
            """
        )
        assert result.request_count == 50
        assert result.has_server_metrics

        # Verify server metrics export JSONL file exists
        export_file = result.artifacts_dir / "server_metrics_export.jsonl"
        assert export_file.exists(), "Server metrics export file should exist"

        # Read and validate JSONL content
        content = export_file.read_text()
        lines = content.splitlines()
        assert len(lines) > 0, "Export file should contain server metric records"

        # Collect server data for validation
        server_ids = set()
        timestamps = []

        # Validate each line is valid JSON and can be parsed as ServerMetricRecord
        for line in lines:
            record_dict = orjson.loads(line)
            record = ServerMetricRecord.model_validate(record_dict)

            # Verify required fields are present
            assert record.timestamp_ns > 0
            assert record.server_url is not None
            assert record.server_id is not None
            assert record.metrics_data is not None

            # Collect data for validation
            server_ids.add(record.server_id)
            timestamps.append(record.timestamp_ns)

            # Verify metrics_data contains some metrics
            metrics_dict = record.metrics_data.model_dump()
            valid_metrics = {v for v in metrics_dict.values() if v is not None}
            assert len(valid_metrics) > 0, "Each record should have some metrics"

        # Verify we captured data from servers
        assert len(server_ids) >= 2, "Should have records from at least two servers"

        # Verify records are chronologically ordered by timestamp
        assert timestamps == sorted(timestamps), "Records should be in timestamp order"

    async def test_server_metrics_export_with_custom_prefix(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test server metrics export with custom filename prefix."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --server-metrics {" ".join(aiperf_mock_server.server_metrics_urls)} \
                --streaming \
                --request-count 25 \
                --concurrency 1 \
                --workers-max 1 \
                --profile-export-prefix custom_test
            """
        )

        # Verify custom filename is used
        export_file = result.artifacts_dir / "custom_test_server_metrics.jsonl"
        if export_file.exists():
            # Verify content is valid
            content = export_file.read_text()
            lines = content.splitlines()
            assert len(lines) > 0, "Export file should contain server metric records"

            # Validate first record
            first_record = ServerMetricRecord.model_validate_json(lines[0])
            assert first_record.timestamp_ns > 0
            assert first_record.server_url is not None
            assert first_record.metrics_data is not None

    async def test_multiple_server_endpoints(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test that multiple server endpoints are collected separately."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --server-metrics {" ".join(aiperf_mock_server.server_metrics_urls)} \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2
            """
        )
        assert result.request_count == 50
        assert result.has_server_metrics

        # Verify we have multiple endpoints configured
        endpoints = result.json.server_metrics_data.summary.endpoints_configured
        assert len(endpoints) >= 2, "Should have at least 2 server endpoints configured"

        # Verify endpoints are successful
        successful = result.json.server_metrics_data.summary.endpoints_successful
        assert len(successful) >= 2, "At least 2 server endpoints should be successful"

        # Verify each endpoint has its own server data
        server_endpoints = result.json.server_metrics_data.endpoints
        assert len(server_endpoints) >= 2, (
            "Should have data for at least 2 server endpoints"
        )

        # Verify each endpoint has servers with metrics
        for _, endpoint_data in server_endpoints.items():
            assert len(endpoint_data.servers) >= 1, (
                "Endpoint should have at least 1 server"
            )
            for server_summary in endpoint_data.servers.values():
                assert len(server_summary.metrics) > 0, "Server should have metrics"

    async def test_server_metrics_with_ai_specific_metrics(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test that AI-specific server metrics (Dynamo, KV cache) are collected."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --server-metrics {aiperf_mock_server.server_metrics_urls[0]} \
                --streaming \
                --request-count 30 \
                --concurrency 1 \
                --workers-max 1
            """
        )
        assert result.request_count == 30
        assert result.has_server_metrics

        # Find a server and check for AI-specific metrics
        server_endpoints = result.json.server_metrics_data.endpoints
        assert len(server_endpoints) > 0

        # Get first server's data
        first_endpoint_data = next(iter(server_endpoints.values()))
        first_server_summary = next(iter(first_endpoint_data.servers.values()))

        # Check metrics contain AI-specific metrics
        metrics = first_server_summary.metrics
        assert len(metrics) > 0

        # Verify some AI-specific metrics exist in the statistical summary
        # Note: With dynamic field discovery, we use Prometheus metric names directly
        ai_metrics_found = []

        # Look for Dynamo frontend metrics (Prometheus names)
        if "dynamo_frontend_requests" in metrics:
            ai_metrics_found.append("dynamo_frontend_requests")

        # Look for Dynamo component metrics (Prometheus names)
        if "dynamo_component_requests" in metrics:
            ai_metrics_found.append("dynamo_component_requests")

        # Look for KV cache metrics (Prometheus names)
        if "dynamo_component_kvstats_active_blocks" in metrics:
            ai_metrics_found.append("dynamo_component_kvstats_active_blocks")

        assert len(ai_metrics_found) > 0, (
            f"Should find at least some AI-specific metrics (frontend, component, or kvstats). "
            f"Found metrics: {sorted(metrics.keys())}"
        )
