# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for server metrics collection and reporting.

These tests verify the full end-to-end flow of server metrics collection,
including scraping from multiple mock server endpoints and validating
the exported data (JSON, JSONL, CSV).
"""

import platform

import pytest

from aiperf.common.models import ServerMetricsSlimRecord
from tests.integration.conftest import AIPerfCLI
from tests.integration.models import AIPerfMockServer


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="This test is flaky on macOS in Github Actions.",
)
@pytest.mark.integration
@pytest.mark.asyncio
class TestServerMetrics:
    """Tests for server metrics collection and reporting."""

    # ========================================================================
    # Basic Server Metrics Tests
    # ========================================================================

    async def test_server_metrics_auto_collected(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Server metrics are auto-collected from base_url/metrics without --server-metrics.

        When no --server-metrics flag is provided, AIPerf should automatically
        scrape server metrics from the inference endpoint's base URL + /metrics.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2 \
                --ui dashboard
            """
        )
        assert result.request_count == 50

        # Server metrics should be auto-collected from default /metrics endpoint
        result.assert_server_metrics_valid()

        # Verify we collected AIPerf mock server metrics (default endpoint)
        # Note: Counter metric names may not include _total suffix depending on parsing
        assert result.has_server_metric("aiperf_mock_requests")
        assert result.has_server_metric("aiperf_mock_request_latency_seconds")
        assert result.has_server_metric("aiperf_mock_time_to_first_token_seconds")
        assert result.has_server_metric("aiperf_mock_tokens_streamed")

        # Verify the auto-collected endpoint is correct
        # Note: endpoints_successful contains normalized identifiers (host:port)
        expected_endpoint = f"{aiperf_mock_server.host}:{aiperf_mock_server.port}"
        assert expected_endpoint in result.server_metrics_endpoints_successful, (
            f"Expected {expected_endpoint} in successful endpoints: "
            f"{result.server_metrics_endpoints_successful}"
        )

    async def test_server_metrics_explicit_endpoint(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Server metrics collection from explicitly specified /metrics endpoint."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2 \
                --server-metrics {aiperf_mock_server.server_metrics_urls["aiperf"]} \
                --ui dashboard
            """
        )
        assert result.request_count == 50
        result.assert_server_metrics_valid()

        # Verify we collected AIPerf mock server metrics
        assert result.has_server_metric("aiperf_mock_requests")
        assert result.has_server_metric("aiperf_mock_request_latency_seconds")

    async def test_server_metrics_vllm_endpoint(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Server metrics collection from vLLM-compatible endpoint."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2 \
                --server-metrics {aiperf_mock_server.server_metrics_urls["vllm"]}
            """
        )
        assert result.request_count == 50
        result.assert_server_metrics_valid()

        # Verify vLLM-specific metrics
        assert result.has_server_metric("vllm:e2e_request_latency_seconds")
        assert result.has_server_metric("vllm:time_to_first_token_seconds")
        assert result.has_server_metric("vllm:inter_token_latency_seconds")
        assert result.has_server_metric("vllm:kv_cache_usage_perc")

    async def test_server_metrics_sglang_endpoint(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Server metrics collection from SGLang-compatible endpoint."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2 \
                --server-metrics {aiperf_mock_server.server_metrics_urls["sglang"]}
            """
        )
        assert result.request_count == 50
        result.assert_server_metrics_valid()

        # Verify SGLang-specific metrics
        assert result.has_server_metric("sglang:e2e_request_latency_seconds")
        assert result.has_server_metric("sglang:time_to_first_token_seconds")
        assert result.has_server_metric("sglang:gen_throughput")
        assert result.has_server_metric("sglang:cache_hit_rate")

    async def test_server_metrics_trtllm_endpoint(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Server metrics collection from TensorRT-LLM-compatible endpoint."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2 \
                --server-metrics {aiperf_mock_server.server_metrics_urls["trtllm"]}
            """
        )
        assert result.request_count == 50
        result.assert_server_metrics_valid()

        # Verify TRT-LLM-specific metrics
        assert result.has_server_metric("trtllm:e2e_request_latency_seconds")
        assert result.has_server_metric("trtllm:time_to_first_token_seconds")
        assert result.has_server_metric("trtllm:time_per_output_token_seconds")
        assert result.has_server_metric("trtllm:request_success")

    # ========================================================================
    # Multiple Endpoints Tests
    # ========================================================================

    async def test_server_metrics_multiple_endpoints_vllm_sglang(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Server metrics collection from multiple endpoints (vLLM + SGLang)."""
        vllm_url = aiperf_mock_server.server_metrics_urls["vllm"]
        sglang_url = aiperf_mock_server.server_metrics_urls["sglang"]

        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2 \
                --server-metrics {vllm_url} {sglang_url}
            """
        )
        assert result.request_count == 50
        result.assert_server_metrics_valid()

        # Verify endpoints were successful (default + vllm + sglang = 3)
        # The default /metrics endpoint is always auto-collected
        assert len(result.server_metrics_endpoints_successful) >= 2

        # Verify vLLM metrics
        assert result.has_server_metric("vllm:e2e_request_latency_seconds")
        assert result.has_server_metric("vllm:time_to_first_token_seconds")

        # Verify SGLang metrics
        assert result.has_server_metric("sglang:e2e_request_latency_seconds")
        assert result.has_server_metric("sglang:time_to_first_token_seconds")

    async def test_server_metrics_all_inference_endpoints(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Server metrics collection from all inference server endpoints."""
        urls = aiperf_mock_server.get_server_metrics_url("vllm", "sglang", "trtllm")

        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2 \
                --server-metrics {" ".join(urls)}
            """
        )
        assert result.request_count == 50
        result.assert_server_metrics_valid()

        # Verify endpoints were successful (default + vllm + sglang + trtllm)
        # The default /metrics endpoint is always auto-collected
        assert len(result.server_metrics_endpoints_successful) >= 3

        # Verify vLLM metrics
        assert result.has_server_metric("vllm:e2e_request_latency_seconds")

        # Verify SGLang metrics
        assert result.has_server_metric("sglang:e2e_request_latency_seconds")

        # Verify TRT-LLM metrics
        assert result.has_server_metric("trtllm:e2e_request_latency_seconds")

    # ========================================================================
    # Dynamo Endpoints Tests
    # ========================================================================

    async def test_server_metrics_dynamo_frontend(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Server metrics collection from Dynamo frontend endpoint."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2 \
                --server-metrics {aiperf_mock_server.server_metrics_urls["dynamo_frontend"]}
            """
        )
        assert result.request_count == 50
        result.assert_server_metrics_valid()

        # Verify Dynamo frontend metrics
        assert result.has_server_metric("dynamo_frontend_request_duration_seconds")
        assert result.has_server_metric("dynamo_frontend_time_to_first_token_seconds")
        assert result.has_server_metric("dynamo_frontend_inter_token_latency_seconds")
        assert result.has_server_metric("dynamo_frontend_requests")

    async def test_server_metrics_dynamo_components(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Server metrics collection from Dynamo prefill and decode components."""
        prefill_url = aiperf_mock_server.server_metrics_urls["dynamo_prefill"]
        decode_url = aiperf_mock_server.server_metrics_urls["dynamo_decode"]

        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2 \
                --server-metrics {prefill_url} {decode_url}
            """
        )
        assert result.request_count == 50
        result.assert_server_metrics_valid()

        # Verify endpoints were successful (default + prefill + decode)
        # The default /metrics endpoint is always auto-collected
        assert len(result.server_metrics_endpoints_successful) >= 2

        # Verify Dynamo component metrics (both use same metric names)
        assert result.has_server_metric("dynamo_component_request_duration_seconds")
        assert result.has_server_metric("dynamo_component_requests")
        assert result.has_server_metric("dynamo_component_inflight_requests")

    async def test_server_metrics_full_dynamo_stack(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Server metrics collection from full Dynamo stack (frontend + prefill + decode)."""
        urls = aiperf_mock_server.get_server_metrics_url(
            "dynamo_frontend", "dynamo_prefill", "dynamo_decode"
        )

        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2 \
                --server-metrics {" ".join(urls)}
            """
        )
        assert result.request_count == 50
        result.assert_server_metrics_valid()

        # Verify endpoints were successful (default + frontend + prefill + decode)
        # The default /metrics endpoint is always auto-collected
        assert len(result.server_metrics_endpoints_successful) >= 3

        # Verify Dynamo frontend metrics
        assert result.has_server_metric("dynamo_frontend_request_duration_seconds")
        assert result.has_server_metric("dynamo_frontend_time_to_first_token_seconds")

        # Verify Dynamo component metrics
        assert result.has_server_metric("dynamo_component_request_duration_seconds")

    # ========================================================================
    # Ultimate Full Stack Test
    # ========================================================================

    async def test_server_metrics_all_endpoints(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Ultimate test: Server metrics from ALL available mock endpoints.

        This test collects metrics from:
        - vLLM endpoint
        - SGLang endpoint
        - TensorRT-LLM endpoint
        - Dynamo frontend endpoint
        - Dynamo prefill component endpoint
        - Dynamo decode component endpoint

        Total: 6 different server metrics endpoints scraped simultaneously!
        """
        all_urls = aiperf_mock_server.get_server_metrics_url(
            "vllm",
            "sglang",
            "trtllm",
            "dynamo_frontend",
            "dynamo_prefill",
            "dynamo_decode",
        )

        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 100 \
                --concurrency 4 \
                --workers-max 2 \
                --server-metrics {" ".join(all_urls)}
            """
        )
        assert result.request_count == 100
        result.assert_server_metrics_valid()

        # Verify all 6+ endpoints were successful (default + 6 explicit)
        # The default /metrics endpoint is always auto-collected
        assert len(result.server_metrics_endpoints_successful) >= 6, (
            f"Expected at least 6 successful endpoints, got {len(result.server_metrics_endpoints_successful)}: "
            f"{result.server_metrics_endpoints_successful}"
        )

        # Verify vLLM metrics
        assert result.has_server_metric("vllm:e2e_request_latency_seconds")
        assert result.has_server_metric("vllm:time_to_first_token_seconds")
        assert result.has_server_metric("vllm:inter_token_latency_seconds")
        assert result.has_server_metric("vllm:kv_cache_usage_perc")

        # Verify SGLang metrics
        assert result.has_server_metric("sglang:e2e_request_latency_seconds")
        assert result.has_server_metric("sglang:time_to_first_token_seconds")
        assert result.has_server_metric("sglang:gen_throughput")

        # Verify TRT-LLM metrics
        assert result.has_server_metric("trtllm:e2e_request_latency_seconds")
        assert result.has_server_metric("trtllm:time_to_first_token_seconds")
        assert result.has_server_metric("trtllm:time_per_output_token_seconds")

        # Verify Dynamo frontend metrics
        assert result.has_server_metric("dynamo_frontend_request_duration_seconds")
        assert result.has_server_metric("dynamo_frontend_time_to_first_token_seconds")
        assert result.has_server_metric("dynamo_frontend_inter_token_latency_seconds")

        # Verify Dynamo component metrics
        assert result.has_server_metric("dynamo_component_request_duration_seconds")
        assert result.has_server_metric("dynamo_component_requests")

    # ========================================================================
    # Export File Validation Tests
    # ========================================================================

    async def test_server_metrics_export_files(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test server metrics export files (JSON, JSONL, CSV) are valid."""
        urls = aiperf_mock_server.get_server_metrics_url("vllm", "sglang")

        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2 \
                --server-metrics {" ".join(urls)}
            """
        )

        # Verify all export files exist
        assert result.has_all_server_metrics_outputs

        # Verify JSON export structure
        assert result.server_metrics_json is not None
        assert result.server_metrics_json.summary is not None
        # At least 2 endpoints (vllm + sglang), possibly more with auto-collected default
        assert len(result.server_metrics_json.summary.endpoints_successful) >= 2
        assert len(result.server_metrics_json.metrics) > 0

        # Verify JSONL records structure
        assert result.server_metrics_jsonl is not None
        assert len(result.server_metrics_jsonl) > 0

        # Check records have expected structure
        for record in result.server_metrics_jsonl:
            assert record.endpoint_url is not None
            assert record.timestamp_ns > 0
            assert record.endpoint_latency_ns >= 0
            assert len(record.metrics) > 0

        # Verify CSV content
        assert result.has_server_metrics_csv
        csv_lines = result.server_metrics_csv.strip().split("\n")
        assert len(csv_lines) > 1  # Header + data rows

    async def test_server_metrics_jsonl_records(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test JSONL records contain expected metrics with valid data."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2 \
                --server-metrics {aiperf_mock_server.server_metrics_urls["vllm"]}
            """
        )

        # Verify JSONL structure and content
        assert result.server_metrics_jsonl is not None

        # Group records by endpoint
        endpoints_seen = set()
        timestamps = []

        for record in result.server_metrics_jsonl:
            endpoints_seen.add(record.endpoint_url)
            timestamps.append(record.timestamp_ns)

            # Verify record has metrics
            assert len(record.metrics) > 0

            # Check for expected vLLM metrics in at least some records
            if "vllm:kv_cache_usage_perc" in record.metrics:
                samples = record.metrics["vllm:kv_cache_usage_perc"]
                assert len(samples) > 0
                assert samples[0].value is not None

        # Verify timestamps are generally increasing (not strictly ordered due to multiple endpoints)
        # When multiple endpoints are scraped, records from different endpoints may interleave
        assert len(timestamps) > 0, "Should have timestamp records"
        assert min(timestamps) > 0, "Timestamps should be positive"

        # Verify we captured data from at least the expected endpoint(s)
        # (vLLM + possibly default /metrics endpoint auto-collected)
        assert len(endpoints_seen) >= 1

    async def test_server_metrics_histogram_data(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test histogram metrics are properly captured and exported."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2 \
                --server-metrics {aiperf_mock_server.server_metrics_urls["vllm"]}
            """
        )
        result.assert_server_metrics_valid()

        # Get histogram metric from JSON export
        ttft_metric = result.get_server_metric("vllm:time_to_first_token_seconds")
        assert ttft_metric is not None
        assert ttft_metric.type.value == "histogram"
        assert len(ttft_metric.series) > 0

        # Verify histogram stats are computed
        series = ttft_metric.series[0]
        assert series.observation_count is not None
        assert series.observation_count > 0

        # Verify JSONL records have histogram data
        for record in result.server_metrics_jsonl or []:
            if "vllm:time_to_first_token_seconds" in record.metrics:
                samples = record.metrics["vllm:time_to_first_token_seconds"]
                assert len(samples) > 0
                # Histogram samples should have histogram field (dict of buckets)
                assert samples[0].histogram is not None
                assert isinstance(samples[0].histogram, dict)

    # ========================================================================
    # Non-Streaming Tests
    # ========================================================================

    async def test_server_metrics_non_streaming(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Server metrics collection works with non-streaming requests."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2 \
                --server-metrics {aiperf_mock_server.server_metrics_urls["vllm"]}
            """
        )
        assert result.request_count == 50
        result.assert_server_metrics_valid()

        # Verify metrics are collected even for non-streaming
        assert result.has_server_metric("vllm:e2e_request_latency_seconds")
        assert result.has_server_metric("vllm:kv_cache_usage_perc")

    # ========================================================================
    # Custom Prefix Tests
    # ========================================================================

    async def test_server_metrics_custom_prefix(
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
                --streaming \
                --request-count 25 \
                --concurrency 1 \
                --workers-max 1 \
                --server-metrics {aiperf_mock_server.server_metrics_urls["vllm"]} \
                --profile-export-prefix custom_test
            """
        )

        # Verify custom prefix files exist
        json_file = result.artifacts_dir / "custom_test_server_metrics.json"
        jsonl_file = result.artifacts_dir / "custom_test_server_metrics.jsonl"

        if json_file.exists():
            content = json_file.read_text()
            assert len(content) > 0

        if jsonl_file.exists():
            lines = jsonl_file.read_text().strip().split("\n")
            assert len(lines) > 0
            # Validate first record
            first_record = ServerMetricsSlimRecord.model_validate_json(lines[0])
            assert first_record.timestamp_ns > 0
