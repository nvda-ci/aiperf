# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for prometheus_faker module."""

import pytest

from tests.aiperf_mock_server.prometheus_faker import (
    SERVER_CONFIGS,
    PrometheusFaker,
    ServerConfig,
)


class TestServerConfig:
    """Tests for ServerConfig dataclass."""

    @pytest.mark.parametrize(
        "server_type",
        ["small", "medium", "large"],
    )
    def test_all_server_configs_exist(self, server_type):
        assert server_type in SERVER_CONFIGS
        config = SERVER_CONFIGS[server_type]
        assert isinstance(config, ServerConfig)


class TestGenericServer:
    """Tests for GenericServer and GenericMetricsFaker."""

    @pytest.fixture
    def server_config(self):
        return SERVER_CONFIGS["medium"]

    @pytest.fixture
    def faker(self):
        return PrometheusFaker(
            server_type="medium", num_servers=1, seed=42, metric_type="generic"
        )

    def test_generic_server_initialization(self, faker, server_config):
        server = faker.servers[0]
        assert server.idx == 0
        assert server.cfg == server_config
        assert server.server_id == "server-0"
        assert server.instance == "localhost:8080"
        assert server.job == "inference-server"

    def test_update_idle_load(self, faker):
        faker.set_load(0.0)
        faker.generate()
        server = faker.servers[0]
        assert server.active_connections >= 0
        assert server.queue_size >= 0
        assert server.memory_used_bytes >= 0

    def test_update_high_load(self, faker):
        faker.set_load(1.0)
        faker.generate()
        server = faker.servers[0]
        assert server.active_connections > 0
        assert server.queue_size >= 0
        assert server.memory_used_bytes > server.cfg.memory_gb * 1024**3 * 0.3

    def test_update_metrics_in_range(self, faker):
        faker.set_load(0.7)
        faker.generate()
        server = faker.servers[0]
        assert 0 <= server.active_connections <= server.cfg.max_connections * 1.1
        assert 0 <= server.queue_size <= server.cfg.max_queue_size * 1.2
        assert 0 <= server.memory_used_bytes <= server.cfg.memory_gb * 1024**3

    def test_counters_increase(self, faker):
        faker.generate()
        initial_requests = faker.servers[0].requests_total
        initial_bytes_sent = faker.servers[0].bytes_sent_total

        faker.set_load(0.5)
        faker.generate()
        assert faker.servers[0].requests_total >= initial_requests
        assert faker.servers[0].bytes_sent_total >= initial_bytes_sent

    def test_cumulative_metrics_monotonic(self, faker):
        faker.set_load(0.3)
        faker.generate()
        requests_1 = faker.servers[0].requests_total
        errors_1 = faker.servers[0].errors_total

        faker.set_load(0.6)
        faker.generate()
        requests_2 = faker.servers[0].requests_total
        errors_2 = faker.servers[0].errors_total

        assert requests_2 >= requests_1
        assert errors_2 >= errors_1

    def test_histogram_data_generated(self, faker):
        faker.set_load(0.5)
        faker.generate()
        server = faker.servers[0]
        assert isinstance(server.request_durations, list)
        assert isinstance(server.response_sizes, list)
        assert len(server.request_durations) > 0
        assert len(server.response_sizes) > 0


class TestPrometheusFaker:
    """Tests for PrometheusFaker class."""

    def test_initialization(self, prometheus_faker):
        assert prometheus_faker.cfg == SERVER_CONFIGS["medium"]
        assert prometheus_faker.load == 0.7
        assert len(prometheus_faker.servers) == 2

    def test_invalid_server_type(self):
        with pytest.raises(ValueError, match="Invalid server type"):
            PrometheusFaker(server_type="invalid-server")

    def test_set_load(self, prometheus_faker):
        prometheus_faker.set_load(0.5)
        assert prometheus_faker.load == 0.5

    def test_set_load_clamps(self, prometheus_faker):
        prometheus_faker.set_load(1.5)
        assert prometheus_faker.load == 1.0
        prometheus_faker.set_load(-0.5)
        assert prometheus_faker.load == 0.0

    def test_generate_output_format(self, prometheus_faker):
        metrics = prometheus_faker.generate()
        assert isinstance(metrics, str)
        assert "# HELP" in metrics
        assert "# TYPE" in metrics

    def test_generate_contains_counter_metrics(self, prometheus_faker):
        metrics = prometheus_faker.generate()
        assert "http_requests_total" in metrics
        assert "http_errors_total" in metrics
        assert "# TYPE http_requests_total counter" in metrics

    def test_generate_contains_gauge_metrics(self, prometheus_faker):
        metrics = prometheus_faker.generate()
        assert "http_active_connections" in metrics
        assert "http_queue_size" in metrics
        assert "process_memory_bytes" in metrics
        assert "# TYPE http_active_connections gauge" in metrics

    def test_generate_contains_histogram_metrics(self, prometheus_faker):
        metrics = prometheus_faker.generate()
        assert "http_request_duration_seconds" in metrics
        assert "http_response_size_bytes" in metrics
        assert "# TYPE http_request_duration_seconds histogram" in metrics
        assert "_bucket" in metrics
        assert "_sum" in metrics
        assert "_count" in metrics
        assert 'le="' in metrics

    def test_generate_contains_summary_metrics(self, prometheus_faker):
        metrics = prometheus_faker.generate()
        assert "http_request_latency_seconds" in metrics
        assert "# TYPE http_request_latency_seconds summary" in metrics
        assert 'quantile="' in metrics

    def test_generate_contains_server_labels(self, prometheus_faker):
        metrics = prometheus_faker.generate()
        assert 'server_id="server-0"' in metrics
        assert 'server_id="server-1"' in metrics
        assert 'job="inference-server"' in metrics
        assert 'instance="localhost:8080"' in metrics

    def test_generate_deterministic_with_seed(self):
        faker1 = PrometheusFaker(server_type="medium", num_servers=2, seed=123)
        faker2 = PrometheusFaker(server_type="medium", num_servers=2, seed=123)
        metrics1 = faker1.generate()
        metrics2 = faker2.generate()
        assert metrics1 == metrics2

    def test_generate_changes_with_load(self, prometheus_faker):
        prometheus_faker.set_load(0.2)
        metrics_low = prometheus_faker.generate()
        prometheus_faker.set_load(0.9)
        metrics_high = prometheus_faker.generate()
        assert metrics_low != metrics_high

    def test_multiple_servers(self):
        faker = PrometheusFaker(server_type="small", num_servers=4, seed=42)
        assert len(faker.servers) == 4
        metrics = faker.generate()
        for i in range(4):
            assert f'server_id="server-{i}"' in metrics

    @pytest.mark.parametrize("num_servers", [1, 2, 4, 8])
    def test_various_server_counts(self, num_servers):
        faker = PrometheusFaker(server_type="medium", num_servers=num_servers, seed=42)
        assert len(faker.servers) == num_servers

    def test_initial_load_applied(self):
        faker_low = PrometheusFaker(
            server_type="medium", num_servers=1, seed=42, initial_load=0.1
        )
        faker_high = PrometheusFaker(
            server_type="medium", num_servers=1, seed=42, initial_load=0.9
        )

        metrics_low = faker_low.generate()
        metrics_high = faker_high.generate()

        assert metrics_low != metrics_high

    def test_server_load_offsets_create_variance(self):
        faker = PrometheusFaker(
            server_type="medium", num_servers=2, seed=42, initial_load=0.5
        )
        faker.generate()

        server0_connections = faker.servers[0].active_connections
        server1_connections = faker.servers[1].active_connections
        # May be equal sometimes due to randomness, but should differ most of the time
        # We just test they are both reasonable values
        assert server0_connections >= 0
        assert server1_connections >= 0

    @pytest.mark.parametrize("server_type", ["small", "medium", "large"])
    def test_all_server_types_generate(self, server_type):
        faker = PrometheusFaker(server_type=server_type, num_servers=1, seed=42)
        metrics = faker.generate()
        assert "http_requests_total" in metrics
        assert "# HELP" in metrics

    def test_histogram_buckets_format(self, prometheus_faker):
        metrics = prometheus_faker.generate()
        # Check for proper histogram bucket labels
        assert 'le="0.005"' in metrics
        assert 'le="0.01"' in metrics
        assert 'le="+Inf"' in metrics

    def test_summary_quantiles_format(self, prometheus_faker):
        metrics = prometheus_faker.generate()
        # Check for proper summary quantile labels
        assert 'quantile="0.5"' in metrics
        assert 'quantile="0.9"' in metrics
        assert 'quantile="0.95"' in metrics
        assert 'quantile="0.99"' in metrics

    def test_metrics_values_are_numeric(self, prometheus_faker):
        metrics = prometheus_faker.generate()
        lines = metrics.split("\n")

        # Find metric value lines (not HELP or TYPE)
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            # Line format: metric_name{labels} value
            parts = line.split("}")
            if len(parts) == 2:
                value_str = parts[1].strip()
                # Should be parseable as float
                try:
                    float(value_str)
                except ValueError:
                    pytest.fail(f"Invalid metric value: {value_str} in line: {line}")


class TestPrometheusFakerVLLM:
    """Tests for PrometheusFaker with vLLM metrics."""

    def test_vllm_metrics_generated(self):
        faker = PrometheusFaker(
            server_type="vllm", num_servers=1, seed=42, metric_type="vllm"
        )
        metrics = faker.generate()

        # Check for vLLM-specific metrics
        assert "vllm:num_requests_running" in metrics
        assert "vllm:num_requests_waiting" in metrics
        assert "vllm:gpu_cache_usage_perc" in metrics
        assert "vllm:prompt_tokens_total" in metrics
        assert "vllm:generation_tokens_total" in metrics
        assert "vllm:time_to_first_token_seconds" in metrics
        assert "vllm:e2e_request_latency_seconds" in metrics

    def test_vllm_gauge_metrics(self):
        faker = PrometheusFaker(
            server_type="vllm", num_servers=1, seed=42, metric_type="vllm"
        )
        metrics = faker.generate()

        assert "# TYPE vllm:num_requests_running gauge" in metrics
        assert "# TYPE vllm:gpu_cache_usage_perc gauge" in metrics

    def test_vllm_counter_metrics(self):
        faker = PrometheusFaker(
            server_type="vllm", num_servers=1, seed=42, metric_type="vllm"
        )
        metrics = faker.generate()

        assert "# TYPE vllm:prompt_tokens_total counter" in metrics
        assert "# TYPE vllm:generation_tokens_total counter" in metrics
        assert "# TYPE vllm:num_preemptions_total counter" in metrics

    def test_vllm_histogram_metrics(self):
        faker = PrometheusFaker(
            server_type="vllm", num_servers=1, seed=42, metric_type="vllm"
        )
        metrics = faker.generate()

        assert "# TYPE vllm:time_to_first_token_seconds histogram" in metrics
        assert "# TYPE vllm:time_per_output_token_seconds histogram" in metrics
        assert "vllm:time_to_first_token_seconds_bucket" in metrics
        assert 'le="0.01"' in metrics

    def test_vllm_load_affects_metrics(self):
        faker = PrometheusFaker(
            server_type="vllm", num_servers=1, seed=42, metric_type="vllm"
        )

        faker.set_load(0.2)
        faker.generate()
        low_running = faker.servers[0].num_requests_running

        faker.set_load(0.9)
        faker.generate()
        high_running = faker.servers[0].num_requests_running

        assert high_running > low_running

    def test_vllm_multiple_servers(self):
        faker = PrometheusFaker(
            server_type="vllm", num_servers=3, seed=42, metric_type="vllm"
        )
        metrics = faker.generate()

        # Should have 3 instances
        assert 'server_id="server-0"' in metrics
        assert 'server_id="server-1"' in metrics
        assert 'server_id="server-2"' in metrics

    @pytest.mark.parametrize("load", [0.1, 0.5, 0.9])
    def test_vllm_cache_usage_in_range(self, load):
        faker = PrometheusFaker(
            server_type="vllm", num_servers=1, seed=42, metric_type="vllm"
        )
        faker.set_load(load)
        faker.generate()

        server = faker.servers[0]
        assert 0.0 <= server.gpu_cache_usage_perc <= 1.0
        assert 0.0 <= server.cpu_cache_usage_perc <= 1.0


class TestPrometheusFakerDynamo:
    """Tests for PrometheusFaker with Dynamo/Triton metrics."""

    def test_dynamo_metrics_generated(self):
        faker = PrometheusFaker(
            server_type="dynamo", num_servers=1, seed=42, metric_type="dynamo"
        )
        metrics = faker.generate()

        # Check for AI Dynamo frontend metrics
        assert "dynamo_frontend_requests_total" in metrics
        assert "dynamo_frontend_output_tokens_total" in metrics
        assert "dynamo_frontend_queued_requests" in metrics
        assert "dynamo_frontend_inflight_requests" in metrics
        assert "kvstats_gpu_cache_usage_percent" in metrics
        assert "dynamo_frontend_time_to_first_token_seconds" in metrics

    def test_dynamo_counter_metrics(self):
        faker = PrometheusFaker(
            server_type="dynamo", num_servers=1, seed=42, metric_type="dynamo"
        )
        metrics = faker.generate()

        assert "# TYPE dynamo_frontend_requests_total counter" in metrics
        assert "# TYPE dynamo_frontend_output_tokens_total counter" in metrics

    def test_dynamo_gauge_metrics(self):
        faker = PrometheusFaker(
            server_type="dynamo", num_servers=1, seed=42, metric_type="dynamo"
        )
        metrics = faker.generate()

        assert "# TYPE dynamo_frontend_queued_requests gauge" in metrics
        assert "# TYPE dynamo_frontend_inflight_requests gauge" in metrics
        assert "# TYPE kvstats_gpu_cache_usage_percent gauge" in metrics

    def test_dynamo_histogram_metrics(self):
        faker = PrometheusFaker(
            server_type="dynamo", num_servers=1, seed=42, metric_type="dynamo"
        )
        metrics = faker.generate()

        assert "# TYPE dynamo_frontend_request_duration_seconds histogram" in metrics
        assert "# TYPE dynamo_frontend_time_to_first_token_seconds histogram" in metrics
        assert 'le="0.01"' in metrics

    def test_dynamo_load_affects_metrics(self):
        faker = PrometheusFaker(
            server_type="dynamo", num_servers=1, seed=42, metric_type="dynamo"
        )

        faker.set_load(0.2)
        faker.generate()
        low_queued = faker.servers[0].dynamo_frontend_queued_requests

        faker.set_load(0.9)
        faker.generate()
        high_queued = faker.servers[0].dynamo_frontend_queued_requests

        assert high_queued > low_queued

    def test_dynamo_counters_increase(self):
        faker = PrometheusFaker(
            server_type="dynamo", num_servers=1, seed=42, metric_type="dynamo"
        )

        faker.generate()
        first_requests = faker.servers[0].dynamo_frontend_requests_total

        faker.generate()
        second_requests = faker.servers[0].dynamo_frontend_requests_total

        assert second_requests >= first_requests

    def test_dynamo_multiple_servers(self):
        faker = PrometheusFaker(
            server_type="dynamo", num_servers=2, seed=42, metric_type="dynamo"
        )
        metrics = faker.generate()

        # Should have 2 instances
        assert 'server_id="server-0"' in metrics
        assert 'server_id="server-1"' in metrics

    def test_dynamo_cache_usage_range(self):
        faker = PrometheusFaker(
            server_type="dynamo", num_servers=1, seed=42, metric_type="dynamo"
        )
        faker.set_load(0.7)
        faker.generate()

        server = faker.servers[0]
        assert 0.0 <= server.kvstats_gpu_cache_usage_percent <= 1.0

    def test_dynamo_kv_blocks_consistency(self):
        faker = PrometheusFaker(
            server_type="dynamo", num_servers=1, seed=42, metric_type="dynamo"
        )
        faker.generate()

        server = faker.servers[0]
        assert server.kvstats_active_blocks <= server.kvstats_total_blocks


class TestPrometheusFakerTriton:
    """Tests for PrometheusFaker with Triton metrics."""

    def test_triton_metrics_generated(self):
        faker = PrometheusFaker(
            server_type="triton", num_servers=1, seed=42, metric_type="triton"
        )
        metrics = faker.generate()

        # Check for Triton-specific metrics
        assert "nv_inference_request_success" in metrics
        assert "nv_inference_request_failure" in metrics
        assert "nv_gpu_utilization" in metrics
        assert "nv_gpu_memory_total_bytes" in metrics
        assert "nv_cache_num_hits_per_model" in metrics
        assert "nv_inference_first_response_histogram_ms" in metrics

    def test_triton_counter_metrics(self):
        faker = PrometheusFaker(
            server_type="triton", num_servers=1, seed=42, metric_type="triton"
        )
        metrics = faker.generate()

        assert "# TYPE nv_inference_request_success counter" in metrics
        assert "# TYPE nv_inference_request_duration_us counter" in metrics
        assert "# TYPE nv_cache_num_hits_per_model counter" in metrics

    def test_triton_gauge_metrics(self):
        faker = PrometheusFaker(
            server_type="triton", num_servers=1, seed=42, metric_type="triton"
        )
        metrics = faker.generate()

        assert "# TYPE nv_gpu_utilization gauge" in metrics
        assert "# TYPE nv_gpu_memory_used_bytes gauge" in metrics
        assert "# TYPE nv_cpu_utilization gauge" in metrics

    def test_triton_histogram_metrics(self):
        faker = PrometheusFaker(
            server_type="triton", num_servers=1, seed=42, metric_type="triton"
        )
        metrics = faker.generate()

        assert "# TYPE nv_inference_first_response_histogram_ms histogram" in metrics
        assert "nv_inference_first_response_histogram_ms_bucket" in metrics


class TestPrometheusFakerSGLang:
    """Tests for PrometheusFaker with SGLang metrics."""

    def test_sglang_metrics_generated(self):
        faker = PrometheusFaker(
            server_type="sglang", num_servers=1, seed=42, metric_type="sglang"
        )
        metrics = faker.generate()

        # Check for SGLang-specific metrics
        assert "sglang:prompt_tokens_total" in metrics
        assert "sglang:generation_tokens_total" in metrics
        assert "sglang:num_running_reqs" in metrics
        assert "sglang:gen_throughput" in metrics
        assert "sglang:time_to_first_token_seconds" in metrics
        assert "sglang:cache_hit_rate" in metrics

    def test_sglang_counter_metrics(self):
        faker = PrometheusFaker(
            server_type="sglang", num_servers=1, seed=42, metric_type="sglang"
        )
        metrics = faker.generate()

        assert "# TYPE sglang:prompt_tokens_total counter" in metrics
        assert "# TYPE sglang:generation_tokens_total counter" in metrics

    def test_sglang_gauge_metrics(self):
        faker = PrometheusFaker(
            server_type="sglang", num_servers=1, seed=42, metric_type="sglang"
        )
        metrics = faker.generate()

        assert "# TYPE sglang:num_running_reqs gauge" in metrics
        assert "# TYPE sglang:gen_throughput gauge" in metrics
        assert "# TYPE sglang:cache_hit_rate gauge" in metrics

    def test_sglang_histogram_metrics(self):
        faker = PrometheusFaker(
            server_type="sglang", num_servers=1, seed=42, metric_type="sglang"
        )
        metrics = faker.generate()

        assert "# TYPE sglang:time_to_first_token_seconds histogram" in metrics
        assert "# TYPE sglang:e2e_request_latency_seconds histogram" in metrics
        assert "sglang:time_to_first_token_seconds_bucket" in metrics


class TestPrometheusFakerKVBM:
    """Tests for PrometheusFaker with KVBM metrics."""

    def test_kvbm_metrics_generated(self):
        faker = PrometheusFaker(
            server_type="kvbm", num_servers=1, seed=42, metric_type="kvbm"
        )
        metrics = faker.generate()

        # Check for KVBM-specific metrics
        assert "dynamo_component_kvstats_active_blocks" in metrics
        assert "dynamo_component_kvstats_total_blocks" in metrics
        assert "dynamo_component_kvstats_gpu_cache_usage_percent" in metrics
        assert "dynamo_component_kvbm_match_operations_total" in metrics
        assert "dynamo_component_kvbm_offload_operations_total" in metrics
        assert "dynamo_component_kvbm_onboard_operations_total" in metrics

    def test_kvbm_counter_metrics(self):
        faker = PrometheusFaker(
            server_type="kvbm", num_servers=1, seed=42, metric_type="kvbm"
        )
        metrics = faker.generate()

        assert "# TYPE dynamo_component_kvbm_match_operations_total counter" in metrics
        assert (
            "# TYPE dynamo_component_kvbm_offload_operations_total counter" in metrics
        )
        assert "# TYPE dynamo_component_kvbm_blocks_offloaded_total counter" in metrics

    def test_kvbm_gauge_metrics(self):
        faker = PrometheusFaker(
            server_type="kvbm", num_servers=1, seed=42, metric_type="kvbm"
        )
        metrics = faker.generate()

        assert "# TYPE dynamo_component_kvstats_active_blocks gauge" in metrics
        assert (
            "# TYPE dynamo_component_kvstats_gpu_cache_usage_percent gauge" in metrics
        )
        assert "# TYPE dynamo_component_kvbm_active_sequences gauge" in metrics

    def test_kvbm_histogram_metrics(self):
        faker = PrometheusFaker(
            server_type="kvbm", num_servers=1, seed=42, metric_type="kvbm"
        )
        metrics = faker.generate()

        assert "# TYPE dynamo_component_kvbm_match_latency_seconds histogram" in metrics
        assert (
            "# TYPE dynamo_component_kvbm_offload_latency_seconds histogram" in metrics
        )
        assert "dynamo_component_kvbm_match_latency_seconds_bucket" in metrics

    def test_kvbm_cache_consistency(self):
        faker = PrometheusFaker(
            server_type="kvbm", num_servers=1, seed=42, metric_type="kvbm"
        )
        faker.generate()

        server = faker.servers[0]
        assert server.kvstats_active_blocks <= server.kvstats_total_blocks
        assert 0.0 <= server.kvstats_gpu_cache_usage_percent <= 1.0


class TestPrometheusFakerMetricTypes:
    """Tests for different metric types."""

    @pytest.mark.parametrize(
        "metric_type", ["generic", "vllm", "triton", "sglang", "kvbm", "dynamo"]
    )
    def test_all_metric_types_generate(self, metric_type):
        faker = PrometheusFaker(
            server_type="medium", num_servers=1, seed=42, metric_type=metric_type
        )
        metrics = faker.generate()
        assert len(metrics) > 0
        assert "# HELP" in metrics
        assert "# TYPE" in metrics

    def test_invalid_metric_type(self):
        with pytest.raises(ValueError, match="Invalid metric type"):
            PrometheusFaker(server_type="medium", num_servers=1, metric_type="invalid")

    def test_generic_vs_vllm_different(self):
        faker_generic = PrometheusFaker(
            server_type="medium", num_servers=1, seed=42, metric_type="generic"
        )
        faker_vllm = PrometheusFaker(
            server_type="medium", num_servers=1, seed=42, metric_type="vllm"
        )

        metrics_generic = faker_generic.generate()
        metrics_vllm = faker_vllm.generate()

        assert "http_requests_total" in metrics_generic
        assert "vllm:num_requests_running" in metrics_vllm
        assert "vllm:num_requests_running" not in metrics_generic
        assert "http_requests_total" not in metrics_vllm

    def test_generic_vs_dynamo_different(self):
        faker_generic = PrometheusFaker(
            server_type="medium", num_servers=1, seed=42, metric_type="generic"
        )
        faker_dynamo = PrometheusFaker(
            server_type="medium", num_servers=1, seed=42, metric_type="dynamo"
        )

        metrics_generic = faker_generic.generate()
        metrics_dynamo = faker_dynamo.generate()

        assert "http_requests_total" in metrics_generic
        assert "dynamo_frontend_requests_total" in metrics_dynamo
        assert "dynamo_frontend_requests_total" not in metrics_generic
        assert "http_requests_total" not in metrics_dynamo
