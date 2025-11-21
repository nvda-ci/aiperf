# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for PrometheusFaker using real server metrics parsing logic."""

import pytest
from pytest import approx

from aiperf.server_metrics.server_metrics_data_collector import (
    ServerMetricsDataCollector,
)
from tests.aiperf_mock_server.prometheus_faker import SERVER_CONFIGS, PrometheusFaker


class TestPrometheusFaker:
    """Test PrometheusFaker by parsing output with actual ServerMetricsDataCollector."""

    @pytest.mark.parametrize("server_type", SERVER_CONFIGS.keys())
    def test_faker_output_parsed_by_real_collector(self, server_type):
        """Test that faker output is parsed correctly by actual ServerMetricsDataCollector."""
        faker = PrometheusFaker(server_type=server_type, num_servers=2, seed=42)
        metrics_text = faker.generate()

        # Use real ServerMetricsDataCollector to parse the output
        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")
        records = collector._parse_metrics_to_records(metrics_text, latency_ns=1000000)

        # Should get 1 ServerMetricsRecord
        assert len(records) == 1
        record = records[0]
        assert record is not None

        # Verify all expected metric families are present
        # Note: prometheus_client strips "_total" suffix from counter names during parsing
        assert "http_requests" in record.metrics
        assert "http_errors" in record.metrics
        assert "http_active_connections" in record.metrics
        assert "http_queue_size" in record.metrics
        assert "process_memory_bytes" in record.metrics
        assert "http_request_duration_seconds" in record.metrics
        assert "http_response_size_bytes" in record.metrics
        assert "http_request_latency_seconds" in record.metrics

        # Verify metric types
        from aiperf.common.enums import PrometheusMetricType

        assert record.metrics["http_requests"].type == PrometheusMetricType.COUNTER
        assert (
            record.metrics["http_active_connections"].type == PrometheusMetricType.GAUGE
        )
        assert (
            record.metrics["http_request_duration_seconds"].type
            == PrometheusMetricType.HISTOGRAM
        )
        assert (
            record.metrics["http_request_latency_seconds"].type
            == PrometheusMetricType.SUMMARY
        )

        # Verify each metric family has 2 samples (one per server)
        assert len(record.metrics["http_requests"].samples) == 2
        assert len(record.metrics["http_active_connections"].samples) == 2

        # Verify labels are correctly parsed
        for sample in record.metrics["http_requests"].samples:
            assert sample.labels is not None
            assert "server_id" in sample.labels
            assert "job" in sample.labels
            assert "instance" in sample.labels
            assert sample.labels["job"] == "inference-server"

    def test_counter_metrics_parsed_correctly(self):
        """Test that counter metrics are parsed correctly with proper values."""
        faker = PrometheusFaker(server_type="medium", num_servers=1, seed=42)
        metrics_text = faker.generate()

        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")
        records = collector._parse_metrics_to_records(metrics_text, latency_ns=1000000)
        record = records[0]

        # Check counter metrics (note: "_total" suffix is stripped by prometheus_client)
        requests_sample = record.metrics["http_requests"].samples[0]
        errors_sample = record.metrics["http_errors"].samples[0]

        assert requests_sample.value >= 0
        assert errors_sample.value >= 0
        assert errors_sample.value <= requests_sample.value  # Errors <= total requests

    def test_gauge_metrics_parsed_correctly(self):
        """Test that gauge metrics are parsed correctly with proper values."""
        faker = PrometheusFaker(server_type="large", num_servers=1, seed=42)
        metrics_text = faker.generate()

        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")
        records = collector._parse_metrics_to_records(metrics_text, latency_ns=1000000)
        record = records[0]

        # Check gauge metrics
        connections_sample = record.metrics["http_active_connections"].samples[0]
        queue_sample = record.metrics["http_queue_size"].samples[0]
        memory_sample = record.metrics["process_memory_bytes"].samples[0]

        assert connections_sample.value >= 0
        assert queue_sample.value >= 0
        assert memory_sample.value >= 0

        # Verify values are in reasonable ranges
        config = SERVER_CONFIGS["large"]
        assert connections_sample.value <= config.max_connections * 1.5
        assert queue_sample.value <= config.max_queue_size * 1.5
        assert memory_sample.value <= config.memory_gb * 1024**3

    def test_histogram_metrics_parsed_correctly(self):
        """Test that histogram metrics are parsed correctly with proper structure."""
        faker = PrometheusFaker(server_type="medium", num_servers=1, seed=42)
        metrics_text = faker.generate()

        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")
        records = collector._parse_metrics_to_records(metrics_text, latency_ns=1000000)
        record = records[0]

        # Check histogram metrics
        duration_family = record.metrics["http_request_duration_seconds"]
        assert len(duration_family.samples) == 1
        duration_sample = duration_family.samples[0]

        assert duration_sample.histogram is not None
        assert duration_sample.histogram.buckets is not None
        assert duration_sample.histogram.sum is not None
        assert duration_sample.histogram.count is not None

        # Verify bucket structure
        assert "+Inf" in duration_sample.histogram.buckets
        assert len(duration_sample.histogram.buckets) > 0

        # Verify sum and count are non-negative
        assert duration_sample.histogram.sum >= 0
        assert duration_sample.histogram.count >= 0

        # Verify buckets are monotonically increasing
        sorted_buckets = sorted(
            [
                (float(le) if le != "+Inf" else float("inf"), count)
                for le, count in duration_sample.histogram.buckets.items()
            ],
            key=lambda x: x[0],
        )
        prev_count = 0
        for _, count in sorted_buckets:
            assert count >= prev_count
            prev_count = count

    def test_summary_metrics_parsed_correctly(self):
        """Test that summary metrics are parsed correctly with proper structure."""
        faker = PrometheusFaker(server_type="medium", num_servers=1, seed=42)
        metrics_text = faker.generate()

        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")
        records = collector._parse_metrics_to_records(metrics_text, latency_ns=1000000)
        record = records[0]

        # Check summary metrics
        latency_family = record.metrics["http_request_latency_seconds"]
        assert len(latency_family.samples) == 1
        latency_sample = latency_family.samples[0]

        assert latency_sample.summary is not None
        assert latency_sample.summary.quantiles is not None
        assert latency_sample.summary.sum is not None
        assert latency_sample.summary.count is not None

        # Verify quantile structure
        assert "0.5" in latency_sample.summary.quantiles
        assert "0.9" in latency_sample.summary.quantiles
        assert "0.95" in latency_sample.summary.quantiles
        assert "0.99" in latency_sample.summary.quantiles

        # Verify sum and count are non-negative
        assert latency_sample.summary.sum >= 0
        assert latency_sample.summary.count >= 0

        # Verify quantiles are monotonically increasing
        sorted_quantiles = sorted(
            latency_sample.summary.quantiles.items(), key=lambda x: float(x[0])
        )
        prev_value = 0
        for _, value in sorted_quantiles:
            assert value >= prev_value
            prev_value = value

    def test_load_affects_metrics(self):
        """Test that load changes affect metrics when parsed by real collector."""
        faker = PrometheusFaker(server_type="medium", num_servers=1, seed=42)
        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")

        # Low load
        faker.set_load(0.1)
        low_metrics = faker.generate()
        low_records = collector._parse_metrics_to_records(
            low_metrics, latency_ns=1000000
        )
        low_record = low_records[0]

        # High load
        faker.set_load(0.9)
        high_metrics = faker.generate()
        high_records = collector._parse_metrics_to_records(
            high_metrics, latency_ns=1000000
        )
        high_record = high_records[0]

        # High load should produce higher values
        low_connections = low_record.metrics["http_active_connections"].samples[0].value
        high_connections = (
            high_record.metrics["http_active_connections"].samples[0].value
        )
        assert high_connections > low_connections

        low_queue = low_record.metrics["http_queue_size"].samples[0].value
        high_queue = high_record.metrics["http_queue_size"].samples[0].value
        assert high_queue > low_queue

    def test_multiple_servers_generate_separate_samples(self):
        """Test that multiple servers generate separate samples with correct labels."""
        faker = PrometheusFaker(server_type="small", num_servers=4, seed=42)
        metrics_text = faker.generate()

        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")
        records = collector._parse_metrics_to_records(metrics_text, latency_ns=1000000)
        record = records[0]

        # Each metric should have 4 samples (one per server)
        # Note: "_total" suffix is stripped by prometheus_client
        assert len(record.metrics["http_requests"].samples) == 4
        assert len(record.metrics["http_active_connections"].samples) == 4

        # Verify server_ids are unique
        server_ids = set()
        for sample in record.metrics["http_requests"].samples:
            assert sample.labels is not None
            server_ids.add(sample.labels["server_id"])

        assert len(server_ids) == 4
        assert server_ids == {"server-0", "server-1", "server-2", "server-3"}

    def test_deterministic_output_with_seed(self):
        """Test that same seed produces identical output that parses identically."""
        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")

        faker1 = PrometheusFaker(server_type="medium", num_servers=2, seed=123)
        metrics1 = faker1.generate()
        records1 = collector._parse_metrics_to_records(metrics1, latency_ns=1000000)

        faker2 = PrometheusFaker(server_type="medium", num_servers=2, seed=123)
        metrics2 = faker2.generate()
        records2 = collector._parse_metrics_to_records(metrics2, latency_ns=1000000)

        # Compare metric values
        record1 = records1[0]
        record2 = records2[0]

        for metric_name in record1.metrics:
            family1 = record1.metrics[metric_name]
            family2 = record2.metrics[metric_name]

            assert len(family1.samples) == len(family2.samples)

            for sample1, sample2 in zip(family1.samples, family2.samples, strict=False):
                if sample1.value is not None:
                    assert sample1.value == approx(sample2.value)
                if sample1.histogram is not None:
                    assert sample1.histogram.sum == approx(sample2.histogram.sum)
                    assert sample1.histogram.count == approx(sample2.histogram.count)
                if sample1.summary is not None:
                    assert sample1.summary.sum == approx(sample2.summary.sum)
                    assert sample1.summary.count == approx(sample2.summary.count)

    def test_metrics_values_in_bounds(self):
        """Test that all metrics stay within reasonable bounds."""
        faker = PrometheusFaker(server_type="medium", num_servers=2, seed=42)
        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")

        # Test extreme high load
        faker.set_load(1.0)
        for _ in range(10):  # Generate multiple times to test with noise variance
            metrics = faker.generate()
            records = collector._parse_metrics_to_records(metrics, latency_ns=1000000)
            record = records[0]

            # Check all gauge metrics are non-negative and within bounds
            for sample in record.metrics["http_active_connections"].samples:
                assert sample.value >= 0
                assert sample.value <= SERVER_CONFIGS["medium"].max_connections * 1.5

            for sample in record.metrics["http_queue_size"].samples:
                assert sample.value >= 0
                assert sample.value <= SERVER_CONFIGS["medium"].max_queue_size * 1.5

            for sample in record.metrics["process_memory_bytes"].samples:
                assert sample.value >= 0
                assert (
                    sample.value <= SERVER_CONFIGS["medium"].memory_gb * 1024**3 * 1.5
                )

    def test_slim_record_conversion(self):
        """Test that records can be converted to slim format for export."""
        faker = PrometheusFaker(server_type="medium", num_servers=1, seed=42)
        metrics_text = faker.generate()

        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")
        records = collector._parse_metrics_to_records(metrics_text, latency_ns=1000000)
        record = records[0]

        # Convert to slim format
        slim_record = record.to_slim()

        # Verify slim record has all the expected fields
        assert slim_record.endpoint_url == record.endpoint_url
        assert slim_record.timestamp_ns == record.timestamp_ns
        assert slim_record.endpoint_latency_ns == record.endpoint_latency_ns

        # Verify metrics are present
        assert len(slim_record.metrics) == len(record.metrics)

        # Verify histogram is converted to array format
        assert "http_request_duration_seconds" in slim_record.metrics
        histogram_samples = slim_record.metrics["http_request_duration_seconds"]
        assert len(histogram_samples) == 1
        histogram_sample = histogram_samples[0]
        assert histogram_sample.histogram is not None
        assert isinstance(histogram_sample.histogram, list)
        assert histogram_sample.sum is not None
        assert histogram_sample.count is not None

        # Verify summary is converted to array format
        assert "http_request_latency_seconds" in slim_record.metrics
        summary_samples = slim_record.metrics["http_request_latency_seconds"]
        assert len(summary_samples) == 1
        summary_sample = summary_samples[0]
        assert summary_sample.summary is not None
        assert isinstance(summary_sample.summary, list)
        assert summary_sample.sum is not None
        assert summary_sample.count is not None


class TestPrometheusFakerVLLM:
    """Integration tests for vLLM metrics with real parser."""

    def test_vllm_metrics_parsed_by_real_collector(self):
        """Test that vLLM metrics are parsed correctly by ServerMetricsDataCollector."""
        faker = PrometheusFaker(
            server_type="vllm", num_servers=1, seed=42, metric_type="vllm"
        )
        metrics_text = faker.generate()

        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")
        records = collector._parse_metrics_to_records(metrics_text, latency_ns=1000000)

        assert len(records) == 1
        record = records[0]

        # Check for key vLLM metrics (note: prometheus_client strips "_total" suffix)
        assert "vllm:num_requests_running" in record.metrics
        assert "vllm:gpu_cache_usage_perc" in record.metrics
        assert "vllm:prompt_tokens" in record.metrics
        assert "vllm:generation_tokens" in record.metrics
        assert "vllm:time_to_first_token_seconds" in record.metrics

    def test_vllm_gauge_metrics_parsed(self):
        """Test vLLM gauge metrics are parsed with correct type."""
        faker = PrometheusFaker(
            server_type="vllm", num_servers=1, seed=42, metric_type="vllm"
        )
        metrics_text = faker.generate()

        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")
        records = collector._parse_metrics_to_records(metrics_text, latency_ns=1000000)
        record = records[0]

        from aiperf.common.enums import PrometheusMetricType

        assert (
            record.metrics["vllm:num_requests_running"].type
            == PrometheusMetricType.GAUGE
        )
        assert (
            record.metrics["vllm:gpu_cache_usage_perc"].type
            == PrometheusMetricType.GAUGE
        )

        # Check values are in valid range
        running_sample = record.metrics["vllm:num_requests_running"].samples[0]
        cache_sample = record.metrics["vllm:gpu_cache_usage_perc"].samples[0]

        assert running_sample.value >= 0
        assert 0.0 <= cache_sample.value <= 1.0

    def test_vllm_histogram_metrics_parsed(self):
        """Test vLLM histogram metrics are parsed correctly."""
        faker = PrometheusFaker(
            server_type="vllm", num_servers=1, seed=42, metric_type="vllm"
        )
        metrics_text = faker.generate()

        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")
        records = collector._parse_metrics_to_records(metrics_text, latency_ns=1000000)
        record = records[0]

        from aiperf.common.enums import PrometheusMetricType

        # Check histogram metrics exist and have correct structure
        assert (
            record.metrics["vllm:time_to_first_token_seconds"].type
            == PrometheusMetricType.HISTOGRAM
        )

        ttft_sample = record.metrics["vllm:time_to_first_token_seconds"].samples[0]
        assert ttft_sample.histogram is not None
        assert ttft_sample.histogram.sum is not None
        assert ttft_sample.histogram.count is not None
        assert "+Inf" in ttft_sample.histogram.buckets

    def test_vllm_load_affects_parsed_metrics(self):
        """Test that load changes affect vLLM metrics when parsed."""
        faker = PrometheusFaker(
            server_type="vllm", num_servers=1, seed=42, metric_type="vllm"
        )
        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")

        # Low load
        faker.set_load(0.2)
        low_metrics = faker.generate()
        low_records = collector._parse_metrics_to_records(
            low_metrics, latency_ns=1000000
        )
        low_record = low_records[0]

        # High load
        faker.set_load(0.9)
        high_metrics = faker.generate()
        high_records = collector._parse_metrics_to_records(
            high_metrics, latency_ns=1000000
        )
        high_record = high_records[0]

        # Compare metrics
        low_running = low_record.metrics["vllm:num_requests_running"].samples[0].value
        high_running = high_record.metrics["vllm:num_requests_running"].samples[0].value

        assert high_running > low_running


class TestPrometheusFakerDynamo:
    """Integration tests for Dynamo/Triton metrics with real parser."""

    def test_dynamo_metrics_parsed_by_real_collector(self):
        """Test that Dynamo metrics are parsed correctly by ServerMetricsDataCollector."""
        faker = PrometheusFaker(
            server_type="dynamo", num_servers=1, seed=42, metric_type="dynamo"
        )
        metrics_text = faker.generate()

        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")
        records = collector._parse_metrics_to_records(metrics_text, latency_ns=1000000)

        assert len(records) == 1
        record = records[0]

        # Check for key Dynamo frontend metrics (note: prometheus_client strips "_total" suffix where applicable)
        assert "dynamo_frontend_requests" in record.metrics
        assert "dynamo_frontend_output_tokens" in record.metrics
        assert "dynamo_frontend_queued_requests" in record.metrics
        assert "dynamo_frontend_inflight_requests" in record.metrics
        assert "kvstats_gpu_cache_usage_percent" in record.metrics
        assert "dynamo_frontend_request_duration_seconds" in record.metrics

    def test_dynamo_counter_metrics_parsed(self):
        """Test Dynamo counter metrics are parsed with correct type."""
        faker = PrometheusFaker(
            server_type="dynamo", num_servers=1, seed=42, metric_type="dynamo"
        )
        metrics_text = faker.generate()

        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")
        records = collector._parse_metrics_to_records(metrics_text, latency_ns=1000000)
        record = records[0]

        from aiperf.common.enums import PrometheusMetricType

        assert (
            record.metrics["dynamo_frontend_requests"].type
            == PrometheusMetricType.COUNTER
        )
        assert (
            record.metrics["dynamo_frontend_output_tokens"].type
            == PrometheusMetricType.COUNTER
        )

        # Check values are non-negative
        requests_sample = record.metrics["dynamo_frontend_requests"].samples[0]
        tokens_sample = record.metrics["dynamo_frontend_output_tokens"].samples[0]

        assert requests_sample.value >= 0
        assert tokens_sample.value >= 0

    def test_dynamo_gauge_metrics_parsed(self):
        """Test Dynamo gauge metrics are parsed correctly."""
        faker = PrometheusFaker(
            server_type="dynamo", num_servers=1, seed=42, metric_type="dynamo"
        )
        metrics_text = faker.generate()

        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")
        records = collector._parse_metrics_to_records(metrics_text, latency_ns=1000000)
        record = records[0]

        from aiperf.common.enums import PrometheusMetricType

        assert (
            record.metrics["dynamo_frontend_queued_requests"].type
            == PrometheusMetricType.GAUGE
        )
        assert (
            record.metrics["dynamo_frontend_inflight_requests"].type
            == PrometheusMetricType.GAUGE
        )
        assert (
            record.metrics["kvstats_gpu_cache_usage_percent"].type
            == PrometheusMetricType.GAUGE
        )

        # Check GPU cache usage is in valid range
        gpu_cache_sample = record.metrics["kvstats_gpu_cache_usage_percent"].samples[0]
        assert 0.0 <= gpu_cache_sample.value <= 1.0

        # Check KV block metrics
        active_blocks_sample = record.metrics["kvstats_active_blocks"].samples[0]
        total_blocks_sample = record.metrics["kvstats_total_blocks"].samples[0]
        assert active_blocks_sample.value <= total_blocks_sample.value

    def test_dynamo_histogram_metrics_parsed(self):
        """Test Dynamo histogram metrics are parsed correctly."""
        faker = PrometheusFaker(
            server_type="dynamo", num_servers=1, seed=42, metric_type="dynamo"
        )
        metrics_text = faker.generate()

        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")
        records = collector._parse_metrics_to_records(metrics_text, latency_ns=1000000)
        record = records[0]

        from aiperf.common.enums import PrometheusMetricType

        assert (
            record.metrics["dynamo_frontend_request_duration_seconds"].type
            == PrometheusMetricType.HISTOGRAM
        )
        assert (
            record.metrics["dynamo_frontend_time_to_first_token_seconds"].type
            == PrometheusMetricType.HISTOGRAM
        )

        histogram_sample = record.metrics[
            "dynamo_frontend_request_duration_seconds"
        ].samples[0]
        assert histogram_sample.histogram is not None
        assert histogram_sample.histogram.buckets is not None
        assert histogram_sample.histogram.sum is not None
        assert histogram_sample.histogram.count is not None

    def test_dynamo_load_affects_parsed_metrics(self):
        """Test that load changes affect Dynamo metrics when parsed."""
        faker = PrometheusFaker(
            server_type="dynamo", num_servers=1, seed=42, metric_type="dynamo"
        )
        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")

        # Low load
        faker.set_load(0.2)
        low_metrics = faker.generate()
        low_records = collector._parse_metrics_to_records(
            low_metrics, latency_ns=1000000
        )
        low_record = low_records[0]

        # High load
        faker.set_load(0.9)
        high_metrics = faker.generate()
        high_records = collector._parse_metrics_to_records(
            high_metrics, latency_ns=1000000
        )
        high_record = high_records[0]

        # Compare queued requests
        low_queued = (
            low_record.metrics["dynamo_frontend_queued_requests"].samples[0].value
        )
        high_queued = (
            high_record.metrics["dynamo_frontend_queued_requests"].samples[0].value
        )

        assert high_queued > low_queued

    def test_dynamo_counters_increase_over_time(self):
        """Test that Dynamo counter metrics increase monotonically."""
        faker = PrometheusFaker(
            server_type="dynamo", num_servers=1, seed=42, metric_type="dynamo"
        )
        collector = ServerMetricsDataCollector(endpoint_url="http://fake:8080/metrics")

        # First generation
        metrics1 = faker.generate()
        records1 = collector._parse_metrics_to_records(metrics1, latency_ns=1000000)
        count1 = records1[0].metrics["dynamo_frontend_requests"].samples[0].value

        # Second generation
        metrics2 = faker.generate()
        records2 = collector._parse_metrics_to_records(metrics2, latency_ns=1000000)
        count2 = records2[0].metrics["dynamo_frontend_requests"].samples[0].value

        assert count2 >= count1
