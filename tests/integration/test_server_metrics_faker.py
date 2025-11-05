# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for ServerMetricsFaker using real ServerMetricsDataCollector."""

import pytest
from pytest import approx

from aiperf.server_metrics.server_metrics_data_collector import (
    ServerMetricsDataCollector,
)
from tests.aiperf_mock_server.server_metrics_faker import (
    SERVER_CONFIGS,
    ServerMetricsFaker,
)


class TestServerMetricsFaker:
    """Test ServerMetricsFaker by parsing output with actual ServerMetricsDataCollector."""

    @pytest.mark.parametrize("config_name", SERVER_CONFIGS.keys())
    def test_faker_output_parsed_by_real_collector(self, config_name):
        """Test that faker output is parsed correctly by actual ServerMetricsDataCollector."""
        faker = ServerMetricsFaker(
            config_name=config_name,
            num_servers=2,
            seed=42,
            instance_prefix="testnode",
        )
        metrics_text = faker.generate()

        # Use real ServerMetricsDataCollector to parse the output
        collector = ServerMetricsDataCollector(endpoint_url="http://fake")
        records = collector._parse_metrics_to_records(metrics_text)

        # Should get 2 ServerMetricRecord objects (one per server)
        assert len(records) == 2
        assert all(record is not None for record in records)

        # Verify server identifiers
        server_ids = {record.server_id for record in records}
        assert len(server_ids) == 2

        # Verify metadata is correctly parsed
        for record in records:
            assert ":8080" in record.instance
            assert record.server_type == "ai-server"

            # Verify ServerMetrics are correctly parsed
            metrics = record.metrics_data
            assert metrics is not None

            # Check that key metrics exist (use model_dump since ServerMetrics uses extra="allow")
            metrics_dict = metrics.model_dump()
            assert "http_requests_in_flight" in metrics_dict
            assert "process_cpu_usage_percent" in metrics_dict
            assert "memory_usage_bytes" in metrics_dict
            assert "dynamo_frontend_requests" in metrics_dict
            assert "dynamo_component_requests" in metrics_dict

    def test_load_affects_server_metric_records(self):
        """Test that load changes affect ServerMetricRecords when parsed by real collector."""
        faker = ServerMetricsFaker(
            config_name="medium", num_servers=1, seed=42, instance_prefix="testserver"
        )
        collector = ServerMetricsDataCollector(endpoint_url="http://fake")

        # Low load
        faker.set_load(0.1)
        low_metrics = faker.generate()
        low_records = collector._parse_metrics_to_records(low_metrics)
        low_server_metrics = low_records[0].metrics_data

        # High load
        faker.set_load(0.9)
        high_metrics = faker.generate()
        high_records = collector._parse_metrics_to_records(high_metrics)
        high_server_metrics = high_records[0].metrics_data

        # High load should produce higher values
        low_dict = low_server_metrics.model_dump()
        high_dict = high_server_metrics.model_dump()
        assert (
            high_dict["process_cpu_usage_percent"]
            > low_dict["process_cpu_usage_percent"]
        )
        assert high_dict["memory_usage_bytes"] > low_dict["memory_usage_bytes"]
        assert (
            high_dict["http_requests_in_flight"] > low_dict["http_requests_in_flight"]
        )

    def test_metrics_clamped_to_bounds(self):
        """Test that all metrics are clamped to [0, max] bounds."""
        faker = ServerMetricsFaker(
            config_name="large", num_servers=2, seed=42, instance_prefix="node"
        )
        collector = ServerMetricsDataCollector(endpoint_url="http://fake")

        # Test extreme high load
        faker.set_load(1.0)
        for _ in range(10):  # Generate multiple times to test with noise variance
            metrics = faker.generate()
            records = collector._parse_metrics_to_records(metrics)

            for record in records:
                m_dict = record.metrics_data.model_dump()

                # All metrics should be non-negative
                assert m_dict.get("process_cpu_usage_percent", 0) >= 0
                assert m_dict.get("memory_usage_bytes", 0) >= 0
                assert m_dict.get("http_requests_in_flight", 0) >= 0
                assert m_dict.get("dynamo_component_kvstats_active_blocks", 0) >= 0

                # CPU usage as percentage should be 0-100 after scaling
                assert m_dict.get("process_cpu_usage_percent", 0) <= 100

    def test_multiple_servers_parsed_correctly(self):
        """Test that metrics from multiple servers are parsed correctly."""
        num_servers = 4
        faker = ServerMetricsFaker(
            config_name="small",
            num_servers=num_servers,
            seed=42,
            instance_prefix="srv",
        )
        collector = ServerMetricsDataCollector(endpoint_url="http://fake")

        metrics = faker.generate()
        records = collector._parse_metrics_to_records(metrics)

        # Should get one record per server
        assert len(records) == num_servers

        # Each record should have unique server_id and instance
        server_ids = [record.server_id for record in records]
        instances = [record.instance for record in records]
        assert len(set(server_ids)) == num_servers
        assert len(set(instances)) == num_servers

    def test_cumulative_metrics_increase_over_time(self):
        """Test that cumulative metrics (counters) increase over multiple generate() calls."""
        faker = ServerMetricsFaker(
            config_name="medium", num_servers=1, seed=42, instance_prefix="host"
        )
        collector = ServerMetricsDataCollector(endpoint_url="http://fake")

        # Generate metrics multiple times
        faker.set_load(0.7)

        # First collection
        metrics1 = faker.generate()
        records1 = collector._parse_metrics_to_records(metrics1)
        m1 = records1[0].metrics_data

        # Second collection (cumulative metrics should increase)
        metrics2 = faker.generate()
        records2 = collector._parse_metrics_to_records(metrics2)
        m2 = records2[0].metrics_data

        # Cumulative metrics should increase (use model_dump for dynamic fields)
        m1_dict = m1.model_dump()
        m2_dict = m2.model_dump()
        assert m2_dict.get("dynamo_frontend_requests", 0) >= m1_dict.get(
            "dynamo_frontend_requests", 0
        )
        assert m2_dict.get("dynamo_component_requests", 0) >= m1_dict.get(
            "dynamo_component_requests", 0
        )
        assert m2_dict.get("process_cpu_seconds", 0) >= m1_dict.get(
            "process_cpu_seconds", 0
        )

    def test_deterministic_output_with_seed(self):
        """Test that using the same seed produces identical output."""
        faker1 = ServerMetricsFaker(
            config_name="medium", num_servers=2, seed=99, instance_prefix="node"
        )
        faker2 = ServerMetricsFaker(
            config_name="medium", num_servers=2, seed=99, instance_prefix="node"
        )

        metrics1 = faker1.generate()
        metrics2 = faker2.generate()

        # Should produce identical output with same seed
        assert metrics1 == metrics2

    def test_all_config_sizes_produce_valid_metrics(self):
        """Test that all server config sizes produce valid, parseable metrics."""
        collector = ServerMetricsDataCollector(endpoint_url="http://fake")

        for config_name in SERVER_CONFIGS:
            faker = ServerMetricsFaker(
                config_name=config_name, num_servers=1, seed=42, instance_prefix="test"
            )
            metrics = faker.generate()
            records = collector._parse_metrics_to_records(metrics)

            assert len(records) == 1
            assert records[0].metrics_data is not None

            # Verify metrics scale with config size
            m = records[0].metrics_data
            cfg = SERVER_CONFIGS[config_name]

            # Memory should match config
            assert m.memory_total_bytes == approx(cfg.memory_gb * 1024**3, abs=1)

    def test_kv_cache_metrics_parsed_correctly(self):
        """Test that KV cache metrics are parsed correctly."""
        faker = ServerMetricsFaker(
            config_name="medium", num_servers=1, seed=42, instance_prefix="kv-test"
        )
        collector = ServerMetricsDataCollector(endpoint_url="http://fake")

        faker.set_load(0.7)
        metrics = faker.generate()
        records = collector._parse_metrics_to_records(metrics)

        m_dict = records[0].metrics_data.model_dump()

        # KV cache metrics should be present and scaled correctly
        assert m_dict.get("dynamo_component_kvstats_total_blocks") is not None
        assert m_dict.get("dynamo_component_kvstats_active_blocks") is not None
        # Cache usage should be scaled to 0-100
        assert (
            0
            <= m_dict.get("dynamo_component_kvstats_gpu_cache_usage_percent", 0)
            <= 100
        )
        assert (
            0
            <= m_dict.get("dynamo_component_kvstats_gpu_prefix_cache_hit_rate", 0)
            <= 100
        )

    def test_frontend_metrics_parsed_correctly(self):
        """Test that frontend metrics are parsed correctly."""
        faker = ServerMetricsFaker(
            config_name="large", num_servers=1, seed=42, instance_prefix="frontend-test"
        )
        collector = ServerMetricsDataCollector(endpoint_url="http://fake")

        faker.set_load(0.8)
        metrics = faker.generate()
        records = collector._parse_metrics_to_records(metrics)

        m_dict = records[0].metrics_data.model_dump()

        # Frontend metrics should be present
        assert m_dict.get("dynamo_frontend_inflight_requests") is not None
        assert m_dict.get("dynamo_frontend_queued_requests") is not None
        assert m_dict.get("dynamo_frontend_requests") is not None
        assert m_dict.get("dynamo_frontend_time_to_first_token_seconds") is not None
        assert m_dict.get("dynamo_frontend_inter_token_latency_seconds") is not None

    def test_component_metrics_parsed_correctly(self):
        """Test that component metrics are parsed correctly."""
        faker = ServerMetricsFaker(
            config_name="medium",
            num_servers=1,
            seed=42,
            instance_prefix="component-test",
        )
        collector = ServerMetricsDataCollector(endpoint_url="http://fake")

        faker.set_load(0.6)
        metrics = faker.generate()
        records = collector._parse_metrics_to_records(metrics)

        m_dict = records[0].metrics_data.model_dump()

        # Component metrics should be present
        assert m_dict.get("dynamo_component_inflight_requests") is not None
        assert m_dict.get("dynamo_component_requests") is not None
        assert m_dict.get("dynamo_component_request_duration_seconds") is not None
        assert m_dict.get("dynamo_component_system_uptime_seconds") is not None

    def test_model_config_metrics_static(self):
        """Test that model configuration metrics remain static."""
        faker = ServerMetricsFaker(
            config_name="medium", num_servers=1, seed=42, instance_prefix="model-test"
        )
        collector = ServerMetricsDataCollector(endpoint_url="http://fake")

        # Generate metrics twice
        metrics1 = faker.generate()
        records1 = collector._parse_metrics_to_records(metrics1)
        m1 = records1[0].metrics_data

        metrics2 = faker.generate()
        records2 = collector._parse_metrics_to_records(metrics2)
        m2 = records2[0].metrics_data

        # Model config metrics should be identical
        m1_dict = m1.model_dump()
        m2_dict = m2.model_dump()
        assert m1_dict.get("dynamo_frontend_model_total_kv_blocks") == m2_dict.get(
            "dynamo_frontend_model_total_kv_blocks"
        )
        assert m1_dict.get("dynamo_frontend_model_max_num_seqs") == m2_dict.get(
            "dynamo_frontend_model_max_num_seqs"
        )
        assert m1_dict.get("dynamo_frontend_model_workers") == m2_dict.get(
            "dynamo_frontend_model_workers"
        )
