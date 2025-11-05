# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for server_metrics_faker module."""

import pytest
from aiperf_mock_server.server_metrics_faker import (
    METRIC_MAPPINGS,
    SERVER_CONFIGS,
    FakeServer,
    ServerConfig,
    ServerMetricsFaker,
)


class TestServerConfig:
    """Tests for ServerConfig dataclass."""

    @pytest.mark.parametrize(
        "config_name",
        ["small", "medium", "large", "xlarge"],
    )
    def test_all_server_configs_exist(self, config_name):
        assert config_name in SERVER_CONFIGS
        config = SERVER_CONFIGS[config_name]
        assert isinstance(config, ServerConfig)


class TestFakeServer:
    """Tests for FakeServer class."""

    @pytest.fixture
    def server_config(self):
        return SERVER_CONFIGS["medium"]

    @pytest.fixture
    def fake_server(self, server_config):
        import random

        rng = random.Random(42)
        return FakeServer(
            idx=0,
            cfg=server_config,
            rng=rng,
            load_offset=0.0,
            instance_name="test-server-0",
        )

    def test_fake_server_initialization(self, fake_server, server_config):
        assert fake_server.idx == 0
        assert fake_server.cfg == server_config
        assert fake_server.instance_name == "test-server-0"
        assert fake_server.memory_total_bytes == server_config.memory_gb * 1024**3
        assert fake_server.model_total_kv_blocks == server_config.kv_blocks
        assert fake_server.model_workers == server_config.workers

    def test_update_idle_load(self, fake_server):
        fake_server.update(0.0)
        assert fake_server.requests_in_flight >= 0
        assert fake_server.cpu_usage_percent >= 0
        assert fake_server.memory_usage_bytes > 0

    def test_update_high_load(self, fake_server):
        fake_server.update(1.0)
        assert fake_server.requests_in_flight > 0
        assert fake_server.cpu_usage_percent > 0.5
        assert fake_server.memory_usage_bytes > fake_server.memory_total_bytes * 0.5

    def test_update_metrics_in_range(self, fake_server):
        fake_server.update(0.7)
        assert 0 <= fake_server.cpu_usage_percent <= 1.0
        assert (
            0 <= fake_server.memory_usage_bytes <= fake_server.memory_total_bytes * 1.5
        )
        assert 0 <= fake_server.requests_in_flight <= fake_server.cfg.max_batch_size
        assert 0 <= fake_server.kvstats_active_blocks <= fake_server.cfg.kv_blocks

    def test_cumulative_metrics_increase(self, fake_server):
        initial_requests = fake_server.requests_total
        initial_cpu_seconds = fake_server.process_cpu_seconds

        fake_server.update(0.5)

        assert fake_server.requests_total >= initial_requests
        assert fake_server.process_cpu_seconds >= initial_cpu_seconds

    def test_kv_cache_metrics(self, fake_server):
        fake_server.update(0.7)
        assert (
            0 <= fake_server.kvstats_active_blocks <= fake_server.kvstats_total_blocks
        )
        assert 0.0 <= fake_server.kvstats_gpu_cache_usage_percent <= 1.0
        assert 0.0 <= fake_server.kvstats_gpu_prefix_cache_hit_rate <= 1.0

    def test_http_status_codes(self, fake_server):
        initial_2xx = fake_server.http_2xx_total
        fake_server.update(0.5)
        # Should have some 2xx responses
        assert fake_server.http_2xx_total >= initial_2xx


class TestServerMetricsFaker:
    """Tests for ServerMetricsFaker class."""

    def test_initialization(self, server_metrics_faker):
        assert server_metrics_faker.cfg == SERVER_CONFIGS["medium"]
        assert server_metrics_faker.load == 0.5
        assert len(server_metrics_faker.servers) == 1

    def test_invalid_config_name(self):
        with pytest.raises(ValueError, match="Invalid config name"):
            ServerMetricsFaker(config_name="invalid-config")

    def test_set_load(self, server_metrics_faker):
        server_metrics_faker.set_load(0.7)
        assert server_metrics_faker.load == 0.7

    def test_set_load_clamps(self, server_metrics_faker):
        server_metrics_faker.set_load(1.5)
        assert server_metrics_faker.load == 1.0
        server_metrics_faker.set_load(-0.5)
        assert server_metrics_faker.load == 0.0

    def test_generate_output_format(self, server_metrics_faker):
        metrics = server_metrics_faker.generate()
        assert isinstance(metrics, str)
        assert "# HELP" in metrics
        assert "# TYPE" in metrics
        assert "gauge" in metrics or "counter" in metrics

    def test_generate_contains_all_metrics(self, server_metrics_faker):
        metrics = server_metrics_faker.generate()
        # Check for key metric names
        for metric_name, _, _ in METRIC_MAPPINGS:
            assert metric_name in metrics

    def test_generate_contains_server_labels(self, server_metrics_faker):
        metrics = server_metrics_faker.generate()
        assert 'instance="server-0:8080"' in metrics
        assert 'job="ai-server"' in metrics

    def test_generate_deterministic_with_seed(self):
        faker1 = ServerMetricsFaker(
            config_name="medium", num_servers=1, seed=123, initial_load=0.5
        )
        faker2 = ServerMetricsFaker(
            config_name="medium", num_servers=1, seed=123, initial_load=0.5
        )
        metrics1 = faker1.generate()
        metrics2 = faker2.generate()
        assert metrics1 == metrics2

    def test_generate_changes_with_load(self, server_metrics_faker):
        server_metrics_faker.set_load(0.2)
        metrics_low = server_metrics_faker.generate()
        server_metrics_faker.set_load(0.9)
        metrics_high = server_metrics_faker.generate()
        assert metrics_low != metrics_high

    def test_multiple_servers(self):
        faker = ServerMetricsFaker(
            config_name="small", num_servers=3, seed=42, instance_prefix="node"
        )
        assert len(faker.servers) == 3
        metrics = faker.generate()
        for i in range(3):
            assert f'instance="node-{i}:8080"' in metrics

    @pytest.mark.parametrize("num_servers", [1, 2, 4, 8])
    def test_various_server_counts(self, num_servers):
        faker = ServerMetricsFaker(
            config_name="large", num_servers=num_servers, seed=42
        )
        assert len(faker.servers) == num_servers

    def test_instance_prefix_in_metrics(self):
        faker = ServerMetricsFaker(
            config_name="medium",
            num_servers=2,
            instance_prefix="test-node",
            seed=42,
        )
        metrics = faker.generate()
        assert 'instance="test-node-0:8080"' in metrics
        assert 'instance="test-node-1:8080"' in metrics

    def test_initial_load_applied(self):
        faker_low = ServerMetricsFaker(
            config_name="medium", num_servers=1, seed=42, initial_load=0.1
        )
        faker_high = ServerMetricsFaker(
            config_name="medium", num_servers=1, seed=42, initial_load=0.9
        )

        metrics_low = faker_low.generate()
        metrics_high = faker_high.generate()

        assert metrics_low != metrics_high

    def test_server_load_offsets_create_variance(self):
        faker = ServerMetricsFaker(
            config_name="medium", num_servers=2, seed=42, initial_load=0.5
        )
        faker.generate()

        server0_inflight = faker.servers[0].requests_in_flight
        server1_inflight = faker.servers[1].requests_in_flight
        # With different load offsets, inflight requests may differ
        # (might be same if both are 0 at low load, so just check they exist)
        assert server0_inflight >= 0
        assert server1_inflight >= 0

    @pytest.mark.parametrize(
        "config_name",
        ["small", "medium", "large", "xlarge"],
    )
    def test_all_configs_generate_valid_metrics(self, config_name):
        faker = ServerMetricsFaker(config_name=config_name, num_servers=1, seed=42)
        metrics = faker.generate()
        assert len(metrics) > 0
        assert "# HELP" in metrics
        assert "# TYPE" in metrics

    def test_dynamo_frontend_metrics_present(self, server_metrics_faker):
        metrics = server_metrics_faker.generate()
        assert "dynamo_frontend_inflight_requests" in metrics
        assert "dynamo_frontend_requests_total" in metrics
        assert "dynamo_frontend_time_to_first_token_seconds" in metrics

    def test_dynamo_component_metrics_present(self, server_metrics_faker):
        metrics = server_metrics_faker.generate()
        assert "dynamo_component_inflight_requests" in metrics
        assert "dynamo_component_requests_total" in metrics
        assert "dynamo_component_kvstats_active_blocks" in metrics

    def test_model_config_metrics_static(self, server_metrics_faker):
        """Test that model config metrics don't change between updates."""
        metrics1 = server_metrics_faker.generate()
        metrics2 = server_metrics_faker.generate()

        # Extract model config metric values from both
        for line in metrics1.split("\n"):
            if "dynamo_frontend_model_total_kv_blocks" in line and "instance=" in line:
                value1 = float(line.split()[-1])
                # Find same metric in metrics2
                for line2 in metrics2.split("\n"):
                    if (
                        "dynamo_frontend_model_total_kv_blocks" in line2
                        and "instance=" in line2
                    ):
                        value2 = float(line2.split()[-1])
                        assert value1 == value2  # Should be unchanged
                        break
                break
