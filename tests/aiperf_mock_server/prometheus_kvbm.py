# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA KVBM (KV Block Manager) metrics faker."""

from dataclasses import dataclass, field

from aiperf_mock_server.prometheus_base import (
    PrometheusMetricsFaker,
    ServerState,
)


@dataclass
class KVBMServer(ServerState):
    """NVIDIA KVBM (KV Block Manager) server state and metrics."""

    # Gauge metrics - KV cache stats
    kvstats_active_blocks: int = 0
    kvstats_total_blocks: int = 0
    kvstats_gpu_cache_usage_percent: float = 0.0
    kvstats_gpu_prefix_cache_hit_rate: float = 0.0

    # Counter metrics - KVBM operations
    kvbm_match_operations_total: int = 0
    kvbm_offload_operations_total: int = 0
    kvbm_onboard_operations_total: int = 0
    kvbm_eviction_operations_total: int = 0

    # Counter metrics - blocks transferred
    kvbm_blocks_offloaded_total: int = 0
    kvbm_blocks_onboarded_total: int = 0

    # Gauge metrics - operational
    kvbm_active_sequences: int = 0
    kvbm_pending_offload_requests: int = 0
    kvbm_pending_onboard_requests: int = 0

    # Histogram tracking - operation latencies
    kvbm_match_latency_seconds: list[float] = field(default_factory=list)
    kvbm_offload_latency_seconds: list[float] = field(default_factory=list)
    kvbm_onboard_latency_seconds: list[float] = field(default_factory=list)


class KVBMMetricsFaker(PrometheusMetricsFaker):
    """Faker for NVIDIA KVBM (KV Block Manager) metrics."""

    def _create_server(self, idx: int) -> KVBMServer:
        server = KVBMServer(idx, self.cfg, self.rng, self.rng.uniform(-0.05, 0.05))
        # Initialize total blocks based on memory
        server.kvstats_total_blocks = int(self.cfg.memory_gb * 1024 / 16)
        return server

    def _update_server_metrics(self, server: KVBMServer, base_load: float) -> None:
        load = server._calculate_load(base_load)
        c = self.cfg

        current_rps = int(load * c.max_rps)

        # KV cache stats
        server.kvstats_active_blocks = int(
            server._noise(
                load * server.kvstats_total_blocks * 0.7,
                0.05,
                server.kvstats_total_blocks,
            )
        )
        server.kvstats_gpu_cache_usage_percent = (
            server.kvstats_active_blocks / server.kvstats_total_blocks
            if server.kvstats_total_blocks > 0
            else 0.0
        )
        server.kvstats_gpu_prefix_cache_hit_rate = server._noise(
            0.5 + load * 0.3, 0.1, 1.0
        )

        # KVBM operations - increase with load
        match_ops = int(
            server._noise(current_rps * 1.5, 0.1, None)
        )  # Match is most frequent
        offload_ops = int(
            server._noise(current_rps * 0.3, 0.2, None)
        )  # Offload less frequent
        onboard_ops = int(
            server._noise(current_rps * 0.25, 0.2, None)
        )  # Onboard similar to offload
        eviction_ops = int(
            server._noise(current_rps * 0.2, 0.2, None)
        )  # Eviction least frequent

        server.kvbm_match_operations_total += match_ops
        server.kvbm_offload_operations_total += offload_ops
        server.kvbm_onboard_operations_total += onboard_ops
        server.kvbm_eviction_operations_total += eviction_ops

        # Blocks transferred
        avg_blocks_per_offload = 50
        avg_blocks_per_onboard = 45
        server.kvbm_blocks_offloaded_total += int(offload_ops * avg_blocks_per_offload)
        server.kvbm_blocks_onboarded_total += int(onboard_ops * avg_blocks_per_onboard)

        # Operational gauges
        server.kvbm_active_sequences = int(
            server._noise(load * c.max_connections * 0.4, 0.1, c.max_connections)
        )
        server.kvbm_pending_offload_requests = int(
            server._noise(load * c.max_queue_size * 0.1, 0.2, c.max_queue_size * 0.5)
        )
        server.kvbm_pending_onboard_requests = int(
            server._noise(load * c.max_queue_size * 0.08, 0.2, c.max_queue_size * 0.5)
        )

        # Histogram samples - operation latencies
        num_samples = min(current_rps, 1000)
        # Match operations are fast
        server.kvbm_match_latency_seconds = [
            server._noise(0.001 + load * 0.002, 0.2, None)
            for _ in range(min(match_ops, num_samples))
        ]
        # Offload operations are slower (network/storage)
        server.kvbm_offload_latency_seconds = [
            server._noise(0.01 + load * 0.05, 0.3, None)
            for _ in range(min(offload_ops, num_samples))
        ]
        # Onboard operations are similar to offload
        server.kvbm_onboard_latency_seconds = [
            server._noise(0.012 + load * 0.048, 0.3, None)
            for _ in range(min(onboard_ops, num_samples))
        ]

    def _generate_metrics(self) -> list[str]:
        metrics = []

        # KV cache stats (gauges)
        metrics.append(
            self._format_gauge(
                "dynamo_component_kvstats_active_blocks",
                "Number of active KV cache blocks currently in use",
                "kvstats_active_blocks",
            )
        )
        metrics.append(
            self._format_gauge(
                "dynamo_component_kvstats_total_blocks",
                "Total number of KV cache blocks available",
                "kvstats_total_blocks",
            )
        )
        metrics.append(
            self._format_gauge(
                "dynamo_component_kvstats_gpu_cache_usage_percent",
                "GPU cache usage as a percentage (0.0-1.0)",
                "kvstats_gpu_cache_usage_percent",
            )
        )
        metrics.append(
            self._format_gauge(
                "dynamo_component_kvstats_gpu_prefix_cache_hit_rate",
                "GPU prefix cache hit rate as a percentage (0.0-1.0)",
                "kvstats_gpu_prefix_cache_hit_rate",
            )
        )

        # KVBM operation counters
        metrics.append(
            self._format_counter(
                "dynamo_component_kvbm_match_operations_total",
                "Total number of KV block match operations",
                "kvbm_match_operations_total",
            )
        )
        metrics.append(
            self._format_counter(
                "dynamo_component_kvbm_offload_operations_total",
                "Total number of KV block offload operations",
                "kvbm_offload_operations_total",
            )
        )
        metrics.append(
            self._format_counter(
                "dynamo_component_kvbm_onboard_operations_total",
                "Total number of KV block onboard operations",
                "kvbm_onboard_operations_total",
            )
        )
        metrics.append(
            self._format_counter(
                "dynamo_component_kvbm_eviction_operations_total",
                "Total number of KV block eviction operations",
                "kvbm_eviction_operations_total",
            )
        )

        # Block transfer counters
        metrics.append(
            self._format_counter(
                "dynamo_component_kvbm_blocks_offloaded_total",
                "Total number of KV blocks offloaded",
                "kvbm_blocks_offloaded_total",
            )
        )
        metrics.append(
            self._format_counter(
                "dynamo_component_kvbm_blocks_onboarded_total",
                "Total number of KV blocks onboarded",
                "kvbm_blocks_onboarded_total",
            )
        )

        # Operational gauges
        metrics.append(
            self._format_gauge(
                "dynamo_component_kvbm_active_sequences",
                "Number of active sequences using KVBM",
                "kvbm_active_sequences",
            )
        )
        metrics.append(
            self._format_gauge(
                "dynamo_component_kvbm_pending_offload_requests",
                "Number of pending offload requests",
                "kvbm_pending_offload_requests",
            )
        )
        metrics.append(
            self._format_gauge(
                "dynamo_component_kvbm_pending_onboard_requests",
                "Number of pending onboard requests",
                "kvbm_pending_onboard_requests",
            )
        )

        # Histograms - operation latencies
        metrics.append(
            self._format_histogram(
                "dynamo_component_kvbm_match_latency_seconds",
                "Histogram of KV block match operation latency in seconds",
                "kvbm_match_latency_seconds",
                [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
            )
        )
        metrics.append(
            self._format_histogram(
                "dynamo_component_kvbm_offload_latency_seconds",
                "Histogram of KV block offload operation latency in seconds",
                "kvbm_offload_latency_seconds",
                [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            )
        )
        metrics.append(
            self._format_histogram(
                "dynamo_component_kvbm_onboard_latency_seconds",
                "Histogram of KV block onboard operation latency in seconds",
                "kvbm_onboard_latency_seconds",
                [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            )
        )

        return metrics
