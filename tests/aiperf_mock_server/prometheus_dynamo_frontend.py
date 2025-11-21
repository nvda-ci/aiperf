# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AI Dynamo frontend metrics faker."""

from dataclasses import dataclass, field

from aiperf_mock_server.prometheus_base import (
    DURATION_BUCKETS,
    ITL_BUCKETS,
    LATENCY_BUCKETS,
    OUTPUT_TOKEN_BUCKETS,
    TOKEN_BUCKETS,
    PrometheusMetricsFaker,
    ServerState,
)


@dataclass
class DynamoFrontendServer(ServerState):
    """AI Dynamo frontend server state and metrics."""

    # Counter metrics
    dynamo_frontend_requests_total: int = 0
    dynamo_frontend_output_tokens_total: int = 0

    # Gauge metrics
    dynamo_frontend_queued_requests: int = 0
    dynamo_frontend_inflight_requests: int = 0
    dynamo_frontend_disconnected_clients: int = 0
    dynamo_frontend_model_total_kv_blocks: int = 0

    # KV stats gauges (no dynamo prefix!)
    kvstats_active_blocks: int = 0
    kvstats_total_blocks: int = 0
    kvstats_gpu_cache_usage_percent: float = 0.0
    kvstats_gpu_prefix_cache_hit_rate: float = 0.0

    # Histogram tracking
    dynamo_frontend_request_duration_seconds: list[float] = field(default_factory=list)
    dynamo_frontend_time_to_first_token_seconds: list[float] = field(
        default_factory=list
    )
    dynamo_frontend_inter_token_latency_seconds: list[float] = field(
        default_factory=list
    )
    dynamo_frontend_input_sequence_tokens: list[int] = field(default_factory=list)
    dynamo_frontend_output_sequence_tokens: list[int] = field(default_factory=list)


class DynamoFrontendMetricsFaker(PrometheusMetricsFaker):
    """Faker for AI Dynamo frontend metrics."""

    def _create_server(self, idx: int) -> DynamoFrontendServer:
        return DynamoFrontendServer(
            idx, self.cfg, self.rng, self.rng.uniform(-0.05, 0.05)
        )

    def _update_server_metrics(
        self, server: DynamoFrontendServer, base_load: float
    ) -> None:
        load = server._calculate_load(base_load)
        c = self.cfg

        current_rps = int(load * c.max_rps)
        successful_requests = int(current_rps * (1 - c.error_rate))

        # Counter metrics
        server.dynamo_frontend_requests_total += current_rps
        server.dynamo_frontend_output_tokens_total += successful_requests * 30

        # Gauge metrics
        server.dynamo_frontend_queued_requests = int(
            server._noise(load * c.max_queue_size * 0.2, 0.15, c.max_queue_size)
        )
        server.dynamo_frontend_inflight_requests = int(
            server._noise(load * c.max_connections * 0.3, 0.1, c.max_connections)
        )
        server.dynamo_frontend_disconnected_clients = int(
            server._noise(load * 5, 0.3, 20)
        )

        # KV cache metrics
        total_kv_blocks = int(c.memory_gb * 1024 / 16)
        server.dynamo_frontend_model_total_kv_blocks = total_kv_blocks
        server.kvstats_total_blocks = total_kv_blocks
        server.kvstats_active_blocks = int(
            server._noise(load * total_kv_blocks * 0.7, 0.05, total_kv_blocks)
        )
        server.kvstats_gpu_cache_usage_percent = server._noise(
            0.3 + load * 0.6, 0.05, 1.0
        )
        server.kvstats_gpu_prefix_cache_hit_rate = server._noise(
            0.5 + load * 0.3, 0.1, 1.0
        )

        # Histogram samples
        num_samples = min(current_rps, 1000)
        avg_request_duration = 0.1 + load * 0.5
        server.dynamo_frontend_request_duration_seconds = [
            server._noise(avg_request_duration, 0.2, None) for _ in range(num_samples)
        ]
        server.dynamo_frontend_time_to_first_token_seconds = [
            server._noise(0.01 + load * 0.05, 0.2, None) for _ in range(num_samples)
        ]
        server.dynamo_frontend_inter_token_latency_seconds = [
            server._noise(0.005 + load * 0.015, 0.2, None) for _ in range(num_samples)
        ]
        server.dynamo_frontend_input_sequence_tokens = [
            int(server._noise(50, 0.3, None)) for _ in range(num_samples)
        ]
        server.dynamo_frontend_output_sequence_tokens = [
            int(server._noise(30, 0.3, None)) for _ in range(num_samples)
        ]

    def _generate_metrics(self) -> list[str]:
        metrics = []
        # Counters
        metrics.append(
            self._format_counter(
                "dynamo_frontend_requests_total",
                "Total LLM requests processed.",
                "dynamo_frontend_requests_total",
            )
        )
        metrics.append(
            self._format_counter(
                "dynamo_frontend_output_tokens_total",
                "Total generated output tokens.",
                "dynamo_frontend_output_tokens_total",
            )
        )
        # Gauges
        metrics.append(
            self._format_gauge(
                "dynamo_frontend_queued_requests",
                "Requests in HTTP queue.",
                "dynamo_frontend_queued_requests",
            )
        )
        metrics.append(
            self._format_gauge(
                "dynamo_frontend_inflight_requests",
                "Concurrent requests to engine.",
                "dynamo_frontend_inflight_requests",
            )
        )
        metrics.append(
            self._format_gauge(
                "dynamo_frontend_disconnected_clients",
                "Current disconnected clients.",
                "dynamo_frontend_disconnected_clients",
            )
        )
        metrics.append(
            self._format_gauge(
                "dynamo_frontend_model_total_kv_blocks",
                "Available KV blocks per worker.",
                "dynamo_frontend_model_total_kv_blocks",
            )
        )
        metrics.append(
            self._format_gauge(
                "kvstats_active_blocks",
                "Active KV cache blocks.",
                "kvstats_active_blocks",
            )
        )
        metrics.append(
            self._format_gauge(
                "kvstats_total_blocks",
                "Total KV blocks available.",
                "kvstats_total_blocks",
            )
        )
        metrics.append(
            self._format_gauge(
                "kvstats_gpu_cache_usage_percent",
                "GPU cache usage percentage.",
                "kvstats_gpu_cache_usage_percent",
            )
        )
        metrics.append(
            self._format_gauge(
                "kvstats_gpu_prefix_cache_hit_rate",
                "Prefix cache hit rate.",
                "kvstats_gpu_prefix_cache_hit_rate",
            )
        )
        # Histograms
        metrics.append(
            self._format_histogram(
                "dynamo_frontend_request_duration_seconds",
                "LLM request duration in seconds.",
                "dynamo_frontend_request_duration_seconds",
                DURATION_BUCKETS,
            )
        )
        metrics.append(
            self._format_histogram(
                "dynamo_frontend_time_to_first_token_seconds",
                "First token latency in seconds.",
                "dynamo_frontend_time_to_first_token_seconds",
                LATENCY_BUCKETS,
            )
        )
        metrics.append(
            self._format_histogram(
                "dynamo_frontend_inter_token_latency_seconds",
                "Token-to-token latency in seconds.",
                "dynamo_frontend_inter_token_latency_seconds",
                ITL_BUCKETS,
            )
        )
        metrics.append(
            self._format_histogram(
                "dynamo_frontend_input_sequence_tokens",
                "Input token count distribution.",
                "dynamo_frontend_input_sequence_tokens",
                TOKEN_BUCKETS,
            )
        )
        metrics.append(
            self._format_histogram(
                "dynamo_frontend_output_sequence_tokens",
                "Output token count distribution.",
                "dynamo_frontend_output_sequence_tokens",
                OUTPUT_TOKEN_BUCKETS,
            )
        )
        return metrics
