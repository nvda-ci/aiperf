# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SGLang server metrics faker."""

from dataclasses import dataclass, field

from aiperf_mock_server.prometheus_base import (
    PrometheusMetricsFaker,
    ServerState,
)


@dataclass
class SGLangServer(ServerState):
    """SGLang server state and metrics."""

    # Counter metrics
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0

    # Gauge metrics
    token_usage: float = 0.0
    cache_hit_rate: float = 0.0
    num_running_reqs: int = 0
    num_used_tokens: int = 0
    gen_throughput: float = 0.0
    num_queue_reqs: int = 0

    # Histogram tracking
    time_to_first_token_seconds: list[float] = field(default_factory=list)
    e2e_request_latency_seconds: list[float] = field(default_factory=list)
    time_per_output_token_seconds: list[float] = field(default_factory=list)


class SGLangMetricsFaker(PrometheusMetricsFaker):
    """Faker for SGLang metrics."""

    def _create_server(self, idx: int) -> SGLangServer:
        return SGLangServer(idx, self.cfg, self.rng, self.rng.uniform(-0.05, 0.05))

    def _update_server_metrics(self, server: SGLangServer, base_load: float) -> None:
        load = server._calculate_load(base_load)
        c = self.cfg

        current_rps = int(load * c.max_rps)
        successful_requests = int(current_rps * (1 - c.error_rate))

        # Counter metrics
        prompt_tokens = successful_requests * 50
        generation_tokens = successful_requests * 30
        server.prompt_tokens_total += prompt_tokens
        server.generation_tokens_total += generation_tokens

        # Gauge metrics
        server.num_running_reqs = int(
            server._noise(load * c.max_connections * 0.3, 0.1, c.max_connections)
        )
        server.num_queue_reqs = int(
            server._noise(load * c.max_queue_size * 0.2, 0.15, c.max_queue_size)
        )

        # Token usage and cache metrics
        max_tokens = c.memory_gb * 1024 * 200  # Approximate tokens that can fit
        server.num_used_tokens = int(
            server._noise(load * max_tokens * 0.7, 0.05, max_tokens)
        )
        server.token_usage = (
            server.num_used_tokens / max_tokens if max_tokens > 0 else 0.0
        )
        server.cache_hit_rate = server._noise(0.4 + load * 0.4, 0.1, 1.0)

        # Throughput
        server.gen_throughput = server._noise(current_rps * 30, 0.1, None)

        # Histogram samples
        num_samples = min(successful_requests, 1000)
        server.time_to_first_token_seconds = [
            server._noise(0.01 + load * 0.05, 0.2, None) for _ in range(num_samples)
        ]
        server.e2e_request_latency_seconds = [
            server._noise(0.5 + load * 2.0, 0.2, None) for _ in range(num_samples)
        ]
        server.time_per_output_token_seconds = [
            server._noise(0.01 + load * 0.02, 0.2, None) for _ in range(num_samples)
        ]

    def _generate_metrics(self) -> list[str]:
        metrics = []

        # Counters
        metrics.append(
            self._format_counter(
                "sglang:prompt_tokens_total",
                "Number of prefill tokens processed",
                "prompt_tokens_total",
            )
        )
        metrics.append(
            self._format_counter(
                "sglang:generation_tokens_total",
                "Number of generation tokens processed",
                "generation_tokens_total",
            )
        )

        # Gauges
        metrics.append(
            self._format_gauge(
                "sglang:token_usage",
                "Token usage measurement",
                "token_usage",
            )
        )
        metrics.append(
            self._format_gauge(
                "sglang:cache_hit_rate",
                "Cache hit rate measurement",
                "cache_hit_rate",
            )
        )
        metrics.append(
            self._format_gauge(
                "sglang:num_running_reqs",
                "The number of running requests",
                "num_running_reqs",
            )
        )
        metrics.append(
            self._format_gauge(
                "sglang:num_used_tokens",
                "The number of used tokens",
                "num_used_tokens",
            )
        )
        metrics.append(
            self._format_gauge(
                "sglang:gen_throughput",
                "The generate throughput (token/s)",
                "gen_throughput",
            )
        )
        metrics.append(
            self._format_gauge(
                "sglang:num_queue_reqs",
                "The number of requests in the waiting queue",
                "num_queue_reqs",
            )
        )

        # Histograms
        metrics.append(
            self._format_histogram(
                "sglang:time_to_first_token_seconds",
                "Histogram of time to first token in seconds",
                "time_to_first_token_seconds",
                [
                    0.001,
                    0.005,
                    0.01,
                    0.02,
                    0.04,
                    0.06,
                    0.08,
                    0.1,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                    2.5,
                    5.0,
                    7.5,
                    10.0,
                    15.0,
                    20.0,
                    25.0,
                    30.0,
                ],
            )
        )
        metrics.append(
            self._format_histogram(
                "sglang:e2e_request_latency_seconds",
                "Histogram of End-to-end request latency in seconds",
                "e2e_request_latency_seconds",
                [
                    0.3,
                    0.5,
                    0.8,
                    1.0,
                    1.5,
                    2.0,
                    2.5,
                    5.0,
                    10.0,
                    15.0,
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                    60.0,
                ],
            )
        )
        metrics.append(
            self._format_histogram(
                "sglang:time_per_output_token_seconds",
                "Histogram of time per output token in seconds",
                "time_per_output_token_seconds",
                [
                    0.005,
                    0.01,
                    0.015,
                    0.02,
                    0.025,
                    0.03,
                    0.04,
                    0.05,
                    0.075,
                    0.1,
                    0.15,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.75,
                    1.0,
                    2.5,
                ],
            )
        )

        return metrics
