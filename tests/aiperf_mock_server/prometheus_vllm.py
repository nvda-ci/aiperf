# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vLLM server metrics faker."""

from dataclasses import dataclass, field

from aiperf_mock_server.prometheus_base import (
    OUTPUT_TOKEN_BUCKETS,
    TOKEN_BUCKETS,
    PrometheusMetricsFaker,
    ServerState,
)


@dataclass
class VLLMServer(ServerState):
    """vLLM server state and metrics."""

    # Gauge metrics
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    num_requests_swapped: int = 0
    gpu_cache_usage_perc: float = 0.0
    cpu_cache_usage_perc: float = 0.0
    cpu_prefix_cache_hit_rate: float = 0.0
    gpu_prefix_cache_hit_rate: float = 0.0
    avg_prompt_throughput: float = 0.0
    avg_generation_throughput: float = 0.0

    # Counter metrics
    num_preemptions_total: int = 0
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0
    request_success_total: int = 0

    # Histogram tracking
    time_to_first_token: list[float] = field(default_factory=list)
    time_per_output_token: list[float] = field(default_factory=list)
    e2e_request_latency: list[float] = field(default_factory=list)
    request_prompt_tokens: list[int] = field(default_factory=list)
    request_generation_tokens: list[int] = field(default_factory=list)


class VLLMMetricsFaker(PrometheusMetricsFaker):
    """Faker for vLLM metrics."""

    def _create_server(self, idx: int) -> VLLMServer:
        return VLLMServer(idx, self.cfg, self.rng, self.rng.uniform(-0.05, 0.05))

    def _update_server_metrics(self, server: VLLMServer, base_load: float) -> None:
        load = server._calculate_load(base_load)
        c = self.cfg

        # Gauge metrics
        server.num_requests_running = int(
            server._noise(load * c.max_connections * 0.3, 0.1, c.max_connections)
        )
        server.num_requests_waiting = int(
            server._noise(load * c.max_queue_size * 0.2, 0.15, c.max_queue_size)
        )
        server.num_requests_swapped = int(server._noise(load * 10, 0.2, 50))
        server.gpu_cache_usage_perc = server._noise(0.3 + load * 0.6, 0.05, 1.0)
        server.cpu_cache_usage_perc = server._noise(0.1 + load * 0.4, 0.05, 1.0)
        server.gpu_prefix_cache_hit_rate = server._noise(0.5 + load * 0.3, 0.1, 1.0)
        server.cpu_prefix_cache_hit_rate = server._noise(0.4 + load * 0.3, 0.1, 1.0)

        current_rps = load * c.max_rps
        server.avg_prompt_throughput = server._noise(current_rps * 50, 0.1, None)
        server.avg_generation_throughput = server._noise(current_rps * 30, 0.1, None)

        # Counter metrics
        server.prompt_tokens_total += int(current_rps * 50)
        server.generation_tokens_total += int(current_rps * 30)
        server.request_success_total += int(current_rps * (1 - c.error_rate))
        server.num_preemptions_total += int(server._noise(load * 5, 0.5, None))

        # Histogram samples
        num_samples = min(int(current_rps), 1000)
        server.time_to_first_token = [
            server._noise(0.01 + load * 0.05, 0.2, None) for _ in range(num_samples)
        ]
        server.time_per_output_token = [
            server._noise(0.005 + load * 0.015, 0.2, None) for _ in range(num_samples)
        ]
        server.e2e_request_latency = [
            server._noise(0.1 + load * 0.5, 0.2, None) for _ in range(num_samples)
        ]
        server.request_prompt_tokens = [
            int(server._noise(50, 0.3, None)) for _ in range(num_samples)
        ]
        server.request_generation_tokens = [
            int(server._noise(30, 0.3, None)) for _ in range(num_samples)
        ]

    def _generate_metrics(self) -> list[str]:
        metrics = []
        # Gauges
        metrics.append(
            self._format_gauge(
                "vllm:num_requests_running",
                "Number of requests currently running on GPU.",
                "num_requests_running",
            )
        )
        metrics.append(
            self._format_gauge(
                "vllm:num_requests_waiting",
                "Number of requests waiting to be processed.",
                "num_requests_waiting",
            )
        )
        metrics.append(
            self._format_gauge(
                "vllm:num_requests_swapped",
                "Number of requests swapped to CPU.",
                "num_requests_swapped",
            )
        )
        metrics.append(
            self._format_gauge(
                "vllm:gpu_cache_usage_perc",
                "GPU KV-cache usage. 1 means 100 percent usage.",
                "gpu_cache_usage_perc",
            )
        )
        metrics.append(
            self._format_gauge(
                "vllm:cpu_cache_usage_perc",
                "CPU KV-cache usage. 1 means 100 percent usage.",
                "cpu_cache_usage_perc",
            )
        )
        metrics.append(
            self._format_gauge(
                "vllm:gpu_prefix_cache_hit_rate",
                "GPU prefix cache block hit rate.",
                "gpu_prefix_cache_hit_rate",
            )
        )
        metrics.append(
            self._format_gauge(
                "vllm:cpu_prefix_cache_hit_rate",
                "CPU prefix cache block hit rate.",
                "cpu_prefix_cache_hit_rate",
            )
        )
        metrics.append(
            self._format_gauge(
                "vllm:avg_prompt_throughput_toks_per_s",
                "Average prefill throughput in tokens/s.",
                "avg_prompt_throughput",
            )
        )
        metrics.append(
            self._format_gauge(
                "vllm:avg_generation_throughput_toks_per_s",
                "Average generation throughput in tokens/s.",
                "avg_generation_throughput",
            )
        )
        # Counters
        metrics.append(
            self._format_counter(
                "vllm:num_preemptions_total",
                "Cumulative number of preemption from the engine.",
                "num_preemptions_total",
            )
        )
        metrics.append(
            self._format_counter(
                "vllm:prompt_tokens_total",
                "Number of prefill tokens processed.",
                "prompt_tokens_total",
            )
        )
        metrics.append(
            self._format_counter(
                "vllm:generation_tokens_total",
                "Number of generation tokens processed.",
                "generation_tokens_total",
            )
        )
        metrics.append(
            self._format_counter(
                "vllm:request_success_total",
                "Count of successfully processed requests.",
                "request_success_total",
            )
        )
        # Histograms
        metrics.append(
            self._format_histogram(
                "vllm:time_to_first_token_seconds",
                "Histogram of time to first token in seconds.",
                "time_to_first_token",
                [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 1.0],
            )
        )
        metrics.append(
            self._format_histogram(
                "vllm:time_per_output_token_seconds",
                "Histogram of time per output token in seconds.",
                "time_per_output_token",
                [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2],
            )
        )
        metrics.append(
            self._format_histogram(
                "vllm:e2e_request_latency_seconds",
                "Histogram of end to end request latency in seconds.",
                "e2e_request_latency",
                [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1.0, 2.5],
            )
        )
        metrics.append(
            self._format_histogram(
                "vllm:request_prompt_tokens",
                "Distribution of prefill tokens per request.",
                "request_prompt_tokens",
                TOKEN_BUCKETS,
            )
        )
        metrics.append(
            self._format_histogram(
                "vllm:request_generation_tokens",
                "Distribution of generation tokens per request.",
                "request_generation_tokens",
                OUTPUT_TOKEN_BUCKETS,
            )
        )
        return metrics
