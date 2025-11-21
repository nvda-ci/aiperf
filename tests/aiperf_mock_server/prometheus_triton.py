# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Triton Inference Server metrics faker."""

from dataclasses import dataclass, field

from aiperf_mock_server.prometheus_base import (
    PrometheusMetricsFaker,
    ServerState,
)


@dataclass
class TritonServer(ServerState):
    """NVIDIA Triton Inference Server state and metrics."""

    # Counter metrics - request counts
    inference_request_success: int = 0
    inference_request_failure: int = 0
    inference_count: int = 0
    inference_exec_count: int = 0

    # Counter metrics - latencies (cumulative microseconds)
    inference_request_duration_us: int = 0
    inference_queue_duration_us: int = 0
    inference_compute_input_duration_us: int = 0
    inference_compute_infer_duration_us: int = 0
    inference_compute_output_duration_us: int = 0

    # Counter metrics - cache
    cache_num_hits: int = 0
    cache_num_misses: int = 0
    cache_hit_duration_us: int = 0
    cache_miss_duration_us: int = 0

    # Gauge metrics - GPU
    gpu_utilization: float = 0.0
    gpu_memory_total_bytes: int = 0
    gpu_memory_used_bytes: int = 0
    gpu_power_usage: float = 0.0
    gpu_power_limit: float = 0.0
    energy_consumption: float = 0.0

    # Gauge metrics - CPU
    cpu_utilization: float = 0.0
    cpu_memory_total_bytes: int = 0
    cpu_memory_used_bytes: int = 0

    # Gauge metrics - pending requests
    pending_request_count: int = 0

    # Histogram tracking - first response time (ms)
    first_response_times_ms: list[float] = field(default_factory=list)


class TritonMetricsFaker(PrometheusMetricsFaker):
    """Faker for NVIDIA Triton Inference Server metrics."""

    def _create_server(self, idx: int) -> TritonServer:
        server = TritonServer(idx, self.cfg, self.rng, self.rng.uniform(-0.05, 0.05))
        # Initialize fixed values
        server.gpu_memory_total_bytes = self.cfg.memory_gb * 1024**3
        server.gpu_power_limit = 400.0
        server.cpu_memory_total_bytes = self.cfg.memory_gb * 2 * 1024**3
        return server

    def _update_server_metrics(self, server: TritonServer, base_load: float) -> None:
        load = server._calculate_load(base_load)
        c = self.cfg

        current_rps = int(load * c.max_rps)
        successful_requests = int(current_rps * (1 - c.error_rate))
        failed_requests = current_rps - successful_requests

        # Update request counters
        server.inference_request_success += successful_requests
        server.inference_request_failure += failed_requests
        server.inference_count += successful_requests
        server.inference_exec_count += int(successful_requests * 0.8)  # Batching

        # Update latency counters (cumulative microseconds)
        avg_request_duration_us = (10000 + load * 90000) * (
            1 + server.rng.uniform(-0.1, 0.1)
        )
        server.inference_request_duration_us += int(
            successful_requests * avg_request_duration_us
        )
        server.inference_queue_duration_us += int(
            successful_requests * avg_request_duration_us * 0.1
        )
        server.inference_compute_input_duration_us += int(
            successful_requests * avg_request_duration_us * 0.05
        )
        server.inference_compute_infer_duration_us += int(
            successful_requests * avg_request_duration_us * 0.8
        )
        server.inference_compute_output_duration_us += int(
            successful_requests * avg_request_duration_us * 0.05
        )

        # Cache metrics
        cache_hit_rate = 0.5 + load * 0.3
        cache_hits = int(successful_requests * cache_hit_rate)
        cache_misses = successful_requests - cache_hits
        server.cache_num_hits += cache_hits
        server.cache_num_misses += cache_misses
        server.cache_hit_duration_us += cache_hits * int(server._noise(500, 0.2, None))
        server.cache_miss_duration_us += cache_misses * int(
            server._noise(2000, 0.2, None)
        )

        # GPU metrics
        server.gpu_utilization = server._noise(0.3 + load * 0.65, 0.05, 1.0)
        server.gpu_memory_used_bytes = int(
            server._noise(
                (0.2 + load * 0.7) * server.gpu_memory_total_bytes,
                0.05,
                server.gpu_memory_total_bytes,
            )
        )
        server.gpu_power_usage = server._noise(
            100 + load * 250, 0.1, server.gpu_power_limit
        )
        server.energy_consumption += server.gpu_power_usage * 1.0  # Joules per second

        # CPU metrics
        server.cpu_utilization = server._noise(0.2 + load * 0.5, 0.05, 1.0)
        server.cpu_memory_used_bytes = int(
            server._noise(
                (0.1 + load * 0.4) * server.cpu_memory_total_bytes,
                0.05,
                server.cpu_memory_total_bytes,
            )
        )

        # Pending requests
        server.pending_request_count = int(
            server._noise(load * c.max_queue_size * 0.15, 0.2, c.max_queue_size)
        )

        # Histogram samples - first response time
        num_samples = min(successful_requests, 1000)
        server.first_response_times_ms = [
            server._noise(10 + load * 50, 0.2, None) for _ in range(num_samples)
        ]

    def _generate_metrics(self) -> list[str]:
        metrics = []

        # Request count metrics
        metrics.append(
            self._format_counter(
                "nv_inference_request_success",
                "Number of successful inference requests per model",
                "inference_request_success",
            )
        )
        metrics.append(
            self._format_counter(
                "nv_inference_request_failure",
                "Number of failed inference requests per model",
                "inference_request_failure",
            )
        )
        metrics.append(
            self._format_counter(
                "nv_inference_count",
                "Number of inferences performed per model",
                "inference_count",
            )
        )
        metrics.append(
            self._format_counter(
                "nv_inference_exec_count",
                "Number of model batch executions performed",
                "inference_exec_count",
            )
        )

        # Latency metrics (cumulative microseconds)
        metrics.append(
            self._format_counter(
                "nv_inference_request_duration_us",
                "Cumulative end-to-end inference request handling time in microseconds",
                "inference_request_duration_us",
            )
        )
        metrics.append(
            self._format_counter(
                "nv_inference_queue_duration_us",
                "Cumulative time requests wait in scheduling queue in microseconds",
                "inference_queue_duration_us",
            )
        )
        metrics.append(
            self._format_counter(
                "nv_inference_compute_input_duration_us",
                "Cumulative time to prepare input tensors in microseconds",
                "inference_compute_input_duration_us",
            )
        )
        metrics.append(
            self._format_counter(
                "nv_inference_compute_infer_duration_us",
                "Cumulative inference compute time in microseconds",
                "inference_compute_infer_duration_us",
            )
        )
        metrics.append(
            self._format_counter(
                "nv_inference_compute_output_duration_us",
                "Cumulative time to extract output tensors in microseconds",
                "inference_compute_output_duration_us",
            )
        )

        # Cache metrics
        metrics.append(
            self._format_counter(
                "nv_cache_num_hits_per_model",
                "Number of cache hits per model",
                "cache_num_hits",
            )
        )
        metrics.append(
            self._format_counter(
                "nv_cache_num_misses_per_model",
                "Number of cache misses per model",
                "cache_num_misses",
            )
        )
        metrics.append(
            self._format_counter(
                "nv_cache_hit_duration_per_model",
                "Total cache hit lookup time in microseconds",
                "cache_hit_duration_us",
            )
        )
        metrics.append(
            self._format_counter(
                "nv_cache_miss_duration_per_model",
                "Total cache miss insertion time in microseconds",
                "cache_miss_duration_us",
            )
        )

        # GPU metrics
        metrics.append(
            self._format_gauge(
                "nv_gpu_utilization",
                "GPU utilization rate [0.0 - 1.0]",
                "gpu_utilization",
            )
        )
        metrics.append(
            self._format_gauge(
                "nv_gpu_memory_total_bytes",
                "Total GPU memory in bytes",
                "gpu_memory_total_bytes",
            )
        )
        metrics.append(
            self._format_gauge(
                "nv_gpu_memory_used_bytes",
                "Used GPU memory in bytes",
                "gpu_memory_used_bytes",
            )
        )
        metrics.append(
            self._format_gauge(
                "nv_gpu_power_usage",
                "GPU instantaneous power usage in watts",
                "gpu_power_usage",
            )
        )
        metrics.append(
            self._format_gauge(
                "nv_gpu_power_limit",
                "GPU maximum power limit in watts",
                "gpu_power_limit",
            )
        )
        metrics.append(
            self._format_gauge(
                "nv_energy_consumption",
                "GPU energy consumption in joules since server start",
                "energy_consumption",
            )
        )

        # CPU metrics
        metrics.append(
            self._format_gauge(
                "nv_cpu_utilization",
                "CPU utilization rate [0.0 - 1.0]",
                "cpu_utilization",
            )
        )
        metrics.append(
            self._format_gauge(
                "nv_cpu_memory_total_bytes",
                "Total CPU memory in bytes",
                "cpu_memory_total_bytes",
            )
        )
        metrics.append(
            self._format_gauge(
                "nv_cpu_memory_used_bytes",
                "Used CPU memory in bytes",
                "cpu_memory_used_bytes",
            )
        )

        # Pending requests
        metrics.append(
            self._format_gauge(
                "nv_inference_pending_request_count",
                "Number of inference requests awaiting execution per model",
                "pending_request_count",
            )
        )

        # Histogram - first response time
        metrics.append(
            self._format_histogram(
                "nv_inference_first_response_histogram_ms",
                "Histogram of time to first response in milliseconds",
                "first_response_times_ms",
                [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
            )
        )

        return metrics
