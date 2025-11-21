# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic HTTP server metrics faker."""

from dataclasses import dataclass, field

from aiperf_mock_server.prometheus_base import (
    SUMMARY_QUANTILES,
    PrometheusMetricsFaker,
    ServerState,
)


@dataclass
class GenericServer(ServerState):
    """Generic HTTP server state and metrics."""

    # Counter metrics
    requests_total: int = 0
    errors_total: int = 0
    bytes_sent_total: int = 0
    bytes_received_total: int = 0

    # Gauge metrics
    active_connections: int = 0
    queue_size: int = 0
    memory_used_bytes: int = 0

    # Histogram/Summary tracking
    request_durations: list[float] = field(default_factory=list)
    response_sizes: list[int] = field(default_factory=list)


class GenericMetricsFaker(PrometheusMetricsFaker):
    """Faker for generic HTTP server metrics."""

    def _create_server(self, idx: int) -> GenericServer:
        return GenericServer(idx, self.cfg, self.rng, self.rng.uniform(-0.05, 0.05))

    def _update_server_metrics(self, server: GenericServer, base_load: float) -> None:
        load = server._calculate_load(base_load)
        c = self.cfg

        current_rps = int(server._noise(load * c.max_rps, 0.05, c.max_rps))

        # Update counters
        server.requests_total += current_rps
        errors = int(current_rps * c.error_rate * server.rng.uniform(0.5, 1.5))
        server.errors_total += errors

        # Update gauges
        server.active_connections = int(
            server._noise(load * c.max_connections, 0.1, c.max_connections)
        )
        server.queue_size = int(
            server._noise(load * c.max_queue_size, 0.15, c.max_queue_size)
        )
        server.memory_used_bytes = int(
            server._noise(
                (0.3 + load * 0.6) * c.memory_gb * 1024**3,
                0.05,
                c.memory_gb * 1024**3,
            )
        )

        # Generate request samples
        server.request_durations = [
            server._noise(0.01 + load * 0.1, 0.2, None)
            for _ in range(min(current_rps, 1000))
        ]
        server.response_sizes = [
            int(server._noise(1024 + load * 4096, 0.3, None))
            for _ in range(min(current_rps, 1000))
        ]

        # Update bytes transferred
        for size in server.response_sizes:
            server.bytes_sent_total += size
            server.bytes_received_total += int(size * 0.1)

    def _generate_metrics(self) -> list[str]:
        metrics = []
        metrics.append(
            self._format_counter(
                "http_requests_total",
                "Total number of HTTP requests processed.",
                "requests_total",
            )
        )
        metrics.append(
            self._format_counter(
                "http_errors_total",
                "Total number of HTTP errors encountered.",
                "errors_total",
            )
        )
        metrics.append(
            self._format_counter(
                "http_bytes_sent_total",
                "Total bytes sent in HTTP responses.",
                "bytes_sent_total",
            )
        )
        metrics.append(
            self._format_counter(
                "http_bytes_received_total",
                "Total bytes received in HTTP requests.",
                "bytes_received_total",
            )
        )
        metrics.append(
            self._format_gauge(
                "http_active_connections",
                "Current number of active HTTP connections.",
                "active_connections",
            )
        )
        metrics.append(
            self._format_gauge(
                "http_queue_size", "Current size of the request queue.", "queue_size"
            )
        )
        metrics.append(
            self._format_gauge(
                "process_memory_bytes",
                "Current memory usage in bytes.",
                "memory_used_bytes",
            )
        )
        metrics.append(
            self._format_histogram(
                "http_request_duration_seconds",
                "HTTP request duration in seconds.",
                "request_durations",
                [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            )
        )
        metrics.append(
            self._format_histogram(
                "http_response_size_bytes",
                "HTTP response size in bytes.",
                "response_sizes",
                [100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000],
            )
        )
        metrics.append(
            self._format_summary(
                "http_request_latency_seconds",
                "HTTP request latency in seconds.",
                "request_durations",
                SUMMARY_QUANTILES,
            )
        )
        return metrics
