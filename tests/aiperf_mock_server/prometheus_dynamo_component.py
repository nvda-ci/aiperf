# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AI Dynamo component/work handler metrics faker."""

from dataclasses import dataclass, field

from aiperf_mock_server.prometheus_base import (
    DURATION_BUCKETS,
    PrometheusMetricsFaker,
    ServerState,
)


@dataclass
class DynamoComponentServer(ServerState):
    """AI Dynamo component/work handler state and metrics."""

    # Counter metrics
    dynamo_component_requests_total: int = 0
    dynamo_component_request_bytes_total: int = 0
    dynamo_component_response_bytes_total: int = 0
    dynamo_component_errors_total: int = 0
    dynamo_component_tasks_issued_total: int = 0
    dynamo_component_tasks_success_total: int = 0
    dynamo_component_tasks_failed_total: int = 0

    # Gauge metrics
    dynamo_component_inflight_requests: int = 0
    dynamo_component_uptime_seconds: float = 0.0

    # Histogram tracking
    dynamo_component_request_duration_seconds: list[float] = field(default_factory=list)


class DynamoComponentMetricsFaker(PrometheusMetricsFaker):
    """Faker for AI Dynamo component/work handler metrics."""

    def _create_server(self, idx: int) -> DynamoComponentServer:
        return DynamoComponentServer(
            idx, self.cfg, self.rng, self.rng.uniform(-0.05, 0.05)
        )

    def _update_server_metrics(
        self, server: DynamoComponentServer, base_load: float
    ) -> None:
        load = server._calculate_load(base_load)
        c = self.cfg

        current_rps = int(load * c.max_rps)
        successful_requests = int(current_rps * (1 - c.error_rate))

        # Counter metrics
        server.dynamo_component_requests_total += current_rps
        server.dynamo_component_request_bytes_total += current_rps * 500
        server.dynamo_component_response_bytes_total += successful_requests * 2000
        server.dynamo_component_errors_total += int(current_rps * c.error_rate)
        server.dynamo_component_tasks_issued_total += current_rps
        server.dynamo_component_tasks_success_total += successful_requests
        server.dynamo_component_tasks_failed_total += int(current_rps * c.error_rate)

        # Gauge metrics
        server.dynamo_component_inflight_requests = int(
            server._noise(load * c.max_connections * 0.25, 0.1, c.max_connections)
        )
        server.dynamo_component_uptime_seconds += 1.0

        # Histogram samples
        num_samples = min(current_rps, 1000)
        avg_request_duration = 0.1 + load * 0.5
        server.dynamo_component_request_duration_seconds = [
            server._noise(avg_request_duration * 0.9, 0.2, None)
            for _ in range(num_samples)
        ]

    def _generate_metrics(self) -> list[str]:
        metrics = []
        # Counters
        metrics.append(
            self._format_counter(
                "dynamo_component_requests_total",
                "Total component requests processed.",
                "dynamo_component_requests_total",
            )
        )
        metrics.append(
            self._format_counter(
                "dynamo_component_request_bytes_total",
                "Total request bytes received.",
                "dynamo_component_request_bytes_total",
            )
        )
        metrics.append(
            self._format_counter(
                "dynamo_component_response_bytes_total",
                "Total response bytes sent.",
                "dynamo_component_response_bytes_total",
            )
        )
        metrics.append(
            self._format_counter(
                "dynamo_component_errors_total",
                "Total processing errors.",
                "dynamo_component_errors_total",
            )
        )
        metrics.append(
            self._format_counter(
                "dynamo_component_tasks_issued_total",
                "Total tasks submitted.",
                "dynamo_component_tasks_issued_total",
            )
        )
        metrics.append(
            self._format_counter(
                "dynamo_component_tasks_success_total",
                "Successfully completed tasks.",
                "dynamo_component_tasks_success_total",
            )
        )
        metrics.append(
            self._format_counter(
                "dynamo_component_tasks_failed_total",
                "Failed tasks.",
                "dynamo_component_tasks_failed_total",
            )
        )
        # Gauges
        metrics.append(
            self._format_gauge(
                "dynamo_component_inflight_requests",
                "Concurrent requests being processed.",
                "dynamo_component_inflight_requests",
            )
        )
        metrics.append(
            self._format_gauge(
                "dynamo_component_uptime_seconds",
                "Component uptime in seconds.",
                "dynamo_component_uptime_seconds",
            )
        )
        # Histograms
        metrics.append(
            self._format_histogram(
                "dynamo_component_request_duration_seconds",
                "Component request duration in seconds.",
                "dynamo_component_request_duration_seconds",
                DURATION_BUCKETS,
            )
        )
        return metrics
