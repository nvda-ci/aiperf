# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base classes and shared configuration for Prometheus metrics fakers."""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

# =============================================================================
# Configuration and Constants
# =============================================================================


@dataclass
class ServerConfig:
    """Configuration for simulated server characteristics."""

    name: str
    max_rps: int
    max_connections: int
    memory_gb: int
    max_queue_size: int
    error_rate: float


SERVER_CONFIGS = {
    "small": ServerConfig("small-server", 100, 50, 8, 100, 0.001),
    "medium": ServerConfig("medium-server", 1000, 500, 32, 1000, 0.0005),
    "large": ServerConfig("large-server", 10000, 5000, 128, 10000, 0.0001),
    "vllm": ServerConfig("vllm-server", 500, 200, 48, 500, 0.0002),
    "dynamo": ServerConfig("dynamo-server", 2000, 1000, 80, 2000, 0.0001),
    "triton": ServerConfig("triton-server", 800, 400, 64, 800, 0.0003),
    "sglang": ServerConfig("sglang-server", 600, 300, 56, 600, 0.0002),
    "kvbm": ServerConfig("kvbm-server", 1500, 750, 96, 1500, 0.00015),
}

# Standard histogram buckets for different metric types
DURATION_BUCKETS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
LATENCY_BUCKETS = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5]
ITL_BUCKETS = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15]
TOKEN_BUCKETS = [1, 10, 50, 100, 200, 500, 1000, 2000, 5000]
OUTPUT_TOKEN_BUCKETS = [1, 10, 50, 100, 200, 500, 1000, 2000]
SUMMARY_QUANTILES = [0.5, 0.9, 0.95, 0.99]


# =============================================================================
# Base Server State
# =============================================================================


@dataclass
class ServerState:
    """Base class for server state with common attributes and utilities."""

    idx: int
    cfg: ServerConfig
    rng: random.Random
    load_offset: float

    def __post_init__(self):
        """Initialize common server attributes."""
        self.server_id = f"server-{self.idx}"
        self.instance = f"localhost:{8080 + self.idx}"
        self.job = "inference-server"

    def _noise(
        self, val: float, variance: float, max_val: float | None = None
    ) -> float:
        """Add random noise to a value.

        Args:
            val: Base value
            variance: Variance as fraction (e.g., 0.1 for ±10%)
            max_val: Optional maximum value to clamp to

        Returns:
            Noisy value, clamped to [0, max_val]
        """
        noisy = val * self.rng.uniform(1 - variance, 1 + variance)
        result = max(0.0, noisy)
        if max_val is not None:
            result = min(result, max_val)
        return result

    def _calculate_load(self, base_load: float) -> float:
        """Calculate effective load for this server including offset."""
        return max(0.0, min(1.0, base_load + self.load_offset))


class PrometheusMetricsFaker(ABC):
    """Abstract base class for Prometheus metrics generation.

    Subclasses should implement:
    - _create_server(): Create server instances with metric-specific fields
    - _update_server_metrics(): Update metric-specific fields
    - _generate_metrics(): Generate Prometheus format metrics
    """

    def __init__(
        self,
        server_type: str,
        num_servers: int = 2,
        seed: int | None = None,
        initial_load: float = 0.7,
    ):
        """Initialize faker with load level (0.0=idle, 1.0=max).

        Args:
            server_type: Type of server config (small, medium, large, vllm, dynamo)
            num_servers: Number of server instances to simulate
            seed: Random seed for reproducibility
            initial_load: Initial load level (0.0=idle, 1.0=max)
        """
        if server_type not in SERVER_CONFIGS:
            raise ValueError(f"Invalid server type: {server_type}")
        self.cfg = SERVER_CONFIGS[server_type]
        self.load = max(0.0, min(1.0, initial_load))
        self.rng = random.Random(seed)
        self.servers = [self._create_server(i) for i in range(num_servers)]

    @abstractmethod
    def _create_server(self, idx: int):
        """Create a server instance with metric-specific fields."""
        pass

    @abstractmethod
    def _update_server_metrics(self, server, base_load: float) -> None:
        """Update metric-specific fields for a server."""
        pass

    @abstractmethod
    def _generate_metrics(self) -> list[str]:
        """Generate Prometheus format metrics."""
        pass

    def set_load(self, load: float) -> None:
        """Set load level (0.0=idle, 1.0=max). Affects all metrics."""
        self.load = max(0.0, min(1.0, load))

    def generate(self) -> str:
        """Generate complete Prometheus metrics snapshot based on current load."""
        for server in self.servers:
            self._update_server_metrics(server, self.load)

        metrics = self._generate_metrics()
        return "\n".join(metrics) + "\n"

    # Common formatting methods used by all subclasses
    def _format_counter(self, name: str, help_text: str, attr: str) -> str:
        """Format Prometheus counter metric."""
        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} counter"]
        for server in self.servers:
            value = getattr(server, attr)
            lines.append(
                f'{name}{{job="{server.job}",instance="{server.instance}",server_id="{server.server_id}"}} {value}'
            )
        return "\n".join(lines)

    def _format_gauge(self, name: str, help_text: str, attr: str) -> str:
        """Format Prometheus gauge metric."""
        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} gauge"]
        for server in self.servers:
            value = getattr(server, attr)
            lines.append(
                f'{name}{{job="{server.job}",instance="{server.instance}",server_id="{server.server_id}"}} {value}'
            )
        return "\n".join(lines)

    def _format_histogram(
        self, name: str, help_text: str, values_attr: str, buckets: list[float]
    ) -> str:
        """Format Prometheus histogram metric."""
        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]

        for server in self.servers:
            values = getattr(server, values_attr)
            if not values:
                for le in buckets:
                    lines.append(
                        f'{name}_bucket{{job="{server.job}",instance="{server.instance}",server_id="{server.server_id}",le="{le}"}} 0'
                    )
                lines.append(
                    f'{name}_bucket{{job="{server.job}",instance="{server.instance}",server_id="{server.server_id}",le="+Inf"}} 0'
                )
                lines.append(
                    f'{name}_sum{{job="{server.job}",instance="{server.instance}",server_id="{server.server_id}"}} 0.0'
                )
                lines.append(
                    f'{name}_count{{job="{server.job}",instance="{server.instance}",server_id="{server.server_id}"}} 0'
                )
                continue

            sorted_values = sorted(values)
            total_sum = sum(values)
            total_count = len(values)

            cumulative_count = 0
            for le in buckets:
                cumulative_count = sum(1 for v in sorted_values if v <= le)
                lines.append(
                    f'{name}_bucket{{job="{server.job}",instance="{server.instance}",server_id="{server.server_id}",le="{le}"}} {cumulative_count}'
                )

            lines.append(
                f'{name}_bucket{{job="{server.job}",instance="{server.instance}",server_id="{server.server_id}",le="+Inf"}} {total_count}'
            )
            lines.append(
                f'{name}_sum{{job="{server.job}",instance="{server.instance}",server_id="{server.server_id}"}} {total_sum:.6f}'
            )
            lines.append(
                f'{name}_count{{job="{server.job}",instance="{server.instance}",server_id="{server.server_id}"}} {total_count}'
            )

        return "\n".join(lines)

    def _format_summary(
        self, name: str, help_text: str, values_attr: str, quantiles: list[float]
    ) -> str:
        """Format Prometheus summary metric."""
        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} summary"]

        for server in self.servers:
            values = getattr(server, values_attr)
            if not values:
                for q in quantiles:
                    lines.append(
                        f'{name}{{job="{server.job}",instance="{server.instance}",server_id="{server.server_id}",quantile="{q}"}} 0.0'
                    )
                lines.append(
                    f'{name}_sum{{job="{server.job}",instance="{server.instance}",server_id="{server.server_id}"}} 0.0'
                )
                lines.append(
                    f'{name}_count{{job="{server.job}",instance="{server.instance}",server_id="{server.server_id}"}} 0'
                )
                continue

            sorted_values = sorted(values)
            total_sum = sum(values)
            total_count = len(values)

            for q in quantiles:
                idx = int(q * (total_count - 1))
                quantile_value = sorted_values[idx]
                lines.append(
                    f'{name}{{job="{server.job}",instance="{server.instance}",server_id="{server.server_id}",quantile="{q}"}} {quantile_value:.6f}'
                )

            lines.append(
                f'{name}_sum{{job="{server.job}",instance="{server.instance}",server_id="{server.server_id}"}} {total_sum:.6f}'
            )
            lines.append(
                f'{name}_count{{job="{server.job}",instance="{server.instance}",server_id="{server.server_id}"}} {total_count}'
            )

        return "\n".join(lines)
