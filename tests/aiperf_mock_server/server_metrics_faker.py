# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Server metrics faker that generates realistic load-driven AI server metrics."""

import random
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """AI server hardware configuration."""

    name: str
    cpu_cores: int
    memory_gb: int
    max_batch_size: int
    max_tokens: int
    kv_blocks: int
    workers: int


SERVER_CONFIGS = {
    "small": ServerConfig("Small AI Server", 16, 128, 32, 4096, 1000, 2),
    "medium": ServerConfig("Medium AI Server", 32, 256, 64, 8192, 2000, 4),
    "large": ServerConfig("Large AI Server", 64, 512, 128, 16384, 4000, 8),
    "xlarge": ServerConfig("XLarge AI Server", 128, 1024, 256, 32768, 8000, 16),
}  # fmt: skip
"""Server configuration for realistic AI metrics generation."""

# Metric mappings: (prometheus_name, help_text, internal_attr)
METRIC_MAPPINGS = [
    # Generic HTTP/Request metrics
    ("http_requests_total", "Total HTTP requests", "requests_total"),
    ("http_request_duration_seconds", "HTTP request duration in seconds", "request_duration_seconds"),
    ("http_requests_in_flight", "Number of HTTP requests in flight", "requests_in_flight"),
    ("http_response_size_bytes", "HTTP response size in bytes", "response_size_bytes"),
    ("http_responses_2xx", "Total 2xx HTTP responses", "http_2xx_total"),
    ("http_responses_4xx", "Total 4xx HTTP responses", "http_4xx_total"),
    ("http_responses_5xx", "Total 5xx HTTP responses", "http_5xx_total"),
    # Generic CPU metrics
    ("process_cpu_usage_percent", "Process CPU usage percentage", "cpu_usage_percent"),
    ("process_cpu_seconds_total", "Total process CPU seconds", "process_cpu_seconds"),
    ("cpu_system_seconds_total", "Total CPU system seconds", "cpu_system_seconds"),
    ("cpu_user_seconds_total", "Total CPU user seconds", "cpu_user_seconds"),
    # Generic Memory metrics
    ("process_resident_memory_bytes", "Process resident memory in bytes", "process_resident_memory_bytes"),
    ("process_virtual_memory_bytes", "Process virtual memory in bytes", "process_virtual_memory_bytes"),
    ("memory_usage_bytes", "Memory usage in bytes", "memory_usage_bytes"),
    ("memory_total_bytes", "Total memory in bytes", "memory_total_bytes"),
    # Generic Process metrics
    ("process_open_fds", "Number of open file descriptors", "process_open_fds"),
    # Generic Network metrics
    ("network_receive_bytes_total", "Total bytes received", "network_receive_bytes"),
    ("network_transmit_bytes_total", "Total bytes transmitted", "network_transmit_bytes"),
    # Dynamo Backend Component Metrics
    ("dynamo_component_inflight_requests", "Component inflight requests", "component_inflight_requests"),
    ("dynamo_component_request_bytes_total", "Component request bytes", "component_request_bytes_total"),
    ("dynamo_component_request_duration_seconds", "Component request duration", "component_request_duration_seconds"),
    ("dynamo_component_requests_total", "Total component requests", "component_requests_total"),
    ("dynamo_component_response_bytes_total", "Component response bytes", "component_response_bytes_total"),
    ("dynamo_component_system_uptime_seconds", "Component system uptime", "component_system_uptime_seconds"),
    # Dynamo KV Router Statistics
    ("dynamo_component_kvstats_active_blocks", "KV cache active blocks", "kvstats_active_blocks"),
    ("dynamo_component_kvstats_total_blocks", "KV cache total blocks", "kvstats_total_blocks"),
    ("dynamo_component_kvstats_gpu_cache_usage_percent", "KV GPU cache usage %", "kvstats_gpu_cache_usage_percent"),
    ("dynamo_component_kvstats_gpu_prefix_cache_hit_rate", "KV prefix cache hit rate", "kvstats_gpu_prefix_cache_hit_rate"),
    # Dynamo Frontend Metrics
    ("dynamo_frontend_inflight_requests", "Frontend inflight requests", "frontend_inflight_requests"),
    ("dynamo_frontend_queued_requests", "Frontend queued requests", "frontend_queued_requests"),
    ("dynamo_frontend_input_sequence_tokens_total", "Frontend input tokens", "frontend_input_sequence_tokens"),
    ("dynamo_frontend_inter_token_latency_seconds", "Frontend inter-token latency", "frontend_inter_token_latency_seconds"),
    ("dynamo_frontend_output_sequence_tokens_total", "Frontend output tokens", "frontend_output_sequence_tokens"),
    ("dynamo_frontend_request_duration_seconds", "Frontend request duration", "frontend_request_duration_seconds"),
    ("dynamo_frontend_requests_total", "Total frontend requests", "frontend_requests_total"),
    ("dynamo_frontend_time_to_first_token_seconds", "Frontend TTFT", "frontend_time_to_first_token_seconds"),
    # Dynamo Model Configuration Metrics
    ("dynamo_frontend_model_total_kv_blocks", "Model total KV blocks", "model_total_kv_blocks"),
    ("dynamo_frontend_model_max_num_seqs", "Model max sequences", "model_max_num_seqs"),
    ("dynamo_frontend_model_max_num_batched_tokens", "Model max batched tokens", "model_max_num_batched_tokens"),
    ("dynamo_frontend_model_context_length", "Model context length", "model_context_length"),
    ("dynamo_frontend_model_kv_cache_block_size", "Model KV cache block size", "model_kv_cache_block_size"),
    ("dynamo_frontend_model_migration_limit", "Model migration limit", "model_migration_limit"),
    ("dynamo_frontend_model_workers", "Model workers", "model_workers"),
]  # fmt: skip
"""Metric mappings for AI server metrics to internal FakeServer attributes."""


@dataclass
class FakeServer:
    """Single AI server state and metrics."""

    idx: int
    cfg: ServerConfig
    rng: random.Random
    load_offset: float
    instance_name: str

    # Cumulative metrics (counters)
    requests_total: int = 0
    http_2xx_total: int = 0
    http_4xx_total: int = 0
    http_5xx_total: int = 0
    process_cpu_seconds: float = 0.0
    cpu_system_seconds: float = 0.0
    cpu_user_seconds: float = 0.0
    network_receive_bytes: float = 0.0
    network_transmit_bytes: float = 0.0
    component_request_bytes_total: float = 0.0
    component_requests_total: int = 0
    component_response_bytes_total: float = 0.0
    frontend_input_sequence_tokens: int = 0
    frontend_output_sequence_tokens: int = 0
    frontend_requests_total: int = 0

    # Current/gauge metrics
    requests_in_flight: int = 0
    request_duration_seconds: float = 0.0
    response_size_bytes: float = 0.0
    cpu_usage_percent: float = 0.0
    process_resident_memory_bytes: float = 0.0
    process_virtual_memory_bytes: float = 0.0
    memory_usage_bytes: float = 0.0
    process_open_fds: int = 0
    component_inflight_requests: int = 0
    component_request_duration_seconds: float = 0.0
    component_system_uptime_seconds: float = 0.0
    kvstats_active_blocks: int = 0
    kvstats_total_blocks: int = 0
    kvstats_gpu_cache_usage_percent: float = 0.0
    kvstats_gpu_prefix_cache_hit_rate: float = 0.0
    frontend_inflight_requests: int = 0
    frontend_queued_requests: int = 0
    frontend_inter_token_latency_seconds: float = 0.0
    frontend_request_duration_seconds: float = 0.0
    frontend_time_to_first_token_seconds: float = 0.0

    # Model config (static)
    model_total_kv_blocks: int = 0
    model_max_num_seqs: int = 0
    model_max_num_batched_tokens: int = 0
    model_context_length: int = 0
    model_kv_cache_block_size: int = 0
    model_migration_limit: int = 0
    model_workers: int = 0

    def __post_init__(self):
        """Initialize computed values."""
        self.memory_total_bytes = float(self.cfg.memory_gb * 1024**3)
        self.component_system_uptime_seconds = self.rng.uniform(
            3600, 86400 * 7
        )  # 1hr to 7 days

        # Initialize model config (static values)
        self.model_total_kv_blocks = self.cfg.kv_blocks
        self.model_max_num_seqs = self.cfg.max_batch_size
        self.model_max_num_batched_tokens = self.cfg.max_tokens
        self.model_context_length = self.cfg.max_tokens
        self.model_kv_cache_block_size = 16
        self.model_migration_limit = 100
        self.model_workers = self.cfg.workers

        self.kvstats_total_blocks = self.cfg.kv_blocks

    def _noise(
        self, val: float, variance: float, max_val: float | None = None
    ) -> float:
        """Add noise to a value and optionally clamp to [0, max]."""
        noisy = val * self.rng.uniform(1 - variance, 1 + variance)
        if max_val is not None:
            return max(0.0, min(noisy, max_val))
        return max(0.0, noisy)

    def update(self, base_load: float) -> None:
        """Update all metrics based on current load (0.0=idle, 1.0=max)."""
        load = max(0.0, min(1.0, base_load + self.load_offset))
        c = self.cfg

        # Update inflight/queue metrics
        self.requests_in_flight = int(self._noise(load * c.max_batch_size * 0.5, 0.1))
        self.component_inflight_requests = int(
            self._noise(load * c.max_batch_size * 0.3, 0.1)
        )
        self.frontend_inflight_requests = int(
            self._noise(load * c.max_batch_size * 0.4, 0.1)
        )
        self.frontend_queued_requests = int(
            self._noise(load * c.max_batch_size * 0.2, 0.15)
        )

        # Update duration/latency metrics
        self.request_duration_seconds = self._noise(
            0.05 + load * 0.5, 0.1
        )  # 50ms-550ms
        self.component_request_duration_seconds = self._noise(0.02 + load * 0.2, 0.1)
        self.frontend_request_duration_seconds = self._noise(0.03 + load * 0.3, 0.1)
        self.frontend_time_to_first_token_seconds = self._noise(0.02 + load * 0.1, 0.1)
        self.frontend_inter_token_latency_seconds = self._noise(
            0.005 + load * 0.015, 0.1
        )

        # Update CPU metrics (as decimal 0.0-1.0)
        cpu_usage_decimal = 0.1 + load * 0.8  # 10%-90%
        self.cpu_usage_percent = (
            cpu_usage_decimal  # Store as decimal, will be scaled by collector
        )

        # Update memory metrics
        mem_usage_pct = 0.30 + load * 0.50  # 30%-80%
        self.memory_usage_bytes = self._noise(
            self.memory_total_bytes * mem_usage_pct, 0.02, self.memory_total_bytes
        )
        self.process_resident_memory_bytes = self._noise(
            self.memory_usage_bytes * 0.8, 0.05, self.memory_usage_bytes
        )
        self.process_virtual_memory_bytes = self._noise(
            self.memory_usage_bytes * 1.2, 0.05, self.memory_total_bytes * 1.5
        )

        # Update KV cache metrics
        self.kvstats_active_blocks = int(
            self._noise(load * c.kv_blocks * 0.7, 0.1, c.kv_blocks)
        )
        kv_usage_decimal = (
            self.kvstats_active_blocks / c.kv_blocks if c.kv_blocks > 0 else 0.0
        )
        self.kvstats_gpu_cache_usage_percent = kv_usage_decimal  # Store as decimal
        self.kvstats_gpu_prefix_cache_hit_rate = self._noise(
            0.3 + load * 0.5, 0.1, 1.0
        )  # Store as decimal

        # Update process metrics
        self.process_open_fds = int(self._noise(100 + load * 400, 0.1))
        self.response_size_bytes = self._noise(1024 + load * 4096, 0.2)

        # Update cumulative metrics (1 tick = 1 second worth of activity)
        requests_per_tick = int(
            self._noise(load * 100, 0.3)
        )  # Up to 100 req/s at max load
        self.requests_total += requests_per_tick
        self.frontend_requests_total += requests_per_tick
        self.component_requests_total += int(
            requests_per_tick * 1.5
        )  # Components get more calls

        # Distribute HTTP status codes (mostly 2xx)
        self.http_2xx_total += int(requests_per_tick * 0.95)
        self.http_4xx_total += int(requests_per_tick * 0.04)
        self.http_5xx_total += int(requests_per_tick * 0.01)

        # Update CPU time (1 second distributed across modes)
        self.process_cpu_seconds += cpu_usage_decimal * c.cpu_cores
        self.cpu_user_seconds += cpu_usage_decimal * c.cpu_cores * 0.7
        self.cpu_system_seconds += cpu_usage_decimal * c.cpu_cores * 0.3

        # Update network metrics
        bytes_per_request = 2048  # Average request size
        self.network_receive_bytes += requests_per_tick * bytes_per_request
        self.network_transmit_bytes += (
            requests_per_tick * bytes_per_request * 2
        )  # Response is larger

        # Update component metrics
        self.component_request_bytes_total += int(
            requests_per_tick * 1.5 * bytes_per_request * 0.8
        )
        self.component_response_bytes_total += int(
            requests_per_tick * 1.5 * bytes_per_request * 1.2
        )

        # Update token metrics
        tokens_per_request_in = int(self._noise(100 + load * 200, 0.2))
        tokens_per_request_out = int(self._noise(50 + load * 150, 0.2))
        self.frontend_input_sequence_tokens += requests_per_tick * tokens_per_request_in
        self.frontend_output_sequence_tokens += (
            requests_per_tick * tokens_per_request_out
        )

        # Increment uptime
        self.component_system_uptime_seconds += 1.0


class ServerMetricsFaker:
    """Simulated AI server metrics generator (Prometheus format)."""

    def __init__(
        self,
        config_name: str = "medium",
        num_servers: int = 1,
        seed: int | None = None,
        instance_prefix: str = "server",
        initial_load: float = 0.5,
    ):
        """Initialize faker with load level (0.0=idle, 1.0=max)."""
        if config_name not in SERVER_CONFIGS:
            raise ValueError(f"Invalid config name: {config_name}")
        self.cfg = SERVER_CONFIGS[config_name]
        self.instance_prefix = instance_prefix
        self.load = max(0.0, min(1.0, initial_load))
        self.rng = random.Random(seed)
        self.servers = [
            FakeServer(
                i,
                self.cfg,
                self.rng,
                self.rng.uniform(-0.05, 0.05),
                f"{instance_prefix}-{i}",
            )
            for i in range(num_servers)
        ]

    def set_load(self, load: float) -> None:
        """Set load level (0.0=idle, 1.0=max). Affects all metrics."""
        self.load = max(0.0, min(1.0, load))

    def _format_metric(self, name: str, help_text: str, attr: str) -> str:
        """Format Prometheus metric block."""
        # Determine metric type
        metric_type = (
            "counter"
            if "_total" in name
            or attr
            in [
                "requests_total",
                "http_2xx_total",
                "http_4xx_total",
                "http_5xx_total",
                "process_cpu_seconds",
                "cpu_system_seconds",
                "cpu_user_seconds",
                "network_receive_bytes",
                "network_transmit_bytes",
                "component_request_bytes_total",
                "component_requests_total",
                "component_response_bytes_total",
                "frontend_input_sequence_tokens",
                "frontend_output_sequence_tokens",
                "frontend_requests_total",
            ]
            else "gauge"
        )

        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} {metric_type}"]
        for server in self.servers:
            value = getattr(server, attr)
            lines.append(
                f'{name}{{instance="{server.instance_name}:8080",job="ai-server"}} {float(value):.6f}'
            )
        return "\n".join(lines)

    def generate(self) -> str:
        """Generate complete AI server metrics snapshot based on current load."""
        for server in self.servers:
            server.update(self.load)

        metrics = []
        for name, help_text, attr in METRIC_MAPPINGS:
            metrics.append(self._format_metric(name, help_text, attr))

        return "\n".join(metrics) + "\n"
