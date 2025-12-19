# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live metrics HTTP server for real-time metrics exposure.

Provides a simple HTTP server that exposes AIPerf realtime metrics
in Prometheus Exposition Format (PEF) for scraping by PodMonitors.
"""

from __future__ import annotations

import errno
import inspect
import re
from collections.abc import Awaitable, Callable
from importlib.metadata import version

import orjson
from aiohttp import web

from aiperf.common.config import UserConfig, coerce_value
from aiperf.common.exceptions import MetricTypeError
from aiperf.common.hooks import on_start, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import MetricResult
from aiperf.metrics.metric_registry import MetricRegistry

# Type alias for info labels dict
InfoLabels = dict[str, str]

# Type alias for metrics callback (can be sync or async)
MetricsCallback = Callable[[], list[MetricResult] | Awaitable[list[MetricResult]]]

# Stats to expose with human-readable names for HELP text
STAT_DISPLAY_NAMES = {
    "avg": "average",
    "sum": "total",
    "p1": "1st percentile",
    "p5": "5th percentile",
    "p10": "10th percentile",
    "p25": "25th percentile",
    "p50": "50th percentile",
    "p75": "75th percentile",
    "p90": "90th percentile",
    "p95": "95th percentile",
    "p99": "99th percentile",
    "min": "minimum",
    "max": "maximum",
    "std": "standard deviation",
}

# Mapping of strings to replace in unit display names
_REPLACE_MAP = {
    "/": "_per_",
    "tokens_per_sec": "tps",
    "__": "_",
}

# Regex for sanitizing metric names to Prometheus format
_METRIC_NAME_RE = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize_metric_name(name: str) -> str:
    """Sanitize a metric name for Prometheus compatibility.

    Prometheus metric names must match [a-zA-Z_:][a-zA-Z0-9_:]*.
    We replace invalid characters with underscores.

    Args:
        name: The raw metric name/tag

    Returns:
        A sanitized metric name valid for Prometheus
    """
    sanitized = _METRIC_NAME_RE.sub("_", name.lower())
    # Ensure it starts with a letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


def _format_info_metric(info_labels: InfoLabels) -> str:
    """Format the aiperf_info metric with benchmark metadata.

    Args:
        info_labels: Dict of label name to value for the info metric

    Returns:
        Prometheus Exposition Format text for the info metric
    """
    if not info_labels:
        return ""

    # Add version to info metric only
    labels = {"version": version("aiperf"), **info_labels}

    # Format labels as key="value" pairs, escaping quotes in values
    label_pairs = []
    for key, value in labels.items():
        escaped_value = str(value).replace("\\", "\\\\").replace('"', '\\"')
        label_pairs.append(f'{key}="{escaped_value}"')

    labels_str = ",".join(label_pairs)

    lines = [
        "# HELP aiperf_info AIPerf benchmark information",
        "# TYPE aiperf_info gauge",
        f"aiperf_info{{{labels_str}}} 1",
    ]
    return "\n".join(lines) + "\n"


def _format_labels(labels: InfoLabels) -> str:
    """Format labels dict as Prometheus label string.

    Args:
        labels: Dict of label name to value

    Returns:
        Formatted label string like {key1="value1",key2="value2"}
    """
    if not labels:
        return ""
    label_pairs = []
    for key, value in labels.items():
        escaped_value = str(value).replace("\\", "\\\\").replace('"', '\\"')
        label_pairs.append(f'{key}="{escaped_value}"')
    return "{" + ",".join(label_pairs) + "}"


def format_as_prometheus(
    metrics: list[MetricResult], info_labels: InfoLabels | None = None
) -> str:
    """Convert MetricResult list to Prometheus Exposition Format text.

    Generates raw PEF text with gauges for each metric stat.
    Creates fresh output on each call (no caching/registry reuse).
    Converts metrics to display units for human-readable values.

    Args:
        metrics: List of MetricResult objects from realtime metrics
        info_labels: Optional dict of labels for the aiperf_info metric.
            Key labels (excluding 'config') are also added to all metrics.

    Returns:
        Prometheus Exposition Format text string
    """
    lines: list[str] = []

    # Add info metric first if labels provided
    if info_labels:
        lines.append(_format_info_metric(info_labels))

    # Extract metric labels (exclude 'config' and 'version' which are info-only)
    metric_labels = {
        k: v for k, v in (info_labels or {}).items() if k not in ("config", "version")
    }
    labels_str = _format_labels(metric_labels)

    for raw_metric in metrics:
        # Convert to display units (e.g., ns -> ms, bytes -> MB)
        try:
            metric = raw_metric.to_display_unit()
            metric_cls = MetricRegistry.get_class(metric.tag)
            display_unit = metric_cls.display_unit or metric_cls.unit
            unit_suffix = str(display_unit)
            for old, new in _REPLACE_MAP.items():
                unit_suffix = unit_suffix.replace(old, new)
            unit_display = display_unit.display_name()
        except MetricTypeError:
            metric = raw_metric
            unit_suffix = ""
            unit_display = metric.unit

        base_name = f"aiperf_{_sanitize_metric_name(metric.tag)}"

        for stat, stat_display in STAT_DISPLAY_NAMES.items():
            value = getattr(metric, stat, None)
            if value is None:
                continue

            if stat == "sum":
                metric_name = f"{base_name}_{unit_suffix}_total"
            else:
                metric_name = f"{base_name}_{stat}_{unit_suffix}"

            lines.append(
                f"# HELP {metric_name} {metric.header} {stat_display} (in {unit_display})"
            )
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name}{labels_str} {value}")

    return "\n".join(lines) + "\n" if lines else ""


class LiveMetricsServer(AIPerfLifecycleMixin):
    """Live metrics HTTP server component.

    Exposes AIPerf realtime metrics via HTTP in Prometheus Exposition Format.
    Integrates with the AIPerf lifecycle for proper startup/shutdown handling.
    """

    def __init__(
        self,
        user_config: UserConfig,
        metrics_callback: MetricsCallback,
        **kwargs,
    ) -> None:
        """Initialize the live metrics server.

        Args:
            user_config: UserConfig for building aiperf_info labels, host, and port
            metrics_callback: Callable that returns current MetricResult list
        """
        super().__init__(**kwargs)
        self._user_config = user_config
        self._host = user_config.live_metrics_host
        self._port = user_config.live_metrics_port
        self._metrics_callback = metrics_callback
        self._info_labels = self._build_info_labels(user_config)
        self._runner: web.AppRunner | None = None

    @staticmethod
    def _build_info_labels(user_config: UserConfig) -> InfoLabels:
        """Build info labels for the aiperf_info metric from UserConfig.

        Args:
            user_config: The user configuration for the benchmark

        Returns:
            Dictionary of label names to values for the info metric
        """
        labels: InfoLabels = {}

        # Key labels for easy querying
        if user_config.benchmark_id:
            labels["benchmark_id"] = user_config.benchmark_id
        labels["model"] = ",".join(user_config.endpoint.model_names)
        labels["endpoint_type"] = user_config.endpoint.type
        labels["streaming"] = str(user_config.endpoint.streaming).lower()
        if user_config.loadgen.concurrency is not None:
            labels["concurrency"] = str(user_config.loadgen.concurrency)
        if user_config.loadgen.request_rate is not None:
            labels["request_rate"] = str(user_config.loadgen.request_rate)

        # Full config as JSON for complete information
        labels["config"] = user_config.model_dump_json(exclude_unset=True)

        return labels

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """Handle GET /metrics endpoint."""
        result = self._metrics_callback()

        # Handle both sync and async callbacks
        if inspect.isawaitable(result):
            metrics = await result
        else:
            metrics = result

        content = format_as_prometheus(metrics, self._info_labels)
        return web.Response(
            body=content,
            content_type="text/plain",
            charset="utf-8",
        )

    async def _handle_config(self, request: web.Request) -> web.Response:
        """Handle GET /config endpoint."""
        return web.Response(
            body=self._user_config.model_dump_json(exclude_unset=True),
            content_type="application/json",
            charset="utf-8",
        )

    async def _handle_metrics_json(self, request: web.Request) -> web.Response:
        """Handle GET /metrics.json endpoint."""
        callback_result = self._metrics_callback()
        if inspect.isawaitable(callback_result):
            metrics = await callback_result
        else:
            metrics = callback_result
        result = {
            "aiperf_version": version("aiperf"),
            "benchmark_id": self._user_config.benchmark_id,
        }
        if self._info_labels:
            result.update(
                {
                    key: coerce_value(value)
                    for key, value in self._info_labels.items()
                    if key not in ("config", "version")
                }
            )
        metrics_dict = {}
        for metric in metrics:
            try:
                display_metric = metric.to_display_unit()
            except MetricTypeError:
                display_metric = metric
            metrics_dict[metric.tag] = display_metric.model_dump(
                mode="json", exclude_none=True, exclude={"tag"}
            )
        result["metrics"] = metrics_dict
        content = orjson.dumps(result, option=orjson.OPT_INDENT_2)
        return web.Response(
            body=content,
            content_type="application/json",
            charset="utf-8",
        )

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle GET /health endpoint for liveness probes."""
        return web.Response(text="ok")

    @on_start
    async def _start_server(self) -> None:
        """Start the HTTP server."""
        app = web.Application()
        app.router.add_get("/metrics", self._handle_metrics)
        app.router.add_get("/metrics.json", self._handle_metrics_json)
        app.router.add_get("/config", self._handle_config)
        app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)

        try:
            await site.start()
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                raise OSError(
                    f"Port {self._port} is already in use. "
                    f"Use --live-metrics-port to specify a different port."
                ) from None
            raise

        self.info(f"Live metrics available at http://{self._host}:{self._port}/metrics")

    @on_stop
    async def _stop_server(self) -> None:
        """Stop the HTTP server."""
        if self._runner:
            runner = self._runner
            self._runner = None
            await runner.cleanup()
            self.info("Live metrics server stopped")
