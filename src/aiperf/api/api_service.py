# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified AIPerf API Service.

Provides both HTTP endpoints and WebSocket streaming on a single port:

HTTP Endpoints:
- GET / - API documentation page
- GET /metrics - Prometheus format (for scrapers)
- GET /api/metrics - JSON metrics
- GET /api/status - Benchmark status
- GET /api/progress - Current progress
- GET /api/workers - Worker status
- GET /api/config - Benchmark configuration
- GET /health - Health check

WebSocket Endpoint:
- WS /ws - Real-time ZMQ message stream with dynamic subscriptions
"""

from __future__ import annotations

import pathlib
import time
import uuid

import aiohttp.web
import orjson

from aiperf.api.connection_manager import ConnectionManager
from aiperf.api.metrics_utils import build_info_labels, format_metrics_json
from aiperf.api.prometheus_formatter import InfoLabels, format_as_prometheus
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import CreditPhase, MessageType, MetricFlags, WorkerStatus
from aiperf.common.hooks import on_message, on_start, on_stop
from aiperf.common.messages import (
    CommandErrorResponse,
    GetAPIStatusCommand,
    GetAPIStatusResponse,
    Message,
    RealtimeMetricsMessage,
)
from aiperf.common.mixins import (
    ProgressTrackerMixin,
    RealtimeMetricsMixin,
    WorkerTrackerMixin,
)


class APIService(
    RealtimeMetricsMixin, ProgressTrackerMixin, WorkerTrackerMixin, BaseComponentService
):
    """Unified API Service providing HTTP and WebSocket endpoints.

    Combines HTTP endpoints for metrics scraping and JSON APIs with
    WebSocket streaming for real-time ZMQ message forwarding.

    Uses ProgressTrackerMixin and WorkerTrackerMixin to track benchmark
    progress and worker status via message subscriptions.
    HTTP endpoints for metrics/status use command pattern.
    WebSocket streaming uses dynamic ZMQ subscriptions based on client requests.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the API service.

        Args:
            service_config: Service configuration with API host/port.
            user_config: User configuration for benchmark settings.
            service_id: Optional service ID.
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            **kwargs,
        )

        self.api_host = service_config.api_host or "127.0.0.1"
        self.api_port = service_config.api_port or 9090

        self.connection_manager = ConnectionManager()
        self.app: aiohttp.web.Application | None = None
        self.runner: aiohttp.web.AppRunner | None = None
        self.site: aiohttp.web.TCPSite | None = None

        self.client_subscriptions: dict[str, set[str]] = {}
        self.zmq_subscriptions: set[str] = set()

        # Message types with @on_message handlers - no dynamic subscription needed
        self._handled_message_types: set[str] = {
            str(MessageType.REALTIME_METRICS),
            str(MessageType.REALTIME_TELEMETRY_METRICS),
            str(MessageType.CREDIT_PHASE_START),
            str(MessageType.CREDIT_PHASE_PROGRESS),
            str(MessageType.CREDIT_PHASE_COMPLETE),
            str(MessageType.WORKER_STATUS_SUMMARY),
            str(MessageType.PROCESSING_STATS),
            str(MessageType.ALL_RECORDS_RECEIVED),
        }

        self._info_labels: InfoLabels | None = None

    def _get_info_labels(self) -> InfoLabels:
        """Get cached info labels for metrics."""
        if self._info_labels is None:
            self._info_labels = build_info_labels(self.user_config)
        return self._info_labels

    @on_start
    async def _start_api_server(self) -> None:
        """Start the unified HTTP + WebSocket server."""
        self.info(f"Starting AIPerf API at http://{self.api_host}:{self.api_port}/")

        self.app = aiohttp.web.Application()

        self.app.router.add_get("/", self._handle_index)
        self.app.router.add_get("/dashboard", self._handle_dashboard)
        self.app.router.add_get("/metrics", self._handle_prometheus_metrics)
        self.app.router.add_get("/api/metrics", self._handle_json_metrics)
        self.app.router.add_get("/api/status", self._handle_status)
        self.app.router.add_get("/api/progress", self._handle_progress)
        self.app.router.add_get("/api/workers", self._handle_workers)
        self.app.router.add_get("/api/config", self._handle_config)
        self.app.router.add_get("/health", self._handle_health)

        self.app.router.add_get("/ws", self._handle_websocket)

        self.runner = aiohttp.web.AppRunner(self.app)
        await self.runner.setup()

        self.site = aiohttp.web.TCPSite(self.runner, self.api_host, self.api_port)
        await self.site.start()

        self.info(f"AIPerf API started at http://{self.api_host}:{self.api_port}/")
        self.info("  Dashboard: /dashboard (live metrics)")
        self.info("  HTTP: /metrics (Prometheus), /api/metrics (JSON)")
        self.info("  WebSocket: /ws (real-time stream)")

    @on_stop
    async def _stop_api_server(self) -> None:
        """Stop the API server."""
        self.info("Stopping AIPerf API server...")
        await self.connection_manager.close_all()

        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

        self.info("AIPerf API server stopped")

    async def _handle_index(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """Serve API documentation page."""
        return await self._serve_static("index.html")

    async def _handle_dashboard(
        self, request: aiohttp.web.Request
    ) -> aiohttp.web.Response:
        """Serve live dashboard page."""
        return await self._serve_static("dashboard.html")

    async def _serve_static(self, filename: str) -> aiohttp.web.Response:
        """Serve static file.

        Args:
            filename: Name of the static file to serve.

        Returns:
            HTTP response with file contents.
        """
        static_dir = pathlib.Path(__file__).parent / "static"
        try:
            with open(static_dir / filename) as f:
                return aiohttp.web.Response(text=f.read(), content_type="text/html")
        except FileNotFoundError:
            return aiohttp.web.Response(
                text=f"<h1>404 - {filename} not found</h1>", status=404
            )

    async def _handle_prometheus_metrics(
        self, request: aiohttp.web.Request
    ) -> aiohttp.web.Response:
        """Handle /metrics endpoint (Prometheus format) using local metrics state."""
        metrics = list(self._metrics)

        content = format_as_prometheus(
            metrics=metrics,
            info_labels=self._get_info_labels(),
        )
        return aiohttp.web.Response(
            body=content,
            content_type="text/plain",
            charset="utf-8",
        )

    async def _handle_json_metrics(
        self, request: aiohttp.web.Request
    ) -> aiohttp.web.Response:
        """Handle /api/metrics endpoint (JSON format) using local metrics state."""
        metrics = list(self._metrics)

        content = format_metrics_json(
            metrics=metrics,
            info_labels=self._get_info_labels(),
            benchmark_id=self.user_config.benchmark_id,
        )
        return aiohttp.web.Response(
            body=content,
            content_type="application/json",
            charset="utf-8",
        )

    async def _handle_status(
        self, request: aiohttp.web.Request
    ) -> aiohttp.web.Response:
        """Handle /api/status endpoint."""
        response = await self.send_command_and_wait_for_response(
            GetAPIStatusCommand(service_id=self.service_id),
            timeout=5.0,
        )

        if isinstance(response, GetAPIStatusResponse):
            content = orjson.dumps(
                {
                    "state": str(response.state),
                    "phase": response.phase,
                    "profile_id": response.profile_id,
                    "error": response.error,
                },
                option=orjson.OPT_INDENT_2,
            )
            return aiohttp.web.Response(
                body=content,
                content_type="application/json",
                charset="utf-8",
            )
        elif isinstance(response, CommandErrorResponse):
            return aiohttp.web.Response(
                text=f'{{"error": "{response.error.message}"}}',
                content_type="application/json",
                status=500,
            )
        else:
            return aiohttp.web.Response(
                text='{"error": "unexpected response"}',
                content_type="application/json",
                status=500,
            )

    async def _handle_progress(
        self, request: aiohttp.web.Request
    ) -> aiohttp.web.Response:
        """Handle /api/progress endpoint using local progress state."""
        total = 0
        completed = 0
        warmup = False
        start_ns = None
        elapsed_ns = None

        for phase, progress in self._phase_progress_map.items():
            if progress.requests.total_expected_requests:
                total += progress.requests.total_expected_requests
            completed += progress.requests.completed
            if phase == CreditPhase.WARMUP:
                warmup = True
            if progress.requests.start_ns and (
                start_ns is None or progress.requests.start_ns < start_ns
            ):
                start_ns = progress.requests.start_ns

        if start_ns:
            elapsed_ns = time.time_ns() - start_ns

        content = orjson.dumps(
            {
                "total": total,
                "completed": completed,
                "warmup": warmup,
                "start_ns": start_ns,
                "elapsed_ns": elapsed_ns,
                "percent_complete": (
                    round(completed / total * 100, 2) if total > 0 else 0
                ),
            },
            option=orjson.OPT_INDENT_2,
        )
        return aiohttp.web.Response(
            body=content,
            content_type="application/json",
            charset="utf-8",
        )

    async def _handle_workers(
        self, request: aiohttp.web.Request
    ) -> aiohttp.web.Response:
        """Handle /api/workers endpoint using local worker state."""
        total_workers = len(self._workers_stats)
        active_workers = sum(
            1
            for stats in self._workers_stats.values()
            if stats.status in (WorkerStatus.HEALTHY, WorkerStatus.HIGH_LOAD)
        )
        worker_statuses = {
            worker_id: str(stats.status) if stats.status else "unknown"
            for worker_id, stats in self._workers_stats.items()
        }

        content = orjson.dumps(
            {
                "total_workers": total_workers,
                "active_workers": active_workers,
                "worker_statuses": worker_statuses,
            },
            option=orjson.OPT_INDENT_2,
        )
        return aiohttp.web.Response(
            body=content,
            content_type="application/json",
            charset="utf-8",
        )

    async def _handle_config(
        self, request: aiohttp.web.Request
    ) -> aiohttp.web.Response:
        """Handle /api/config endpoint."""
        return aiohttp.web.Response(
            body=self.user_config.model_dump_json(exclude_unset=True),
            content_type="application/json",
            charset="utf-8",
        )

    async def _handle_health(
        self, request: aiohttp.web.Request
    ) -> aiohttp.web.Response:
        """Handle /health endpoint."""
        return aiohttp.web.Response(text="ok")

    async def _handle_websocket(
        self, request: aiohttp.web.Request
    ) -> aiohttp.web.WebSocketResponse:
        """Handle WebSocket connections for real-time streaming."""
        ws = aiohttp.web.WebSocketResponse()
        await ws.prepare(request)

        client_id = f"{request.remote}:{uuid.uuid4().hex[:8]}"
        self.info(f"WebSocket client connected: {client_id}")

        self.connection_manager.add_client(client_id, ws)
        self.client_subscriptions[client_id] = set()

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = msg.json()
                    await self._handle_client_message(client_id, ws, data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.warning(f"WebSocket error from {client_id}: {ws.exception()}")

        except Exception as e:
            self.error(f"WebSocket error for {client_id}: {e}")

        finally:
            self.connection_manager.remove_client(client_id)
            if client_id in self.client_subscriptions:
                del self.client_subscriptions[client_id]
            self.info(f"WebSocket client disconnected: {client_id}")

        return ws

    async def _handle_client_message(
        self,
        client_id: str,
        ws: aiohttp.web.WebSocketResponse,
        data: dict,
    ) -> None:
        """Handle client subscription requests.

        Args:
            client_id: The client identifier.
            ws: The WebSocket connection.
            data: The message data from the client.
        """
        msg_type = data.get("type")

        if msg_type == "subscribe":
            message_types = data.get("message_types", [])
            await self._subscribe_client(client_id, message_types)
            await ws.send_json(
                {"type": "subscribed", "message_types": list(message_types)}
            )

        elif msg_type == "unsubscribe":
            message_types = data.get("message_types", [])
            self.client_subscriptions[client_id] -= set(message_types)
            self.info(f"Client {client_id} unsubscribed from: {message_types}")

        elif msg_type == "ping":
            await ws.send_json({"type": "pong"})

    async def _subscribe_client(self, client_id: str, message_types: list[str]) -> None:
        """Subscribe a client to message types and ensure ZMQ subscriptions exist.

        For message types with existing @on_message handlers, we only track the
        client subscription (the handler already forwards to WebSocket clients).
        For other message types, we create a dynamic ZMQ subscription.

        Args:
            client_id: The client ID.
            message_types: List of message types to subscribe to.
        """
        for msg_type_str in message_types:
            self.client_subscriptions[client_id].add(msg_type_str)

            # Skip dynamic subscription for types that already have @on_message handlers
            # or are already subscribed, or are wildcards
            if (
                msg_type_str in self._handled_message_types
                or msg_type_str in self.zmq_subscriptions
                or msg_type_str == "*"
            ):
                continue

            try:
                msg_type_enum = MessageType(msg_type_str)

                # Use self.subscribe() from MessageBusClientMixin for dynamic subscriptions
                await self.subscribe(
                    message_type=msg_type_enum,
                    callback=self._forward_message,
                )

                self.zmq_subscriptions.add(msg_type_str)
                self.info(f"Subscribed to ZMQ message type: {msg_type_str}")

            except ValueError as e:
                self.warning(f"Invalid message type '{msg_type_str}': {e}")

        self.info(f"Client {client_id} subscribed to: {message_types}")

    async def _forward_message(self, message: Message) -> None:
        """Forward a ZMQ message to subscribed WebSocket clients.

        Args:
            message: The ZMQ message to forward.
        """
        message_type = str(message.message_type)
        message_dict = message.model_dump(exclude_none=True)

        sent_count = 0
        for client_id, subscriptions in self.client_subscriptions.items():
            if "*" in subscriptions or message_type in subscriptions:
                success = await self.connection_manager.send_to_client(
                    client_id, message_dict
                )
                if success:
                    sent_count += 1

        self.debug(lambda: f"Forwarded {message_type} to {sent_count} clients")

    # -------------------------------------------------------------------------
    # WebSocket forwarding handlers - forward ZMQ messages to WebSocket clients
    # -------------------------------------------------------------------------

    @on_message(MessageType.REALTIME_METRICS)
    async def _ws_forward_realtime_metrics(
        self, message: RealtimeMetricsMessage
    ) -> None:
        """Forward realtime metrics to WebSocket clients with display unit conversion.

        Filters out internal and experimental metrics before forwarding.
        """
        from aiperf.metrics.metric_registry import MetricRegistry

        # Filter out internal and experimental metrics, then convert to display units
        hidden_flags = MetricFlags.INTERNAL | MetricFlags.EXPERIMENTAL
        display_metrics = []
        for m in message.metrics:
            try:
                metric_cls = MetricRegistry.get_class(m.tag)
                if metric_cls.flags.has_any_flags(hidden_flags):
                    continue
            except Exception:
                pass  # If we can't find the metric class, include it anyway
            display_metrics.append(m.to_display_unit())

        # Build message dict with filtered and converted metrics
        message_dict = message.model_dump(exclude_none=True)
        message_dict["metrics"] = [
            m.model_dump(exclude_none=True) for m in display_metrics
        ]

        message_type = str(message.message_type)
        sent_count = 0
        for client_id, subscriptions in self.client_subscriptions.items():
            if "*" in subscriptions or message_type in subscriptions:
                success = await self.connection_manager.send_to_client(
                    client_id, message_dict
                )
                if success:
                    sent_count += 1

        self.debug(
            lambda: f"Forwarded {message_type} (display units) to {sent_count} clients"
        )

    @on_message(MessageType.CREDIT_PHASE_START)
    async def _ws_forward_credit_phase_start(self, message: Message) -> None:
        """Forward credit phase start to WebSocket clients."""
        await self._forward_message(message)

    @on_message(MessageType.CREDIT_PHASE_PROGRESS)
    async def _ws_forward_credit_phase_progress(self, message: Message) -> None:
        """Forward credit phase progress to WebSocket clients."""
        await self._forward_message(message)

    @on_message(MessageType.CREDIT_PHASE_COMPLETE)
    async def _ws_forward_credit_phase_complete(self, message: Message) -> None:
        """Forward credit phase complete to WebSocket clients."""
        await self._forward_message(message)

    @on_message(MessageType.WORKER_STATUS_SUMMARY)
    async def _ws_forward_worker_status(self, message: Message) -> None:
        """Forward worker status to WebSocket clients."""
        await self._forward_message(message)

    @on_message(MessageType.PROCESSING_STATS)
    async def _ws_forward_processing_stats(self, message: Message) -> None:
        """Forward records processing stats to WebSocket clients."""
        await self._forward_message(message)

    @on_message(MessageType.ALL_RECORDS_RECEIVED)
    async def _ws_forward_all_records_received(self, message: Message) -> None:
        """Forward all records received to WebSocket clients."""
        await self._forward_message(message)

    @on_message(MessageType.REALTIME_TELEMETRY_METRICS)
    async def _ws_forward_realtime_telemetry(self, message: Message) -> None:
        """Forward realtime GPU telemetry metrics to WebSocket clients."""
        await self._forward_message(message)


def main() -> None:
    """Main entry point."""
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.API)


if __name__ == "__main__":
    main()
