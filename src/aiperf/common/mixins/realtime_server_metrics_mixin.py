# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio

from aiperf.common.config import ServiceConfig
from aiperf.common.enums import MessageType
from aiperf.common.hooks import AIPerfHook, on_message, provides_hooks
from aiperf.common.messages import RealtimeServerMetricsMessage
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models import MetricResult
from aiperf.controller.system_controller import SystemController


@provides_hooks(AIPerfHook.ON_REALTIME_SERVER_METRICS)
class RealtimeServerMetricsMixin(MessageBusClientMixin):
    """A mixin that provides a hook for real-time server metrics."""

    def __init__(
        self, service_config: ServiceConfig, controller: SystemController, **kwargs
    ):
        super().__init__(service_config=service_config, controller=controller, **kwargs)
        self._controller = controller
        self._server_metrics: list[MetricResult] = []
        self._server_metrics_lock = asyncio.Lock()

    @on_message(MessageType.REALTIME_SERVER_METRICS)
    async def _on_realtime_server_metrics(self, message: RealtimeServerMetricsMessage):
        """Update the server metrics from a real-time server metrics message."""
        self.debug(
            f"Mixin received server metrics message with {len(message.metrics)} metrics, triggering hook"
        )

        async with self._server_metrics_lock:
            self._server_metrics = message.metrics
        await self.run_hooks(
            AIPerfHook.ON_REALTIME_SERVER_METRICS,
            metrics=message.metrics,
        )
