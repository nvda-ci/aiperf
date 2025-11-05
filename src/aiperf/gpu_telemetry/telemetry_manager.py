# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ServiceType
from aiperf.common.environment import Environment
from aiperf.common.factories import ServiceFactory
from aiperf.common.messages import (
    TelemetryRecordsMessage,
    TelemetryStatusMessage,
)
from aiperf.common.metrics.base_metrics_manager import BaseMetricsManager
from aiperf.common.models import TelemetryRecord
from aiperf.common.protocols import ServiceProtocol
from aiperf.gpu_telemetry.telemetry_data_collector import TelemetryDataCollector

__all__ = ["TelemetryManager"]


@implements_protocol(ServiceProtocol)
@ServiceFactory.register(ServiceType.TELEMETRY_MANAGER)
class TelemetryManager(BaseMetricsManager[TelemetryDataCollector, TelemetryRecord]):
    """Coordinates multiple TelemetryDataCollector instances for GPU telemetry collection.

    Extends BaseMetricsManager to provide DCGM-specific endpoint management,
    collector creation, and message handling.

    This service:
    - Manages lifecycle of TelemetryDataCollector instances
    - Collects telemetry from multiple DCGM endpoints
    - Sends TelemetryRecordsMessage to RecordsManager via message system
    - Handles errors gracefully with ErrorDetails
    - Follows centralized architecture patterns

    Args:
        service_config: Service-level configuration (logging, communication, etc.)
        user_config: User-provided configuration including gpu_telemetry list
        service_id: Optional unique identifier for this service instance
    """

    METRICS_TYPE_NAME = "GPU Telemetry"
    COLLECTOR_CLASS = TelemetryDataCollector
    RECORDS_MESSAGE_CLASS = TelemetryRecordsMessage
    RECORDS_ENDPOINT_FIELD = "dcgm_url"
    STATUS_MESSAGE_CLASS = TelemetryStatusMessage
    COLLECTION_INTERVAL = Environment.GPU.COLLECTION_INTERVAL
    SHUTDOWN_DELAY = Environment.GPU.SHUTDOWN_DELAY

    def _get_raw_default_endpoints(self, user_config: UserConfig) -> list[str]:
        """Get raw default DCGM endpoints from environment."""
        return list(Environment.GPU.DEFAULT_DCGM_ENDPOINTS)

    def _get_user_endpoints_from_config(self, user_config: UserConfig) -> list[str]:
        """Get user-provided GPU telemetry endpoints from config."""
        return user_config.gpu_telemetry_urls
