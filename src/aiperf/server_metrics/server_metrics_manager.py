# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ServiceType
from aiperf.common.environment import Environment
from aiperf.common.factories import ServiceFactory
from aiperf.common.messages import (
    ServerMetricRecordsMessage,
    ServerMetricsStatusMessage,
)
from aiperf.common.metrics.base_metrics_manager import BaseMetricsManager
from aiperf.common.models import ServerMetricRecord
from aiperf.common.protocols import ServiceProtocol
from aiperf.server_metrics.server_metrics_data_collector import (
    ServerMetricsDataCollector,
)

__all__ = ["ServerMetricsManager"]


@implements_protocol(ServiceProtocol)
@ServiceFactory.register(ServiceType.SERVER_METRICS_MANAGER)
class ServerMetricsManager(
    BaseMetricsManager[ServerMetricsDataCollector, ServerMetricRecord]
):
    """Coordinates multiple ServerMetricsDataCollector instances for server metrics collection.

    Extends BaseMetricsManager to provide server metrics-specific endpoint management,
    collector creation, and message handling.

    This service:
    - ALWAYS attempts to collect server metrics from multiple default endpoints:
      * Inference endpoint URL + /metrics (auto-derived from --url)
      * Environment default endpoints (localhost:8081, :7777, :2379)
    - Collection happens automatically even without --server-metrics flag (like GPU telemetry)
    - Console display only enabled when --server-metrics flag is provided
    - JSONL export always happens if data is collected
    - Manages lifecycle of ServerMetricsDataCollector instances
    - Collects metrics from multiple server endpoints (defaults + user-specified)
    - Sends ServerMetricRecordsMessage to RecordsManager via message system
    - Handles errors gracefully with ErrorDetails
    - Follows centralized architecture patterns

    Args:
        service_config: Service-level configuration (logging, communication, etc.)
        user_config: User-provided configuration including server_metrics list
        service_id: Optional unique identifier for this service instance
    """

    METRICS_TYPE_NAME = "Server Metrics"
    COLLECTOR_CLASS = ServerMetricsDataCollector
    RECORDS_MESSAGE_CLASS = ServerMetricRecordsMessage
    RECORDS_ENDPOINT_FIELD = "server_url"
    STATUS_MESSAGE_CLASS = ServerMetricsStatusMessage
    COLLECTION_INTERVAL = Environment.SERVER_METRICS.COLLECTION_INTERVAL
    SHUTDOWN_DELAY = Environment.SERVER_METRICS.SHUTDOWN_DELAY

    def _get_raw_default_endpoints(self, user_config: UserConfig) -> list[str]:
        """Get raw default server endpoints (inference endpoint + environment defaults)."""
        # ALWAYS derive server metrics endpoint from inference endpoint URL
        # This ensures server metrics are attempted even without --server-metrics flag
        inference_endpoint = user_config.endpoint.url

        # Get default endpoints from environment (localhost:8081, :7777, :2379)
        # These are ALWAYS attempted in addition to the inference endpoint
        env_defaults = list(Environment.SERVER_METRICS.DEFAULT_ENDPOINTS)

        return [inference_endpoint] + env_defaults

    def _get_user_endpoints_from_config(self, user_config: UserConfig) -> list[str]:
        """Get user-provided server metrics endpoints from config."""
        return user_config.server_metrics_urls
