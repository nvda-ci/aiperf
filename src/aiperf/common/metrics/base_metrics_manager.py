# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class for metrics manager services that orchestrate data collectors.

This module provides a reusable abstract base class for managing multiple metrics
data collectors and coordinating their lifecycle.
"""

import asyncio
from abc import abstractmethod
from typing import ClassVar, Generic, TypeVar

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import (
    CommAddress,
    CommandType,
)
from aiperf.common.hooks import on_command, on_init, on_stop
from aiperf.common.messages import (
    ProfileCancelCommand,
    ProfileConfigureCommand,
)
from aiperf.common.models import ErrorDetails
from aiperf.common.protocols import (
    PushClientProtocol,
    ServiceProtocol,
)

__all__ = ["BaseMetricsManager"]

# Generic types
CollectorT = TypeVar(
    "CollectorT"
)  # Type of data collector (e.g., TelemetryDataCollector)
RecordT = TypeVar("RecordT")  # Type of record (e.g., TelemetryRecord)


@implements_protocol(ServiceProtocol)
class BaseMetricsManager(BaseComponentService, Generic[CollectorT, RecordT]):
    """Abstract base class for metrics manager services.

    This class provides common functionality for:
    - Managing multiple data collector instances
    - Lifecycle coordination (configure, start, stop)
    - Endpoint reachability testing
    - Status message broadcasting
    - Error handling
    - Push client communication with RecordsManager

    Subclasses must implement:
    - _get_raw_default_endpoints(): Get raw default endpoint URLs
    - _get_user_endpoints_from_config(): Get user-provided endpoint URLs
    - _create_collector(): Create a data collector instance

    ClassVars to override:
        METRICS_TYPE_NAME: Human-readable name for logging (e.g., "GPU Telemetry")
        COLLECTOR_CLASS: Collector class to instantiate
        RECORDS_MESSAGE_CLASS: Records message class to instantiate
        RECORDS_ENDPOINT_FIELD: Field name for endpoint in records message (e.g., "dcgm_url")
        STATUS_MESSAGE_CLASS: Status message class to instantiate
        COLLECTION_INTERVAL: Collection interval in seconds
        SHUTDOWN_DELAY: Delay before shutdown to allow command response transmission

    Args:
        service_config: Service-level configuration (logging, communication, etc.)
        user_config: User-provided configuration
        service_id: Optional unique identifier for this service instance
    """

    # Subclasses must override these ClassVars
    METRICS_TYPE_NAME: ClassVar[str]
    COLLECTOR_CLASS: ClassVar[type[CollectorT]]
    RECORDS_MESSAGE_CLASS: ClassVar[type]
    RECORDS_ENDPOINT_FIELD: ClassVar[str]
    STATUS_MESSAGE_CLASS: ClassVar[type]
    COLLECTION_INTERVAL: ClassVar[float]
    SHUTDOWN_DELAY: ClassVar[float]

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
        )

        self.records_push_client: PushClientProtocol = self.comms.create_push_client(
            CommAddress.RECORDS,
        )

        self._collectors: dict[str, CollectorT] = {}
        self._collector_id_to_url: dict[str, str] = {}

        # Setup endpoint tracking with simplified logic
        raw_defaults = self._get_raw_default_endpoints(user_config)
        self._default_endpoints = [
            self._normalize_url_with_suffix(url)
            for url in self._ensure_list(raw_defaults)
        ]

        raw_user_endpoints = self._get_user_endpoints_from_config(user_config)
        normalized_user_endpoints = [
            self._normalize_url_with_suffix(url)
            for url in self._ensure_list(raw_user_endpoints)
        ]
        self._user_provided_endpoints = [
            ep for ep in normalized_user_endpoints if ep not in self._default_endpoints
        ]

        # Combine and deduplicate all endpoints
        self._configured_endpoints = list(
            dict.fromkeys(self._default_endpoints + self._user_provided_endpoints)
        )

        self._collection_interval = self.COLLECTION_INTERVAL

    @on_init
    async def _initialize(self) -> None:
        """Initialize metrics manager.

        Called automatically during service startup via @on_init hook.
        Actual collector initialization happens in _profile_configure_command
        after configuration is received from SystemController.
        """
        pass

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the metrics collectors but don't start them yet.

        Creates collector instances for each configured endpoint,
        tests reachability, and sends status message.
        If no endpoints are reachable, disables metrics and prepares for shutdown.

        Args:
            message: Profile configuration command from SystemController
        """
        self._collectors.clear()
        self._collector_id_to_url.clear()

        for endpoint_url in self._configured_endpoints:
            self.debug(
                f"{self.METRICS_TYPE_NAME}: Testing reachability of {endpoint_url}"
            )
            collector_id = (
                f"collector_{endpoint_url.replace(':', '_').replace('/', '_')}"
            )
            self._collector_id_to_url[collector_id] = endpoint_url

            collector = self._create_collector(
                endpoint_url=endpoint_url,
                collection_interval=self._collection_interval,
                collector_id=collector_id,
            )

            try:
                is_reachable = await collector.is_url_reachable()
                if is_reachable:
                    self._collectors[endpoint_url] = collector
                    self.debug(
                        f"{self.METRICS_TYPE_NAME}: Endpoint {endpoint_url} is reachable"
                    )
                else:
                    self.debug(
                        f"{self.METRICS_TYPE_NAME}: Endpoint {endpoint_url} is not reachable"
                    )
            except Exception as e:
                self.error(
                    f"{self.METRICS_TYPE_NAME}: Exception testing {endpoint_url}: {e}"
                )

        reachable_endpoints = list(self._collectors.keys())

        if not self._collectors:
            await self._send_status_message(
                enabled=False,
                reason="no endpoints reachable",
                endpoints_configured=self._configured_endpoints,
                endpoints_reachable=[],
            )
            return

        await self._send_status_message(
            enabled=True,
            reason=None,
            endpoints_configured=self._configured_endpoints,
            endpoints_reachable=reachable_endpoints,
        )

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, message) -> None:
        """Start all metrics collectors.

        Initializes and starts each configured collector.
        If no collectors start successfully, sends disabled status.

        Args:
            message: Profile start command from SystemController
        """
        if not self._collectors:
            # Metrics disabled status already sent in _profile_configure_command
            self._shutdown_task = asyncio.create_task(self._delayed_shutdown())
            return

        started_count = 0
        for endpoint_url, collector in self._collectors.items():
            try:
                await collector.initialize()
                await collector.start()
                started_count += 1
            except Exception as e:
                self.error(f"Failed to start collector for {endpoint_url}: {e}")

        if started_count == 0:
            self.warning(f"No {self.METRICS_TYPE_NAME} collectors successfully started")
            await self._send_status_message(
                enabled=False,
                reason="all collectors failed to start",
                endpoints_configured=self._configured_endpoints,
                endpoints_reachable=[],
            )
            self._shutdown_task = asyncio.create_task(self._delayed_shutdown())
            return

    @on_command(CommandType.PROFILE_CANCEL)
    async def _handle_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> None:
        """Stop all metrics collectors when profiling is cancelled.

        Called when user cancels profiling or an error occurs during profiling.
        Stops all running collectors gracefully and cleans up resources.

        Args:
            message: Profile cancel command from SystemController
        """
        await self._stop_all_collectors()

    @on_stop
    async def _metrics_manager_stop(self) -> None:
        """Stop all metrics collectors during service shutdown.

        Called automatically by BaseComponentService lifecycle management via @on_stop hook.
        Ensures all collectors are properly stopped and cleaned up even if shutdown
        command was not received.
        """
        await self._stop_all_collectors()

    async def _delayed_shutdown(self) -> None:
        """Shutdown service after a delay to allow command response to be sent.

        Waits before calling stop() to ensure the command response
        has time to be published and transmitted to the SystemController.
        """
        await asyncio.sleep(self.SHUTDOWN_DELAY)
        await self.stop()

    async def _stop_all_collectors(self) -> None:
        """Stop all metrics collectors.

        Attempts to stop each collector gracefully, logging errors but continuing with
        remaining collectors to ensure all resources are released. Does nothing if no
        collectors are configured.

        Errors during individual collector shutdown do not prevent other collectors
        from being stopped.
        """
        if not self._collectors:
            return

        for endpoint_url, collector in self._collectors.items():
            try:
                await collector.stop()
            except Exception as e:
                self.error(f"Failed to stop collector for {endpoint_url}: {e}")

    async def _on_metric_records(
        self, records: list[RecordT], collector_id: str
    ) -> None:
        """Async callback for receiving metric records from collectors.

        Sends records message to RecordsManager via message system.
        Empty record lists are ignored.

        Args:
            records: List of record objects from a collector
            collector_id: Unique identifier of the collector that sent the records
        """
        if not records:
            return

        try:
            endpoint_url = self._collector_id_to_url.get(collector_id, "")
            message = self._create_records_message(
                collector_id=collector_id,
                endpoint_url=endpoint_url,
                records=records,
                error=None,
            )

            await self.records_push_client.push(message)

        except Exception as e:
            self.error(f"Failed to send metric records: {e}")

    async def _on_metric_error(self, error: ErrorDetails, collector_id: str) -> None:
        """Async callback for receiving metric errors from collectors.

        Sends error message to RecordsManager via message system.
        The message contains an empty records list and the error details.

        Args:
            error: ErrorDetails describing the collection error
            collector_id: Unique identifier of the collector that encountered the error
        """
        try:
            endpoint_url = self._collector_id_to_url.get(collector_id, "")
            error_message = self._create_records_message(
                collector_id=collector_id,
                endpoint_url=endpoint_url,
                records=[],
                error=error,
            )

            await self.records_push_client.push(error_message)

        except Exception as e:
            self.error(f"Failed to send metric error message: {e}")

    async def _send_status_message(
        self,
        enabled: bool,
        reason: str | None = None,
        endpoints_configured: list[str] | None = None,
        endpoints_reachable: list[str] | None = None,
    ) -> None:
        """Send metrics status message to SystemController.

        Publishes status message to inform SystemController about metrics
        availability and endpoint reachability. Applies display filtering to show
        user-provided endpoints and reachable defaults only.

        Args:
            enabled: Whether metrics collection is enabled/available
            reason: Optional human-readable reason for status
            endpoints_configured: List of endpoint URLs configured (unused, kept for API compatibility)
            endpoints_reachable: List of endpoint URLs that are accessible
        """
        try:
            # Apply display filtering: filter to show only reachable defaults
            endpoints_reachable = endpoints_reachable or []
            reachable_defaults = [
                ep for ep in self._default_endpoints if ep in endpoints_reachable
            ]
            endpoints_for_display = self._compute_endpoints_for_display(
                reachable_defaults
            )

            status_message = self._create_status_message(
                enabled=enabled,
                reason=reason,
                endpoints_configured=endpoints_for_display,
                endpoints_reachable=endpoints_reachable,
            )

            await self.publish(status_message)

        except Exception as e:
            self.error(f"Failed to send metrics status message: {e}")

    def _compute_endpoints_for_display(
        self, reachable_defaults: list[str]
    ) -> list[str]:
        """Compute which endpoints should be displayed to the user.

        Filters endpoints for clean console output based on user configuration
        and reachability. This intentional filtering prevents cluttering the UI
        with unreachable default endpoints that the user didn't explicitly configure.

        Args:
            reachable_defaults: List of default endpoints that are reachable

        Returns:
            List of endpoint URLs to display in console/export output
        """
        return list(self._user_provided_endpoints) + reachable_defaults

    @staticmethod
    def _normalize_url_with_suffix(
        url: str, suffix: str = "/metrics", add_http: bool = True
    ) -> str:
        """Normalize URL with configurable suffix and protocol.

        Args:
            url: Base URL to normalize
            suffix: Suffix to append (default: "/metrics")
            add_http: Whether to add http:// if no protocol present (default: True)

        Returns:
            str: Normalized URL
        """
        url = url.strip()

        # Add http:// if no protocol specified and add_http is True
        if add_http and not url.startswith(("http://", "https://")):
            url = f"http://{url}"

        # Remove trailing slash
        url = url.rstrip("/")

        # Add suffix if not present
        if suffix and not url.endswith(suffix):
            url = f"{url}{suffix}"

        return url

    @staticmethod
    def _ensure_list(value: str | list[str] | None) -> list[str]:
        """Convert string or None to list for consistent handling.

        Args:
            value: String, list of strings, or None

        Returns:
            list[str]: Empty list if None, single-item list if string, original list otherwise
        """
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    # Abstract methods that subclasses must implement

    @abstractmethod
    def _get_raw_default_endpoints(self, user_config: UserConfig) -> list[str]:
        """Get raw default endpoint URLs (will be normalized by base class).

        This method returns the default endpoints for this metrics type without normalization.
        The base class will handle URL normalization automatically.

        Args:
            user_config: User configuration object

        Returns:
            list[str]: List of raw default endpoint URLs

        Example (GPU telemetry):
            return list(Environment.GPU.DEFAULT_DCGM_ENDPOINTS)

        Example (Server metrics with derived endpoint):
            inference_endpoint = user_config.endpoint.url
            env_defaults = list(Environment.SERVER_METRICS.DEFAULT_ENDPOINTS)
            return [inference_endpoint] + env_defaults
        """
        pass

    @abstractmethod
    def _get_user_endpoints_from_config(self, user_config: UserConfig) -> list[str]:
        """Get user-provided endpoint URLs from config (will be normalized by base class).

        This method extracts the user-configured endpoints for this metrics type.
        The base class will handle URL normalization automatically.

        Args:
            user_config: User configuration object

        Returns:
            list[str]: List of user-provided endpoint URLs

        Example (GPU telemetry):
            return user_config.gpu_telemetry_urls

        Example (Server metrics):
            return user_config.server_metrics_urls
        """
        pass

    def _create_collector(
        self,
        endpoint_url: str,
        collection_interval: float,
        collector_id: str,
    ) -> CollectorT:
        """Create a data collector instance.

        Uses COLLECTOR_CLASS class attribute to instantiate the appropriate
        collector type with standardized callbacks.

        Args:
            endpoint_url: The metrics endpoint URL
            collection_interval: Collection interval in seconds
            collector_id: Unique identifier for the collector

        Returns:
            CollectorT: A configured collector instance
        """
        return self.COLLECTOR_CLASS(
            endpoint_url=endpoint_url,
            collection_interval=collection_interval,
            record_callback=self._on_metric_records,
            error_callback=self._on_metric_error,
            collector_id=collector_id,
        )

    def _create_records_message(
        self,
        collector_id: str,
        endpoint_url: str,
        records: list[RecordT],
        error: ErrorDetails | None,
    ):
        """Create a records message to send to RecordsManager.

        Uses RECORDS_MESSAGE_CLASS and RECORDS_ENDPOINT_FIELD class attributes
        to instantiate the appropriate message type with the correct field name.

        Args:
            collector_id: ID of the collector that produced the records
            endpoint_url: The endpoint URL
            records: List of collected records (empty if error occurred)
            error: Error details if collection failed, None otherwise

        Returns:
            A message object (e.g., TelemetryRecordsMessage, ServerMetricRecordsMessage)
        """
        return self.RECORDS_MESSAGE_CLASS(
            service_id=self.service_id,
            collector_id=collector_id,
            **{self.RECORDS_ENDPOINT_FIELD: endpoint_url},
            records=records,
            error=error,
        )

    def _create_status_message(
        self,
        enabled: bool,
        reason: str | None,
        endpoints_configured: list[str],
        endpoints_reachable: list[str],
    ):
        """Create a status message to send to SystemController.

        Uses STATUS_MESSAGE_CLASS class attribute to instantiate the
        appropriate message type.

        Args:
            enabled: Whether metrics collection is enabled
            reason: Reason for disabled status (if applicable)
            endpoints_configured: List of configured endpoint URLs
            endpoints_reachable: List of reachable endpoint URLs

        Returns:
            A status message object (e.g., TelemetryStatusMessage, ServerMetricsStatusMessage)
        """
        return self.STATUS_MESSAGE_CLASS(
            service_id=self.service_id,
            enabled=enabled,
            reason=reason,
            endpoints_configured=endpoints_configured,
            endpoints_reachable=endpoints_reachable,
        )
