# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class for metrics data collectors that fetch from Prometheus endpoints.

This module provides a reusable abstract base class for collecting metrics from
Prometheus-compatible /metrics endpoints using async architecture.
"""

import asyncio
import time
from abc import abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

import aiohttp
from prometheus_client.parser import text_string_to_metric_families

from aiperf.common.hooks import background_task, on_init, on_stop
from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin
from aiperf.common.models import ErrorDetails, HistogramComponents

__all__ = ["BaseMetricsDataCollector"]

# Generic type for the record type (e.g., TelemetryRecord, ServerMetricRecord)
RecordT = TypeVar("RecordT")


class BaseMetricsDataCollector(AIPerfLifecycleMixin, Generic[RecordT]):
    """Abstract base class for collecting metrics from Prometheus endpoints.

    This class provides common functionality for:
    - HTTP session management with aiohttp
    - Endpoint reachability testing
    - Background periodic collection
    - Prometheus format parsing
    - Error handling and callbacks
    - Lifecycle management

    Subclasses must implement:
    - _extract_resource_info(): Extract resource ID and metadata from labels
    - _create_records(): Create typed records from parsed metrics

    ClassVars to override:
        FIELD_MAPPING: Dict mapping Prometheus metric names to record field names
        SCALING_FACTORS: Dict mapping field names to scaling multipliers for unit conversion
        DEFAULT_COLLECTOR_ID: Default collector ID if not specified
        DEFAULT_COLLECTION_INTERVAL: Default collection interval in seconds
        REACHABILITY_TIMEOUT: Timeout for endpoint reachability checks in seconds

    Args:
        endpoint_url: URL of the metrics endpoint (e.g., "http://server:8080/metrics")
        collection_interval: Interval in seconds between collections (default: from ClassVar)
        record_callback: Async callback to receive collected records
            Signature: async (records: list[RecordT], collector_id: str) -> None
        error_callback: Async callback to receive collection errors
            Signature: async (error: ErrorDetails, collector_id: str) -> None
        collector_id: Unique identifier for this collector instance
    """

    # Subclasses must override these ClassVars
    FIELD_MAPPING: dict[str, str]
    SCALING_FACTORS: dict[str, float]
    DEFAULT_COLLECTOR_ID: str = "metrics_collector"
    DEFAULT_COLLECTION_INTERVAL: float
    REACHABILITY_TIMEOUT: float

    def __init__(
        self,
        endpoint_url: str,
        collection_interval: float | None = None,
        record_callback: Callable[[list[RecordT], str], Awaitable[None]] | None = None,
        error_callback: Callable[[ErrorDetails, str], Awaitable[None]] | None = None,
        collector_id: str | None = None,
    ) -> None:
        self._endpoint_url = endpoint_url
        self._field_mapping = self.FIELD_MAPPING
        self._scaling_factors = self.SCALING_FACTORS
        self._collection_interval = (
            collection_interval
            if collection_interval is not None
            else self.DEFAULT_COLLECTION_INTERVAL
        )
        self._record_callback = record_callback
        self._error_callback = error_callback
        self._session: aiohttp.ClientSession | None = None

        super().__init__(id=collector_id or self.DEFAULT_COLLECTOR_ID)

    @property
    def endpoint_url(self) -> str:
        """The metrics endpoint URL being monitored."""
        return self._endpoint_url

    @on_init
    async def _initialize_http_client(self) -> None:
        """Initialize the aiohttp client session.

        Called automatically by AIPerfLifecycleMixin during initialization phase.
        Creates an aiohttp ClientSession with appropriate timeout settings.
        """
        timeout = aiohttp.ClientTimeout(total=self.REACHABILITY_TIMEOUT)
        self._session = aiohttp.ClientSession(timeout=timeout)

    @on_stop
    async def _cleanup_http_client(self) -> None:
        """Clean up the aiohttp client session.

        Called automatically by AIPerfLifecycleMixin during shutdown phase.
        Race conditions with background tasks are handled by checking
        self.stop_requested in the background task itself.

        Raises:
            Exception: Any exception from session.close() is allowed to propagate
        """
        if self._session:
            await self._session.close()
            self._session = None

    async def is_url_reachable(self) -> bool:
        """Check if metrics endpoint is accessible.

        Attempts HEAD request first for efficiency, falls back to GET if HEAD is not supported.
        Uses existing session if available, otherwise creates a temporary session.

        Returns:
            bool: True if endpoint responds with HTTP 200, False for any error or other status
        """
        if not self._endpoint_url:
            return False

        # Use existing session if available, otherwise create a temporary one
        if self._session:
            try:
                # Try HEAD first for efficiency
                async with self._session.head(
                    self._endpoint_url, allow_redirects=False
                ) as response:
                    if response.status == 200:
                        return True
                # Fall back to GET if HEAD is not supported
                async with self._session.get(self._endpoint_url) as response:
                    return response.status == 200
            except (aiohttp.ClientError, asyncio.TimeoutError):
                return False
        else:
            # Create a temporary session for reachability check
            timeout = aiohttp.ClientTimeout(total=self.REACHABILITY_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as temp_session:
                try:
                    # Try HEAD first for efficiency
                    async with temp_session.head(
                        self._endpoint_url, allow_redirects=False
                    ) as response:
                        if response.status == 200:
                            return True
                    # Fall back to GET if HEAD is not supported
                    async with temp_session.get(self._endpoint_url) as response:
                        return response.status == 200
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    return False

    @background_task(immediate=True, interval=lambda self: self._collection_interval)
    async def _collect_metrics_task(self) -> None:
        """Background task for collecting metrics at regular intervals.

        This uses the @background_task decorator which automatically handles
        lifecycle management and stopping when the collector is stopped.
        The interval is set to the collection_interval so this runs periodically.

        Errors during collection are caught and sent via error_callback if configured.
        CancelledError is propagated to allow graceful shutdown.

        Raises:
            asyncio.CancelledError: Propagated to signal task cancellation during shutdown
        """
        try:
            await self._collect_and_process_metrics()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self._error_callback:
                try:
                    await self._error_callback(ErrorDetails.from_exception(e), self.id)
                except Exception as callback_error:
                    self.error(f"Failed to send error via callback: {callback_error}")
            else:
                self.error(f"Metrics collection error: {e}")

    async def _collect_and_process_metrics(self) -> None:
        """Collect metrics from endpoint and process them into typed records.

        Orchestrates the full collection flow:
        1. Fetches raw metrics data from endpoint
        2. Parses Prometheus-format data into typed record objects
        3. Sends records via callback (if configured and records are not empty)

        Callback failures are caught and logged as warnings without stopping collection.

        Raises:
            Exception: Any exception from fetch or parse is logged and re-raised
        """
        try:
            metrics_data = await self._fetch_metrics()
            records = self._parse_metrics_to_records(metrics_data)

            if records and self._record_callback:
                try:
                    await self._record_callback(records, self.id)
                except Exception as e:
                    self.warning(f"Failed to send metric records via callback: {e}")

        except Exception as e:
            self.error(f"Error collecting and processing metrics: {e}")
            raise

    async def _fetch_metrics(self) -> str:
        """Fetch raw metrics data from endpoint using aiohttp.

        Performs safety checks before making HTTP request:
        - Verifies stop_requested flag to allow graceful shutdown
        - Checks session is initialized and not closed

        Returns:
            str: Raw metrics text in Prometheus exposition format

        Raises:
            RuntimeError: If HTTP session is not initialized
            aiohttp.ClientError: If HTTP request fails (4xx, 5xx, network errors)
            asyncio.CancelledError: If collector is being stopped or session is closed
        """
        if self.stop_requested:
            raise asyncio.CancelledError

        if not self._session:
            raise RuntimeError("HTTP session not initialized. Call initialize() first.")

        if self._session.closed:
            raise asyncio.CancelledError

        async with self._session.get(self._endpoint_url) as response:
            response.raise_for_status()
            text = await response.text()
            return text

    def _parse_metrics_to_records(self, metrics_data: str) -> list[RecordT]:
        """Parse Prometheus metrics text into typed record objects.

        This is the main parsing method that coordinates:
        1. Parsing metric families using prometheus_client
        2. Extracting resource data and metadata from labels
        3. Mapping metric names to field names
        4. Applying scaling factors
        5. Creating typed record objects

        Args:
            metrics_data: Raw metrics text from endpoint in Prometheus format

        Returns:
            list[RecordT]: List of typed record objects (e.g., TelemetryRecord, ServerMetricRecord)
                Returns empty list if metrics_data is empty or parsing fails.
        """
        if not metrics_data.strip():
            return []

        current_timestamp = time.time_ns()
        resource_data: dict[Any, dict[str, float]] = {}
        resource_metadata: dict[Any, dict[str, Any]] = {}
        # Track histogram data: dict[resource_id, dict[histogram_name, HistogramComponents]]
        histogram_data: dict[Any, dict[str, HistogramComponents]] = {}

        try:
            for family in text_string_to_metric_families(metrics_data):
                # Handle histogram metrics specially
                if family.type == "histogram":
                    for sample in family.samples:
                        metric_name = sample.name
                        labels = sample.labels
                        value = sample.value

                        # Extract resource identifier and metadata
                        resource_id, metadata = self._extract_resource_info(labels)
                        if resource_id is None:
                            continue

                        # Store metadata for this resource
                        if resource_id not in resource_metadata:
                            resource_metadata[resource_id] = metadata

                        # Capture _created timestamp (histogram creation time)
                        if metric_name.endswith("_created"):
                            base_name = metric_name.removesuffix("_created")
                            if value is None:
                                continue
                            if resource_id not in histogram_data:
                                histogram_data[resource_id] = {}
                            if base_name not in histogram_data[resource_id]:
                                histogram_data[resource_id][base_name] = (
                                    HistogramComponents(buckets=[])
                                )
                            # Convert seconds to nanoseconds
                            histogram_data[resource_id][base_name].created_at_ns = int(
                                value * 1_000_000_000
                            )
                            continue

                        # Determine histogram component type
                        if metric_name.endswith("_bucket"):
                            base_name = metric_name.removesuffix("_bucket")
                            le = labels.get("le")

                            # Skip buckets without 'le' label or with None/invalid values
                            # Be explicit about all invalid cases
                            if le is None or le == "" or le == "None" or le == "null":
                                self.warning(
                                    f"Skipping histogram bucket {metric_name} with missing/invalid le label: le={le!r}"
                                )
                                continue

                            # Initialize histogram data structure if needed
                            if resource_id not in histogram_data:
                                histogram_data[resource_id] = {}
                            if base_name not in histogram_data[resource_id]:
                                histogram_data[resource_id][base_name] = (
                                    HistogramComponents(buckets=[])
                                )

                            # Parse le value (handle "+Inf" specially)
                            le_value: float
                            if le == "+Inf":
                                le_value = float("inf")
                            else:
                                try:
                                    le_value = float(le)
                                except (ValueError, TypeError) as e:
                                    self.debug(
                                        f"Skipping histogram bucket {metric_name} with invalid le value '{le}': {e}"
                                    )
                                    continue

                            # Skip buckets with None or invalid count values
                            if value is None:
                                self.debug(
                                    f"Skipping histogram bucket {metric_name} with None count value"
                                )
                                continue

                            # Final safety check before appending
                            if le_value is None:
                                self.error(
                                    f"ERROR: le_value is None after parsing! metric={metric_name}, le={le!r}, le_value={le_value!r}"
                                )
                                continue

                            histogram_data[resource_id][base_name].buckets.append(
                                (le_value, value)
                            )

                        elif metric_name.endswith("_sum"):
                            base_name = metric_name.removesuffix("_sum")
                            if value is None:
                                continue
                            if resource_id not in histogram_data:
                                histogram_data[resource_id] = {}
                            if base_name not in histogram_data[resource_id]:
                                histogram_data[resource_id][base_name] = (
                                    HistogramComponents(buckets=[])
                                )
                            histogram_data[resource_id][base_name].sum = float(value)

                        elif metric_name.endswith("_count"):
                            base_name = metric_name.removesuffix("_count")
                            if value is None:
                                continue
                            if resource_id not in histogram_data:
                                histogram_data[resource_id] = {}
                            if base_name not in histogram_data[resource_id]:
                                histogram_data[resource_id][base_name] = (
                                    HistogramComponents(buckets=[])
                                )
                            try:
                                histogram_data[resource_id][base_name].count = int(
                                    value
                                )
                            except (ValueError, TypeError):
                                self.debug(
                                    f"Skipping histogram count with invalid value: {value}"
                                )
                                continue

                else:
                    # Handle non-histogram metrics (gauges, counters, summaries)
                    for sample in family.samples:
                        metric_name = sample.name
                        labels = sample.labels
                        value = sample.value

                        # Skip non-finite values early (value != value checks for NaN)
                        if isinstance(value, float) and (
                            value != value or value in (float("inf"), float("-inf"))
                        ):
                            continue

                        # Extract resource identifier and metadata (subclass-specific)
                        resource_id, metadata = self._extract_resource_info(labels)
                        if resource_id is None:
                            continue

                        # Store metadata for this resource
                        if resource_id not in resource_metadata:
                            resource_metadata[resource_id] = metadata

                        # Map metric name to field name
                        # If FIELD_MAPPING is empty, use Prometheus names directly
                        base_metric_name = metric_name.removesuffix("_total")
                        if self._field_mapping:
                            # Using explicit field mapping
                            if base_metric_name in self._field_mapping:
                                field_name = self._field_mapping[base_metric_name]
                                resource_data.setdefault(resource_id, {})[
                                    field_name
                                ] = value
                        else:
                            # Use Prometheus metric name directly (no mapping)
                            field_name = base_metric_name
                            resource_data.setdefault(resource_id, {})[field_name] = (
                                value
                            )

        except ValueError:
            self.warning("Failed to parse Prometheus metrics - invalid format")
            return []

        # Convert resource data to typed records (subclass-specific)
        return self._create_records(
            resource_data, resource_metadata, current_timestamp, histogram_data
        )

    def _apply_scaling_factors(self, metrics: dict[str, float]) -> dict[str, float]:
        """Apply scaling factors to convert raw units to display units.

        Converts metrics from their native units to human-readable units based on
        the scaling factors provided in constructor.

        Only applies scaling to metrics present in the input dict. None values are preserved.

        Args:
            metrics: Dict of metric_name -> raw_value

        Returns:
            dict: New dict with scaled values ready for display. Unscaled metrics are copied as-is.
        """
        scaled_metrics = metrics.copy()
        for metric, factor in self._scaling_factors.items():
            if metric in scaled_metrics and scaled_metrics[metric] is not None:
                scaled_metrics[metric] *= factor
        return scaled_metrics

    # Abstract methods that subclasses must implement

    @abstractmethod
    def _extract_resource_info(
        self, labels: dict[str, str]
    ) -> tuple[Any, dict[str, Any]]:
        """Extract resource identifier and metadata from Prometheus labels.

        Args:
            labels: Prometheus metric labels (e.g., {'gpu': '0', 'UUID': '...', ...})

        Returns:
            tuple: (resource_id, metadata_dict)
                - resource_id: Unique identifier for the resource (e.g., GPU index, server ID)
                              Return None to skip this metric
                - metadata_dict: Dictionary of metadata fields for this resource

        Example for GPU telemetry:
            return (gpu_index, {'uuid': labels.get('UUID'), 'model': labels.get('modelName'), ...})

        Example for server metrics:
            return (server_id, {'hostname': labels.get('hostname'), 'instance': labels.get('instance'), ...})
        """
        pass

    @abstractmethod
    def _create_records(
        self,
        resource_data: dict[Any, dict[str, float]],
        resource_metadata: dict[Any, dict[str, Any]],
        timestamp_ns: int,
        histogram_data: dict[Any, dict[str, HistogramComponents]] | None = None,
    ) -> list[RecordT]:
        """Create typed record objects from parsed metrics.

        Args:
            resource_data: Dict mapping resource_id -> {field_name: value}
            resource_metadata: Dict mapping resource_id -> {metadata_key: metadata_value}
            timestamp_ns: Timestamp when metrics were collected
            histogram_data: Optional dict mapping resource_id -> histogram_name -> HistogramComponents
                          Each HistogramComponents contains buckets, sum, count, and optional created_at_ns

        Returns:
            list[RecordT]: List of typed record objects

        Example implementation:
            records = []
            for resource_id, metrics in resource_data.items():
                metadata = resource_metadata.get(resource_id, {})
                scaled_metrics = self._apply_scaling_factors(metrics)
                record = MyRecordType(
                    timestamp_ns=timestamp_ns,
                    resource_id=resource_id,
                    metrics_data=MyMetricsType(**scaled_metrics),
                    **metadata
                )
                records.append(record)
            return records
        """
        pass
