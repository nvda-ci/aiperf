# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import PrometheusMetricType, ResultsProcessorType
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.models.processor_summary_results import ServerMetricsSummaryResult
from aiperf.common.models.server_metrics_models import (
    ServerMetricsHierarchy,
    ServerMetricsMetadata,
    ServerMetricsRecord,
)
from aiperf.common.protocols import (
    ServerMetricsResultsProcessorProtocol,
)
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


@implements_protocol(ServerMetricsResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.SERVER_METRICS_RESULTS)
class ServerMetricsResultsProcessor(BaseMetricsProcessor):
    """Process individual ServerMetricsRecord objects into hierarchical storage.

    This processor receives ServerMetricsRecord objects from the ServerMetricsManager
    and accumulates them in hierarchical storage (endpoint_url -> time series).
    All metrics are automatically discovered from Prometheus endpoints.
    The summarize() method generates MetricResult objects for display and export.
    """

    def __init__(self, user_config: UserConfig, **kwargs: Any):
        super().__init__(user_config=user_config, **kwargs)
        self._server_metrics_hierarchy = ServerMetricsHierarchy()
        self._discovered_metrics: set[tuple[str, str, frozenset]] = set()
        self._metadata_by_endpoint: dict[str, ServerMetricsMetadata] = {}

    def get_server_metrics_hierarchy(self) -> ServerMetricsHierarchy:
        """Get the accumulated server metrics hierarchy."""
        return self._server_metrics_hierarchy

    async def process_server_metrics_record(self, record: ServerMetricsRecord) -> None:
        """Process individual server metrics record into hierarchical storage.

        Args:
            record: ServerMetricsRecord containing Prometheus metrics snapshot and metadata
        """
        self._server_metrics_hierarchy.add_record(record)

        # If we have stored metadata for this endpoint (with metric_schemas),
        # apply it to the hierarchy to ensure the complete metadata is available
        endpoint_url = record.endpoint_url
        if (
            endpoint_url in self._metadata_by_endpoint
            and endpoint_url in self._server_metrics_hierarchy.endpoints
        ):
            stored_metadata = self._metadata_by_endpoint[endpoint_url]
            # Only update if the stored metadata has metric_schemas
            if stored_metadata.metric_schemas:
                self._server_metrics_hierarchy.endpoints[
                    endpoint_url
                ].metadata = stored_metadata

        self._discover_metrics_from_record(record)

    async def process_server_metrics_metadata(
        self, collector_id: str, metadata: ServerMetricsMetadata
    ) -> None:
        """Process server metrics metadata and update hierarchy.

        Stores the metadata (including metric_schemas) and applies it to the
        hierarchy if endpoint data already exists.

        Args:
            collector_id: Unique identifier for the server metrics data collector
            metadata: ServerMetricsMetadata containing static endpoint information and metric_schemas
        """
        endpoint_url = metadata.endpoint_url
        self._metadata_by_endpoint[endpoint_url] = metadata

        # If endpoint already exists in hierarchy, update its metadata with the full metadata
        # (including metric_schemas that were sent separately)
        if endpoint_url in self._server_metrics_hierarchy.endpoints:
            self._server_metrics_hierarchy.endpoints[endpoint_url].metadata = metadata
            self.debug(
                f"Updated metadata for endpoint {endpoint_url} "
                f"with {len(metadata.metric_schemas)} metric schemas"
            )
        else:
            self.debug(
                f"Stored metadata for endpoint {endpoint_url} (will apply when first record arrives)"
            )

    def _discover_metrics_from_record(self, record: ServerMetricsRecord) -> None:
        """Discover metrics from a record for auto-aggregation.

        Args:
            record: ServerMetricsRecord to discover metrics from
        """
        for metric_name, metric_family in record.metrics.items():
            # Auto-discover counters, gauges, histograms, and summaries
            if metric_family.type in (
                PrometheusMetricType.COUNTER,
                PrometheusMetricType.GAUGE,
                PrometheusMetricType.HISTOGRAM,
                PrometheusMetricType.SUMMARY,
            ):
                for sample in metric_family.samples:
                    # Check if sample has data: value, histogram, or summary
                    has_data = (
                        sample.value is not None
                        or (
                            metric_family.type == PrometheusMetricType.HISTOGRAM
                            and sample.histogram is not None
                        )
                        or (
                            metric_family.type == PrometheusMetricType.SUMMARY
                            and sample.summary is not None
                        )
                    )
                    if has_data:
                        labels_key = frozenset(sample.labels.items())
                        self._discovered_metrics.add(
                            (metric_name, metric_family.type, labels_key)
                        )

    async def summarize(
        self,
        min_timestamp_ns: int | None = None,
        max_timestamp_ns: int | None = None,
    ) -> ServerMetricsSummaryResult:
        """Generate summary with server metrics hierarchy.

        This method is called by RecordsManager after profiling completes.
        The server metrics hierarchy contains all Prometheus data in its native
        structured format. MetricResult objects can be generated on-demand from
        the hierarchy when needed for display or export.

        Args:
            min_timestamp_ns: Optional start of inference time window (min request start)
            max_timestamp_ns: Optional end of inference time window (max response end)

        Returns:
            ServerMetricsSummaryResult containing the server metrics hierarchy.

        Note:
            The time window parameters are stored for use by consumers that need
            to filter server metrics to align with the inference benchmark window.
        """
        # Log time window if provided
        if min_timestamp_ns:
            if max_timestamp_ns:
                duration_sec = (max_timestamp_ns - min_timestamp_ns) / 1e9
                self.info(
                    f"Server metrics time window: "
                    f"{duration_sec:.3f}s ({min_timestamp_ns} to {max_timestamp_ns})"
                )
            else:
                self.info(
                    f"Server metrics time window: "
                    f"[{min_timestamp_ns}, end of collection]"
                )

        # Debug: Log hierarchy state
        self.debug(
            lambda: f"Server metrics hierarchy has {len(self._server_metrics_hierarchy.endpoints)} endpoint(s)"
        )
        for endpoint_url in self._server_metrics_hierarchy.endpoints:
            endpoint_data = self._server_metrics_hierarchy.endpoints[endpoint_url]
            self.debug(
                lambda url=endpoint_url,
                data=endpoint_data: f"  Endpoint {url}: {len(data.time_series.snapshots)} snapshots total"
            )

        # Apply time-window filtering to align server metrics with inference window
        filtered_hierarchy = self._server_metrics_hierarchy
        if min_timestamp_ns is not None or max_timestamp_ns is not None:
            filtered_hierarchy = ServerMetricsHierarchy()
            for (
                endpoint_url,
                endpoint_data,
            ) in self._server_metrics_hierarchy.endpoints.items():
                # Filter time series to inference window
                filtered_time_series = endpoint_data.time_series.filter_by_time_window(
                    min_timestamp_ns, max_timestamp_ns
                )
                # Create filtered endpoint data with same metadata
                from aiperf.common.models.server_metrics_models import ServerMetricsData

                filtered_hierarchy.endpoints[endpoint_url] = ServerMetricsData(
                    metadata=endpoint_data.metadata,
                    time_series=filtered_time_series,
                )
                self.debug(
                    lambda url=endpoint_url,
                    orig_data=endpoint_data,
                    filt_ts=filtered_time_series: (
                        f"  Endpoint {url}: filtered {len(orig_data.time_series.snapshots)} "
                        f"-> {len(filt_ts.snapshots)} snapshots"
                    )
                )

        # Get endpoints tested and successful from hierarchy
        endpoints = list(filtered_hierarchy.endpoints.keys())

        return ServerMetricsSummaryResult(
            server_metrics_data=filtered_hierarchy,
            endpoints_tested=endpoints,
            endpoints_successful=endpoints,
            error_summary=[],
        )

    @staticmethod
    def _extract_help_text_from_endpoint(endpoint_data) -> dict[str, str]:
        """Extract help text for all metrics from the first snapshot.

        Args:
            endpoint_data: ServerMetricsData containing snapshots

        Returns:
            Dict mapping metric_name to help text
        """
        help_text_map = {}
        if endpoint_data.time_series.snapshots:
            _, first_metrics = endpoint_data.time_series.snapshots[0]
            for metric_name, metric_family in first_metrics.items():
                help_text_map[metric_name] = metric_family.help
        return help_text_map

    @staticmethod
    def _infer_unit_from_metric_name(metric_name: str) -> str:
        """Infer unit from metric name based on common naming conventions.

        Args:
            metric_name: Name of the metric

        Returns:
            Inferred unit string
        """
        name_lower = metric_name.lower()

        if any(x in name_lower for x in ["_seconds", "_duration_s"]):
            return "s"
        if any(x in name_lower for x in ["_milliseconds", "_duration_ms", "_ms"]):
            return "ms"
        if any(x in name_lower for x in ["_microseconds", "_duration_us", "_us"]):
            return "us"
        if any(x in name_lower for x in ["_bytes", "_size"]):
            return "bytes"
        if any(x in name_lower for x in ["_percent", "_perc", "_usage"]):
            return "%"
        if any(x in name_lower for x in ["_rate", "_per_s", "_toks_per_s"]):
            return "/s"
        if any(x in name_lower for x in ["_count", "_total", "_requests"]):
            return "count"

        return ""
