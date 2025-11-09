# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import PrometheusMetricType, ResultsProcessorType
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.models.processor_summary_results import ServerMetricsSummaryResult
from aiperf.common.models.server_metrics_models import (
    ServerMetricsHierarchy,
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

    def get_server_metrics_hierarchy(self) -> ServerMetricsHierarchy:
        """Get the accumulated server metrics hierarchy."""
        return self._server_metrics_hierarchy

    async def process_server_metrics_record(self, record: ServerMetricsRecord) -> None:
        """Process individual server metrics record into hierarchical storage.

        Args:
            record: ServerMetricsRecord containing Prometheus metrics snapshot and metadata
        """
        self._server_metrics_hierarchy.add_record(record)
        self._discover_metrics_from_record(record)

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
        """Generate MetricResult list with aggregated statistics for auto-discovered metrics.

        This method aggregates all metrics discovered from Prometheus endpoints across
        the entire benchmark run, computing statistics (min, max, avg, percentiles) for
        each metric. Optionally filters server metrics to align with the actual inference
        time window.

        Args:
            min_timestamp_ns: Optional start of inference time window (min request start)
            max_timestamp_ns: Optional end of inference time window (max response end)

        Returns:
            ServerMetricsSummaryResult containing MetricResult objects and server metrics hierarchy.
        """
        results = []

        # Log time window if provided
        if min_timestamp_ns:
            if max_timestamp_ns:
                duration_sec = (max_timestamp_ns - min_timestamp_ns) / 1e9
                self.info(
                    f"Aggregating server metrics with time window filter: "
                    f"{duration_sec:.3f}s ({min_timestamp_ns} to {max_timestamp_ns})"
                )
            else:
                self.info(
                    f"Aggregating server metrics with time window filter: "
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

        for (
            endpoint_url,
            endpoint_data,
        ) in self._server_metrics_hierarchy.endpoints.items():
            if not endpoint_data.time_series.snapshots:
                self.debug(f"Skipping endpoint {endpoint_url}: no snapshots")
                continue

            endpoint_display = endpoint_data.metadata.endpoint_display

            # Get help text from first snapshot
            help_text_map = self._extract_help_text_from_endpoint(endpoint_data)

            # Process auto-discovered metrics
            self.debug(
                lambda url=endpoint_url,
                metrics=self._discovered_metrics: f"Processing {len(metrics)} auto-discovered metrics for {url}"
            )

            for (
                metric_name,
                metric_type,
                labels_frozen,
            ) in self._discovered_metrics:
                labels = dict(labels_frozen)
                try:
                    # Create tag and header from metric name and labels
                    labels_str = (
                        "_" + "_".join(f"{k}_{v}" for k, v in sorted(labels.items()))
                        if labels
                        else ""
                    )
                    tag = f"server_metrics.{endpoint_display}.{metric_name}{labels_str}"
                    header = f"{metric_name} ({endpoint_display})"

                    # Determine unit based on metric name conventions
                    unit = self._infer_unit_from_metric_name(metric_name)

                    metric_result = endpoint_data.get_metric_result(
                        metric_name=metric_name,
                        labels=labels,
                        tag=tag,
                        header=header,
                        unit=unit,
                        min_timestamp_ns=min_timestamp_ns,
                        max_timestamp_ns=max_timestamp_ns,
                    )
                    # Add server metrics metadata
                    metric_result.metric_name = metric_name
                    metric_result.metric_type = metric_type
                    metric_result.metric_labels = labels
                    metric_result.metric_help = help_text_map.get(metric_name, "")
                    results.append(metric_result)
                except NoMetricValue:
                    continue
                except Exception as e:
                    self.warning(
                        f"Error aggregating discovered metric '{metric_name}': {e!r}"
                    )

        self.info(f"Generated {len(results)} aggregated metric results")

        # Get endpoints tested and successful from hierarchy
        endpoints = list(self._server_metrics_hierarchy.endpoints.keys())

        return ServerMetricsSummaryResult(
            results=results,
            server_metrics_data=self._server_metrics_hierarchy,
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
