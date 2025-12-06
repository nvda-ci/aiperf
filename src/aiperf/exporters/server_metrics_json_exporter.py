# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import DataExporterType, PrometheusMetricType
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.factories import DataExporterFactory
from aiperf.common.models.export_data import (
    FlatSeriesStats,
    HybridMetricData,
    ServerMetricsEndpointInfo,
    ServerMetricsHybridExportData,
    ServerMetricsSummary,
)
from aiperf.common.protocols import DataExporterProtocol
from aiperf.exporters.display_units_utils import (
    normalize_endpoint_display,
    parse_unit_from_metric_name,
)
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter


@DataExporterFactory.register(DataExporterType.SERVER_METRICS_JSON)
@implements_protocol(DataExporterProtocol)
class ServerMetricsJsonExporter(MetricsBaseExporter):
    """Export server metrics to a separate JSON file in hybrid format.

    Exports server metrics with metrics keyed by name for O(1) lookup,
    while keeping stats flat within each series for easy access.

    Format: data["metrics"]["metric_name"]["series"][0]["p99"]
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        # Check if server metrics data is available before initializing
        if not exporter_config.server_metrics_results:
            raise DataExporterDisabled(
                "Server metrics JSON export disabled: no server metrics data available"
            )

        super().__init__(exporter_config, **kwargs)
        self._file_path = (
            exporter_config.user_config.output.server_metrics_export_json_file
        )
        self.trace_or_debug(
            lambda: f"Initializing ServerMetricsJsonExporter with config: {exporter_config}",
            lambda: f"Initializing ServerMetricsJsonExporter with file path: {self._file_path}",
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Server Metrics JSON Export",
            file_path=self._file_path,
        )

    def _generate_content(self) -> str:
        """Generate JSON content for server metrics data in hybrid format.

        The hybrid format provides:
        - O(1) metric lookup by name (metrics keyed by name)
        - Flat stats within each series (FlatSeriesStats is canonical model)
        - Unit parsed from metric name suffix
        - Both normalized endpoint and full endpoint_url

        Returns:
            str: JSON content with hybrid server metrics format
        """
        if not self._server_metrics_results:
            return "{}"

        metrics, endpoint_info = self._build_hybrid_metrics()

        # Normalize endpoint URLs in configured/successful lists
        endpoints_configured = [
            normalize_endpoint_display(url)
            for url in self._server_metrics_results.endpoints_configured
        ]
        endpoints_successful = [
            normalize_endpoint_display(url)
            for url in self._server_metrics_results.endpoints_successful
        ]

        summary = ServerMetricsSummary(
            endpoints_configured=endpoints_configured,
            endpoints_successful=endpoints_successful,
            start_time=datetime.fromtimestamp(
                self._server_metrics_results.start_ns / NANOS_PER_SECOND
            ),
            end_time=datetime.fromtimestamp(
                self._server_metrics_results.end_ns / NANOS_PER_SECOND
            ),
            endpoint_info=endpoint_info if endpoint_info else None,
        )

        export_data = ServerMetricsHybridExportData(
            summary=summary,
            metrics=metrics,
        )

        return export_data.model_dump_json(
            indent=2, exclude_unset=True, exclude_none=True
        )

    def _build_hybrid_metrics(
        self,
    ) -> tuple[
        dict[str, HybridMetricData], dict[str, ServerMetricsEndpointInfo] | None
    ]:
        """Build hybrid metrics dict from endpoint summaries.

        Merges metrics from all endpoints into a single dict keyed by metric name.
        ServerMetricSummary.series already contains FlatSeriesStats, so we just
        need to add endpoint info and merge across endpoints.

        Returns:
            Tuple of (metrics dict, endpoint info dict)
        """
        if not self._server_metrics_results:
            return {}, None

        endpoint_summaries = self._server_metrics_results.endpoint_summaries
        if not endpoint_summaries:
            self.warning(
                "No pre-computed server metrics summaries available. "
                "This may indicate a ZMQ serialization issue."
            )
            return {}, None

        metrics: dict[str, HybridMetricData] = {}
        endpoint_info: dict[str, ServerMetricsEndpointInfo] = {}

        for endpoint_summary in endpoint_summaries.values():
            endpoint_url = endpoint_summary.endpoint_url
            normalized_endpoint = normalize_endpoint_display(endpoint_url)

            # Collect endpoint metadata for summary
            endpoint_info[normalized_endpoint] = ServerMetricsEndpointInfo(
                endpoint_url=endpoint_url,
                duration_seconds=endpoint_summary.duration_seconds,
                scrape_count=endpoint_summary.scrape_count,
                avg_scrape_latency_ms=endpoint_summary.avg_scrape_latency_ms,
                avg_scrape_period_ms=endpoint_summary.avg_scrape_period_ms,
            )

            # Process info metrics as gauges with labels only
            if endpoint_summary.info_metrics:
                for metric_name, info_data in endpoint_summary.info_metrics.items():
                    unit = parse_unit_from_metric_name(metric_name)

                    # Get or create metric entry
                    if metric_name not in metrics:
                        metrics[metric_name] = HybridMetricData(
                            type=PrometheusMetricType.GAUGE,
                            description=info_data.description,
                            unit=unit,
                            series=[],
                        )

                    # Add series for each label set
                    for label_set in info_data.labels:
                        metrics[metric_name].series.append(
                            FlatSeriesStats(
                                endpoint=normalized_endpoint,
                                endpoint_url=endpoint_url,
                                labels=label_set if label_set else None,
                            )
                        )

            # Process regular metrics - series already contains FlatSeriesStats
            for metric_name, metric_summary in endpoint_summary.metrics.items():
                unit = parse_unit_from_metric_name(metric_name)

                # Get or create metric entry
                if metric_name not in metrics:
                    metrics[metric_name] = HybridMetricData(
                        type=metric_summary.type,
                        description=metric_summary.description,
                        unit=unit,
                        series=[],
                    )

                # Add endpoint info to each FlatSeriesStats and append to series
                for flat_stats in metric_summary.series:
                    # Set endpoint info on the stats
                    flat_stats.endpoint = normalized_endpoint
                    flat_stats.endpoint_url = endpoint_url
                    metrics[metric_name].series.append(flat_stats)

        # Sort metrics alphabetically by name for deterministic output and easier lookup
        sorted_metrics = dict(sorted(metrics.items()))

        # Sort series within each metric by endpoint, then by labels
        for metric_data in sorted_metrics.values():
            metric_data.series.sort(
                key=lambda s: (s.endpoint or "", str(s.labels) if s.labels else "")
            )

        # Sort endpoint_info for consistency
        sorted_endpoint_info = (
            dict(sorted(endpoint_info.items())) if endpoint_info else None
        )

        return sorted_metrics, sorted_endpoint_info
