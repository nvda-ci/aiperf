# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import DataExporterType
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.factories import DataExporterFactory
from aiperf.common.models.export_data import (
    ServerMetricLabeledStats,
    ServerMetricsEndpointInfo,
    ServerMetricsMergedExportData,
    ServerMetricsSummary,
    ServerMetricSummary,
)
from aiperf.common.models.metric_info_models import InfoMetricData
from aiperf.common.protocols import DataExporterProtocol
from aiperf.exporters.display_units_utils import normalize_endpoint_display
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter


@DataExporterFactory.register(DataExporterType.SERVER_METRICS_JSON)
@implements_protocol(DataExporterProtocol)
class ServerMetricsJsonExporter(MetricsBaseExporter):
    """Export server metrics to a separate JSON file.

    Exports server metrics in a merged format where all endpoints' series are
    combined under each metric name, with each series item containing a
    normalized 'endpoint' field (without http:// prefix or /metrics suffix).
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
        """Generate JSON content for server metrics data in merged format.

        The merged format combines all endpoints' series under each metric name,
        with each series item containing a normalized 'endpoint' field.
        Endpoint metadata (duration, scrape count, latency) is included in the summary.

        Returns:
            str: JSON content with merged server metrics data
        """
        if not self._server_metrics_results:
            return "{}"

        merged_metrics, merged_info, endpoint_info = self._merge_endpoint_summaries()

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

        export_data = ServerMetricsMergedExportData(
            summary=summary,
            info_metrics=merged_info if merged_info else None,
            metrics=merged_metrics,
        )

        return export_data.model_dump_json(
            indent=2, exclude_unset=True, exclude_none=True
        )

    def _merge_endpoint_summaries(
        self,
    ) -> tuple[
        dict[str, ServerMetricSummary],
        dict[str, InfoMetricData] | None,
        dict[str, ServerMetricsEndpointInfo] | None,
    ]:
        """Merge all endpoint summaries into a single metrics dict.

        Each metric's series list contains items from all endpoints, with each
        series item tagged with its normalized source endpoint.

        Returns:
            Tuple of (merged metrics dict, merged info metrics dict, endpoint info dict)
        """
        if not self._server_metrics_results:
            return {}, None, None

        endpoint_summaries = self._server_metrics_results.endpoint_summaries
        if not endpoint_summaries:
            self.warning(
                "No pre-computed server metrics summaries available. "
                "This may indicate a ZMQ serialization issue."
            )
            return {}, None, None

        merged_metrics: dict[str, ServerMetricSummary] = {}
        merged_info: dict[str, InfoMetricData] = {}
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
            )

            # Merge info metrics
            if endpoint_summary.info_metrics:
                for metric_name, info_data in endpoint_summary.info_metrics.items():
                    if metric_name not in merged_info:
                        merged_info[metric_name] = InfoMetricData(
                            description=info_data.description,
                            labels=[],
                        )
                    # Add labels with normalized endpoint info
                    for label_set in info_data.labels:
                        label_with_endpoint = {
                            "endpoint": normalized_endpoint,
                            **label_set,
                        }
                        merged_info[metric_name].labels.append(label_with_endpoint)

            # Merge metrics
            for metric_name, metric_summary in endpoint_summary.metrics.items():
                if metric_name not in merged_metrics:
                    merged_metrics[metric_name] = ServerMetricSummary(
                        description=metric_summary.description,
                        type=metric_summary.type,
                        series=[],
                    )

                # Add each series with normalized endpoint field
                for series_item in metric_summary.series:
                    merged_series = ServerMetricLabeledStats(
                        endpoint=normalized_endpoint,
                        labels=series_item.labels,
                        stats=series_item.stats,
                    )
                    merged_metrics[metric_name].series.append(merged_series)

        return (
            merged_metrics,
            merged_info if merged_info else None,
            endpoint_info if endpoint_info else None,
        )
