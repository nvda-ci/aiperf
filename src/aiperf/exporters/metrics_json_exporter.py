# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from datetime import datetime

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import DataExporterType
from aiperf.common.factories import DataExporterFactory
from aiperf.common.models import MetricResult
from aiperf.common.models.export_models import (
    AggregatedMetricFamily,
    AggregatedMetricSample,
    EndpointData,
    GpuSummary,
    JsonExportData,
    JsonMetricResult,
    ServerMetricsEndpointData,
    ServerMetricsExportData,
    ServerMetricsSummary,
    TelemetryExportData,
    TelemetrySummary,
)
from aiperf.common.protocols import DataExporterProtocol
from aiperf.exporters.display_units_utils import normalize_endpoint_display
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter
from aiperf.gpu_telemetry.constants import get_gpu_telemetry_metrics_config


@DataExporterFactory.register(DataExporterType.JSON)
@implements_protocol(DataExporterProtocol)
class MetricsJsonExporter(MetricsBaseExporter):
    """
    A class to export records to a JSON file.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)
        self.debug(
            lambda: f"Initializing MetricsJsonExporter with config: {exporter_config}"
        )
        self._file_path = exporter_config.user_config.output.profile_export_json_file

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="JSON Export",
            file_path=self._file_path,
        )

    def _generate_content(self) -> str:
        """Generate JSON content string from inference and telemetry data.

        Uses instance data members from process_records_result.

        Returns:
            str: Complete JSON content with all sections formatted and ready to write
        """
        # Extract metric results from summary_results
        metric_results = self._get_metric_results()
        prepared_json_metrics = self._prepare_metrics_for_json(metric_results)

        # Get timing data from profile_summary
        profile_summary = self._process_records_result.profile_summary
        start_time = (
            datetime.fromtimestamp(profile_summary.start_ns / NANOS_PER_SECOND)
            if profile_summary.start_ns
            else None
        )
        end_time = (
            datetime.fromtimestamp(profile_summary.end_ns / NANOS_PER_SECOND)
            if profile_summary.end_ns
            else None
        )

        telemetry_export_data = None
        telemetry_results = self._get_telemetry_results()
        if telemetry_results:
            summary = TelemetrySummary(
                endpoints_configured=telemetry_results.endpoints_tested,
                endpoints_successful=telemetry_results.endpoints_successful,
                start_time=start_time,
                end_time=end_time,
            )
            telemetry_export_data = TelemetryExportData(
                summary=summary,
                endpoints=self._generate_telemetry_statistical_summary(
                    telemetry_results
                ),
            )

        server_metrics_export_data = None
        server_metrics_results = self._get_server_metrics_results()
        if server_metrics_results:
            summary = ServerMetricsSummary(
                endpoints_configured=server_metrics_results.endpoints_tested,
                endpoints_successful=server_metrics_results.endpoints_successful,
                start_time=start_time,
                end_time=end_time,
            )
            server_metrics_export_data = ServerMetricsExportData(
                summary=summary,
                endpoints=self._generate_server_metrics_statistical_summary(
                    server_metrics_results
                ),
            )

        export_data = JsonExportData(
            input_config=self._user_config,
            was_cancelled=profile_summary.was_cancelled,
            error_summary=profile_summary.error_summary,
            start_time=start_time,
            end_time=end_time,
            telemetry_data=telemetry_export_data,
            server_metrics_data=server_metrics_export_data,
        )

        # Add all prepared metrics dynamically
        for metric_tag, json_result in prepared_json_metrics.items():
            setattr(export_data, metric_tag, json_result)

        self.debug(lambda: f"Exporting data to JSON file: {export_data}")
        return export_data.model_dump_json(
            indent=2, exclude_unset=True, exclude_none=True
        )

    def _prepare_metrics_for_json(
        self, metric_results: Iterable[MetricResult]
    ) -> dict[str, JsonMetricResult]:
        """Prepare and convert metrics to JsonMetricResult objects.

        Applies unit conversion, filtering, and conversion to JSON format.

        Args:
            metric_results: Raw metric results to prepare

        Returns:
            dict mapping metric tags to JsonMetricResult objects ready for export
        """
        prepared = self._prepare_metrics(metric_results)
        return {tag: result.to_json_result() for tag, result in prepared.items()}

    def _generate_telemetry_statistical_summary(
        self, telemetry_results
    ) -> dict[str, EndpointData]:
        """Generate clean statistical summary of telemetry data for JSON export.

        Args:
            telemetry_results: TelemetrySummaryResult containing telemetry data

        Processes telemetry hierarchy into a structured dict with:
        - Endpoints organized by normalized display name (e.g., "localhost:9400")
        - GPU data with metadata (index, name, UUID, hostname)
        - Metric statistics (avg, min, max, p99, p90, p75, std, count) per GPU
        - Only includes metrics with available data

        Returns:
            dict: Nested structure of endpoints -> gpus -> metrics with statistics.
                Empty dict if no telemetry data available.
        """
        summary = {}

        if not telemetry_results or not telemetry_results.telemetry_data:
            return summary

        for (
            dcgm_url,
            gpus_data,
        ) in telemetry_results.telemetry_data.dcgm_endpoints.items():
            endpoint_display = normalize_endpoint_display(dcgm_url)
            gpus_dict = {}

            for gpu_uuid, gpu_data in gpus_data.items():
                metrics_dict = {}

                for (
                    _metric_display,
                    metric_key,
                    unit_enum,
                ) in get_gpu_telemetry_metrics_config():
                    try:
                        unit = unit_enum.value
                        metric_result = gpu_data.get_metric_result(
                            metric_key, metric_key, metric_key, unit
                        )
                        metrics_dict[metric_key] = metric_result.to_json_result()
                    except Exception:
                        continue

                gpu_summary = GpuSummary(
                    gpu_index=gpu_data.metadata.gpu_index,
                    gpu_name=gpu_data.metadata.model_name,
                    gpu_uuid=gpu_uuid,
                    hostname=gpu_data.metadata.hostname,
                    metrics=metrics_dict,
                )

                gpus_dict[f"gpu_{gpu_data.metadata.gpu_index}"] = gpu_summary

            summary[endpoint_display] = EndpointData(gpus=gpus_dict)

        return summary

    def _generate_server_metrics_statistical_summary(
        self, server_metrics_results
    ) -> dict[str, ServerMetricsEndpointData]:
        """Generate clean statistical summary of server metrics data for JSON export.

        Args:
            server_metrics_results: ServerMetricsSummaryResult containing server metrics data

        Creates hierarchical structure similar to raw snapshots but with aggregated statistics:
        - Endpoints organized by display name
        - Metrics grouped by metric_name (preserving Prometheus naming)
        - Samples contain labels and aggregated statistics (avg, min, max, percentiles)

        Returns:
            dict: Nested structure endpoints -> metrics -> samples with labels and statistics.
                Empty dict if no server metrics data available.
        """
        from collections import defaultdict

        summary = {}

        if not server_metrics_results or not server_metrics_results.server_metrics_data:
            return summary

        # Create a mapping of endpoint display name to endpoint URL
        endpoint_metadata = {}
        for (
            endpoint_url,
            endpoint_data,
        ) in server_metrics_results.server_metrics_data.endpoints.items():
            endpoint_display = endpoint_data.metadata.endpoint_display
            endpoint_metadata[endpoint_display] = endpoint_url

        # Group metrics by endpoint -> metric_name -> labels
        # Structure: endpoint_display -> metric_name -> list of (labels, metric_result)
        endpoint_metrics: dict[str, dict[str, list[tuple[dict, MetricResult]]]] = (
            defaultdict(lambda: defaultdict(list))
        )

        for metric in server_metrics_results.results:
            # Skip if missing required metadata
            if not metric.metric_name or metric.metric_labels is None:
                continue

            # Extract endpoint from header: "metric_name (endpoint_display)"
            if "(" in metric.header and ")" in metric.header:
                endpoint_display = metric.header.split("(")[-1].rstrip(")")
                endpoint_metrics[endpoint_display][metric.metric_name].append(
                    (metric.metric_labels, metric)
                )

        # Build ServerMetricsEndpointData for each endpoint
        for endpoint_display in sorted(endpoint_metrics.keys()):
            if endpoint_display not in endpoint_metadata:
                continue

            endpoint_url = endpoint_metadata[endpoint_display]
            metrics_by_name = endpoint_metrics[endpoint_display]

            # Build AggregatedMetricFamily for each metric
            aggregated_families = {}
            for metric_name, samples_list in metrics_by_name.items():
                # All samples of the same metric should have same type and help
                first_metric = samples_list[0][1]
                metric_type = first_metric.metric_type or ""
                metric_help = first_metric.metric_help or ""
                metric_unit = first_metric.unit

                # Create AggregatedMetricSample for each label combination
                aggregated_samples = []
                for labels, metric_result in samples_list:
                    # Build sample based on metric type
                    if metric_type == "histogram" and metric_result.raw_histogram_delta:
                        # For histograms: use raw bucket deltas
                        buckets_delta, sum_delta, count_delta = (
                            metric_result.raw_histogram_delta
                        )
                        # Add avg for user convenience (sum/count)
                        avg = sum_delta / count_delta if count_delta > 0 else 0.0
                        histogram_data = {
                            "buckets": buckets_delta,
                            "sum": sum_delta,
                            "count": count_delta,
                            "avg": avg,
                        }
                        sample = AggregatedMetricSample(
                            labels=labels, histogram=histogram_data
                        )
                    elif metric_type == "summary" and metric_result.raw_summary_delta:
                        # For summaries: use quantiles and deltas
                        quantiles, sum_delta, count_delta = (
                            metric_result.raw_summary_delta
                        )
                        # Add avg for user convenience (sum/count)
                        avg = sum_delta / count_delta if count_delta > 0 else 0.0
                        summary_data = {
                            "quantiles": quantiles,
                            "sum": sum_delta,
                            "count": count_delta,
                            "avg": avg,
                        }
                        sample = AggregatedMetricSample(
                            labels=labels, summary=summary_data
                        )
                    elif (
                        metric_type == "counter"
                        and metric_result.raw_counter_delta is not None
                    ):
                        # For counters: use raw delta value
                        sample = AggregatedMetricSample(
                            labels=labels, value=metric_result.raw_counter_delta
                        )
                    else:
                        # For gauges: use computed statistics
                        sample = AggregatedMetricSample(
                            labels=labels, statistics=metric_result.to_json_result()
                        )
                    aggregated_samples.append(sample)

                family = AggregatedMetricFamily(
                    type=metric_type,
                    help=metric_help,
                    unit=metric_unit,
                    samples=aggregated_samples,
                )
                aggregated_families[metric_name] = family

            server_endpoint_data = ServerMetricsEndpointData(
                endpoint_url=endpoint_url,
                endpoint_display=endpoint_display,
                metrics=aggregated_families,
            )
            summary[endpoint_display] = server_endpoint_data

        return summary
