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

        Generates MetricResults on-demand from the hierarchy for each metric/label combination.

        Returns:
            dict: Nested structure endpoints -> metrics -> samples with labels and statistics.
                Empty dict if no server metrics data available.
        """
        from aiperf.common.enums import PrometheusMetricType

        summary = {}

        if not server_metrics_results or not server_metrics_results.server_metrics_data:
            return summary

        for (
            endpoint_url,
            endpoint_data,
        ) in server_metrics_results.server_metrics_data.endpoints.items():
            if not endpoint_data.time_series.snapshots:
                continue

            endpoint_display = endpoint_data.metadata.endpoint_display

            # Discover metrics from the first snapshot
            discovered_metrics = self._discover_metrics_from_endpoint_data(
                endpoint_data
            )

            # Build AggregatedMetricFamily for each metric
            aggregated_families = {}
            for metric_name, metric_type, labels, help_text in discovered_metrics:
                # Generate MetricResult on-demand
                try:
                    # Infer unit from metric name
                    unit = self._infer_unit_from_metric_name(metric_name)

                    # Create tag and header (consistent with console exporter)
                    labels_str = (
                        "_" + "_".join(f"{k}_{v}" for k, v in sorted(labels.items()))
                        if labels
                        else ""
                    )
                    tag = f"server_metrics.{endpoint_display}.{metric_name}{labels_str}"
                    header = f"{metric_name} ({endpoint_display})"

                    # Generate MetricResult on-demand from hierarchy
                    metric_result = endpoint_data.get_metric_result(
                        metric_name=metric_name,
                        labels=labels,
                        tag=tag,
                        header=header,
                        unit=unit,
                    )

                    # Initialize metric family if not exists
                    if metric_name not in aggregated_families:
                        aggregated_families[metric_name] = {
                            "type": metric_type,
                            "help": help_text,
                            "unit": unit,
                            "samples": [],
                        }

                    # Create AggregatedMetricSample based on metric type
                    if metric_type == PrometheusMetricType.HISTOGRAM and hasattr(
                        metric_result, "raw_histogram_delta"
                    ):
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
                    elif metric_type == PrometheusMetricType.SUMMARY and hasattr(
                        metric_result, "raw_summary_delta"
                    ):
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
                    elif metric_type == PrometheusMetricType.COUNTER and hasattr(
                        metric_result, "raw_counter_delta"
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

                    aggregated_families[metric_name]["samples"].append(sample)

                except Exception as e:
                    self.debug(
                        lambda err=e,
                        name=metric_name: f"Failed to generate metric result for {name}: {err}"
                    )
                    continue

            # Convert dict format to AggregatedMetricFamily objects
            final_families = {}
            for metric_name, family_data in aggregated_families.items():
                family = AggregatedMetricFamily(
                    type=family_data["type"],
                    help=family_data["help"],
                    unit=family_data["unit"],
                    samples=family_data["samples"],
                )
                final_families[metric_name] = family

            # Convert KubernetesPodInfo to dict for JSON serialization
            k8s_pod_info_dict = None
            if endpoint_data.metadata.kubernetes_pod_info:
                k8s_info = endpoint_data.metadata.kubernetes_pod_info
                k8s_pod_info_dict = {
                    "pod_name": k8s_info.pod_name,
                    "namespace": k8s_info.namespace,
                    "node_name": k8s_info.node_name,
                    "container_name": k8s_info.container_name,
                    "service_name": k8s_info.service_name,
                    "pod_ip": k8s_info.pod_ip,
                    "labels": k8s_info.labels,
                }

            server_endpoint_data = ServerMetricsEndpointData(
                endpoint_url=endpoint_url,
                kubernetes_pod_info=k8s_pod_info_dict,
                metrics=final_families,
            )
            summary[endpoint_display] = server_endpoint_data

        return summary

    def _discover_metrics_from_endpoint_data(
        self, endpoint_data
    ) -> list[tuple[str, str, dict[str, str], str]]:
        """Discover metrics from the first snapshot of an endpoint.

        Args:
            endpoint_data: ServerMetricsData containing snapshots

        Returns:
            List of tuples: (metric_name, metric_type, labels, help_text)
        """
        from aiperf.common.enums import PrometheusMetricType

        discovered = []

        if not endpoint_data.time_series.snapshots:
            return discovered

        # Use first snapshot to discover metrics
        _, first_metrics = endpoint_data.time_series.snapshots[0]

        for metric_name, metric_family in first_metrics.items():
            # Only include counters, gauges, histograms, and summaries
            if metric_family.type not in (
                PrometheusMetricType.COUNTER,
                PrometheusMetricType.GAUGE,
                PrometheusMetricType.HISTOGRAM,
                PrometheusMetricType.SUMMARY,
            ):
                continue

            help_text = metric_family.help or ""

            for sample in metric_family.samples:
                # Check if sample has data
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
                    discovered.append(
                        (metric_name, metric_family.type, sample.labels, help_text)
                    )

        return discovered

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
