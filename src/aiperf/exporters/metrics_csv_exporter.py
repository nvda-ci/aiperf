# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import io
import numbers
from collections.abc import Mapping, Sequence
from decimal import Decimal

from aiperf.common.constants import STAT_KEYS
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums.data_exporter_enums import DataExporterType
from aiperf.common.factories import DataExporterFactory
from aiperf.common.models import MetricResult
from aiperf.common.protocols import DataExporterProtocol
from aiperf.exporters.display_units_utils import normalize_endpoint_display
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter
from aiperf.gpu_telemetry.constants import get_gpu_telemetry_metrics_config


def _percentile_keys_from(stat_keys: Sequence[str]) -> list[str]:
    # e.g., ["avg","min","max","p50","p90","p95","p99"] -> ["p50","p90","p95","p99"]
    return [k for k in stat_keys if len(k) >= 2 and k[0] == "p" and k[1:].isdigit()]


@DataExporterFactory.register(DataExporterType.CSV)
@implements_protocol(DataExporterProtocol)
class MetricsCsvExporter(MetricsBaseExporter):
    """Exports records to a CSV file in a legacy, two-section format."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)
        self.debug(
            lambda: f"Initializing MetricsCsvExporter with config: {exporter_config}"
        )
        self._file_path = exporter_config.user_config.output.profile_export_csv_file
        self._percentile_keys = _percentile_keys_from(STAT_KEYS)

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="CSV Export",
            file_path=self._file_path,
        )

    def _generate_content(self) -> str:
        """Generate CSV content string from inference and telemetry data.

        Uses instance data members from process_records_result.

        Returns:
            str: Complete CSV content with all sections formatted and ready to write
        """
        buf = io.StringIO()
        writer = csv.writer(buf)

        # Extract metric results from summary_results
        metric_results = self._get_metric_results()
        prepared_metrics = self._prepare_metrics(metric_results)

        request_metrics, system_metrics = self._split_metrics(prepared_metrics)

        if request_metrics:
            self._write_request_metrics(writer, request_metrics)
            if system_metrics:  # blank line between sections
                writer.writerow([])

        if system_metrics:
            self._write_system_metrics(writer, system_metrics)

        # Add telemetry data section if available
        telemetry_results = self._get_telemetry_results()
        if telemetry_results:
            self._write_telemetry_section(writer, telemetry_results)

        # Add server metrics section if available
        server_metrics_results = self._get_server_metrics_results()
        if server_metrics_results:
            self._write_server_metrics_section(writer, server_metrics_results)

        return buf.getvalue()

    def _split_metrics(
        self, records: Mapping[str, MetricResult]
    ) -> tuple[dict[str, MetricResult], dict[str, MetricResult]]:
        """Split metrics into request metrics (with percentiles) and system metrics (single values)."""
        request_metrics: dict[str, MetricResult] = {}
        system_metrics: dict[str, MetricResult] = {}

        for tag, metric in records.items():
            if self._has_percentiles(metric):
                request_metrics[tag] = metric
            else:
                system_metrics[tag] = metric

        return request_metrics, system_metrics

    def _has_percentiles(self, metric: MetricResult) -> bool:
        """Check if a metric has any percentile data."""
        return any(getattr(metric, k, None) is not None for k in self._percentile_keys)

    def _write_request_metrics(
        self,
        writer: csv.writer,
        records: Mapping[str, MetricResult],
    ) -> None:
        header = ["Metric"] + list(STAT_KEYS)
        writer.writerow(header)

        for _, metric in sorted(records.items(), key=lambda kv: kv[0]):
            if not self._should_export(metric):
                continue
            row = [self._format_metric_name(metric)]
            for stat_name in STAT_KEYS:
                value = getattr(metric, stat_name, None)
                row.append(self._format_number(value))
            writer.writerow(row)

    def _write_system_metrics(
        self,
        writer: csv.writer,
        records: Mapping[str, MetricResult],
    ) -> None:
        writer.writerow(["Metric", "Value"])
        for _, metric in sorted(records.items(), key=lambda kv: kv[0]):
            if not self._should_export(metric):
                continue
            writer.writerow(
                [self._format_metric_name(metric), self._format_number(metric.avg)]
            )

    def _format_metric_name(self, metric: MetricResult) -> str:
        """Format metric name with its unit."""
        name = metric.header or ""
        if metric.unit and metric.unit.lower() not in {"count", "requests"}:
            name = f"{name} ({metric.unit})" if name else f"({metric.unit})"
        return name

    def _format_number(self, value) -> str:
        """Format a number for CSV output."""
        if value is None:
            return ""
        # Handle bools explicitly (bool is a subclass of int)
        if isinstance(value, bool):
            return str(value)
        # Integers (covers built-in int and other Integral implementations)
        if isinstance(value, numbers.Integral):
            return f"{int(value)}"
        # Real numbers (covers built-in float and many Real implementations) and Decimal
        if isinstance(value, numbers.Real | Decimal):
            return f"{float(value):.2f}"

        return str(value)

    def _write_telemetry_section(self, writer: csv.writer, telemetry_results) -> None:
        """Write GPU telemetry data section to CSV in structured table format.

        Args:
            writer: CSV writer object
            telemetry_results: TelemetrySummaryResult containing telemetry data

        Creates a single flat table with all GPU telemetry metrics that's easy to
        parse programmatically for visualization platforms (pandas, Tableau, Excel, etc.).

        Each row represents one metric for one GPU with all statistics in columns.
        """

        writer.writerow([])
        writer.writerow([])

        # Write header row for GPU telemetry table
        header_row = [
            "Endpoint",
            "GPU_Index",
            "GPU_Name",
            "GPU_UUID",
            "Metric",
        ]
        header_row.extend(STAT_KEYS)
        writer.writerow(header_row)

        for (
            dcgm_url,
            gpus_data,
        ) in telemetry_results.telemetry_data.dcgm_endpoints.items():
            if not gpus_data:
                continue

            endpoint_display = normalize_endpoint_display(dcgm_url)

            for gpu_uuid, gpu_data in gpus_data.items():
                for (
                    metric_display,
                    metric_key,
                    unit_enum,
                ) in get_gpu_telemetry_metrics_config():
                    if not self._gpu_has_metric(gpu_data, metric_key):
                        continue

                    self._write_gpu_metric_row_structured(
                        writer,
                        endpoint_display,
                        gpu_data,
                        gpu_uuid,
                        metric_key,
                        metric_display,
                        unit_enum.value,
                    )

    def _write_gpu_metric_row_structured(
        self,
        writer,
        endpoint_display,
        gpu_data,
        gpu_uuid,
        metric_key,
        metric_display,
        unit,
    ):
        """Write a single GPU metric row in structured table format.

        Each row contains: endpoint, GPU info, metric name with unit, and all stats.
        This format is optimized for programmatic extraction and visualization.

        Args:
            writer: CSV writer object
            endpoint_display: Display name of the DCGM endpoint
            gpu_data: GpuTelemetryData containing metric time series
            gpu_uuid: UUID identifier for the GPU
            metric_key: Internal metric name (e.g., "gpu_power_usage")
            metric_display: Display name for the metric (e.g., "GPU Power Usage")
            unit: Unit of measurement (e.g., "W", "GB", "%")
        """
        try:
            metric_result = gpu_data.get_metric_result(
                metric_key, metric_key, metric_display, unit
            )

            # Format metric name with unit like inference metrics
            metric_with_unit = f"{metric_display} ({unit})"

            row = [
                endpoint_display,
                str(gpu_data.metadata.gpu_index),
                gpu_data.metadata.model_name,
                gpu_uuid,
                metric_with_unit,
            ]

            for stat in STAT_KEYS:
                value = getattr(metric_result, stat, None)
                row.append(self._format_number(value))

            writer.writerow(row)
        except Exception as e:
            self.warning(
                f"Failed to write metric row for GPU {gpu_uuid}, metric {metric_key}: {e}"
            )

    def _gpu_has_metric(self, gpu_data, metric_key: str) -> bool:
        """Check if GPU has data for the specified metric.

        Attempts to retrieve metric result to determine if the metric has any data.
        Used to filter out metrics with no collected data.

        Args:
            gpu_data: GpuTelemetryData containing metric time series
            metric_key: Internal metric name to check (e.g., "gpu_power_usage")

        Returns:
            bool: True if metric has data, False if metric is unavailable or has no data
        """
        try:
            gpu_data.get_metric_result(metric_key, metric_key, "test", "test")
            return True
        except Exception as e:
            self.debug(lambda err=e: f"GPU metric {metric_key} not available: {err}")
            return False

    def _write_server_metrics_section(
        self, writer: csv.writer, server_metrics_results
    ) -> None:
        """Write server metrics data section to CSV in structured table format.

        Args:
            writer: CSV writer object
            server_metrics_results: ServerMetricsSummaryResult containing server metrics data

        Generates MetricResults on-demand from the hierarchy and exports only non-histogram
        metrics. Histogram metrics are skipped for CSV export.
        """
        from aiperf.common.enums import PrometheusMetricType

        writer.writerow([])
        writer.writerow([])

        # Generate MetricResults on-demand, filtering out histograms
        non_histogram_metrics = []

        for (
            _endpoint_url,
            endpoint_data,
        ) in server_metrics_results.server_metrics_data.endpoints.items():
            if not endpoint_data.time_series.snapshots:
                continue

            endpoint_display = endpoint_data.metadata.endpoint_display

            # Discover metrics from the first snapshot
            discovered_metrics = self._discover_metrics_from_endpoint(endpoint_data)

            for metric_name, metric_type, labels, _help_text in discovered_metrics:
                # Skip histograms for CSV export
                if metric_type == PrometheusMetricType.HISTOGRAM:
                    continue

                try:
                    # Infer unit from metric name
                    unit = self._infer_unit_from_metric_name(metric_name)

                    # Create tag and header
                    labels_str = (
                        "_" + "_".join(f"{k}_{v}" for k, v in sorted(labels.items()))
                        if labels
                        else ""
                    )
                    tag = f"server_metrics.{endpoint_display}.{metric_name}{labels_str}"
                    header = f"{metric_name} ({endpoint_display})"

                    # Generate MetricResult on-demand
                    metric_result = endpoint_data.get_metric_result(
                        metric_name=metric_name,
                        labels=labels,
                        tag=tag,
                        header=header,
                        unit=unit,
                    )

                    non_histogram_metrics.append(metric_result)
                except Exception as e:
                    self.debug(
                        lambda err=e,
                        name=metric_name: f"Failed to generate metric result for {name}: {err}"
                    )
                    continue

        if not non_histogram_metrics:
            return

        # Write header row for server metrics table
        header_row = ["Metric"]
        header_row.extend(STAT_KEYS)
        writer.writerow(header_row)

        # Write rows for non-histogram metrics
        for metric in non_histogram_metrics:
            try:
                metric_with_unit = self._format_metric_name(metric)
                row = [metric_with_unit]

                for stat in STAT_KEYS:
                    value = getattr(metric, stat, None)
                    row.append(self._format_number(value))

                writer.writerow(row)
            except Exception as e:
                self.warning(f"Failed to write metric row for metric {metric.tag}: {e}")

    def _discover_metrics_from_endpoint(
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
