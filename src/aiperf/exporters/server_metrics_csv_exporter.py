# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import io
import numbers
from collections import defaultdict
from decimal import Decimal

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import DataExporterType, PrometheusMetricType
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.factories import DataExporterFactory
from aiperf.common.protocols import DataExporterProtocol
from aiperf.exporters.display_units_utils import normalize_endpoint_display
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter

# Stat keys for each metric type (using FlatSeriesStats field names)
GAUGE_STAT_KEYS = ["avg", "min", "max", "std", "p50", "p90", "p95", "p99"]
COUNTER_STAT_KEYS = [
    "delta",
    "rate_per_second",
    "rate_avg",
    "rate_min",
    "rate_max",
    "rate_std",
]
HISTOGRAM_STAT_KEYS = [
    "observation_count",
    "delta",
    "avg",
    "observations_per_second",
    "rate_per_second",
    "p50",
    "p90",
    "p95",
    "p99",
]
SUMMARY_STAT_KEYS = [
    "observation_count",
    "delta",
    "avg",
    "observations_per_second",
    "rate_per_second",
]


@DataExporterFactory.register(DataExporterType.SERVER_METRICS_CSV)
@implements_protocol(DataExporterProtocol)
class ServerMetricsCsvExporter(MetricsBaseExporter):
    """Export server metrics to a separate CSV file organized by metric type.

    Exports server metrics in sections separated by Prometheus metric type
    (gauge, counter, histogram, summary), with appropriate columns for each type.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        # Check if server metrics data is available before initializing
        if not exporter_config.server_metrics_results:
            raise DataExporterDisabled(
                "Server metrics CSV export disabled: no server metrics data available"
            )

        super().__init__(exporter_config, **kwargs)
        self._file_path = (
            exporter_config.user_config.output.server_metrics_export_csv_file
        )
        self.trace_or_debug(
            lambda: f"Initializing ServerMetricsCsvExporter with config: {exporter_config}",
            lambda: f"Initializing ServerMetricsCsvExporter with file path: {self._file_path}",
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Server Metrics CSV Export",
            file_path=self._file_path,
        )

    def _generate_content(self) -> str:
        """Generate CSV content for server metrics data organized by metric type.

        Creates separate sections for each metric type (gauge, counter, histogram,
        summary) with appropriate headers and statistics columns for each type.

        Returns:
            str: CSV content with server metrics organized by type
        """
        if not self._server_metrics_results:
            return ""

        endpoint_summaries = self._server_metrics_results.endpoint_summaries
        if not endpoint_summaries:
            self.warning(
                "No pre-computed server metrics summaries available. "
                "This may indicate a ZMQ serialization issue."
            )
            return ""

        # Group metrics by type across all endpoints
        metrics_by_type = self._group_metrics_by_type(endpoint_summaries)

        buf = io.StringIO()
        writer = csv.writer(buf)

        # Write sections in order: gauge, counter, histogram, summary
        section_order = [
            PrometheusMetricType.GAUGE,
            PrometheusMetricType.COUNTER,
            PrometheusMetricType.HISTOGRAM,
            PrometheusMetricType.SUMMARY,
        ]
        first_section = True

        for metric_type in section_order:
            if metric_type not in metrics_by_type:
                continue

            if not first_section:
                writer.writerow([])

            self._write_section(writer, metric_type, metrics_by_type[metric_type])
            first_section = False

        return buf.getvalue()

    def _group_metrics_by_type(
        self, endpoint_summaries: dict
    ) -> dict[PrometheusMetricType, list[tuple[str, str, str, object]]]:
        """Group all metrics by their type across all endpoints.

        Returns:
            Dict mapping metric type to list of (endpoint, metric_name, description, stats) tuples.
            Stats is a FlatSeriesStats object with all fields including labels.
        """
        from aiperf.common.models.export_data import FlatSeriesStats

        metrics_by_type: dict[
            PrometheusMetricType, list[tuple[str, str, str, FlatSeriesStats]]
        ] = defaultdict(list)

        for endpoint_summary in endpoint_summaries.values():
            endpoint_url = endpoint_summary.endpoint_url
            normalized_endpoint = normalize_endpoint_display(endpoint_url)

            for metric_name, metric_summary in endpoint_summary.metrics.items():
                metric_type = metric_summary.type

                for series_item in metric_summary.series:
                    # series_item is FlatSeriesStats with all stats directly on it
                    metrics_by_type[metric_type].append(
                        (
                            normalized_endpoint,
                            metric_name,
                            metric_summary.description,
                            series_item,
                        )
                    )

        return dict(metrics_by_type)

    def _write_section(
        self,
        writer: csv.writer,
        metric_type: PrometheusMetricType,
        metrics: list[tuple[str, str, str, object]],
    ) -> None:
        """Write a section for a specific metric type.

        Args:
            writer: CSV writer object
            metric_type: Prometheus metric type enum
            metrics: List of (endpoint, metric_name, description, stats) tuples
        """
        # All metric types now use the same simple section format
        # Histograms have buckets in a final JSON column, summaries have quantiles
        self._write_simple_section(writer, metric_type, metrics)

    def _write_simple_section(
        self,
        writer: csv.writer,
        metric_type: PrometheusMetricType,
        metrics: list[tuple[str, str, str, object]],
    ) -> None:
        """Write a section for any metric type.

        For histograms, adds a 'buckets' column at the end with bucket boundary=count pairs.
        For summaries, adds a 'quantiles' column at the end with quantile=value pairs.
        """
        stat_keys = self._get_stat_keys_for_type(metric_type)
        header = ["Endpoint", "Type", "Metric", "Labels"] + stat_keys

        # Add metadata column for histogram/summary
        if metric_type == PrometheusMetricType.HISTOGRAM:
            header.append("buckets")
        elif metric_type == PrometheusMetricType.SUMMARY:
            header.append("quantiles")

        writer.writerow(header)

        # Sort by metric name, endpoint, then labels
        sorted_metrics = sorted(
            metrics,
            key=lambda x: (x[1], x[0], str(x[3].labels) if x[3].labels else ""),
        )

        for endpoint, metric_name, _description, stats in sorted_metrics:
            labels = stats.labels
            labels_str = (
                ";".join(f"{k}={v}" for k, v in sorted(labels.items()))
                if labels
                else ""
            )

            row = [endpoint, metric_type, metric_name, labels_str]

            # Add stat values
            for stat in stat_keys:
                stat_value = getattr(stats, stat, None)
                row.append(self._format_number(stat_value))

            # Add metadata column for histogram/summary (key=value;key2=value2 format)
            if metric_type == PrometheusMetricType.HISTOGRAM:
                buckets = getattr(stats, "buckets", None) or {}
                row.append(
                    ";".join(
                        f"{k}={self._format_number(v)}" for k, v in buckets.items()
                    )
                    if buckets
                    else ""
                )
            elif metric_type == PrometheusMetricType.SUMMARY:
                quantiles = getattr(stats, "quantiles", None) or {}
                row.append(
                    ";".join(
                        f"{k}={self._format_number(v)}" for k, v in quantiles.items()
                    )
                    if quantiles
                    else ""
                )

            writer.writerow(row)

    def _get_stat_keys_for_type(self, metric_type: PrometheusMetricType) -> list[str]:
        """Get the stat keys for a given metric type."""
        stat_keys_map = {
            PrometheusMetricType.GAUGE: GAUGE_STAT_KEYS,
            PrometheusMetricType.COUNTER: COUNTER_STAT_KEYS,
            PrometheusMetricType.HISTOGRAM: HISTOGRAM_STAT_KEYS,
            PrometheusMetricType.SUMMARY: SUMMARY_STAT_KEYS,
        }
        return stat_keys_map.get(metric_type, GAUGE_STAT_KEYS)

    def _format_number(self, value) -> str:
        """Format a number for CSV output."""
        if value is None:
            return ""
        # Handle bools explicitly (bool is a subclass of int)
        if isinstance(value, bool):
            return str(value)
        # Integers
        if isinstance(value, numbers.Integral):
            return f"{int(value)}"
        # Real numbers and Decimal
        if isinstance(value, numbers.Real | Decimal):
            return f"{float(value):.4f}"
        return str(value)
