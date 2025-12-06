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

# Stat keys for each metric type
GAUGE_STAT_KEYS = ["avg", "min", "max", "std", "p50", "p90", "p95", "p99"]
COUNTER_STAT_KEYS = [
    "delta",
    "rate_overall",
    "rate_avg",
    "rate_min",
    "rate_max",
    "rate_std",
]
# Histogram base stats (before bucket columns)
HISTOGRAM_BASE_STAT_KEYS = ["count_delta", "sum_delta", "avg", "count_rate", "sum_rate"]
# Histogram percentile columns (nested under percentiles.bucket, percentiles.observed, percentiles.best_guess)
HISTOGRAM_PERCENTILE_KEYS = [
    "bucket.p50",
    "bucket.p90",
    "bucket.p95",
    "bucket.p99",
    "observed.p50",
    "observed.p90",
    "observed.p95",
    "observed.p99",
    "observed.exact_count",
    "observed.bucket_placed_count",
    "observed.coverage",
    "best_guess.p50",
    "best_guess.p90",
    "best_guess.p95",
    "best_guess.p99",
    "best_guess.p999",
    "best_guess.inf_bucket_count",
    "best_guess.inf_bucket_estimated_mean",
    "best_guess.estimation_confidence",
]
# Combined histogram stat keys for CSV export
HISTOGRAM_STAT_KEYS = HISTOGRAM_BASE_STAT_KEYS + HISTOGRAM_PERCENTILE_KEYS
SUMMARY_STAT_KEYS = ["count_delta", "sum_delta", "avg", "count_rate", "sum_rate"]


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
        section_order = ["gauge", "counter", "histogram", "summary"]
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
    ) -> dict[str, list[tuple[str, str, str, dict | None, object]]]:
        """Group all metrics by their type across all endpoints.

        Returns:
            Dict mapping metric type to list of (endpoint, metric_name, description, labels, stats) tuples
        """
        metrics_by_type: dict[str, list[tuple[str, str, str, dict | None, object]]] = (
            defaultdict(list)
        )

        for endpoint_summary in endpoint_summaries.values():
            endpoint_url = endpoint_summary.endpoint_url
            normalized_endpoint = normalize_endpoint_display(endpoint_url)

            for metric_name, metric_summary in endpoint_summary.metrics.items():
                metric_type = metric_summary.type.lower()

                for series_item in metric_summary.series:
                    metrics_by_type[metric_type].append(
                        (
                            normalized_endpoint,
                            metric_name,
                            metric_summary.description,
                            series_item.labels,
                            series_item.stats,
                            series_item.value,  # For constant gauges
                            series_item.count_delta,  # For empty histograms/summaries
                        )
                    )

        return dict(metrics_by_type)

    def _write_section(
        self,
        writer: csv.writer,
        metric_type: str,
        metrics: list[
            tuple[str, str, str, dict | None, object | None, float | None, float | None]
        ],
    ) -> None:
        """Write a section for a specific metric type.

        Args:
            writer: CSV writer object
            metric_type: Type of metrics (gauge, counter, histogram, summary)
            metrics: List of (endpoint, metric_name, description, labels, stats, value, count_delta) tuples
        """
        # For histogram and summary, group by bucket/quantile keys and write sub-sections
        if metric_type == PrometheusMetricType.HISTOGRAM:
            self._write_histogram_sections(writer, metrics)
        elif metric_type == PrometheusMetricType.SUMMARY:
            self._write_summary_sections(writer, metrics)
        else:
            self._write_simple_section(writer, metric_type, metrics)

    def _write_simple_section(
        self,
        writer: csv.writer,
        metric_type: str,
        metrics: list[
            tuple[str, str, str, dict | None, object | None, float | None, float | None]
        ],
    ) -> None:
        """Write a simple section for gauge or counter metrics."""
        stat_keys = self._get_stat_keys_for_type(metric_type)
        header = ["Endpoint", "Type", "Metric", "Labels"] + stat_keys
        writer.writerow(header)

        sorted_metrics = sorted(
            metrics, key=lambda x: (x[1], x[0], str(x[3]) if x[3] else "")
        )

        for (
            endpoint,
            metric_name,
            _description,
            labels,
            stats,
            value,
            _count_delta,
        ) in sorted_metrics:
            self._write_metric_row(
                writer,
                metric_type,
                endpoint,
                metric_name,
                labels,
                stats,
                value,
                stat_keys,
            )

    def _write_histogram_sections(
        self,
        writer: csv.writer,
        metrics: list[
            tuple[str, str, str, dict | None, object | None, float | None, float | None]
        ],
    ) -> None:
        """Write histogram sections grouped by bucket boundaries."""
        # Group metrics by their bucket keys
        grouped = self._group_by_keys(metrics, "buckets")

        first_group = True
        for bucket_keys, group_metrics in sorted(grouped.items(), key=lambda x: x[0]):
            if not first_group:
                writer.writerow([])

            stat_keys = HISTOGRAM_STAT_KEYS
            header = (
                ["Endpoint", "Type", "Metric", "Labels"] + stat_keys + list(bucket_keys)
            )
            writer.writerow(header)

            sorted_metrics = sorted(
                group_metrics, key=lambda x: (x[1], x[0], str(x[3]) if x[3] else "")
            )

            for (
                endpoint,
                metric_name,
                _description,
                labels,
                stats,
                _value,
                _count_delta,
            ) in sorted_metrics:
                self._write_metric_row_with_dict(
                    writer,
                    "histogram",
                    endpoint,
                    metric_name,
                    labels,
                    stats,
                    stat_keys,
                    "buckets",
                )

            first_group = False

    def _write_summary_sections(
        self,
        writer: csv.writer,
        metrics: list[
            tuple[str, str, str, dict | None, object | None, float | None, float | None]
        ],
    ) -> None:
        """Write summary sections grouped by quantile keys."""
        # Group metrics by their quantile keys
        grouped = self._group_by_keys(metrics, "quantiles")

        first_group = True
        for quantile_keys, group_metrics in sorted(grouped.items(), key=lambda x: x[0]):
            if not first_group:
                writer.writerow([])

            stat_keys = SUMMARY_STAT_KEYS
            header = (
                ["Endpoint", "Type", "Metric", "Labels"]
                + stat_keys
                + list(quantile_keys)
            )
            writer.writerow(header)

            sorted_metrics = sorted(
                group_metrics, key=lambda x: (x[1], x[0], str(x[3]) if x[3] else "")
            )

            for (
                endpoint,
                metric_name,
                _description,
                labels,
                stats,
                _value,
                _count_delta,
            ) in sorted_metrics:
                self._write_metric_row_with_dict(
                    writer,
                    "summary",
                    endpoint,
                    metric_name,
                    labels,
                    stats,
                    stat_keys,
                    "quantiles",
                )

            first_group = False

    def _group_by_keys(
        self,
        metrics: list[
            tuple[str, str, str, dict | None, object | None, float | None, float | None]
        ],
        dict_attr: str,
    ) -> dict[
        tuple,
        list[
            tuple[str, str, str, dict | None, object | None, float | None, float | None]
        ],
    ]:
        """Group metrics by the keys of a dict attribute (buckets or quantiles)."""
        grouped: dict[
            tuple,
            list[
                tuple[
                    str,
                    str,
                    str,
                    dict | None,
                    object | None,
                    float | None,
                    float | None,
                ]
            ],
        ] = defaultdict(list)

        for metric in metrics:
            stats = metric[4]
            d = getattr(stats, dict_attr, None) or {}
            keys = tuple(d.keys())
            grouped[keys].append(metric)

        return dict(grouped)

    def _get_stat_keys_for_type(self, metric_type: str) -> list[str]:
        """Get the stat keys for a given metric type."""
        stat_keys_map = {
            "gauge": GAUGE_STAT_KEYS,
            "counter": COUNTER_STAT_KEYS,
            "histogram": HISTOGRAM_STAT_KEYS,
            "summary": SUMMARY_STAT_KEYS,
        }
        return stat_keys_map.get(metric_type, GAUGE_STAT_KEYS)

    def _write_metric_row(
        self,
        writer: csv.writer,
        metric_type: str,
        endpoint: str,
        metric_name: str,
        labels: dict | None,
        stats: object | None,
        constant_value: float | None,
        stat_keys: list[str],
    ) -> None:
        """Write a single metric row for gauge/counter metrics.

        Args:
            writer: CSV writer object
            metric_type: Type of metric (gauge, counter)
            endpoint: Normalized endpoint display name
            metric_name: Name of the metric
            labels: Label dict or None
            stats: Stats object with values, or None for constant metrics
            constant_value: Constant value for metrics that didn't change
            stat_keys: List of stat keys for this metric type
        """
        labels_str = (
            ",".join(f"{k}={v}" for k, v in sorted(labels.items())) if labels else ""
        )

        row = [endpoint, metric_type, metric_name, labels_str]

        # For constant gauges (stats is None, value is set), synthesize the row
        # All stat values equal the constant value, with std=0
        if (
            stats is None
            and constant_value is not None
            and metric_type == PrometheusMetricType.GAUGE
        ):
            for stat in stat_keys:
                if stat == "std":
                    row.append(self._format_number(0.0))
                else:
                    row.append(self._format_number(constant_value))
        else:
            for stat in stat_keys:
                stat_value = getattr(stats, stat, None) if stats else None
                row.append(self._format_number(stat_value))

        writer.writerow(row)

    def _write_metric_row_with_dict(
        self,
        writer: csv.writer,
        metric_type: str,
        endpoint: str,
        metric_name: str,
        labels: dict | None,
        stats: object,
        stat_keys: list[str],
        dict_attr: str,
    ) -> None:
        """Write a metric row with bucket/quantile values as columns.

        Args:
            writer: CSV writer object
            metric_type: Type of metric (histogram, summary)
            endpoint: Normalized endpoint display name
            metric_name: Name of the metric
            labels: Label dict or None
            stats: Stats object with values
            stat_keys: List of stat keys for this metric type
            dict_attr: Attribute name for dict values ('buckets' or 'quantiles')
        """
        labels_str = (
            ",".join(f"{k}={v}" for k, v in sorted(labels.items())) if labels else ""
        )

        row = [endpoint, metric_type, metric_name, labels_str]

        for stat in stat_keys:
            value = self._get_stat_value(stats, stat)
            row.append(self._format_number(value))

        # Add bucket/quantile values as separate columns
        d = getattr(stats, dict_attr, None) or {}
        for value in d.values():
            row.append(self._format_number(value))

        writer.writerow(row)

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

    def _get_stat_value(self, stats: object, stat_key: str) -> object:
        """Get a stat value, handling nested paths for percentile fields.

        For keys like "bucket.p50", accesses stats.percentiles.bucket.p50.
        For simple keys like "avg", accesses stats.avg.

        Args:
            stats: Stats object with values
            stat_key: Stat key, possibly with dot notation for nested access

        Returns:
            The stat value or None if not found
        """
        if "." in stat_key:
            # Nested path like "bucket.p50" -> stats.percentiles.bucket.p50
            parts = stat_key.split(".")
            # First get the percentiles object
            percentiles = getattr(stats, "percentiles", None)
            if percentiles is None:
                return None
            # Then traverse the nested path
            obj = percentiles
            for part in parts:
                obj = getattr(obj, part, None)
                if obj is None:
                    return None
            return obj
        else:
            # Simple attribute access
            return getattr(stats, stat_key, None)
