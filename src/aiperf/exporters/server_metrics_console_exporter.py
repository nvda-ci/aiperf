# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rich.console import Console, Group, RenderableType
from rich.table import Table
from rich.text import Text

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import PrometheusMetricType, ResultsProcessorType
from aiperf.common.enums.data_exporter_enums import ConsoleExporterType
from aiperf.common.factories import ConsoleExporterFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models.processor_summary_results import ServerMetricsSummaryResult
from aiperf.common.protocols import ConsoleExporterProtocol
from aiperf.exporters.display_units_utils import normalize_endpoint_display
from aiperf.exporters.exporter_config import ExporterConfig


@implements_protocol(ConsoleExporterProtocol)
@ConsoleExporterFactory.register(ConsoleExporterType.SERVER_METRICS)
class ServerMetricsConsoleExporter(AIPerfLoggerMixin):
    """Console exporter for server metrics data from Prometheus endpoints.

    Displays server metrics in a table format showing aggregated statistics
    for auto-discovered metrics across all Prometheus endpoints.
    """

    STAT_COLUMN_KEYS = ["avg", "min", "max", "p99", "p90", "p50", "std"]

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._process_records_result = exporter_config.process_records_result
        self._user_config = exporter_config.user_config
        self._service_config = exporter_config.service_config
        self._exporter_config = exporter_config

    async def export(self, console: Console) -> None:
        """Export server metrics data to console.

        Displays server metrics only if server metrics collection was enabled
        and data is available.

        Args:
            console: Rich Console instance for formatted output
        """
        # Extract server metrics results from summary_results
        server_metrics_results = self._extract_server_metrics_results()
        if not server_metrics_results:
            return

        self._print_renderable(console, self.get_renderable(server_metrics_results))

    def _extract_server_metrics_results(self) -> ServerMetricsSummaryResult | None:
        """Extract server metrics results from summary_results dictionary."""
        summary_results = self._process_records_result.summary_results
        if ResultsProcessorType.SERVER_METRICS_RESULTS in summary_results:
            server_metrics_summary = summary_results[
                ResultsProcessorType.SERVER_METRICS_RESULTS
            ]
            if isinstance(server_metrics_summary, ServerMetricsSummaryResult):
                return server_metrics_summary
        return None

    def _print_renderable(self, console: Console, renderable: RenderableType) -> None:
        """Print the renderable to the console with formatting.

        Adds blank line before output and flushes console buffer after printing.

        Args:
            console: Rich Console instance for formatted output
            renderable: Rich renderable object (Table, Group, Text, etc.) to display
        """
        console.print("\n")
        console.print(renderable)
        console.file.flush()

    def get_renderable(
        self, server_metrics_results: ServerMetricsSummaryResult
    ) -> RenderableType:
        """Create Rich tables showing server metrics with endpoint status.

        Args:
            server_metrics_results: ServerMetricsSummaryResult containing hierarchy and endpoint status

        Generates formatted output with:
        - Summary header showing endpoint reachability status
        - Per-endpoint tables with metrics organized by type
        - Statistical summaries (avg, min, max, p99, p90, p50, std) for each metric
        - Error summary if no data was collected

        MetricResults are generated on-demand from the hierarchy for display.

        Returns:
            RenderableType: Rich Group containing multiple Tables, or Text message if no data
        """
        renderables = []
        first_table = True
        server_metrics_data = server_metrics_results.server_metrics_data

        for _endpoint_url, endpoint_data in server_metrics_data.endpoints.items():
            if not endpoint_data.time_series.snapshots:
                continue

            endpoint_display = endpoint_data.metadata.endpoint_display

            # Discover metrics from the first snapshot
            discovered_metrics = self._discover_metrics_from_endpoint(endpoint_data)
            if not discovered_metrics:
                continue

            # Generate MetricResults on-demand and group by type
            metrics_by_type = {
                PrometheusMetricType.COUNTER: [],
                PrometheusMetricType.GAUGE: [],
                PrometheusMetricType.HISTOGRAM: [],
                PrometheusMetricType.SUMMARY: [],
            }

            for metric_name, metric_type, labels, help_text in discovered_metrics:
                try:
                    # Generate tag and header
                    labels_str = (
                        "_" + "_".join(f"{k}_{v}" for k, v in sorted(labels.items()))
                        if labels
                        else ""
                    )
                    tag = f"server_metrics.{endpoint_display}.{metric_name}{labels_str}"
                    header = f"{metric_name} ({endpoint_display})"

                    # Infer unit from metric name
                    unit = self._infer_unit_from_metric_name(metric_name)

                    # Generate MetricResult on-demand
                    metric_result = endpoint_data.get_metric_result(
                        metric_name=metric_name,
                        labels=labels,
                        tag=tag,
                        header=header,
                        unit=unit,
                    )

                    # Add metadata for display
                    metric_result.metric_name = metric_name
                    metric_result.metric_type = metric_type
                    metric_result.metric_labels = labels
                    metric_result.metric_help = help_text

                    metrics_by_type[metric_type].append(metric_result)
                except Exception as e:
                    self.debug(
                        f"Failed to generate metric result for {metric_name}: {e}"
                    )
                    continue

            table_title_base = f"Server Metrics | {endpoint_display}"

            if first_table:
                first_table = False
                table_title = self._create_summary_header(
                    table_title_base, server_metrics_results
                )
            else:
                renderables.append(Text(""))
                table_title = table_title_base

            # Create table for each metric type that has metrics
            for metric_type, type_metrics in metrics_by_type.items():
                if not type_metrics:
                    continue

                metrics_table = self._create_metrics_table(
                    f"{table_title}\n{metric_type.value.upper()} Metrics",
                    type_metrics,
                    metric_type,
                )
                renderables.append(metrics_table)

        if not renderables:
            return self._create_no_data_message(server_metrics_results)

        return Group(*renderables)

    def _create_summary_header(
        self, table_title_base: str, server_metrics_results: ServerMetricsSummaryResult
    ) -> str:
        """Create the summary header with endpoint reachability status.

        Args:
            table_title_base: Base title for the first table
            server_metrics_results: ServerMetricsSummaryResult containing endpoint status

        Returns:
            Formatted title string with endpoint status
        """
        title_lines = ["NVIDIA AIPerf | Server Metrics Summary"]

        endpoints_tested = server_metrics_results.endpoints_tested
        endpoints_successful = server_metrics_results.endpoints_successful
        total_count = len(endpoints_tested)
        successful_count = len(endpoints_successful)
        failed_count = total_count - successful_count

        if failed_count == 0:
            title_lines.append(
                f"[bold green]{successful_count}/{total_count} Prometheus endpoints reachable[/bold green]"
            )
        elif successful_count == 0:
            title_lines.append(
                f"[bold red]{successful_count}/{total_count} Prometheus endpoints reachable[/bold red]"
            )
        else:
            title_lines.append(
                f"[bold yellow]{successful_count}/{total_count} Prometheus endpoints reachable[/bold yellow]"
            )

        for endpoint in endpoints_tested:
            clean_endpoint = normalize_endpoint_display(endpoint)
            if endpoint in endpoints_successful:
                title_lines.append(f"[green]• {clean_endpoint} \u2714 [/green]")
            else:
                title_lines.append(
                    f"[red]• {clean_endpoint} \u2718 (unreachable)[/red]"
                )

        title_lines.append("")
        title_lines.append(table_title_base)
        return "\n".join(title_lines)

    def _format_number(self, value) -> str:
        """Format a number for console output with adaptive formatting.

        Args:
            value: The value to format

        Returns:
            Formatted string representation of the value
        """
        if value is None:
            return "N/A"

        # Use scientific notation for very large numbers (> 1 million)
        if abs(value) >= 1_000_000:
            return f"{value:.2e}"

        # Use comma-separated format for smaller numbers
        return f"{value:,.2f}"

    def _create_metrics_table(
        self, table_title: str, metrics: list, metric_type: PrometheusMetricType
    ) -> Table:
        """Create a metrics table for a specific metric type.

        Args:
            table_title: Title for the table
            metrics: List of MetricResult objects
            metric_type: Type of Prometheus metric

        Returns:
            Rich Table with server metrics
        """
        metrics_table = Table(show_header=True, title=table_title, title_style="italic")
        metrics_table.add_column("Metric", justify="left", style="cyan")

        # For counters and histograms, show only avg column (they are deltas)
        # For gauges and summaries, show full statistics
        if metric_type in (
            PrometheusMetricType.COUNTER,
            PrometheusMetricType.HISTOGRAM,
        ):
            stat_columns = ["avg"]
        else:
            stat_columns = self.STAT_COLUMN_KEYS

        for stat in stat_columns:
            metrics_table.add_column(stat, justify="right", style="green")

        for metric_result in metrics:
            metric_name = getattr(metric_result, "metric_name", metric_result.header)
            metric_help = getattr(metric_result, "metric_help", "")
            unit = metric_result.unit

            # Format metric name with labels
            labels = getattr(metric_result, "metric_labels", {})
            if labels:
                labels_str = ", ".join(f"{k}={v}" for k, v in sorted(labels.items()))
                display_name = f"{metric_name}{{{labels_str}}}"
            else:
                display_name = metric_name

            # Add help text if available
            if metric_help:
                display_name = f"{display_name}\n[dim]{metric_help}[/dim]"

            # Add unit if available
            if unit:
                display_name = f"{display_name} ({unit})"

            row = [display_name]
            for stat in stat_columns:
                value = getattr(metric_result, stat, None)
                row.append(self._format_number(value))

            metrics_table.add_row(*row)

        return metrics_table

    def _discover_metrics_from_endpoint(
        self, endpoint_data
    ) -> list[tuple[str, str, dict[str, str], str]]:
        """Discover metrics from the first snapshot of an endpoint.

        Args:
            endpoint_data: ServerMetricsData containing snapshots

        Returns:
            List of tuples: (metric_name, metric_type, labels, help_text)
        """
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

    def _create_no_data_message(
        self, server_metrics_results: ServerMetricsSummaryResult
    ) -> Text:
        """Create error message when no server metrics data is available.

        Args:
            server_metrics_results: ServerMetricsSummaryResult containing endpoint status and errors

        Returns:
            Rich Text with error message and endpoint status
        """
        message_parts = [
            "No server metrics data collected during the benchmarking run."
        ]

        endpoints_tested = server_metrics_results.endpoints_tested
        endpoints_successful = server_metrics_results.endpoints_successful
        failed_endpoints = [
            ep for ep in endpoints_tested if ep not in endpoints_successful
        ]

        if failed_endpoints:
            message_parts.append("\n\nUnreachable endpoints:")
            for endpoint in failed_endpoints:
                clean_endpoint = normalize_endpoint_display(endpoint)
                message_parts.append(f"  • {clean_endpoint}")

        if server_metrics_results.error_summary:
            message_parts.append("\n\nErrors encountered:")
            for error_count in server_metrics_results.error_summary:
                error = error_count.error_details
                count = error_count.count
                if count > 1:
                    message_parts.append(f"  • {error.message} ({count} occurrences)")
                else:
                    message_parts.append(f"  • {error.message}")

        return Text("".join(message_parts), style="dim italic")
