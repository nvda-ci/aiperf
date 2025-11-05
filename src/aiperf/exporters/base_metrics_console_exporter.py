# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Any, ClassVar

from rich.console import Console, Group, RenderableType
from rich.table import Table
from rich.text import Text

from aiperf.common.metrics import format_metric_display_name, infer_metric_unit
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.exporters.display_units_utils import normalize_endpoint_display
from aiperf.exporters.exporter_config import ExporterConfig

__all__ = ["BaseMetricsConsoleExporter"]


class BaseMetricsConsoleExporter(AIPerfLoggerMixin):
    """Base class for metrics console exporters (GPU telemetry, server metrics, etc.).

    Provides common functionality for displaying metrics in Rich console format.
    Subclasses implement specific rendering logic for their metrics type by:
    - Setting ClassVars for simple string constants
    - Implementing abstract methods for logic that requires computation

    ClassVars to override:
        SUMMARY_TITLE: Title for summary section (e.g., "GPU Telemetry Summary")
        ENDPOINT_TYPE_NAME: Name for endpoint type (e.g., "DCGM endpoints")
        NO_DATA_TYPE_NAME: Data type name for no-data messages (e.g., "GPU telemetry data")
    """

    # Standard statistical column keys used across all metrics exporters
    STAT_COLUMN_KEYS = ["avg", "min", "max", "p99", "p90", "p50", "std"]

    # Subclasses must override these ClassVars
    SUMMARY_TITLE: ClassVar[str]
    ENDPOINT_TYPE_NAME: ClassVar[str]
    NO_DATA_TYPE_NAME: ClassVar[str]

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._results = exporter_config.results
        self._user_config = exporter_config.user_config
        self._service_config = exporter_config.service_config
        self._exporter_config = exporter_config

    async def export(self, console: Console) -> None:
        """Export metrics data to console if user requested this metrics type.

        Uses template method pattern to check conditions and render metrics.
        Subclasses define what constitutes "enabled" and which results to display.

        Args:
            console: Rich Console instance for formatted output
        """
        if not self._should_export():
            return

        metrics_results = self._get_metrics_results()
        if not metrics_results:
            return

        self._print_renderable(console, self.get_renderable())

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

    @abstractmethod
    def _should_export(self) -> bool:
        """Check if metrics should be exported based on user configuration.

        Returns:
            bool: True if user requested this metrics type (e.g., --gpu-telemetry flag)

        Example:
            return self._user_config.gpu_telemetry is not None
        """
        pass

    @abstractmethod
    def _get_metrics_results(self):
        """Get the metrics results to display.

        Returns:
            Metrics results object, or None if no data available

        Example:
            return self._exporter_config.telemetry_results
        """
        pass

    def get_renderable(self) -> RenderableType:
        """Create Rich tables showing metrics with consolidated single-table format.

        This template method provides common rendering logic for all metrics types.
        Subclasses only need to implement abstract methods for domain-specific details.

        Generates formatted output with:
        - Summary header showing endpoint reachability status
        - Per-resource tables with metrics
        - Statistical summaries (avg, min, max, p99, p90, p75, std) for each metric
        - Error summary if no data was collected

        Returns:
            RenderableType: Rich Group containing multiple Tables, or Text message if no data
        """
        renderables = []
        metrics_data = self._get_metrics_data()
        first_table = True

        endpoints_dict = self._get_endpoints_dict(metrics_data)
        for endpoint_url, resources_data in endpoints_dict.items():
            if not resources_data:
                continue

            endpoint_display = normalize_endpoint_display(endpoint_url)

            for resource_id, resource_data in resources_data.items():
                title_parts = self._extract_title_parts(resource_data)
                table_title_base = f"{endpoint_display} | {title_parts}"

                if first_table:
                    first_table = False
                    table_title = self._create_summary_header(table_title_base)
                else:
                    renderables.append(Text(""))
                    table_title = table_title_base

                metrics_table = self._create_metrics_table(
                    table_title, resource_data, resource_id
                )
                renderables.append(metrics_table)

        if not renderables:
            return self._create_no_data_message()

        return Group(*renderables)

    @abstractmethod
    def _get_metrics_data(self) -> Any:
        """Get the metrics data hierarchy from results.

        Returns:
            Metrics data object (e.g., telemetry_data, metrics_data)

        Example:
            return self._telemetry_results.telemetry_data
        """
        pass

    @abstractmethod
    def _get_endpoints_dict(self, metrics_data: Any) -> dict[str, dict[str, Any]]:
        """Get endpoints dictionary from metrics data.

        Args:
            metrics_data: Metrics data hierarchy

        Returns:
            Dict mapping endpoint URL to resource data dict

        Example:
            return metrics_data.dcgm_endpoints
        """
        pass

    @abstractmethod
    def _extract_title_parts(self, resource_data: Any) -> str:
        """Extract title parts from resource metadata.

        Args:
            resource_data: Resource data containing metadata

        Returns:
            Title parts string (e.g., "GPU 0 | RTX 6000" or "frontend | host1")

        Example:
            gpu_index = resource_data.metadata.gpu_index
            gpu_name = resource_data.metadata.model_name
            return f"GPU {gpu_index} | {gpu_name}"
        """
        pass

    @abstractmethod
    def _get_metrics_config(self) -> list[tuple[str, str, Any]]:
        """Get metrics configuration list.

        Returns:
            List of (metric_display, metric_key, unit_enum) tuples
        """
        pass

    @abstractmethod
    def _format_resource_error(self, metric_key: str, resource_id: str) -> str:
        """Format error message for failed metric retrieval.

        Args:
            metric_key: Metric key that failed
            resource_id: Resource identifier

        Returns:
            Error message string
        """
        pass

    def _create_summary_header(self, table_title_base: str) -> str:
        """Create the summary header with endpoint reachability status.

        Args:
            table_title_base: Base title for the first table

        Returns:
            Formatted title string with endpoint status
        """
        title_lines = [f"NVIDIA AIPerf | {self.SUMMARY_TITLE}"]

        metrics_results = self._get_metrics_results()
        endpoints_configured = metrics_results.endpoints_configured
        endpoints_successful = metrics_results.endpoints_successful
        total_count = len(endpoints_configured)
        successful_count = len(endpoints_successful)
        failed_count = total_count - successful_count

        endpoint_type_name = self.ENDPOINT_TYPE_NAME

        if failed_count == 0:
            title_lines.append(
                f"[bold green]{successful_count}/{total_count} {endpoint_type_name} reachable[/bold green]"
            )
        elif successful_count == 0:
            title_lines.append(
                f"[bold red]{successful_count}/{total_count} {endpoint_type_name} reachable[/bold red]"
            )
        else:
            title_lines.append(
                f"[bold yellow]{successful_count}/{total_count} {endpoint_type_name} reachable[/bold yellow]"
            )

        for endpoint in endpoints_configured:
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

    def _create_metrics_table(
        self, table_title: str, resource_data: Any, resource_id: str
    ) -> Table:
        """Create a metrics table for a single resource with dynamic field discovery.

        Discovers metrics from the actual data instead of using pre-defined config.
        Supports both scalar metrics (gauges, counters) and histogram metrics.

        Args:
            table_title: Title for the table
            resource_data: Resource data containing metrics
            resource_id: Resource identifier for error messages

        Returns:
            Rich Table with resource metrics
        """
        metrics_table = Table(show_header=True, title=table_title, title_style="italic")
        metrics_table.add_column("Metric", justify="right", style="cyan")
        for stat in self.STAT_COLUMN_KEYS:
            metrics_table.add_column(stat, justify="right", style="green")

        # Dynamically discover metrics from resource data
        if (
            hasattr(resource_data, "time_series")
            and resource_data.time_series.snapshots
        ):
            first_snapshot = resource_data.time_series.snapshots[0]

            # Display scalar metrics (gauges, counters)
            field_names = sorted(first_snapshot.metrics.keys())
            for field_name in field_names:
                try:
                    display_name = format_metric_display_name(field_name)
                    unit_enum = infer_metric_unit(field_name)
                    unit = unit_enum.value

                    metric_result = resource_data.get_metric_result(
                        field_name, field_name, display_name, unit
                    )

                    row = [f"{display_name} ({unit})"]
                    for stat in self.STAT_COLUMN_KEYS:
                        value = getattr(metric_result, stat, None)
                        row.append(self._format_number(value))

                    metrics_table.add_row(*row)
                except Exception:
                    self.debug(self._format_resource_error(field_name, resource_id))
                    continue

            # Display histogram metrics (if any)
            if hasattr(first_snapshot, "histograms"):
                histogram_names = sorted(first_snapshot.histograms.keys())
                for histogram_name in histogram_names:
                    try:
                        display_name = format_metric_display_name(histogram_name)
                        unit_enum = infer_metric_unit(histogram_name)
                        unit = unit_enum.value

                        metric_result = (
                            resource_data.time_series.histogram_to_metric_result(
                                histogram_name, histogram_name, display_name, unit
                            )
                        )

                        row = [f"{display_name} ({unit}) [histogram]"]
                        for stat in self.STAT_COLUMN_KEYS:
                            value = getattr(metric_result, stat, None)
                            row.append(self._format_number(value))

                        metrics_table.add_row(*row)
                    except Exception:
                        self.debug(
                            self._format_resource_error(histogram_name, resource_id)
                        )
                        continue

        return metrics_table

    def _create_no_data_message(self) -> Text:
        """Create error message when no metrics data is available.

        Returns:
            Rich Text with error message and endpoint status
        """
        message_parts = [
            f"No {self.NO_DATA_TYPE_NAME} collected during the benchmarking run."
        ]

        metrics_results = self._get_metrics_results()
        endpoints_configured = metrics_results.endpoints_configured
        endpoints_successful = metrics_results.endpoints_successful
        failed_endpoints = [
            ep for ep in endpoints_configured if ep not in endpoints_successful
        ]

        if failed_endpoints:
            message_parts.append("\n\nUnreachable endpoints:")
            for endpoint in failed_endpoints:
                clean_endpoint = normalize_endpoint_display(endpoint)
                message_parts.append(f"  • {clean_endpoint}")

        if metrics_results.error_summary:
            message_parts.append("\n\nErrors encountered:")
            for error_count in metrics_results.error_summary:
                error = error_count.error_details
                count = error_count.count
                if count > 1:
                    message_parts.append(f"  • {error.message} ({count} occurrences)")
                else:
                    message_parts.append(f"  • {error.message}")

        return Text("".join(message_parts), style="dim italic")

    def _format_number(self, value) -> str:
        """Format a number for console output with adaptive formatting.

        This is a shared utility for all metrics console exporters to ensure
        consistent number formatting across GPU telemetry, server metrics, etc.

        Args:
            value: The value to format (int, float, or None)

        Returns:
            Formatted string representation of the value:
            - "N/A" for None values
            - Scientific notation (e.g., "1.23e+06") for values >= 1 million
            - Comma-separated format (e.g., "1,234.56") for smaller values
        """
        if value is None:
            return "N/A"

        # Use scientific notation for very large numbers (> 1 million)
        if abs(value) >= 1_000_000:
            return f"{value:.2e}"

        # Use comma-separated format for smaller numbers
        return f"{value:,.2f}"
