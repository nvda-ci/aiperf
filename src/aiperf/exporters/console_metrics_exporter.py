# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from datetime import datetime

from rich.console import Console, RenderableType
from rich.table import Table

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import MetricFlags, ResultsProcessorType
from aiperf.common.enums.data_exporter_enums import ConsoleExporterType
from aiperf.common.factories import ConsoleExporterFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import MetricResult
from aiperf.common.models.processor_summary_results import (
    MetricSummaryResult,
    TimesliceSummaryResult,
)
from aiperf.common.protocols import ConsoleExporterProtocol
from aiperf.exporters.display_units_utils import to_display_unit
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.metrics.metric_registry import MetricRegistry


@implements_protocol(ConsoleExporterProtocol)
@ConsoleExporterFactory.register(ConsoleExporterType.METRICS)
class ConsoleMetricsExporter(AIPerfLoggerMixin):
    """A class that exports data to the console"""

    STAT_COLUMN_KEYS = ["avg", "min", "max", "p99", "p90", "p50", "std"]

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._process_records_result = exporter_config.process_records_result
        self._endpoint_type = exporter_config.user_config.endpoint.type

    async def export(self, console: Console) -> None:
        # Extract records from summary_results
        records = self._extract_records()
        if not records:
            self.debug("No records to export")
            return

        self._print_renderable(console, self.get_renderable(records, console))

    def _extract_records(self) -> list[MetricResult]:
        """Extract metric records from summary_results dictionary."""
        summary_results = self._process_records_result.summary_results

        # Check for regular metric results
        if ResultsProcessorType.METRIC_RESULTS in summary_results:
            metric_summary = summary_results[ResultsProcessorType.METRIC_RESULTS]
            if isinstance(metric_summary, MetricSummaryResult):
                return metric_summary.results

        # Check for timeslice results
        if ResultsProcessorType.TIMESLICE in summary_results:
            timeslice_summary = summary_results[ResultsProcessorType.TIMESLICE]
            if isinstance(timeslice_summary, TimesliceSummaryResult):
                # Flatten all timeslice results into a single list
                all_results = []
                for timeslice_results in timeslice_summary.timeslice_results.values():
                    all_results.extend(timeslice_results)
                return all_results

        return []

    def _print_renderable(self, console: Console, renderable: RenderableType) -> None:
        console.print("\n")
        console.print(renderable)
        console.file.flush()

    def get_renderable(
        self, records: list[MetricResult], console: Console
    ) -> RenderableType:
        table = Table(title=self._get_title())
        table.add_column("Metric", justify="right", style="cyan")
        for key in self.STAT_COLUMN_KEYS:
            table.add_column(key, justify="right", style="green")
        self._construct_table(table, records)
        return table

    def _construct_table(self, table: Table, records: list[MetricResult]) -> None:
        records = sorted(
            (to_display_unit(r, MetricRegistry) for r in records),
            key=lambda x: MetricRegistry.get_class(x.tag).display_order or sys.maxsize,
        )
        for record in records:
            if not self._should_show(record):
                continue
            table.add_row(*self._format_row(record))

    def _should_show(self, record: MetricResult) -> bool:
        # Only show metrics that are not error-only or hidden
        metric_class = MetricRegistry.get_class(record.tag)
        return metric_class.missing_flags(
            MetricFlags.ERROR_ONLY
            | MetricFlags.NO_CONSOLE
            | MetricFlags.INTERNAL
            | MetricFlags.EXPERIMENTAL
        )

    def _format_row(self, record: MetricResult) -> list[str]:
        metric_class = MetricRegistry.get_class(record.tag)
        display_unit = metric_class.display_unit or metric_class.unit
        delimiter = "\n" if len(record.header) > 30 else " "
        row = [f"{record.header}{delimiter}({display_unit})"]
        for stat in self.STAT_COLUMN_KEYS:
            value = getattr(record, stat, None)
            if value is None:
                row.append("[dim]N/A[/dim]")
                continue

            if isinstance(value, datetime):
                value = value.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(value, int | float):
                value = f"{value:,.2f}"
            else:
                value = str(value)
            row.append(value)
        return row

    def _get_title(self) -> str:
        from aiperf.common.factories import EndpointFactory

        metadata = EndpointFactory.get_metadata(self._endpoint_type)
        return f"NVIDIA AIPerf | {metadata.metrics_title}"
