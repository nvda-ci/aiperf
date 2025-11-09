# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rich.console import Console
from rich.panel import Panel

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ConsoleExporterType, ResultsProcessorType
from aiperf.common.environment import Environment
from aiperf.common.factories import ConsoleExporterFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import MetricResult
from aiperf.common.models.processor_summary_results import (
    MetricSummaryResult,
    TimesliceSummaryResult,
)
from aiperf.common.protocols import ConsoleExporterProtocol
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.usage_diff_metrics import UsageDiscrepancyCountMetric


@implements_protocol(ConsoleExporterProtocol)
@ConsoleExporterFactory.register(ConsoleExporterType.USAGE_DISCREPANCY_WARNING)
class ConsoleUsageDiscrepancyExporter(AIPerfLoggerMixin):
    """Display warning panel when API usage tokens differ significantly from client token counts.

    This exporter checks if any requests have token count discrepancies exceeding the
    configured threshold and displays a prominent warning panel with:
    - Number and percentage of affected requests
    - Possible causes of discrepancies
    - Investigation steps for users
    - Configuration information
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._process_records_result = exporter_config.process_records_result
        self._threshold = Environment.METRICS.USAGE_PCT_DIFF_THRESHOLD

    async def export(self, console: Console) -> None:
        """Export usage discrepancy warning to console if discrepancies detected."""
        metric = self._get_discrepancy_metric()
        if not metric or not metric.avg or metric.avg <= 0:
            self.debug(
                "No usage discrepancies detected, skipping token count discrepancy warning"
            )
            return

        discrepancy_count = int(metric.avg)
        total_records = self._get_total_records()
        if not total_records:
            self.debug(
                "No valid records detected, skipping token count discrepancy warning"
            )
            return
        percentage = (discrepancy_count / total_records) * 100

        panel = Panel(
            self._create_warning_text(discrepancy_count, total_records, percentage),
            title="Token Count Discrepancy Warning",
            border_style="bold yellow",
            title_align="center",
            padding=(0, 2),
            expand=False,
        )

        console.print()
        console.print(panel)
        console.file.flush()

    def _get_discrepancy_metric(self) -> MetricResult | None:
        """Extract the discrepancy metric from results."""
        records = self._extract_records()
        return next(
            (r for r in records if r.tag == UsageDiscrepancyCountMetric.tag),
            None,
        )

    def _get_total_records(self) -> int:
        """Get the total number of valid records from results."""
        records = self._extract_records()
        return int(
            next(
                (r.avg for r in records if r.tag == RequestCountMetric.tag),
                0,
            )
        )

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

    def _create_warning_text(
        self, discrepancy_count: int, total_records: int, percentage: float
    ) -> str:
        """Create the formatted warning text with details and recommendations."""
        return f"""\
[bold]{discrepancy_count:,} of {total_records:,} requests ({percentage:.1f}%) show a difference exceeding {self._threshold:g}% between:[/bold]
  • API-reported usage tokens (from 'usage' field)
  • Client-computed token counts (from tokenization)

[bold]Possible Causes:[/bold]
  • Different tokenization methods (API vs client)
  • API special tokens or preprocessing

[bold]Investigation Steps:[/bold]
  1. Review [cyan]profile_export.jsonl[/cyan] for per-request [cyan]usage_*_diff_pct[/cyan] values
  2. Verify client tokenizer matches the model's tokenizer
  4. Adjust threshold: [green]AIPERF_METRICS_USAGE_PCT_DIFF_THRESHOLD={self._threshold:g}[/green]\
"""
