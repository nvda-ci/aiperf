# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Iterable

import aiofiles

from aiperf.common.enums import MetricFlags, ResultsProcessorType
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import MetricResult
from aiperf.common.models.processor_summary_results import (
    MetricSummaryResult,
    ServerMetricsSummaryResult,
    TelemetrySummaryResult,
    TimesliceSummaryResult,
)
from aiperf.exporters.display_units_utils import convert_all_metrics_to_display_units
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.metrics.metric_registry import MetricRegistry


class MetricsBaseExporter(AIPerfLoggerMixin, ABC):
    """Base class for all metrics exporters with common functionality."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._process_records_result = exporter_config.process_records_result
        self._user_config = exporter_config.user_config
        self._metric_registry = MetricRegistry
        self._output_directory = exporter_config.user_config.output.artifact_directory

    def _get_metric_results(self) -> list[MetricResult]:
        """Extract metric results from summary_results dictionary."""
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

    def _get_telemetry_results(self) -> TelemetrySummaryResult | None:
        """Extract telemetry results from summary_results dictionary."""
        summary_results = self._process_records_result.summary_results
        if ResultsProcessorType.TELEMETRY_RESULTS in summary_results:
            telemetry_summary = summary_results[ResultsProcessorType.TELEMETRY_RESULTS]
            if isinstance(telemetry_summary, TelemetrySummaryResult):
                return telemetry_summary
        return None

    def _get_server_metrics_results(self) -> ServerMetricsSummaryResult | None:
        """Extract server metrics results from summary_results dictionary."""
        summary_results = self._process_records_result.summary_results
        if ResultsProcessorType.SERVER_METRICS_RESULTS in summary_results:
            server_metrics_summary = summary_results[
                ResultsProcessorType.SERVER_METRICS_RESULTS
            ]
            if isinstance(server_metrics_summary, ServerMetricsSummaryResult):
                return server_metrics_summary
        return None

    def _prepare_metrics(
        self, metric_results: Iterable[MetricResult]
    ) -> dict[str, MetricResult]:
        """Convert to display units and filter exportable metrics.

        Args:
            metric_results: Raw metric results to prepare

        Returns:
            dict of filtered and converted metrics ready for export
        """
        converted = convert_all_metrics_to_display_units(
            metric_results, self._metric_registry
        )
        return {
            tag: result
            for tag, result in converted.items()
            if self._should_export(result)
        }

    def _should_export(self, metric: MetricResult) -> bool:
        """Check if a metric should be exported.

        Filters out experimental and internal metrics.

        Args:
            metric: MetricResult to check

        Returns:
            bool: True if metric should be exported
        """
        metric_class = MetricRegistry.get_class(metric.tag)
        res = metric_class.missing_flags(
            MetricFlags.EXPERIMENTAL | MetricFlags.INTERNAL
        )
        self.debug(lambda: f"Metric '{metric.tag}' should be exported: {res}")
        return res

    @abstractmethod
    def _generate_content(self) -> str:
        """Generate export content string.

        Subclasses must implement this to generate format-specific content
        using instance data members (self._results, self._telemetry_results, etc.).

        Returns:
            str: Complete content string ready to write to file
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _generate_content()"
        )

    async def export(self) -> None:
        """Export inference and telemetry data to file.

        Creates output directory, generates content, and writes to file.
        Handles common file writing logic for all exporters.

        Raises:
            Exception: If file writing fails
        """
        self._output_directory.mkdir(parents=True, exist_ok=True)

        self.debug(lambda: f"Exporting data to file: {self._file_path}")

        try:
            content = self._generate_content()

            async with aiofiles.open(
                self._file_path, "w", newline="", encoding="utf-8"
            ) as f:
                await f.write(content)

        except Exception as e:
            self.error(lambda: f"Failed to export to {self._file_path}: {e}")
            raise
