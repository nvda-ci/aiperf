# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import io
import numbers
from decimal import Decimal

from aiperf.common.constants import STAT_KEYS
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import DataExporterType, ResultsProcessorType
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.factories import DataExporterFactory
from aiperf.common.models.processor_summary_results import TimesliceSummaryResult
from aiperf.common.protocols import DataExporterProtocol
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter


@DataExporterFactory.register(DataExporterType.TIMESLICE_CSV)
@implements_protocol(DataExporterProtocol)
class TimesliceMetricsCsvExporter(MetricsBaseExporter):
    """Exports timeslice metrics to a single CSV file in tidy/long format.

    Creates one CSV file with all timeslices in a tidy data format:
        Timeslice,Metric,Unit,Stat,Value
        0,Request Latency,ms,avg,45.2
        0,Request Latency,ms,min,12.1
        1,Request Latency,ms,avg,48.5
        ...

    This format is optimal for data science tools (pandas, R, Tableau, etc.)
    System metrics (with single values) use stat='avg'.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)
        self.debug(
            lambda: f"Initializing TimesliceMetricsCsvExporter with config: {exporter_config}"
        )

        # Check if timeslice results exist in summary_results
        summary_results = self._process_records_result.summary_results
        if ResultsProcessorType.TIMESLICE not in summary_results or not isinstance(
            summary_results[ResultsProcessorType.TIMESLICE], TimesliceSummaryResult
        ):
            raise DataExporterDisabled(
                "TimesliceMetricsCsvExporter disabled: no timeslice metric results found"
            )

        # Extract base filename from configured CSV path
        self._file_path = (
            exporter_config.user_config.output.profile_export_timeslices_csv_file
        )

        self.debug(
            lambda: f"Initialized TimesliceMetricsCsvExporter: file={self._file_path}"
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Timeslice CSV Export",
            file_path=self._file_path,
        )

    def _generate_content(self) -> str:
        """Generate tidy/long format CSV content from all timeslices.

        Extracts timeslice results from summary_results dictionary.

        Returns:
            str: Complete CSV content in tidy format
        """
        buf = io.StringIO()
        writer = csv.writer(buf)

        # Write header with 5 columns
        writer.writerow(["Timeslice", "Metric", "Unit", "Stat", "Value"])

        # Extract timeslice results from summary_results
        summary_results = self._process_records_result.summary_results
        timeslice_summary = summary_results[ResultsProcessorType.TIMESLICE]

        if not isinstance(timeslice_summary, TimesliceSummaryResult):
            return buf.getvalue()

        # Process each timeslice in sorted order
        for timeslice_index in sorted(timeslice_summary.timeslice_results.keys()):
            metric_results_list = timeslice_summary.timeslice_results[timeslice_index]

            # Convert to display units and filter exportable metrics
            prepared_metrics = self._prepare_metrics(metric_results_list)

            # Write rows for each metric
            for tag, metric in sorted(prepared_metrics.items()):
                metric_name = metric.header or tag
                unit = metric.unit or ""

                # Write a row for each stat that has a value
                for stat in STAT_KEYS:
                    value = getattr(metric, stat, None)
                    if value is not None:
                        writer.writerow(
                            [
                                timeslice_index,
                                metric_name,
                                unit,
                                stat,
                                self._format_number(value),
                            ]
                        )

        return buf.getvalue()

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
