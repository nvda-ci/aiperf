# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import DataExporterType
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.factories import DataExporterFactory
from aiperf.common.models.export_models import (
    ServerMetricsEndpointSummary,
    ServerMetricsExportData,
    ServerMetricsSummary,
)
from aiperf.common.protocols import DataExporterProtocol
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter


@DataExporterFactory.register(DataExporterType.SERVER_METRICS_JSON)
@implements_protocol(DataExporterProtocol)
class ServerMetricsJsonExporter(MetricsBaseExporter):
    """Export server metrics to a separate JSON file."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        # Check if server metrics data is available before initializing
        if not exporter_config.server_metrics_results:
            raise DataExporterDisabled(
                "Server metrics JSON export disabled: no server metrics data available"
            )

        super().__init__(exporter_config, **kwargs)
        self._file_path = (
            exporter_config.user_config.output.server_metrics_export_json_file
        )
        self.trace_or_debug(
            lambda: f"Initializing ServerMetricsJsonExporter with config: {exporter_config}",
            lambda: f"Initializing ServerMetricsJsonExporter with file path: {self._file_path}",
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Server Metrics JSON Export",
            file_path=self._file_path,
        )

    def _generate_content(self) -> str:
        """Generate JSON content for server metrics data.

        Returns:
            str: JSON content with server metrics data
        """
        if not self._server_metrics_results:
            return "{}"

        summary = ServerMetricsSummary(
            endpoints_configured=self._server_metrics_results.endpoints_configured,
            endpoints_successful=self._server_metrics_results.endpoints_successful,
            start_time=datetime.fromtimestamp(
                self._server_metrics_results.start_ns / NANOS_PER_SECOND
            ),
            end_time=datetime.fromtimestamp(
                self._server_metrics_results.end_ns / NANOS_PER_SECOND
            ),
        )

        export_data = ServerMetricsExportData(
            summary=summary,
            endpoints=self._get_endpoint_summaries(),
        )

        return export_data.model_dump_json(
            indent=2, exclude_unset=True, exclude_none=True
        )

    def _get_endpoint_summaries(self) -> dict[str, ServerMetricsEndpointSummary]:
        """Get pre-computed server metrics summaries.

        Returns:
            dict: Endpoint summaries from pre-computed data
        """
        if not self._server_metrics_results:
            return {}

        if self._server_metrics_results.endpoint_summaries:
            return self._server_metrics_results.endpoint_summaries

        self.warning(
            "No pre-computed server metrics summaries available. "
            "This may indicate a ZMQ serialization issue."
        )
        return {}
