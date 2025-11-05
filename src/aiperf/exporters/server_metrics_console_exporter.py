# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums.data_exporter_enums import ConsoleExporterType
from aiperf.common.factories import ConsoleExporterFactory
from aiperf.common.protocols import ConsoleExporterProtocol
from aiperf.exporters.base_metrics_console_exporter import BaseMetricsConsoleExporter
from aiperf.exporters.exporter_config import ExporterConfig


@implements_protocol(ConsoleExporterProtocol)
@ConsoleExporterFactory.register(ConsoleExporterType.SERVER_METRICS)
class ServerMetricsConsoleExporter(BaseMetricsConsoleExporter):
    """Console exporter for server metrics data.

    Displays server metrics in a table format using dynamic field discovery.
    Metrics are discovered from the actual server response data without requiring
    pre-defined configuration. Only displays when --server-metrics flag is explicitly
    provided by the user.
    """

    # ClassVars for display strings
    SUMMARY_TITLE = "Server Metrics Summary"
    ENDPOINT_TYPE_NAME = "Server endpoints"
    NO_DATA_TYPE_NAME = "server metrics data"

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)
        self._server_metrics_results = exporter_config.server_metrics_results

    def _should_export(self) -> bool:
        """Check if server metrics should be exported."""
        return self._user_config.server_metrics is not None

    def _get_metrics_results(self):
        """Get the server metrics results to display."""
        return self._server_metrics_results

    def _get_metrics_data(self) -> Any:
        """Get the server metrics data hierarchy from results."""
        return self._server_metrics_results.metrics_data

    def _get_endpoints_dict(self, metrics_data: Any) -> dict[str, dict[str, Any]]:
        """Get server endpoints dictionary from metrics data."""
        return metrics_data.server_endpoints

    def _extract_title_parts(self, resource_data: Any) -> str:
        """Extract server title parts from resource metadata."""
        server_type = resource_data.metadata.server_type or "server"
        hostname = resource_data.metadata.hostname or "unknown"
        return f"{server_type} | {hostname}"

    def _get_metrics_config(self) -> list[tuple[str, str, Any]]:
        """Get server metrics configuration.

        This method is not used in this exporter - we override _create_metrics_table
        to dynamically discover metrics for each resource.
        """
        return []

    def _format_resource_error(self, metric_key: str, resource_id: str) -> str:
        """Format error message for failed server metric retrieval."""
        return f"Failed to retrieve metric {metric_key} for server: {resource_id}"
