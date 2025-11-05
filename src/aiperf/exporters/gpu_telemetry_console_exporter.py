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
@ConsoleExporterFactory.register(ConsoleExporterType.TELEMETRY)
class GPUTelemetryConsoleExporter(BaseMetricsConsoleExporter):
    """Console exporter for GPU telemetry data.

    Displays GPU metrics in a table format using dynamic field discovery.
    Metrics are discovered from the actual GPU response data without requiring
    pre-defined configuration. Only displays when --gpu-telemetry flag is explicitly
    provided by the user.
    """

    # ClassVars for display strings
    SUMMARY_TITLE = "GPU Telemetry Summary"
    ENDPOINT_TYPE_NAME = "DCGM endpoints"
    NO_DATA_TYPE_NAME = "GPU telemetry data"

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)
        self._telemetry_results = exporter_config.telemetry_results

    def _should_export(self) -> bool:
        """Check if GPU telemetry should be exported."""
        return self._user_config.gpu_telemetry is not None

    def _get_metrics_results(self):
        """Get the telemetry results to display."""
        return self._telemetry_results

    def _get_metrics_data(self) -> Any:
        """Get the GPU telemetry data hierarchy from results."""
        return self._telemetry_results.telemetry_data

    def _get_endpoints_dict(self, metrics_data: Any) -> dict[str, dict[str, Any]]:
        """Get DCGM endpoints dictionary from telemetry data."""
        return metrics_data.dcgm_endpoints

    def _extract_title_parts(self, resource_data: Any) -> str:
        """Extract GPU title parts from resource metadata."""
        gpu_index = resource_data.metadata.gpu_index
        gpu_name = resource_data.metadata.model_name
        return f"GPU {gpu_index} | {gpu_name}"

    def _get_metrics_config(self) -> list[tuple[str, str, Any]]:
        """Get GPU telemetry metrics configuration.

        This method is not used in this exporter - we override _create_metrics_table
        to dynamically discover metrics for each GPU.
        """
        return []

    def _format_resource_error(self, metric_key: str, resource_id: str) -> str:
        """Format error message for failed GPU metric retrieval."""
        return f"Failed to retrieve metric {metric_key} for GPU {resource_id}"
