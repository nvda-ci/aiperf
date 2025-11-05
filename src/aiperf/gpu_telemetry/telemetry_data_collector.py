# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.environment import Environment
from aiperf.common.metrics.base_metrics_data_collector import BaseMetricsDataCollector
from aiperf.common.models import TelemetryMetrics, TelemetryRecord
from aiperf.gpu_telemetry.constants import (
    DCGM_TO_FIELD_MAPPING,
    SCALING_FACTORS,
)

__all__ = ["TelemetryDataCollector"]


class TelemetryDataCollector(BaseMetricsDataCollector[TelemetryRecord]):
    """Collects telemetry metrics from DCGM metrics endpoint using async architecture.

    Extends BaseMetricsDataCollector to provide DCGM-specific metric parsing and
    record creation. Fetches GPU metrics from DCGM exporter and converts them to
    TelemetryRecord objects.

    Args:
        endpoint_url: URL of the DCGM metrics endpoint (e.g., "http://localhost:9400/metrics")
        collection_interval: Interval in seconds between metric collections (default: 1.0)
        record_callback: Optional async callback to receive collected records.
            Signature: async (records: list[TelemetryRecord], collector_id: str) -> None
        error_callback: Optional async callback to receive collection errors.
            Signature: async (error: ErrorDetails, collector_id: str) -> None
        collector_id: Unique identifier for this collector instance
    """

    FIELD_MAPPING = DCGM_TO_FIELD_MAPPING
    SCALING_FACTORS = SCALING_FACTORS
    DEFAULT_COLLECTOR_ID = "telemetry_collector"
    DEFAULT_COLLECTION_INTERVAL = Environment.GPU.COLLECTION_INTERVAL
    REACHABILITY_TIMEOUT = Environment.GPU.REACHABILITY_TIMEOUT

    def _extract_resource_info(
        self, labels: dict[str, str]
    ) -> tuple[int | None, dict[str, str]]:
        """Extract GPU index and metadata from DCGM labels.

        Args:
            labels: Prometheus metric labels from DCGM exporter

        Returns:
            tuple: (gpu_index, metadata_dict) or (None, {}) to skip this metric
        """
        gpu_index = labels.get("gpu")
        if gpu_index is not None:
            try:
                gpu_index = int(gpu_index)
            except ValueError:
                return None, {}
        else:
            return None, {}

        metadata = {
            "model_name": labels.get("modelName"),
            "uuid": labels.get("UUID"),
            "pci_bus_id": labels.get("pci_bus_id"),
            "device": labels.get("device"),
            "hostname": labels.get("Hostname"),
        }

        return gpu_index, metadata

    def _create_records(
        self,
        resource_data: dict[int, dict[str, float]],
        resource_metadata: dict[int, dict[str, str]],
        timestamp_ns: int,
        histogram_data: dict[int, dict[str, dict[str, any]]] | None = None,
    ) -> list[TelemetryRecord]:
        """Create TelemetryRecord objects from parsed GPU metrics.

        Args:
            resource_data: Dict mapping gpu_index -> {field_name: value}
            resource_metadata: Dict mapping gpu_index -> metadata
            timestamp_ns: Timestamp when metrics were collected
            histogram_data: Optional histogram data (not used for DCGM telemetry)

        Returns:
            list[TelemetryRecord]: List of TelemetryRecord objects, one per GPU

        Note:
            DCGM telemetry does not expose histogram metrics, so histogram_data is ignored.
        """
        records = []
        for gpu_index, metrics in resource_data.items():
            metadata = resource_metadata.get(gpu_index, {})
            scaled_metrics = self._apply_scaling_factors(metrics)

            record = TelemetryRecord(
                timestamp_ns=timestamp_ns,
                dcgm_url=self.endpoint_url,
                gpu_index=gpu_index,
                gpu_uuid=metadata.get("uuid") or f"unknown-gpu-{gpu_index}",
                gpu_model_name=metadata.get("model_name") or f"GPU {gpu_index}",
                pci_bus_id=metadata.get("pci_bus_id"),
                device=metadata.get("device"),
                hostname=metadata.get("hostname"),
                telemetry_data=TelemetryMetrics(**scaled_metrics),
            )
            records.append(record)

        return records
