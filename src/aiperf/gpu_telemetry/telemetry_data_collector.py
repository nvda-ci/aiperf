# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from collections.abc import Awaitable, Callable

from prometheus_client.parser import text_string_to_metric_families

from aiperf.common.environment import Environment
from aiperf.common.mixins import BaseMetricsCollectorMixin
from aiperf.common.models import ErrorDetails, TelemetryMetrics, TelemetryRecord
from aiperf.gpu_telemetry.constants import (
    DCGM_TO_FIELD_MAPPING,
    SCALING_FACTORS,
)

__all__ = ["TelemetryDataCollector"]


class TelemetryDataCollector(BaseMetricsCollectorMixin[TelemetryRecord]):
    """Collects telemetry metrics from DCGM metrics endpoint using async architecture.

    Modern async collector that fetches GPU metrics from DCGM exporter and converts them to
    TelemetryRecord objects. Uses BaseMetricsCollectorMixin for HTTP collection patterns.
    - Extends BaseMetricsCollectorMixin for async HTTP collection
    - Uses prometheus_client for robust metric parsing
    - Sends TelemetryRecord list via callback function

    Args:
        dcgm_url: URL of the DCGM metrics endpoint (e.g., "http://localhost:9400/metrics")
        collection_interval: Interval in seconds between metric collections (default: from Environment)
        reachability_timeout: Timeout in seconds for reachability checks (default: from Environment)
        record_callback: Optional async callback to receive collected records.
            Signature: async (records: list[TelemetryRecord], collector_id: str) -> None
        error_callback: Optional async callback to receive collection errors.
            Signature: async (error: ErrorDetails, collector_id: str) -> None
        collector_id: Unique identifier for this collector instance
    """

    def __init__(
        self,
        dcgm_url: str,
        collection_interval: float | None = None,
        reachability_timeout: float | None = None,
        record_callback: Callable[[list[TelemetryRecord], str], Awaitable[None]]
        | None = None,
        error_callback: Callable[[ErrorDetails, str], Awaitable[None]] | None = None,
        collector_id: str = "telemetry_collector",
    ) -> None:
        self._scaling_factors = SCALING_FACTORS
        super().__init__(
            endpoint_url=dcgm_url,
            collection_interval=collection_interval
            or Environment.GPU.COLLECTION_INTERVAL,
            reachability_timeout=reachability_timeout
            or Environment.GPU.REACHABILITY_TIMEOUT,
            record_callback=record_callback,
            error_callback=error_callback,
            id=collector_id,
        )

    async def _collect_and_process_metrics(self) -> None:
        """Collect metrics from DCGM endpoint and process them into TelemetryRecord objects.

        Implements the abstract method from BaseMetricsCollectorMixin.

        Orchestrates the full collection flow:
        1. Fetches raw metrics data from DCGM endpoint (via mixin's _fetch_metrics_text)
        2. Parses Prometheus-format data into TelemetryRecord objects
        3. Sends records via callback (via mixin's _send_records_via_callback)

        Raises:
            Exception: Any exception from fetch or parse is logged and re-raised
        """
        metrics_data = await self._fetch_metrics_text()
        records = self._parse_metrics_to_records(metrics_data)
        await self._send_records_via_callback(records)

    def _parse_metrics_to_records(self, metrics_data: str) -> list[TelemetryRecord]:
        """Parse DCGM metrics text into TelemetryRecord objects using prometheus_client.

        Processes Prometheus exposition format metrics from DCGM exporter:
        1. Parses metric families using prometheus_client parser
        2. Extracts GPU metadata (UUID, model name, hostname, etc.) from labels
        3. Maps DCGM metric names to TelemetryRecord field names
        4. Applies scaling factors to convert units (e.g., milliwatts to watts)
        5. Aggregates metrics by GPU index into TelemetryRecord objects

        Skips non-finite values (NaN, inf) and metrics without valid GPU index.

        Args:
            metrics_data: Raw metrics text from DCGM exporter in Prometheus format

        Returns:
            list[TelemetryRecord]: List of TelemetryRecord objects, one per GPU with valid data.
                Returns empty list if metrics_data is empty or parsing fails.
        """
        if not metrics_data.strip():
            return []

        current_timestamp = time.time_ns()
        gpu_data = {}
        gpu_metadata = {}

        try:
            for family in text_string_to_metric_families(metrics_data):
                for sample in family.samples:
                    metric_name = sample.name
                    labels = sample.labels
                    value = sample.value

                    # Skip non-finite values early (value != value checks for NaN)
                    if isinstance(value, float) and (
                        value != value or value in (float("inf"), float("-inf"))
                    ):
                        continue

                    gpu_index = labels.get("gpu")
                    if gpu_index is not None:
                        try:
                            gpu_index = int(gpu_index)
                        except ValueError:
                            continue
                    else:
                        continue

                    gpu_metadata[gpu_index] = {
                        "model_name": labels.get("modelName"),
                        "uuid": labels.get("UUID"),
                        "pci_bus_id": labels.get("pci_bus_id"),
                        "device": labels.get("device"),
                        "hostname": labels.get("Hostname"),
                    }

                    base_metric_name = metric_name.removesuffix("_total")
                    if base_metric_name in DCGM_TO_FIELD_MAPPING:
                        field_name = DCGM_TO_FIELD_MAPPING[base_metric_name]
                        gpu_data.setdefault(gpu_index, {})[field_name] = value
        except ValueError:
            self.warning("Failed to parse Prometheus metrics - invalid format")
            return []

        records = []
        for gpu_index, metrics in gpu_data.items():
            metadata = gpu_metadata.get(gpu_index, {})
            scaled_metrics = self._apply_scaling_factors(metrics)

            record = TelemetryRecord(
                timestamp_ns=current_timestamp,
                dcgm_url=self.endpoint_url,
                gpu_index=gpu_index,
                gpu_uuid=metadata.get("uuid", f"unknown-gpu-{gpu_index}"),
                gpu_model_name=metadata.get("model_name", f"GPU {gpu_index}"),
                pci_bus_id=metadata.get("pci_bus_id"),
                device=metadata.get("device"),
                hostname=metadata.get("hostname"),
                telemetry_data=TelemetryMetrics(**scaled_metrics),
            )
            records.append(record)

        return records

    def _apply_scaling_factors(self, metrics: dict) -> dict:
        """Apply scaling factors to convert raw DCGM units to display units.

        Converts metrics from DCGM's native units to human-readable units:
        - Power: milliwatts -> watts (multiply by 0.001)
        - Memory: bytes -> megabytes (multiply by 1e-6)
        - Frequency: MHz values (no scaling needed)

        Only applies scaling to metrics present in the input dict. None values are preserved.

        Args:
            metrics: Dict of metric_name -> raw_value from DCGM

        Returns:
            dict: New dict with scaled values ready for display. Unscaled metrics are copied as-is.
        """
        scaled_metrics = metrics.copy()
        for metric, factor in self._scaling_factors.items():
            if metric in scaled_metrics and scaled_metrics[metric] is not None:
                scaled_metrics[metric] *= factor
        return scaled_metrics
