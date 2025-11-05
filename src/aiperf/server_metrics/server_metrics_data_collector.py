# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.environment import Environment
from aiperf.common.metrics.base_metrics_data_collector import BaseMetricsDataCollector
from aiperf.common.models import ServerMetricRecord, ServerMetrics
from aiperf.common.models.metrics_hierarchy_base import (
    HistogramComponents,
    HistogramSnapshot,
)
from aiperf.server_metrics.constants import SCALING_FACTORS

__all__ = ["ServerMetricsDataCollector"]


class ServerMetricsDataCollector(BaseMetricsDataCollector[ServerMetricRecord]):
    """Collects server metrics from Prometheus /metrics endpoint using async architecture.

    Extends BaseMetricsDataCollector to provide server metrics-specific parsing and
    record creation. Fetches server metrics from Prometheus endpoints and converts them to
    ServerMetricRecord objects.

    Uses dynamic field discovery - metric names from Prometheus are used directly without
    pre-defined mapping. Display names and units are inferred automatically from field names.

    Args:
        endpoint_url: URL of the Prometheus metrics endpoint (e.g., "http://frontend:8080/metrics")
        collection_interval: Interval in seconds between metric collections (default: 1.0)
        record_callback: Optional async callback to receive collected records.
            Signature: async (records: list[ServerMetricRecord], collector_id: str) -> None
        error_callback: Optional async callback to receive collection errors.
            Signature: async (error: ErrorDetails, collector_id: str) -> None
        collector_id: Unique identifier for this collector instance
    """

    FIELD_MAPPING = {}  # Empty mapping enables dynamic field discovery
    SCALING_FACTORS = SCALING_FACTORS
    DEFAULT_COLLECTOR_ID = "server_metrics_collector"
    DEFAULT_COLLECTION_INTERVAL = Environment.SERVER_METRICS.COLLECTION_INTERVAL
    REACHABILITY_TIMEOUT = Environment.SERVER_METRICS.REACHABILITY_TIMEOUT

    def _extract_resource_info(
        self, labels: dict[str, str]
    ) -> tuple[str | None, dict[str, str]]:
        """Extract server identifier and metadata from Prometheus labels.

        Args:
            labels: Prometheus metric labels

        Returns:
            tuple: (server_id, metadata_dict) or (None, {}) to skip this metric
        """
        # Get server identifier from labels (instance, job, etc.)
        instance = labels.get("instance", "unknown")
        job = labels.get("job", "server")
        hostname = labels.get("hostname", labels.get("instance"))

        # Create a unique server_id from job and instance
        server_id = f"{job}-{instance}".replace(":", "-").replace("/", "-")

        metadata = {
            "instance": instance,
            "hostname": hostname,
            "server_type": job,
        }

        return server_id, metadata

    def _create_records(
        self,
        resource_data: dict[str, dict[str, float]],
        resource_metadata: dict[str, dict[str, str]],
        timestamp_ns: int,
        histogram_data: dict[str, dict[str, HistogramComponents]] | None = None,
    ) -> list[ServerMetricRecord]:
        """Create ServerMetricRecord objects from parsed server metrics.

        Args:
            resource_data: Dict mapping server_id -> {field_name: value}
            resource_metadata: Dict mapping server_id -> metadata
            timestamp_ns: Timestamp when metrics were collected
            histogram_data: Optional dict mapping server_id -> histogram_name -> HistogramComponents

        Returns:
            list[ServerMetricRecord]: List of ServerMetricRecord objects, one per server
        """
        records = []
        all_server_ids = set(resource_data.keys())
        if histogram_data:
            all_server_ids.update(histogram_data.keys())

        for server_id in all_server_ids:
            metadata = resource_metadata.get(server_id, {})
            metrics = resource_data.get(server_id, {})
            scaled_metrics = self._apply_scaling_factors(metrics)

            # Create HistogramSnapshot objects from HistogramComponents
            histograms: dict[str, HistogramSnapshot] = {}
            if histogram_data and server_id in histogram_data:
                for histogram_name, components in histogram_data[server_id].items():
                    # Convert HistogramComponents to HistogramSnapshot
                    histograms[histogram_name] = components.to_snapshot()

            record = ServerMetricRecord(
                timestamp_ns=timestamp_ns,
                server_url=self.endpoint_url,
                server_id=server_id,
                server_type=metadata.get("server_type"),
                hostname=metadata.get("hostname"),
                instance=metadata.get("instance"),
                metrics_data=ServerMetrics(**scaled_metrics),
                histograms_data=histograms,
            )
            records.append(record)

        return records
