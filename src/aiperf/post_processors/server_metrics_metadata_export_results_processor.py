# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ResultsProcessorType
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.models.processor_summary_results import (
    ServerMetricsExportSummaryResult,
)
from aiperf.common.models.server_metrics_models import (
    ServerMetricsMetadata,
    ServerMetricsRecord,
)
from aiperf.common.protocols import ServerMetricsResultsProcessorProtocol
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


@implements_protocol(ServerMetricsResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.SERVER_METRICS_METADATA_EXPORT)
class ServerMetricsMetadataExportResultsProcessor(BaseMetricsProcessor):
    """Exports server metrics metadata to JSONL files.

    This processor writes static endpoint metadata once per collector when it starts
    collecting. The metadata includes endpoint information, Kubernetes pod details,
    and metric schemas (type, help text, bucket/quantile labels) that don't change
    during the collection period.

    Each line contains:
        - endpoint_url: Prometheus endpoint URL
        - endpoint_display: Human-readable endpoint display name
        - kubernetes_pod_info: Kubernetes POD information if available
        - metric_schemas: Dict of metric name -> MetricSchema (type, help, bucket/quantile labels)
    """

    def __init__(
        self,
        user_config: UserConfig,
        **kwargs,
    ):
        output_file = user_config.output.server_metrics_metadata_jsonl_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.unlink(missing_ok=True)

        super().__init__(
            output_file=output_file,
            user_config=user_config,
            **kwargs,
        )

        self.info(f"Server metrics metadata export enabled: {self.output_file}")
        self._metadata_endpoints_written: set[str] = set()

    async def process_server_metrics_record(self, record: ServerMetricsRecord) -> None:
        """Process individual server metrics record (no-op for metadata processor).

        This processor only handles metadata messages, not individual records.

        Args:
            record: ServerMetricsRecord (ignored by this processor)
        """
        # Metadata processor doesn't process individual records
        pass

    async def process_server_metrics_metadata(
        self, collector_id: str, metadata: ServerMetricsMetadata
    ) -> None:
        """Process and write server metrics metadata to JSONL.

        Writes metadata once per endpoint to avoid duplicates.

        Args:
            collector_id: Unique identifier for the server metrics data collector
            metadata: ServerMetricsMetadata containing static endpoint information
        """
        # Write metadata once per endpoint
        if metadata.endpoint_url not in self._metadata_endpoints_written:
            await self._write_metadata(metadata)
            self._metadata_endpoints_written.add(metadata.endpoint_url)

    async def _write_metadata(self, metadata: ServerMetricsMetadata) -> None:
        """Write metadata for an endpoint to the metadata JSONL file.

        Args:
            metadata: ServerMetricsMetadata containing metadata to export
        """
        try:
            metadata_dict = {
                "endpoint_url": metadata.endpoint_url,
                "endpoint_display": metadata.endpoint_display,
                "kubernetes_pod_info": (
                    metadata.kubernetes_pod_info.model_dump()
                    if metadata.kubernetes_pod_info
                    else None
                ),
                "metric_schemas": (
                    {
                        name: schema.model_dump()
                        for name, schema in metadata.metric_schemas.items()
                    }
                    if metadata.metric_schemas
                    else {}
                ),
            }

            # Write directly (not buffered since metadata is written once per endpoint)
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(metadata_dict) + "\n")

            self.debug(
                f"Wrote metadata for endpoint: {metadata.endpoint_url} "
                f"with {len(metadata.metric_schemas)} metric schemas"
            )
        except Exception as e:
            self.error(
                f"Failed to write metadata for endpoint {metadata.endpoint_url}: {e}"
            )

    async def summarize(
        self,
        min_timestamp_ns: int | None = None,
        max_timestamp_ns: int | None = None,
    ) -> ServerMetricsExportSummaryResult:
        """Summarize the results. For this processor, we return export metadata.

        Args:
            min_timestamp_ns: Optional start of inference time window (not used by export processor)
            max_timestamp_ns: Optional end of inference time window (not used by export processor)

        Returns:
            ServerMetricsExportSummaryResult with file path and endpoint count.
        """
        return ServerMetricsExportSummaryResult(
            file_path=self.output_file,
            record_count=len(self._metadata_endpoints_written),
        )
