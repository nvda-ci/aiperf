# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ResultsProcessorType
from aiperf.common.environment import Environment
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.mixins import BufferedJSONLWriterMixin
from aiperf.common.models.processor_summary_results import (
    ServerMetricsExportSummaryResult,
)
from aiperf.common.models.server_metrics_models import (
    ServerMetricsMetadata,
    ServerMetricsRecord,
    ServerMetricsSlimRecord,
)
from aiperf.common.protocols import ServerMetricsResultsProcessorProtocol
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


@implements_protocol(ServerMetricsResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.SERVER_METRICS_EXPORT)
class ServerMetricsExportResultsProcessor(
    BaseMetricsProcessor, BufferedJSONLWriterMixin[ServerMetricsSlimRecord]
):
    """Exports per-record server metrics data to JSONL files in slim format.

    This processor converts full ServerMetricsRecord objects to slim format before writing,
    excluding static metadata (endpoint_url, kubernetes_pod_info, metric types, help text)
    to minimize file size. Writes one JSON line per collection cycle.

    Slim format benefits:
    - Reduced file size (no repeated type/help/endpoint/pod info)
    - Flat structure (metrics map directly to sample lists)
    - Static metadata available in separate metadata JSONL file

    Each line contains:
        - timestamp_ns: Collection timestamp in nanoseconds
        - metrics: Dict mapping metric names to sample lists (flat structure)
    """

    def __init__(
        self,
        user_config: UserConfig,
        **kwargs,
    ):
        output_file = user_config.output.server_metrics_export_jsonl_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.unlink(missing_ok=True)

        super().__init__(
            output_file=output_file,
            batch_size=Environment.RECORD.EXPORT_BATCH_SIZE,
            user_config=user_config,
            **kwargs,
        )

        self.info(f"Server metrics export enabled: {self.output_file}")

    async def process_server_metrics_record(self, record: ServerMetricsRecord) -> None:
        """Process individual server metrics record by converting to slim and writing to JSONL.

        Converts full record to slim format to reduce file size by excluding static metadata.

        Args:
            record: ServerMetricsRecord containing Prometheus metrics snapshot and metadata
        """
        # Convert to slim format before writing to reduce file size
        slim_record = record.to_slim()
        await self.buffered_write(slim_record)

    async def process_server_metrics_metadata(
        self, collector_id: str, metadata: "ServerMetricsMetadata"
    ) -> None:
        """Process server metrics metadata (no-op for this processor).

        This processor only handles full records. Metadata is handled by
        ServerMetricsMetadataExportResultsProcessor.

        Args:
            collector_id: Unique identifier for the server metrics data collector
            metadata: ServerMetricsMetadata containing static endpoint information
        """
        # This processor doesn't handle metadata messages
        pass

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
            ServerMetricsExportSummaryResult with file path and record count.
        """
        return ServerMetricsExportSummaryResult(
            file_path=self.output_file,
            record_count=self.lines_written,
        )
