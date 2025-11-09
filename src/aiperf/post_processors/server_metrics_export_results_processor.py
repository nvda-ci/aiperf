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
from aiperf.common.models.server_metrics_models import ServerMetricsRecord
from aiperf.common.protocols import ServerMetricsResultsProcessorProtocol
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


@implements_protocol(ServerMetricsResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.SERVER_METRICS_EXPORT)
class ServerMetricsExportResultsProcessor(
    BaseMetricsProcessor, BufferedJSONLWriterMixin[ServerMetricsRecord]
):
    """Exports per-record server metrics data to JSONL files.

    This processor streams each ServerMetricsRecord as it arrives from the ServerMetricsManager,
    writing one JSON line per Prometheus endpoint per collection cycle. The output format supports
    multi-endpoint time series analysis.

    Each line contains:
        - timestamp_ns: Collection timestamp in nanoseconds
        - endpoint_url: Prometheus endpoint URL for filtering by endpoint
        - snapshot: Complete Prometheus metrics snapshot (metrics dict with families)
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
        """Process individual server metrics record by writing it to JSONL.

        Args:
            record: ServerMetricsRecord containing Prometheus metrics snapshot and metadata
        """
        await self.buffered_write(record)

    async def summarize(self) -> ServerMetricsExportSummaryResult:
        """Summarize the results. For this processor, we return export metadata."""
        return ServerMetricsExportSummaryResult(
            file_path=self.output_file,
            record_count=self.lines_written,
        )
