# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ResultsProcessorType
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.mixins import BufferedJSONLWriterMixinWithDeduplication
from aiperf.common.models.record_models import MetricResult
from aiperf.common.models.server_metrics_models import (
    ServerMetricsRecord,
    ServerMetricsSlimRecord,
)
from aiperf.common.protocols import ServerMetricsResultsProcessorProtocol
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


@implements_protocol(ServerMetricsResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.SERVER_METRICS_JSONL_WRITER)
class ServerMetricsExportResultsProcessor(
    BaseMetricsProcessor,
    BufferedJSONLWriterMixinWithDeduplication[ServerMetricsSlimRecord],
):
    """Exports per-record server metrics data to JSONL files in slim format.

    This processor converts full ServerMetricsRecord objects to slim format before writing,
    excluding static metadata (metric types, description text) to minimize file size.
    Writes one JSON line per collection cycle.

    Each line contains:
        - timestamp_ns: Collection timestamp in nanoseconds
        - endpoint_latency_ns: Time taken to collect the metrics from the endpoint
        - endpoint_url: Source Prometheus metrics endpoint URL (e.g., 'http://localhost:8081/metrics')
        - metrics: Dict mapping metric names to sample lists (flat structure)
    """

    def __init__(
        self,
        user_config: UserConfig,
        **kwargs,
    ) -> None:
        if user_config.server_metrics_disabled:
            raise PostProcessorDisabled(
                "Server metrics JSONL export is disabled via --no-server-metrics"
            )

        output_file = user_config.output.server_metrics_export_jsonl_file

        super().__init__(
            user_config=user_config,
            output_file=output_file,
            batch_size=Environment.SERVER_METRICS.EXPORT_BATCH_SIZE,
            dedupe_key_function=lambda x: x.endpoint_url,
            dedupe_value_function=lambda x: x.metrics,
            **kwargs,
        )

        self.info(f"Server metrics JSONL export enabled: {self.output_file}")

    async def process_server_metrics_record(self, record: ServerMetricsRecord) -> None:
        """Process individual server metrics record by converting to slim and writing to JSONL.

        Converts full record to slim format to reduce file size by excluding static metadata.

        Args:
            record: ServerMetricsRecord containing Prometheus metrics snapshot and metadata
        """
        # Convert to slim format before writing to reduce file size (will be deduplicated)
        slim_record = record.to_slim()
        await self.buffered_write(slim_record)

    async def summarize(self) -> list[MetricResult]:
        """Summarize the results.

        Returns:
            Empty list (export processors don't generate metric results).
        """
        return []
