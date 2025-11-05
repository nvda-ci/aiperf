# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ResultsProcessorType
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.models.server_metrics_models import ServerMetricRecord
from aiperf.common.protocols import ServerMetricsResultsProcessorProtocol
from aiperf.post_processors.base_jsonl_export_processor import (
    BaseJSONLExportProcessor,
)


@implements_protocol(ServerMetricsResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.SERVER_METRICS_EXPORT)
class ServerMetricsExportResultsProcessor(BaseJSONLExportProcessor[ServerMetricRecord]):
    """Exports per-record server metrics data to JSONL files.

    This processor streams each ServerMetricRecord as it arrives from the ServerMetricsManager,
    writing one JSON line per server per collection cycle. The output format supports
    multi-endpoint and multi-server time series analysis.

    Each line contains:
        - timestamp_ns: Collection timestamp in nanoseconds
        - server_url: Server metrics endpoint URL for filtering by endpoint
        - server_id: Unique server identifier
        - server_type: Type of server (e.g., "frontend", "worker")
        - hostname: Host machine name
        - instance: Server instance identifier
        - metrics_data: Complete metrics snapshot (requests, CPU, memory, Dynamo metrics, etc.)
    """

    def __init__(self, user_config: UserConfig, **kwargs):
        super().__init__(
            user_config=user_config,
            output_file_path=user_config.output.profile_export_server_metrics_jsonl_file,
            **kwargs,
        )

    async def process_server_metric_record(self, record: ServerMetricRecord) -> None:
        """Process individual server metric record by writing it to JSONL.

        Args:
            record: ServerMetricRecord containing server metrics and hierarchical metadata
        """
        await self._process_record_internal(record)
