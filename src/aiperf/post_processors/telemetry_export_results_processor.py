# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ResultsProcessorType
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.models.telemetry_models import TelemetryRecord
from aiperf.common.protocols import TelemetryResultsProcessorProtocol
from aiperf.post_processors.base_jsonl_export_processor import (
    BaseJSONLExportProcessor,
)


@implements_protocol(TelemetryResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.TELEMETRY_EXPORT)
class TelemetryExportResultsProcessor(BaseJSONLExportProcessor[TelemetryRecord]):
    """Exports per-record GPU telemetry data to JSONL files.

    This processor streams each TelemetryRecord as it arrives from the TelemetryManager,
    writing one JSON line per GPU per collection cycle. The output format supports
    multi-endpoint and multi-GPU time series analysis.

    Each line contains:
        - timestamp_ns: Collection timestamp in nanoseconds
        - dcgm_url: DCGM endpoint URL for filtering by endpoint
        - gpu_uuid: Unique GPU identifier
        - gpu_index: GPU index on the host
        - hostname: Host machine name
        - gpu_model_name: GPU model string
        - telemetry_data: Complete metrics snapshot (power, utilization, memory, etc.)
    """

    def __init__(self, user_config: UserConfig, **kwargs):
        super().__init__(
            user_config=user_config,
            output_file_path=user_config.output.profile_export_gpu_telemetry_jsonl_file,
            **kwargs,
        )

    async def process_telemetry_record(self, record: TelemetryRecord) -> None:
        """Process individual telemetry record by writing it to JSONL.

        Args:
            record: TelemetryRecord containing GPU metrics and hierarchical metadata
        """
        await self._process_record_internal(record)
