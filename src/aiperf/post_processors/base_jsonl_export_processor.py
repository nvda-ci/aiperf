# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class for JSONL export processors.

This module provides a generic base class that eliminates duplication between
TelemetryExportResultsProcessor and ServerMetricsExportResultsProcessor. Both
export processors follow identical patterns:
    1. Configure output file path from user config
    2. Initialize buffered JSONL writer with batch size
    3. Stream records to JSONL as they arrive
    4. Return empty list from summarize (no aggregation needed)

The generic base class captures this common structure while allowing concrete
implementations to customize:
    - Output file path extraction from config
    - Display name for logging
"""

from pathlib import Path
from typing import Generic, TypeVar

from aiperf.common.config import UserConfig
from aiperf.common.environment import Environment
from aiperf.common.mixins import BufferedJSONLWriterMixin
from aiperf.common.models import MetricResult
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor

# Type variable for record types
RecordT = TypeVar("RecordT", bound=AIPerfBaseModel)


class BaseJSONLExportProcessor(
    BaseMetricsProcessor, BufferedJSONLWriterMixin[RecordT], Generic[RecordT]
):
    """Base class for processors that export metrics records to JSONL files.

    This class provides common functionality for:
    - Configuring output file paths
    - Initializing buffered JSONL writers
    - Streaming records to file
    - Empty summarization (export processors don't aggregate)

    Subclasses must:
    - Implement protocol-specific process method (process_telemetry_record, etc.)
    - Pass output_file_path and export_name to constructor

    Type Parameters:
        RecordT: The record model type (e.g., TelemetryRecord, ServerMetricRecord)
    """

    def __init__(
        self,
        user_config: UserConfig,
        output_file_path: Path,
        **kwargs,
    ):
        """Initialize the JSONL export processor.

        Args:
            user_config: User configuration
            output_file_path: Path to output JSONL file
            export_name: Display name for logging (e.g., "GPU telemetry", "Server metrics")
            **kwargs: Additional keyword arguments passed to parent
        """
        # Ensure parent directory exists and clean up existing file
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        output_file_path.unlink(missing_ok=True)

        # Initialize with output file and batch size
        super().__init__(
            output_file=output_file_path,
            batch_size=Environment.RECORD.EXPORT_BATCH_SIZE,
            user_config=user_config,
            **kwargs,
        )

        self.info(f"{self.__class__.__name__} export enabled: {self.output_file}")

    async def _process_record_internal(self, record: RecordT) -> None:
        """Internal method to process individual record by writing to JSONL.

        This is the core processing logic that's identical across all JSONL
        export processors. Subclasses should delegate their protocol-specific
        methods (e.g., process_telemetry_record, process_server_metric_record) to this.

        Args:
            record: Record to write to JSONL file
        """
        try:
            await self.buffered_write(record)
        except Exception as e:
            self.error(f"Failed to write {record.__class__.__name__}: {e}")

    async def summarize(self) -> list[MetricResult]:
        """Summarize the results.

        Export processors stream records directly to files without aggregation,
        so they don't produce MetricResult summaries.

        Returns:
            Empty list (no aggregation performed)
        """
        return []
