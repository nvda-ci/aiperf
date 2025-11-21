# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import orjson

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ResultsProcessorType
from aiperf.common.environment import Environment
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.mixins import BufferedJSONLWriterMixinWithDeduplication
from aiperf.common.models.record_models import MetricResult
from aiperf.common.models.server_metrics_models import (
    ServerMetricsMetadata,
    ServerMetricsMetadataFile,
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
        output_file = user_config.output.server_metrics_export_jsonl_file

        super().__init__(
            user_config=user_config,
            output_file=output_file,
            batch_size=Environment.SERVER_METRICS.EXPORT_BATCH_SIZE,
            dedupe_key_function=lambda x: x.endpoint_url,
            dedupe_value_function=lambda x: x.metrics,
            **kwargs,
        )

        self._metadata_file = user_config.output.server_metrics_metadata_json_file
        self._metadata_file.unlink(missing_ok=True)

        # Keep track of metadata for all endpoints over time. Note that the metadata fields
        # can occasionally change, so we need to keep track of it over time.
        self._metadata_file_model = ServerMetricsMetadataFile()
        self._metadata_file_lock = asyncio.Lock()

        self.info(f"Server metrics jsonl writer export enabled: {self.output_file}")
        self.info(f"Server metrics metadata file: {self._metadata_file}")

    async def process_server_metrics_record(self, record: ServerMetricsRecord) -> None:
        """Process individual server metrics record by converting to slim and writing to JSONL.

        Converts full record to slim format to reduce file size by excluding static metadata.
        On first record from each endpoint, extracts metadata and writes metadata file.

        Args:
            record: ServerMetricsRecord containing Prometheus metrics snapshot and metadata
        """
        url = record.endpoint_url
        metadata = record.extract_metadata()
        # Check without lock to avoid unnecessary locking
        if (
            url not in self._metadata_file_model.endpoints
            or self._metadata_file_model.endpoints[url] != metadata
        ):
            async with self._metadata_file_lock:
                # No reason to check again with lock, its very unlikely to have a race condition here.
                # and even if there is, we will just end up writing the file twice.
                self._metadata_file_model.endpoints[url] = metadata
                await self._write_metadata_file()

        # Convert to slim format before writing to reduce file size (will be deduplicated)
        slim_record = record.to_slim()
        await self.buffered_write(slim_record)

    def _should_update_metadata(
        self, record: ServerMetricsRecord, existing_metadata: ServerMetricsMetadata
    ) -> bool:
        """Check if metadata should be updated based on record changes.

        Detects new metric names not in existing metadata.

        Args:
            record: ServerMetricsRecord to check
            existing_metadata: Existing metadata for the endpoint

        Returns:
            True if metadata needs updating, False otherwise
        """
        existing_schemas = existing_metadata.metric_schemas

        # Check for new metric names
        new_metrics = set(record.metrics.keys()) - set(existing_schemas.keys())
        if new_metrics:
            self.debug(
                lambda: f"Detected new metrics for {record.endpoint_url}: {sorted(new_metrics)}"
            )
            return True

        return False

    async def _write_metadata_file(self) -> None:
        """Write the complete metadata file for all seen endpoints atomically.

        Re-writes the entire metadata file with all endpoints seen so far.
        Uses Pydantic model serialization with orjson for efficient JSON writing.
        Uses atomic temp-file + rename pattern to prevent corruption on crash.
        """
        metadata_json = orjson.dumps(
            self._metadata_file_model.model_dump(exclude_none=True, mode="json"),
            option=orjson.OPT_INDENT_2,
        )

        # Write to temp file and atomically rename to prevent corruption
        temp_file = self._metadata_file.with_suffix(".tmp")
        try:
            temp_file.write_bytes(metadata_json)
            temp_file.replace(self._metadata_file)  # Atomic on POSIX
            self.debug(
                lambda: f"Wrote metadata file with {len(self._metadata_file_model.endpoints)} endpoints"
            )
        except Exception as e:
            self.error(f"Failed to write metadata file: {e}")
            temp_file.unlink(missing_ok=True)
            raise

    async def summarize(self) -> list[MetricResult]:
        """Summarize the results.

        Returns:
            Empty list (export processors don't generate metric results).
        """
        return []
