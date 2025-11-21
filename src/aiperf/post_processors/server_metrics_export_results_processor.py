# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ResultsProcessorType
from aiperf.common.environment import Environment
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.mixins import BufferedJSONLWriterMixin
from aiperf.common.models.record_models import MetricResult
from aiperf.common.models.server_metrics_models import (
    MetricSchema,
    ServerMetricsMetadata,
    ServerMetricsMetadataFile,
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
    excluding static metadata (metric types, help text) to minimize file size.
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
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.unlink(missing_ok=True)

        super().__init__(
            output_file=output_file,
            batch_size=Environment.RECORD.EXPORT_BATCH_SIZE,
            user_config=user_config,
            **kwargs,
        )

        self._metadata_file = user_config.output.server_metrics_metadata_json_file
        self._metadata_file.unlink(missing_ok=True)

        # Track seen endpoints to know when to write metadata
        self._seen_endpoints: set[str] = set()
        # Store metadata for all endpoints using Pydantic model
        self._metadata_file_model = ServerMetricsMetadataFile()

        self.info(f"Server metrics export enabled: {self.output_file}")
        self.info(f"Server metrics metadata file: {self._metadata_file}")

    async def process_server_metrics_record(self, record: ServerMetricsRecord) -> None:
        """Process individual server metrics record by converting to slim and writing to JSONL.

        Converts full record to slim format to reduce file size by excluding static metadata.
        On first record from each endpoint, extracts metadata and writes metadata file.

        Args:
            record: ServerMetricsRecord containing Prometheus metrics snapshot and metadata
        """
        # Check if this is a new endpoint
        if record.endpoint_url not in self._seen_endpoints:
            self._seen_endpoints.add(record.endpoint_url)
            self._extract_and_store_metadata(record)
            await self._write_metadata_file()

        # Convert to slim format before writing to reduce file size
        slim_record = record.to_slim()
        await self.buffered_write(slim_record)

    def _extract_and_store_metadata(self, record: ServerMetricsRecord) -> None:
        """Extract metadata from a ServerMetricsRecord and store it.

        Extracts endpoint URL, metric schemas (type, help text, bucket labels, quantile labels)
        from the record.

        Args:
            record: ServerMetricsRecord to extract metadata from
        """
        metric_schemas: dict[str, MetricSchema] = {}

        for metric_name, metric_family in record.metrics.items():
            # Extract bucket labels for histogram metrics
            bucket_labels = None
            if metric_family.samples and metric_family.samples[0].histogram:
                # Get bucket labels from the first sample's histogram buckets
                sorted_buckets = sorted(
                    metric_family.samples[0].histogram.buckets.keys(),
                    key=lambda x: float(x),
                )
                bucket_labels = sorted_buckets

            # Extract quantile labels for summary metrics
            quantile_labels = None
            if metric_family.samples and metric_family.samples[0].summary:
                # Get quantile labels from the first sample's summary quantiles
                sorted_quantiles = sorted(
                    metric_family.samples[0].summary.quantiles.keys(),
                    key=lambda x: float(x),
                )
                quantile_labels = sorted_quantiles

            metric_schemas[metric_name] = MetricSchema(
                type=metric_family.type,
                help=metric_family.help,
                bucket_labels=bucket_labels,
                quantile_labels=quantile_labels,
            )

        metadata = ServerMetricsMetadata(
            endpoint_url=record.endpoint_url,
            endpoint_display=record.endpoint_url,  # Can be enhanced with display name
            metric_schemas=metric_schemas,
        )

        self._metadata_file_model.endpoints[record.endpoint_url] = metadata

    async def _write_metadata_file(self) -> None:
        """Write the complete metadata file for all seen endpoints.

        Re-writes the entire metadata file with all endpoints seen so far.
        Uses Pydantic model serialization with orjson for efficient JSON writing.
        """
        # Serialize the Pydantic model to JSON bytes using orjson
        metadata_json = orjson.dumps(
            self._metadata_file_model.model_dump(exclude_none=True, mode="json"),
            option=orjson.OPT_INDENT_2,
        )

        # Write to file
        self._metadata_file.write_bytes(metadata_json)

        self.debug(
            lambda: f"Wrote metadata file with {len(self._metadata_file_model.endpoints)} endpoints"
        )

    async def summarize(self) -> list[MetricResult]:
        """Summarize the results.

        Returns:
            Empty list (export processors don't generate metric results).
        """
        return []
