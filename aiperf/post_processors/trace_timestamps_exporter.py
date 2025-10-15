# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace timestamps JSONL exporter for comprehensive HTTP performance analysis.

This processor exports detailed trace timestamp data to JSONL files,
capturing every aspect of HTTP request performance for offline analysis.
"""

import json
from pathlib import Path
from typing import Any

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import RecordProcessorType
from aiperf.common.factories import RecordProcessorFactory
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import AioHttpTraceTimestamps, ParsedResponseRecord
from aiperf.common.protocols import RecordProcessorProtocol
from aiperf.metrics.metric_dicts import MetricRecordDict


@implements_protocol(RecordProcessorProtocol)
@RecordProcessorFactory.register(RecordProcessorType.TRACE_TIMESTAMPS_EXPORT)
class TraceTimestampsExporter(AIPerfLifecycleMixin):
    """Exports comprehensive HTTP trace timestamp data to JSONL files.

    Features:
    - Captures all raw timestamp fields (16 nanosecond-precision timestamps)
    - Exports all metadata (DNS, headers, compression, errors)
    - Includes ALL computed properties (42 derived metrics)
    - One JSONL file per record processor instance
    - Appends one JSON object per request record

    File naming: trace_timestamps_{processor_id}.jsonl
    Location: Output directory specified in user config
    """

    def __init__(
        self,
        user_config: UserConfig,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(user_config=user_config, service_id=service_id, **kwargs)
        self.user_config = user_config
        self.service_id = service_id or "default"
        self.output_file: Path | None = None
        self.file_handle = None
        self._records_written = 0

    @on_init
    async def _setup_file(self) -> None:
        """Setup the exporter and create the output file."""
        # Determine output directory
        output_dir = self.user_config.output.artifact_directory / "trace_timestamps"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create unique filename for this processor instance
        self.output_file = output_dir / f"trace_timestamps_{self.processor_id}.jsonl"

        # Open file in append mode
        self.file_handle = self.output_file.open("a", encoding="utf-8")
        self.info(f"Trace timestamps will be exported to: {self.output_file}")

    @on_stop
    async def _close_file(self) -> None:
        """Close the output file and log stats."""
        if self.file_handle:
            self.file_handle.close()
            self.info(
                f"Exported {self._records_written} trace timestamp records to {self.output_file}"
            )

    async def process_record(self, record: ParsedResponseRecord) -> MetricRecordDict:
        """Process a record and export its trace timestamps.

        Args:
            record: The parsed response record to process

        Returns:
            Empty MetricRecordDict (this processor doesn't compute metrics)
        """
        # Extract trace timestamps from the request record
        if not record.request.trace_timestamps:
            # No trace data available for this record
            return MetricRecordDict()

        ts = record.request.trace_timestamps

        # Serialize all data to a comprehensive JSON object
        trace_data = self._serialize_trace_timestamps(ts, record)

        # Write to JSONL file (one JSON object per line)
        if self.file_handle:
            json.dump(trace_data, self.file_handle, default=str)
            self.file_handle.write("\n")
            self.file_handle.flush()  # Ensure data is written immediately
            self._records_written += 1

        return MetricRecordDict()

    def _serialize_trace_timestamps(
        self, ts: AioHttpTraceTimestamps, record: ParsedResponseRecord
    ) -> dict[str, Any]:
        """Serialize trace timestamps with all fields and computed properties.

        Creates a comprehensive JSON-serializable dictionary containing:
        - Record metadata (session, request IDs, timestamps)
        - All raw timestamp fields
        - All metadata fields
        - All computed properties (timing breakdowns, bandwidth, statistics, etc.)

        Args:
            ts: The AioHttpTraceTimestamps object
            record: The full parsed response record for context

        Returns:
            Dictionary ready for JSON serialization
        """
        # ====================================================================
        # RECORD METADATA
        # ====================================================================
        data: dict[str, Any] = {
            # Record identification
            "session_num": getattr(record.request, "credit_num", None),
            "x_request_id": record.request.x_request_id,
            "x_correlation_id": record.request.x_correlation_id,
            "conversation_id": record.request.conversation_id,
            "turn_index": record.request.turn_index,
            # Timestamps
            "timestamp_ns": record.request.timestamp_ns,
            "start_perf_ns": record.request.start_perf_ns,
            "end_perf_ns": record.request.end_perf_ns,
            # Status
            "status": record.request.status,
            "has_error": record.has_error,
        }

        # ====================================================================
        # RAW TIMESTAMP FIELDS (16 fields)
        # ====================================================================
        timestamp_fields = {
            # Connection pool
            "connection_queued_start_ns": ts.connection_queued_start_ns,
            "connection_queued_end_ns": ts.connection_queued_end_ns,
            # Connection creation
            "connection_create_start_ns": ts.connection_create_start_ns,
            "connection_create_end_ns": ts.connection_create_end_ns,
            # Connection reuse
            "connection_reuseconn_ns": ts.connection_reuseconn_ns,
            # DNS resolution
            "dns_resolvehost_start_ns": ts.dns_resolvehost_start_ns,
            "dns_resolvehost_end_ns": ts.dns_resolvehost_end_ns,
            "dns_cache_hit_ns": ts.dns_cache_hit_ns,
            "dns_cache_miss_ns": ts.dns_cache_miss_ns,
            # Request
            "request_start_ns": ts.request_start_ns,
            "request_headers_sent_ns": ts.request_headers_sent_ns,
            "request_end_ns": ts.request_end_ns,
            # Chunks
            "request_chunk_sent_ns": ts.request_chunk_sent_ns,
            "response_chunk_received_ns": ts.response_chunk_received_ns,
            # Redirects & exceptions
            "request_redirect_ns": ts.request_redirect_ns,
            "request_exception_ns": ts.request_exception_ns,
        }
        data["timestamps"] = timestamp_fields

        # ====================================================================
        # METADATA FIELDS (13 fields)
        # ====================================================================
        metadata_fields = {
            # DNS & connection
            "dns_host": ts.dns_host,
            "connection_host": ts.connection_host,
            # Request
            "request_method": ts.request_method,
            "request_url": ts.request_url,
            "request_chunk_sizes": ts.request_chunk_sizes,
            # Response
            "response_chunk_sizes": ts.response_chunk_sizes,
            "response_headers": ts.response_headers,
            "response_status": ts.response_status,
            "response_reason": ts.response_reason,
            # Redirects
            "redirect_urls": ts.redirect_urls,
            "redirect_status_codes": ts.redirect_status_codes,
            # Exceptions
            "exception_type": ts.exception_type,
            "exception_message": ts.exception_message,
        }
        data["metadata"] = metadata_fields

        # ====================================================================
        # COMPUTED PROPERTIES - CRITICAL TIMING BREAKDOWNS
        # ====================================================================
        timing_breakdowns = {
            "time_to_first_byte_ns": ts.time_to_first_byte_ns,
            "time_to_last_byte_ns": ts.time_to_last_byte_ns,
            "server_processing_time_ns": ts.server_processing_time_ns,
            "network_transfer_time_ns": ts.network_transfer_time_ns,
            "connection_establishment_time_ns": ts.connection_establishment_time_ns,
            "request_send_duration_ns": ts.request_send_duration_ns,
            "request_headers_duration_ns": ts.request_headers_duration_ns,
            "request_body_duration_ns": ts.request_body_duration_ns,
            "dns_resolution_duration_ns": ts.dns_resolution_duration_ns,
            "connection_create_duration_ns": ts.connection_create_duration_ns,
            "connection_queue_wait_ns": ts.connection_queue_wait_ns,
        }
        data["timing_breakdowns"] = timing_breakdowns

        # ====================================================================
        # COMPUTED PROPERTIES - BANDWIDTH & THROUGHPUT
        # ====================================================================
        bandwidth_metrics = {
            "upload_rate_bytes_per_sec": ts.upload_rate_bytes_per_sec,
            "download_rate_bytes_per_sec": ts.download_rate_bytes_per_sec,
            "avg_request_chunk_rate_bytes_per_sec": ts.avg_request_chunk_rate_bytes_per_sec,
            "avg_response_chunk_rate_bytes_per_sec": ts.avg_response_chunk_rate_bytes_per_sec,
            "total_request_bytes": ts.total_request_bytes,
            "total_response_bytes": ts.total_response_bytes,
            "total_request_chunks": ts.total_request_chunks,
            "total_response_chunks": ts.total_response_chunks,
        }
        data["bandwidth_metrics"] = bandwidth_metrics

        # ====================================================================
        # COMPUTED PROPERTIES - STATISTICAL ANALYSIS
        # ====================================================================
        statistical_analysis = {
            "response_inter_chunk_latencies_ns": ts.response_inter_chunk_latencies_ns,
            "response_chunk_latency_min_ns": ts.response_chunk_latency_min_ns,
            "response_chunk_latency_max_ns": ts.response_chunk_latency_max_ns,
            "response_chunk_latency_avg_ns": ts.response_chunk_latency_avg_ns,
            "response_chunk_latency_jitter_ns": ts.response_chunk_latency_jitter_ns,
            "is_streaming_response": ts.is_streaming_response,
        }
        data["statistical_analysis"] = statistical_analysis

        # ====================================================================
        # COMPUTED PROPERTIES - HEADER & BODY INTELLIGENCE
        # ====================================================================
        header_intelligence = {
            "request_headers_size_bytes": ts.request_headers_size_bytes,
            "response_headers_size_bytes": ts.response_headers_size_bytes,
            "compression_type": ts.compression_type,
            "response_content_length": ts.response_content_length,
            "response_content_type": ts.response_content_type,
            "transfer_encoding": ts.transfer_encoding,
            "compression_ratio": ts.compression_ratio,
        }
        data["header_intelligence"] = header_intelligence

        # ====================================================================
        # COMPUTED PROPERTIES - CONNECTION INSIGHTS
        # ====================================================================
        connection_insights = {
            "connection_was_reused": ts.connection_was_reused,
            "dns_was_cached": ts.dns_was_cached,
            "connection_overhead_percentage": ts.connection_overhead_percentage,
            "queue_wait_percentage": ts.queue_wait_percentage,
            "total_redirects": ts.total_redirects,
        }
        data["connection_insights"] = connection_insights

        # ====================================================================
        # COMPUTED PROPERTIES - DERIVED QUALITY METRICS
        # ====================================================================
        quality_metrics = {
            "network_vs_server_time_ratio": ts.network_vs_server_time_ratio,
            "request_efficiency_score": ts.request_efficiency_score,
        }
        data["quality_metrics"] = quality_metrics

        return data
