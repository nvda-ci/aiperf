# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the TraceTimestampsExporter."""

import json
import tempfile
from pathlib import Path

import pytest

from aiperf.common.config import UserConfig
from aiperf.common.models import (
    AioHttpTraceTimestamps,
    ParsedResponse,
    ParsedResponseRecord,
    RequestRecord,
    TextResponse,
    TextResponseData,
)
from aiperf.post_processors.trace_timestamps_exporter import TraceTimestampsExporter


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def exporter(user_config: UserConfig, temp_output_dir: Path):
    """Create a TraceTimestampsExporter instance."""
    # Create a modified config with our temp directory
    import copy

    config = copy.deepcopy(user_config)
    config.output.artifact_directory = temp_output_dir
    return TraceTimestampsExporter(
        user_config=config,
        processor_id="test_processor_1",
    )


@pytest.fixture
def sample_trace_timestamps():
    """Create a sample AioHttpTraceTimestamps object with realistic data."""
    return AioHttpTraceTimestamps(
        # Connection pool
        connection_queued_start_ns=1000000000,
        connection_queued_end_ns=1000500000,
        # DNS
        dns_resolvehost_start_ns=1000500000,
        dns_resolvehost_end_ns=1002000000,
        dns_host="api.example.com",
        # Connection
        connection_create_start_ns=1002000000,
        connection_create_end_ns=1005000000,
        # Request
        request_start_ns=1005000000,
        request_headers_sent_ns=1005100000,
        request_end_ns=1005500000,
        request_method="POST",
        request_url="https://api.example.com/v1/chat",
        # Response chunks
        response_chunk_received_ns=[1010000000, 1015000000, 1020000000],
        response_chunk_sizes=[100, 150, 200],
        # Response metadata
        response_status=200,
        response_reason="OK",
        response_headers={
            "Content-Type": "application/json",
            "Content-Length": "450",
            "Content-Encoding": "gzip",
        },
    )


@pytest.fixture
def sample_record(sample_trace_timestamps):
    """Create a sample ParsedResponseRecord with trace timestamps."""
    request = RequestRecord(
        start_perf_ns=1000000000,
        end_perf_ns=1020000000,
        status=200,
        trace_timestamps=sample_trace_timestamps,
    )
    request.responses.append(TextResponse(perf_ns=1010000000, text="test response"))

    return ParsedResponseRecord(
        request=request,
        responses=[
            ParsedResponse(perf_ns=1010000000, data=TextResponseData(text="test"))
        ],
        input_token_count=10,
        output_token_count=50,
    )


class TestTraceTimestampsExporter:
    """Test suite for TraceTimestampsExporter."""

    @pytest.mark.asyncio
    async def test_exporter_setup_creates_file(
        self, exporter: TraceTimestampsExporter, temp_output_dir: Path
    ):
        """Test that setup creates the output file."""
        await exporter.initialize()

        expected_file = (
            temp_output_dir
            / "trace_timestamps"
            / "trace_timestamps_test_processor_1.jsonl"
        )
        assert expected_file.exists()
        assert exporter.output_file == expected_file
        assert exporter.file_handle is not None

        await exporter.stop()

    @pytest.mark.asyncio
    async def test_process_record_writes_jsonl(
        self, exporter: TraceTimestampsExporter, sample_record: ParsedResponseRecord
    ):
        """Test that processing a record writes to JSONL file."""
        await exporter.initialize()

        result = await exporter.process_record(sample_record)

        # Should return empty MetricRecordDict (doesn't compute metrics)
        assert len(result) == 0
        assert exporter._records_written == 1

        await exporter.stop()

        # Verify JSONL content
        lines = exporter.output_file.read_text().strip().split("\n")
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert "timestamps" in data
        assert "metadata" in data
        assert "timing_breakdowns" in data
        assert "bandwidth_metrics" in data
        assert "statistical_analysis" in data
        assert "header_intelligence" in data
        assert "connection_insights" in data
        assert "quality_metrics" in data

    @pytest.mark.asyncio
    async def test_exports_all_timestamp_fields(
        self, exporter: TraceTimestampsExporter, sample_record: ParsedResponseRecord
    ):
        """Test that all timestamp fields are exported."""
        await exporter.initialize()
        await exporter.process_record(sample_record)
        await exporter.stop()

        lines = exporter.output_file.read_text().strip().split("\n")
        data = json.loads(lines[0])

        timestamps = data["timestamps"]
        assert timestamps["connection_queued_start_ns"] == 1000000000
        assert timestamps["connection_queued_end_ns"] == 1000500000
        assert timestamps["dns_resolvehost_start_ns"] == 1000500000
        assert timestamps["dns_resolvehost_end_ns"] == 1002000000
        assert timestamps["request_start_ns"] == 1005000000
        assert timestamps["request_headers_sent_ns"] == 1005100000
        assert timestamps["request_end_ns"] == 1005500000

    @pytest.mark.asyncio
    async def test_exports_computed_properties(
        self, exporter: TraceTimestampsExporter, sample_record: ParsedResponseRecord
    ):
        """Test that computed properties are calculated and exported."""
        await exporter.initialize()
        await exporter.process_record(sample_record)
        await exporter.stop()

        lines = exporter.output_file.read_text().strip().split("\n")
        data = json.loads(lines[0])

        # Check timing breakdowns
        timing = data["timing_breakdowns"]
        assert timing["connection_queue_wait_ns"] == 500000  # 1000500000 - 1000000000
        assert (
            timing["dns_resolution_duration_ns"] == 1500000
        )  # 1002000000 - 1000500000
        assert (
            timing["connection_create_duration_ns"] == 3000000
        )  # 1005000000 - 1002000000
        assert (
            timing["request_headers_duration_ns"] == 100000
        )  # 1005100000 - 1005000000
        assert timing["request_body_duration_ns"] == 400000  # 1005500000 - 1005100000

        # Check bandwidth
        bandwidth = data["bandwidth_metrics"]
        assert bandwidth["total_response_bytes"] == 450  # 100 + 150 + 200
        assert bandwidth["total_response_chunks"] == 3

        # Check statistical analysis
        stats = data["statistical_analysis"]
        assert stats["is_streaming_response"] is True
        assert len(stats["response_inter_chunk_latencies_ns"]) == 2

        # Check header intelligence
        headers = data["header_intelligence"]
        assert headers["compression_type"] == "gzip"
        assert headers["response_content_type"] == "application/json"
        assert headers["response_content_length"] == 450

        # Check connection insights
        connection = data["connection_insights"]
        assert connection["dns_was_cached"] is False

    @pytest.mark.asyncio
    async def test_multiple_records_appends_to_file(
        self, exporter: TraceTimestampsExporter, sample_record: ParsedResponseRecord
    ):
        """Test that processing multiple records appends to the same file."""
        await exporter.initialize()

        # Process 5 records
        for _ in range(5):
            await exporter.process_record(sample_record)

        await exporter.stop()

        # Verify 5 lines in JSONL
        lines = exporter.output_file.read_text().strip().split("\n")
        assert len(lines) == 5
        assert exporter._records_written == 5

        # Verify each line is valid JSON
        for line in lines:
            data = json.loads(line)
            assert "timestamps" in data
            assert "quality_metrics" in data

    @pytest.mark.asyncio
    async def test_handles_record_without_trace_timestamps(
        self, exporter: TraceTimestampsExporter
    ):
        """Test that records without trace timestamps are skipped gracefully."""
        # Create record without trace timestamps
        request = RequestRecord(
            start_perf_ns=1000000000,
            end_perf_ns=1020000000,
            status=200,
            trace_timestamps=None,  # No trace data
        )
        request.responses.append(TextResponse(perf_ns=1010000000, text="test"))

        record = ParsedResponseRecord(
            request=request,
            responses=[
                ParsedResponse(perf_ns=1010000000, data=TextResponseData(text="test"))
            ],
        )

        await exporter.initialize()
        result = await exporter.process_record(record)
        await exporter.stop()

        # Should not write anything
        assert exporter._records_written == 0
        assert len(result) == 0

        # File should be empty or have no lines
        content = exporter.output_file.read_text().strip()
        assert content == ""

    @pytest.mark.asyncio
    async def test_unique_filename_per_processor(
        self, user_config: UserConfig, temp_output_dir: Path
    ):
        """Test that each processor instance gets a unique filename."""
        import copy

        config = copy.deepcopy(user_config)
        config.output.artifact_directory = temp_output_dir

        exporter1 = TraceTimestampsExporter(user_config=config, processor_id="worker_1")
        exporter2 = TraceTimestampsExporter(user_config=config, processor_id="worker_2")

        await exporter1.initialize()
        await exporter2.initialize()

        assert exporter1.output_file != exporter2.output_file
        assert exporter1.output_file.name == "trace_timestamps_worker_1.jsonl"
        assert exporter2.output_file.name == "trace_timestamps_worker_2.jsonl"
        # Both should be in the trace_timestamps subdirectory
        assert exporter1.output_file.parent.name == "trace_timestamps"
        assert exporter2.output_file.parent.name == "trace_timestamps"

        await exporter1.stop()
        await exporter2.stop()
