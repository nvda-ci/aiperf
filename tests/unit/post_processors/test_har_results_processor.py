# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for HAR Results Processor."""

from pathlib import Path

import orjson
import pytest

from aiperf.common.config import (
    EndpointConfig,
    OutputConfig,
    ServiceConfig,
    UserConfig,
)
from aiperf.common.enums import EndpointType, ExportLevel
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.models import (
    AioHttpTraceDataExport,
)
from aiperf.common.models.har_models import HAR
from aiperf.post_processors.har_results_processor import HARResultsProcessor
from tests.unit.post_processors.conftest import (
    create_metric_records_message,
)


@pytest.fixture
def tmp_artifact_dir(tmp_path: Path) -> Path:
    """Create a temporary artifact directory for testing."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


@pytest.fixture
def user_config_with_har_export(tmp_artifact_dir: Path) -> UserConfig:
    """Create a UserConfig with HAR export enabled."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            url="https://api.example.com",
            custom_endpoint="/v1/chat/completions",
        ),
        output=OutputConfig(
            artifact_directory=tmp_artifact_dir,
            export_level=ExportLevel.RECORDS,
        ),
    )


@pytest.fixture
def service_config() -> ServiceConfig:
    """Create a ServiceConfig for testing."""
    return ServiceConfig()


@pytest.fixture
def sample_trace_data() -> AioHttpTraceDataExport:
    """Create sample trace data for testing."""
    base_time = 1704067200000000000  # 2024-01-01 00:00:00 UTC

    return AioHttpTraceDataExport(
        trace_type="aiohttp",
        # Request phase
        request_headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer test",
        },
        request_send_start_ns=base_time,
        request_send_end_ns=base_time + 10_000_000,  # +10ms
        request_write_sizes_bytes=[500],
        # Response phase
        response_status_code=200,
        response_headers={"Content-Type": "application/json", "Content-Length": "1024"},
        response_receive_start_ns=base_time + 60_000_000,  # +60ms
        response_receive_end_ns=base_time + 100_000_000,  # +100ms
        response_receive_timestamps_ns=[
            base_time + 60_000_000,
            base_time + 100_000_000,
        ],
        response_receive_sizes_bytes=[512, 512],
        # Connection phase
        connection_pool_wait_start_ns=base_time - 50_000_000,  # -50ms
        connection_pool_wait_end_ns=base_time - 45_000_000,  # -45ms
        dns_lookup_start_ns=base_time - 45_000_000,
        dns_lookup_end_ns=base_time - 43_000_000,  # 2ms DNS
        tcp_connect_start_ns=base_time - 43_000_000,
        tcp_connect_end_ns=base_time - 28_000_000,  # 15ms connect (includes SSL)
    )


class TestHARResultsProcessorInitialization:
    """Test HARResultsProcessor initialization."""

    @pytest.mark.parametrize(
        "export_level, should_raise",  # fmt: skip
        [
            (ExportLevel.SUMMARY, True),
            (ExportLevel.RECORDS, False),
            (ExportLevel.RAW, False),
        ],
    )
    def test_init_with_export_levels(
        self,
        export_level: ExportLevel,
        should_raise: bool,
        user_config_with_har_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test initialization with different export levels."""
        user_config_with_har_export.output.export_level = export_level

        if should_raise:
            with pytest.raises(PostProcessorDisabled):
                HARResultsProcessor(
                    service_id="records-manager",
                    service_config=service_config,
                    user_config=user_config_with_har_export,
                )
        else:
            processor = HARResultsProcessor(
                service_id="records-manager",
                service_config=service_config,
                user_config=user_config_with_har_export,
            )

            assert processor.output_file.name == "profile_export.har"
            assert processor.output_file.parent.exists()
            assert processor._entry_count == 0
            assert processor._entries_with_trace == 0

    def test_init_creates_output_directory(
        self,
        user_config_with_har_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that initialization creates the output directory."""
        processor = HARResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_with_har_export,
        )

        assert processor.output_file.parent.exists()
        assert processor.output_file.parent.is_dir()

    def test_init_clears_existing_file(
        self,
        user_config_with_har_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that initialization clears existing output file."""
        output_file = user_config_with_har_export.output.profile_export_har_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text('{"existing": "content"}')

        processor = HARResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_with_har_export,
        )

        # File should be cleared
        assert (
            not processor.output_file.exists()
            or processor.output_file.stat().st_size == 0
        )


class TestHARResultsProcessorProcessResult:
    """Test HARResultsProcessor process_result method."""

    @pytest.mark.asyncio
    async def test_process_result_with_trace_data(
        self,
        user_config_with_har_export: UserConfig,
        service_config: ServiceConfig,
        sample_trace_data: AioHttpTraceDataExport,
    ):
        """Test processing a record with trace data."""
        processor = HARResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_with_har_export,
        )

        # Create a record with trace data
        message = create_metric_records_message(
            service_id="worker-1",
            x_request_id="test-123",
            results=[{"request_latency_ns": 100_000_000}],
        )
        record_data = message.to_data()
        record_data.trace_data = sample_trace_data

        await processor.process_result(record_data)

        assert processor._entry_count == 1
        assert processor._entries_with_trace == 1
        assert len(processor._entries) == 1

    @pytest.mark.asyncio
    async def test_process_result_without_trace_data(
        self,
        user_config_with_har_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test processing a record without trace data (should be skipped)."""
        processor = HARResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_with_har_export,
        )

        # Create a record without trace data
        message = create_metric_records_message(
            service_id="worker-1",
            x_request_id="test-123",
            results=[{"request_latency_ns": 100_000_000}],
        )
        record_data = message.to_data()
        record_data.trace_data = None

        await processor.process_result(record_data)

        assert processor._entry_count == 1
        assert processor._entries_with_trace == 0
        assert len(processor._entries) == 0

    @pytest.mark.asyncio
    async def test_process_multiple_results(
        self,
        user_config_with_har_export: UserConfig,
        service_config: ServiceConfig,
        sample_trace_data: AioHttpTraceDataExport,
    ):
        """Test processing multiple records."""
        processor = HARResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_with_har_export,
        )

        # Process 3 records
        for i in range(3):
            message = create_metric_records_message(
                service_id="worker-1",
                x_request_id=f"test-{i}",
                results=[{"request_latency_ns": 100_000_000}],
            )
            record_data = message.to_data()
            record_data.trace_data = sample_trace_data

            await processor.process_result(record_data)

        assert processor._entry_count == 3
        assert processor._entries_with_trace == 3
        assert len(processor._entries) == 3


class TestHARResultsProcessorSummarize:
    """Test HARResultsProcessor summarize method."""

    @pytest.mark.asyncio
    async def test_summarize_creates_har_file(
        self,
        user_config_with_har_export: UserConfig,
        service_config: ServiceConfig,
        sample_trace_data: AioHttpTraceDataExport,
    ):
        """Test that summarize creates a valid HAR file."""
        processor = HARResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_with_har_export,
        )

        # Process a record
        message = create_metric_records_message(
            service_id="worker-1",
            x_request_id="test-123",
            results=[{"request_latency_ns": 100_000_000}],
        )
        record_data = message.to_data()
        record_data.trace_data = sample_trace_data

        await processor.process_result(record_data)

        # Summarize
        results = await processor.summarize()

        # Should return empty list (HAR processor doesn't produce metric results)
        assert results == []

        # HAR file should be created
        assert processor.output_file.exists()

        # File should contain valid JSON
        har_bytes = processor.output_file.read_bytes()
        har_dict = orjson.loads(har_bytes)

        assert "log" in har_dict
        assert har_dict["log"]["version"] == "1.2"

    @pytest.mark.asyncio
    async def test_summarize_with_no_entries(
        self,
        user_config_with_har_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test summarize with no entries (should not create file)."""
        processor = HARResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_with_har_export,
        )

        # Summarize without processing any records
        results = await processor.summarize()

        assert results == []
        # File should not be created
        assert not processor.output_file.exists()

    @pytest.mark.asyncio
    async def test_har_structure_compliance(
        self,
        user_config_with_har_export: UserConfig,
        service_config: ServiceConfig,
        sample_trace_data: AioHttpTraceDataExport,
    ):
        """Test that generated HAR follows HAR 1.2 spec."""
        processor = HARResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_with_har_export,
        )

        # Process a record
        message = create_metric_records_message(
            service_id="worker-1",
            x_request_id="test-123",
            results=[{"request_latency_ns": 100_000_000}],
        )
        record_data = message.to_data()
        record_data.trace_data = sample_trace_data

        await processor.process_result(record_data)
        await processor.summarize()

        # Read and validate HAR structure
        har_bytes = processor.output_file.read_bytes()
        har_dict = orjson.loads(har_bytes)

        # Validate root structure
        assert "log" in har_dict

        log = har_dict["log"]
        assert log["version"] == "1.2"
        assert "creator" in log
        assert log["creator"]["name"] == "AIPerf"
        assert log["creator"]["version"] == "0.3.0"
        assert "entries" in log
        assert len(log["entries"]) == 1

        # Validate entry structure
        entry = log["entries"][0]
        assert "startedDateTime" in entry
        assert "time" in entry
        assert "request" in entry
        assert "response" in entry
        assert "cache" in entry
        assert "timings" in entry

        # Validate request
        request = entry["request"]
        assert request["method"] == "POST"
        assert "url" in request
        assert request["httpVersion"] == "HTTP/1.1"
        assert "headers" in request

        # Validate response
        response = entry["response"]
        assert response["status"] == 200
        assert response["statusText"] == "OK"
        assert "content" in response

        # Validate timings
        timings = entry["timings"]
        assert "send" in timings
        assert "wait" in timings
        assert "receive" in timings

    @pytest.mark.asyncio
    async def test_entries_sorted_by_timestamp(
        self,
        user_config_with_har_export: UserConfig,
        service_config: ServiceConfig,
        sample_trace_data: AioHttpTraceDataExport,
    ):
        """Test that entries are sorted by startedDateTime."""
        processor = HARResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_with_har_export,
        )

        # Process records with different timestamps
        base_time = 1704067200000000000
        for i in range(3):
            message = create_metric_records_message(
                service_id="worker-1",
                x_request_id=f"test-{i}",
                results=[{"request_latency_ns": 100_000_000}],
            )
            record_data = message.to_data()
            # Set different start times (in reverse order)
            record_data.metadata.request_start_ns = base_time + (2 - i) * 1_000_000_000
            record_data.trace_data = sample_trace_data

            await processor.process_result(record_data)

        await processor.summarize()

        # Read HAR
        har_bytes = processor.output_file.read_bytes()
        har_dict = orjson.loads(har_bytes)

        entries = har_dict["log"]["entries"]
        assert len(entries) == 3

        # Verify entries are sorted (oldest first)
        for i in range(len(entries) - 1):
            current_time = entries[i]["startedDateTime"]
            next_time = entries[i + 1]["startedDateTime"]
            assert current_time <= next_time

    @pytest.mark.asyncio
    async def test_har_timing_conversion(
        self,
        user_config_with_har_export: UserConfig,
        service_config: ServiceConfig,
        sample_trace_data: AioHttpTraceDataExport,
    ):
        """Test that timings are correctly converted from nanoseconds to milliseconds."""
        processor = HARResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_with_har_export,
        )

        message = create_metric_records_message(
            service_id="worker-1",
            x_request_id="test-123",
            results=[{"request_latency_ns": 100_000_000}],
        )
        record_data = message.to_data()
        record_data.trace_data = sample_trace_data

        await processor.process_result(record_data)
        await processor.summarize()

        # Read HAR
        har_bytes = processor.output_file.read_bytes()
        har_dict = orjson.loads(har_bytes)

        timings = har_dict["log"]["entries"][0]["timings"]

        # Verify timings are in milliseconds
        assert timings["blocked"] == pytest.approx(5.0)  # 5ms pool wait
        assert timings["dns"] == pytest.approx(2.0)  # 2ms DNS
        assert timings["connect"] == pytest.approx(15.0)  # 15ms TCP+SSL
        assert timings["send"] == pytest.approx(10.0)  # 10ms send
        assert timings["wait"] == pytest.approx(50.0)  # 50ms wait (TTFB)
        assert timings["receive"] == pytest.approx(40.0)  # 40ms receive

    @pytest.mark.asyncio
    async def test_har_can_be_deserialized(
        self,
        user_config_with_har_export: UserConfig,
        service_config: ServiceConfig,
        sample_trace_data: AioHttpTraceDataExport,
    ):
        """Test that generated HAR can be deserialized back to models."""
        processor = HARResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_with_har_export,
        )

        message = create_metric_records_message(
            service_id="worker-1",
            x_request_id="test-123",
            results=[{"request_latency_ns": 100_000_000}],
        )
        record_data = message.to_data()
        record_data.trace_data = sample_trace_data

        await processor.process_result(record_data)
        await processor.summarize()

        # Read and deserialize
        har_bytes = processor.output_file.read_bytes()
        har_dict = orjson.loads(har_bytes)

        # Should be able to create HAR model from it
        har = HAR(**har_dict)

        assert har.log.version == "1.2"
        assert len(har.log.entries) == 1
        assert har.log.entries[0].request.method == "POST"


class TestHARResultsProcessorEdgeCases:
    """Test edge cases for HAR Results Processor."""

    @pytest.mark.asyncio
    async def test_process_result_with_exception(
        self,
        user_config_with_har_export: UserConfig,
        service_config: ServiceConfig,
        sample_trace_data: AioHttpTraceDataExport,
    ):
        """Test that exceptions during processing are handled gracefully."""
        processor = HARResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_with_har_export,
        )

        # Create a record with invalid data
        message = create_metric_records_message(
            service_id="worker-1",
            x_request_id="test-123",
            results=[{"request_latency_ns": 100_000_000}],
        )
        record_data = message.to_data()
        record_data.trace_data = sample_trace_data

        # Patch _create_har_entry to raise an exception
        def raise_exception(*args, **kwargs):
            raise RuntimeError("Test exception")

        processor._create_har_entry = raise_exception

        # Should not raise, but should log error
        await processor.process_result(record_data)

        # Entry should not be added
        assert len(processor._entries) == 0
        # But count should be incremented
        assert processor._entry_count == 1

    @pytest.mark.asyncio
    async def test_url_construction_with_custom_endpoint(
        self,
        user_config_with_har_export: UserConfig,
        service_config: ServiceConfig,
        sample_trace_data: AioHttpTraceDataExport,
    ):
        """Test URL construction with custom endpoint."""
        processor = HARResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_with_har_export,
        )

        message = create_metric_records_message(
            service_id="worker-1",
            x_request_id="test-123",
            results=[{"request_latency_ns": 100_000_000}],
        )
        record_data = message.to_data()
        record_data.trace_data = sample_trace_data

        await processor.process_result(record_data)
        await processor.summarize()

        # Read HAR
        har_bytes = processor.output_file.read_bytes()
        har_dict = orjson.loads(har_bytes)

        url = har_dict["log"]["entries"][0]["request"]["url"]
        assert url == "https://api.example.com/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_missing_optional_trace_fields(
        self,
        user_config_with_har_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test handling of missing optional trace fields."""
        processor = HARResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_with_har_export,
        )

        # Create trace data with only required fields
        minimal_trace = AioHttpTraceDataExport(
            trace_type="aiohttp",
            request_send_start_ns=1704067200000000000,
            request_send_end_ns=1704067200010000000,
            response_receive_timestamps_ns=[1704067200060000000],
            response_receive_end_ns=1704067200100000000,
        )

        message = create_metric_records_message(
            service_id="worker-1",
            x_request_id="test-123",
            results=[{"request_latency_ns": 100_000_000}],
        )
        record_data = message.to_data()
        record_data.trace_data = minimal_trace

        await processor.process_result(record_data)
        await processor.summarize()

        # Should create HAR file without errors
        assert processor.output_file.exists()

        # Read HAR
        har_bytes = processor.output_file.read_bytes()
        har_dict = orjson.loads(har_bytes)

        timings = har_dict["log"]["entries"][0]["timings"]
        # Optional timings should be omitted or null
        assert "blocked" not in timings or timings["blocked"] is None
        assert "dns" not in timings or timings["dns"] is None
        assert "connect" not in timings or timings["connect"] is None
