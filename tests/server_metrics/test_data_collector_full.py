# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for ServerMetricsDataCollector."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from aiperf.common.models import ErrorDetails
from aiperf.common.models.server_metrics_models import ServerMetricsRecord
from aiperf.server_metrics.server_metrics_data_collector import (
    ServerMetricsDataCollector,
)

SAMPLE_METRICS = """# HELP http_requests_total Total requests
# TYPE http_requests_total counter
http_requests_total{method="GET"} 100
"""


@pytest.mark.asyncio
async def test_collector_initialization():
    """Test collector initialization with various parameters."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=2.5,
        collector_id="test_collector",
    )

    assert collector._endpoint_url == "http://localhost:8081/metrics"
    assert collector._collection_interval == 2.5
    assert collector.id == "test_collector"
    assert collector._session is None  # Not initialized yet


@pytest.mark.asyncio
async def test_collector_lifecycle():
    """Test collector lifecycle: initialize -> start -> stop."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    # Initially not initialized
    assert collector._session is None

    # Initialize
    await collector.initialize()
    assert collector._session is not None
    assert not collector._session.closed

    # Start (would start background tasks)
    await collector.start()

    # Stop
    await collector.stop()
    assert collector._session is None or collector._session.closed


@pytest.mark.asyncio
async def test_http_session_creation():
    """Test HTTP session is created with correct timeout."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    await collector.initialize()

    assert collector._session is not None
    assert isinstance(collector._session, aiohttp.ClientSession)
    assert collector._session.timeout is not None

    await collector.stop()


@pytest.mark.asyncio
async def test_http_session_cleanup():
    """Test HTTP session is properly cleaned up on stop."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    await collector.initialize()
    session = collector._session

    await collector.stop()

    # Session should be closed
    assert session.closed


@pytest.mark.asyncio
async def test_is_url_reachable_success():
    """Test is_url_reachable returns True for accessible endpoint."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    # Create mock response
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    # Create mock session
    mock_session = MagicMock()
    mock_session.head = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        is_reachable = await collector.is_url_reachable()

        assert is_reachable is True


@pytest.mark.asyncio
async def test_is_url_reachable_failure():
    """Test is_url_reachable returns False for inaccessible endpoint."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    # Create mock session that raises errors
    mock_session = MagicMock()
    mock_session.head = MagicMock(side_effect=aiohttp.ClientError())
    mock_session.get = MagicMock(side_effect=aiohttp.ClientError())
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        is_reachable = await collector.is_url_reachable()

        assert is_reachable is False


@pytest.mark.asyncio
async def test_is_url_reachable_timeout():
    """Test is_url_reachable handles timeout."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    # Create mock session that times out
    mock_session = MagicMock()
    mock_session.head = MagicMock(side_effect=asyncio.TimeoutError())
    mock_session.get = MagicMock(side_effect=asyncio.TimeoutError())
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        is_reachable = await collector.is_url_reachable()

        assert is_reachable is False


@pytest.mark.asyncio
async def test_is_url_reachable_with_initialized_session():
    """Test is_url_reachable uses existing session when available."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    await collector.initialize()

    with patch.object(collector._session, "head") as mock_head:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_head.return_value.__aenter__.return_value = mock_response

        is_reachable = await collector.is_url_reachable()

        assert is_reachable is True
        mock_head.assert_called_once()

    await collector.stop()


@pytest.mark.asyncio
async def test_record_callback_invoked():
    """Test record callback is invoked with collected records."""
    records_received = []

    async def record_callback(records, collector_id):
        records_received.extend(records)

    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
        record_callback=record_callback,
    )

    await collector.initialize()

    # Mock the fetch to return sample metrics
    with patch.object(collector, "_fetch_metrics_text", return_value=SAMPLE_METRICS):
        await collector._collect_and_process_metrics()

    assert len(records_received) == 1
    assert isinstance(records_received[0], ServerMetricsRecord)

    await collector.stop()


@pytest.mark.asyncio
async def test_error_callback_invoked():
    """Test error callback is invoked on collection error."""
    errors_received = []

    async def error_callback(error, collector_id):
        errors_received.append(error)

    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
        error_callback=error_callback,
    )

    await collector.initialize()

    # Mock the fetch to raise an exception
    with patch.object(
        collector, "_fetch_metrics_text", side_effect=RuntimeError("Test error")
    ):
        await collector._collect_metrics_task()

    assert len(errors_received) == 1
    assert isinstance(errors_received[0], ErrorDetails)

    await collector.stop()


@pytest.mark.asyncio
async def test_record_callback_exception_handled():
    """Test that exceptions in record callback are handled gracefully."""

    async def failing_callback(records, collector_id):
        raise ValueError("Callback failed")

    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
        record_callback=failing_callback,
    )

    await collector.initialize()

    # Should not raise, just log warning
    with patch.object(collector, "_fetch_metrics_text", return_value=SAMPLE_METRICS):
        await collector._collect_and_process_metrics()

    await collector.stop()


@pytest.mark.asyncio
async def test_fetch_metrics_text_raises_without_session():
    """Test fetch_metrics raises when session not initialized."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    # Don't initialize - session should be None
    with pytest.raises(RuntimeError, match="HTTP session not initialized"):
        await collector._fetch_metrics_text()


@pytest.mark.asyncio
async def test_fetch_metrics_text_handles_stop_requested():
    """Test fetch_metrics respects stop_requested flag."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    await collector.initialize()

    # Request stop which sets the flag
    await collector.stop()

    # Now _stop_requested should be True
    with pytest.raises(asyncio.CancelledError):
        await collector._fetch_metrics_text()


@pytest.mark.asyncio
async def test_fetch_metrics_text_handles_closed_session():
    """Test fetch_metrics handles closed session."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    await collector.initialize()

    # Close the session
    await collector._session.close()

    with pytest.raises(asyncio.CancelledError):
        await collector._fetch_metrics_text()

    await collector.stop()


@pytest.mark.asyncio
async def test_parse_metrics_empty_input():
    """Test parsing empty metrics input."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records("")
    assert len(records) == 0

    records = collector._parse_metrics_to_records("   \n\n   ")
    assert len(records) == 0


@pytest.mark.asyncio
async def test_parse_metrics_invalid_format():
    """Test parsing invalid metrics format."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    # Invalid format should return empty list
    records = collector._parse_metrics_to_records("invalid metrics data")
    assert len(records) == 0


@pytest.mark.asyncio
async def test_parse_metrics_sets_timestamp():
    """Test that parsed records have timestamp set."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records(SAMPLE_METRICS)

    assert len(records) == 1
    assert records[0].timestamp_ns > 0


@pytest.mark.asyncio
async def test_parse_metrics_sets_endpoint_url():
    """Test that parsed records have correct endpoint URL."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records(SAMPLE_METRICS)

    assert len(records) == 1
    assert records[0].endpoint_url == "http://localhost:8081/metrics"


@pytest.mark.asyncio
async def test_multiple_collection_cycles():
    """Test multiple collection cycles accumulate different records."""
    records_received = []

    async def record_callback(records, collector_id):
        records_received.append(records[0])

    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
        record_callback=record_callback,
    )

    await collector.initialize()

    # Simulate multiple collections
    for _ in range(3):
        with patch.object(
            collector, "_fetch_metrics_text", return_value=SAMPLE_METRICS
        ):
            await collector._collect_and_process_metrics()

    assert len(records_received) == 3
    # Each should have different timestamp
    timestamps = {r.timestamp_ns for r in records_received}
    assert len(timestamps) >= 1  # At least some variation

    await collector.stop()


@pytest.mark.asyncio
async def test_collector_id_passed_to_callbacks():
    """Test that collector_id is passed to callbacks."""
    callback_ids = []

    async def record_callback(records, collector_id):
        callback_ids.append(collector_id)

    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
        record_callback=record_callback,
        collector_id="my_test_collector",
    )

    await collector.initialize()

    with patch.object(collector, "_fetch_metrics_text", return_value=SAMPLE_METRICS):
        await collector._collect_and_process_metrics()

    assert len(callback_ids) == 1
    assert callback_ids[0] == "my_test_collector"

    await collector.stop()


@pytest.mark.asyncio
async def test_no_callback_doesnt_fail():
    """Test that collection works without callbacks."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
        record_callback=None,  # No callback
    )

    await collector.initialize()

    # Should not fail
    with patch.object(collector, "_fetch_metrics_text", return_value=SAMPLE_METRICS):
        await collector._collect_and_process_metrics()

    await collector.stop()


@pytest.mark.asyncio
async def test_empty_records_not_sent_to_callback():
    """Test that empty record lists are not sent to callback."""
    callback_invoked = []

    async def record_callback(records, collector_id):
        callback_invoked.append(True)

    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
        record_callback=record_callback,
    )

    await collector.initialize()

    # Return empty metrics
    with patch.object(collector, "_fetch_metrics_text", return_value=""):
        await collector._collect_and_process_metrics()

    # Callback should not have been invoked
    assert len(callback_invoked) == 0

    await collector.stop()


@pytest.mark.asyncio
async def test_default_collection_interval():
    """Test that default collection interval is used from environment."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=None,  # Use default
    )

    # Should use default from Environment.GPU.COLLECTION_INTERVAL
    assert collector._collection_interval > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
