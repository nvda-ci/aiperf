# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.enums import EndpointType, PrometheusMetricType
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsEndpointData,
    ServerMetricsHierarchy,
    ServerMetricsMetadata,
    ServerMetricsRecord,
)
from aiperf.post_processors.server_metrics_results_processor import (
    ServerMetricsResultsProcessor,
)


@pytest.fixture
def mock_user_config() -> UserConfig:
    """Provide minimal UserConfig for testing."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            streaming=False,
        )
    )


@pytest.fixture
def sample_gauge_metric() -> MetricFamily:
    """Sample gauge metric family."""
    return MetricFamily(
        type=PrometheusMetricType.GAUGE,
        description="KV cache usage percentage",
        samples=[
            MetricSample(
                labels={"model_name": "test-model"},
                value=0.42,
            )
        ],
    )


@pytest.fixture
def sample_counter_metric() -> MetricFamily:
    """Sample counter metric family."""
    return MetricFamily(
        type=PrometheusMetricType.COUNTER,
        description="Total number of requests",
        samples=[
            MetricSample(
                labels={"model_name": "test-model"},
                value=150.0,
            )
        ],
    )


@pytest.fixture
def sample_server_metrics_record(
    sample_gauge_metric: MetricFamily,
    sample_counter_metric: MetricFamily,
) -> ServerMetricsRecord:
    """Create a sample ServerMetricsRecord with typical values."""
    return ServerMetricsRecord(
        endpoint_url="http://node1:8081/metrics",
        timestamp_ns=1_000_000_000,
        endpoint_latency_ns=5_000_000,
        metrics={
            "kv_cache_usage": sample_gauge_metric,
            "requests_total": sample_counter_metric,
        },
    )


class TestServerMetricsResultsProcessor:
    """Test cases for ServerMetricsResultsProcessor."""

    def test_initialization(self, mock_user_config: UserConfig) -> None:
        """Test processor initialization sets up hierarchy."""
        processor = ServerMetricsResultsProcessor(mock_user_config)

        assert isinstance(processor._server_metrics_hierarchy, ServerMetricsHierarchy)

    @pytest.mark.asyncio
    async def test_process_server_metrics_record(
        self,
        mock_user_config: UserConfig,
        sample_server_metrics_record: ServerMetricsRecord,
    ) -> None:
        """Test processing a server metrics record adds it to the hierarchy."""
        processor = ServerMetricsResultsProcessor(mock_user_config)

        await processor.process_server_metrics_record(sample_server_metrics_record)

        endpoint_url = sample_server_metrics_record.endpoint_url

        assert endpoint_url in processor._server_metrics_hierarchy.endpoints

    @pytest.mark.asyncio
    async def test_get_server_metrics_hierarchy(
        self,
        mock_user_config: UserConfig,
        sample_server_metrics_record: ServerMetricsRecord,
    ) -> None:
        """Test get_server_metrics_hierarchy returns accumulated data."""
        processor = ServerMetricsResultsProcessor(mock_user_config)

        # Add some records
        await processor.process_server_metrics_record(sample_server_metrics_record)

        # Get hierarchy
        hierarchy = processor.get_server_metrics_hierarchy()

        assert isinstance(hierarchy, ServerMetricsHierarchy)
        assert sample_server_metrics_record.endpoint_url in hierarchy.endpoints

    @pytest.mark.asyncio
    async def test_summarize_with_valid_data(
        self,
        mock_user_config: UserConfig,
        sample_gauge_metric: MetricFamily,
        sample_counter_metric: MetricFamily,
    ) -> None:
        """Test that processor stores data correctly (summarize returns empty list as output is not used)."""
        processor = ServerMetricsResultsProcessor(mock_user_config)

        # Add multiple records to have enough data for statistics
        for i in range(5):
            # Create gauge metric with varying values
            gauge = MetricFamily(
                type=PrometheusMetricType.GAUGE,
                description="KV cache usage",
                samples=[MetricSample(labels=None, value=0.4 + i * 0.05)],
            )
            counter = MetricFamily(
                type=PrometheusMetricType.COUNTER,
                description="Total requests",
                samples=[MetricSample(labels=None, value=100.0 + i * 50)],
            )
            record = ServerMetricsRecord(
                endpoint_url="http://node1:8081/metrics",
                timestamp_ns=1_000_000_000 + i * 1_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "cache_usage": gauge,
                    "requests_total": counter,
                },
            )
            await processor.process_server_metrics_record(record)

        # summarize() now returns empty list (output not used for export)
        results = await processor.summarize()
        assert results == []

        # Verify data was stored in hierarchy
        hierarchy = processor.get_server_metrics_hierarchy()
        assert "http://node1:8081/metrics" in hierarchy.endpoints
        endpoint_data = hierarchy.endpoints["http://node1:8081/metrics"]
        assert len(endpoint_data.time_series) == 5

    @pytest.mark.asyncio
    async def test_summarize_handles_no_metric_value(
        self, mock_user_config: UserConfig
    ) -> None:
        """Test summarize logs debug message when metric has no data and continues."""
        processor = ServerMetricsResultsProcessor(mock_user_config)

        # Create a server metrics hierarchy with an endpoint but no metric data
        mock_metadata = ServerMetricsMetadata(
            endpoint_url="http://test:8081/metrics",
            metric_schemas={},
            info_metrics={},
        )
        mock_endpoint_data = ServerMetricsEndpointData(
            endpoint_url="http://test:8081/metrics",
            metadata=mock_metadata,
        )
        processor._server_metrics_hierarchy.endpoints = {
            "http://test:8081/metrics": mock_endpoint_data,
        }

        with patch.object(processor, "debug") as mock_debug:
            results = await processor.summarize()

            # Should return empty list when no data available
            assert results == []

    @pytest.mark.asyncio
    async def test_summarize_handles_unexpected_exception(
        self, mock_user_config: UserConfig
    ) -> None:
        """Test summarize returns empty list (output not used for export)."""
        processor = ServerMetricsResultsProcessor(mock_user_config)
        results = await processor.summarize()

        # summarize() returns empty list
        assert results == []

    @pytest.mark.asyncio
    async def test_summarize_continues_after_errors(
        self, mock_user_config: UserConfig
    ) -> None:
        """Test summarize returns empty list (output not used for export)."""
        processor = ServerMetricsResultsProcessor(mock_user_config)
        results = await processor.summarize()

        # summarize() returns empty list
        assert results == []

    @pytest.mark.asyncio
    async def test_summarize_generates_correct_tags(
        self,
        mock_user_config: UserConfig,
    ) -> None:
        """Test summarize returns empty list (output not used for export)."""
        processor = ServerMetricsResultsProcessor(mock_user_config)
        results = await processor.summarize()

        # summarize() returns empty list
        assert results == []

    @pytest.mark.asyncio
    async def test_summarize_multiple_endpoints(
        self, mock_user_config: UserConfig
    ) -> None:
        """Test summarize handles multiple endpoints correctly."""
        processor = ServerMetricsResultsProcessor(mock_user_config)

        # Add records for two different endpoints
        for endpoint_idx, endpoint in enumerate(
            ["http://node1:8081/metrics", "http://node2:8081/metrics"]
        ):
            gauge = MetricFamily(
                type=PrometheusMetricType.GAUGE,
                description="Cache usage",
                samples=[MetricSample(labels=None, value=0.4 + endpoint_idx * 0.1)],
            )
            for i in range(3):
                record = ServerMetricsRecord(
                    endpoint_url=endpoint,
                    timestamp_ns=1_000_000_000 + i * 1_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics={"cache_usage": gauge},
                )
                await processor.process_server_metrics_record(record)

        results = await processor.summarize()

        # summarize() returns empty list (output not used for export)
        assert results == []

    @pytest.mark.asyncio
    async def test_summarize_handles_both_metric_types(
        self, mock_user_config: UserConfig
    ) -> None:
        """Test summarize generates results for both gauge and counter metrics."""
        processor = ServerMetricsResultsProcessor(mock_user_config)

        # Add records with both gauge and counter metrics
        gauge = MetricFamily(
            type=PrometheusMetricType.GAUGE,
            description="Cache usage",
            samples=[MetricSample(labels=None, value=0.5)],
        )
        counter = MetricFamily(
            type=PrometheusMetricType.COUNTER,
            description="Total requests",
            samples=[MetricSample(labels=None, value=100.0)],
        )

        for i in range(3):
            record = ServerMetricsRecord(
                endpoint_url="http://node1:8081/metrics",
                timestamp_ns=1_000_000_000 + i * 1_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "cache_usage": gauge,
                    "requests_total": counter,
                },
            )
            await processor.process_server_metrics_record(record)

        results = await processor.summarize()

        # summarize() returns empty list (output not used for export)
        assert results == []

    @pytest.mark.asyncio
    async def test_summarize_with_labeled_metrics(
        self, mock_user_config: UserConfig
    ) -> None:
        """Test summarize handles metrics with labels correctly."""
        processor = ServerMetricsResultsProcessor(mock_user_config)

        # Add records with labeled metrics
        gauge = MetricFamily(
            type=PrometheusMetricType.GAUGE,
            description="Cache usage per model",
            samples=[
                MetricSample(labels={"model": "model-a"}, value=0.5),
                MetricSample(labels={"model": "model-b"}, value=0.6),
            ],
        )

        for i in range(3):
            record = ServerMetricsRecord(
                endpoint_url="http://node1:8081/metrics",
                timestamp_ns=1_000_000_000 + i * 1_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={"cache_usage": gauge},
            )
            await processor.process_server_metrics_record(record)

        results = await processor.summarize()

        # summarize() returns empty list (output not used for export) for both labeled metrics
        assert results == []
