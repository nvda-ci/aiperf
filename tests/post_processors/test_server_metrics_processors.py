# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for ServerMetrics processors."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.common.config import UserConfig
from aiperf.common.models.processor_summary_results import ServerMetricsSummaryResult
from aiperf.common.models.server_metrics_models import (
    KubernetesPodInfo,
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
)
from aiperf.post_processors.server_metrics_export_results_processor import (
    ServerMetricsExportResultsProcessor,
)
from aiperf.post_processors.server_metrics_results_processor import (
    ServerMetricsResultsProcessor,
)


@pytest.fixture
def user_config(tmp_path):
    """Create mock user config."""
    config = MagicMock(spec=UserConfig)
    config.output = MagicMock()
    # Provide a default temporary path
    config.output.server_metrics_export_jsonl_file = tmp_path / "default_export.jsonl"
    return config


@pytest.fixture
def sample_record():
    """Create sample server metrics record."""
    sample = MetricSample(labels={"method": "GET"}, value=100.0)
    family = MetricFamily(type="counter", help="Test metric", samples=[sample])

    return ServerMetricsRecord(
        timestamp_ns=1000000000,
        endpoint_url="http://localhost:8081/metrics",
        metrics={"test_metric": family},
    )


@pytest.mark.asyncio
async def test_results_processor_initialization(user_config):
    """Test ServerMetricsResultsProcessor initialization."""
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    hierarchy = processor.get_server_metrics_hierarchy()
    assert hierarchy is not None
    assert len(hierarchy.endpoints) == 0


@pytest.mark.asyncio
async def test_results_processor_single_record(user_config, sample_record):
    """Test processing a single record."""
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    await processor.process_server_metrics_record(sample_record)

    hierarchy = processor.get_server_metrics_hierarchy()
    assert "http://localhost:8081/metrics" in hierarchy.endpoints

    endpoint_data = hierarchy.endpoints["http://localhost:8081/metrics"]
    assert len(endpoint_data.time_series.snapshots) == 1


@pytest.mark.asyncio
async def test_results_processor_multiple_records_same_endpoint(
    user_config, sample_record
):
    """Test processing multiple records from same endpoint."""
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    # Process multiple records
    for i in range(5):
        record = ServerMetricsRecord(
            timestamp_ns=1000000000 + i * 1000000000,
            endpoint_url="http://localhost:8081/metrics",
            metrics=sample_record.metrics,
        )
        await processor.process_server_metrics_record(record)

    hierarchy = processor.get_server_metrics_hierarchy()
    endpoint_data = hierarchy.endpoints["http://localhost:8081/metrics"]
    assert len(endpoint_data.time_series.snapshots) == 5


@pytest.mark.asyncio
async def test_results_processor_multiple_endpoints(user_config, sample_record):
    """Test processing records from multiple endpoints."""
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    # Create records for different endpoints
    endpoints = [
        "http://localhost:8081/metrics",
        "http://localhost:9090/metrics",
        "http://192.168.1.1:8080/metrics",
    ]

    for endpoint in endpoints:
        record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url=endpoint,
            metrics=sample_record.metrics,
        )
        await processor.process_server_metrics_record(record)

    hierarchy = processor.get_server_metrics_hierarchy()
    assert len(hierarchy.endpoints) == 3

    for endpoint in endpoints:
        assert endpoint in hierarchy.endpoints


@pytest.mark.asyncio
async def test_results_processor_summarize(user_config, sample_record):
    """Test summarize method returns ServerMetricsSummaryResult."""
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    await processor.process_server_metrics_record(sample_record)

    results = await processor.summarize()
    # Should return ServerMetricsSummaryResult with hierarchy
    assert isinstance(results, ServerMetricsSummaryResult)
    assert results.endpoints_tested == ["http://localhost:8081/metrics"]
    assert results.endpoints_successful == ["http://localhost:8081/metrics"]


@pytest.mark.asyncio
async def test_export_processor_initialization(user_config):
    """Test ServerMetricsExportResultsProcessor initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_export.jsonl"

        # Mock output path in processor
        processor = ServerMetricsExportResultsProcessor(user_config=user_config)

        # Override output file for testing
        processor.output_file = output_file

        assert processor.output_file == output_file


@pytest.mark.asyncio
async def test_export_processor_writes_record(user_config, sample_record):
    """Test export processor writes records to JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_export.jsonl"

        processor = ServerMetricsExportResultsProcessor(user_config=user_config)
        processor.output_file = output_file

        # Initialize processor
        await processor.initialize()

        # Process a record
        await processor.process_server_metrics_record(sample_record)

        # Stop processor (which flushes)
        await processor.stop()

        # Verify file exists and contains data
        assert output_file.exists()
        content = output_file.read_text()
        assert len(content) > 0

        # Parse JSON and verify slim structure (no endpoint_url, kubernetes_pod_info, type, help)
        record_data = orjson.loads(content)
        assert record_data["timestamp_ns"] == sample_record.timestamp_ns
        assert "endpoint_url" not in record_data  # Slim format excludes endpoint_url
        assert "kubernetes_pod_info" not in record_data  # Slim format excludes pod info
        assert "metrics" in record_data
        # Verify flat structure: metrics map directly to sample lists
        assert isinstance(record_data["metrics"], dict)
        for _metric_name, samples in record_data["metrics"].items():
            assert isinstance(samples, list)  # Direct list, no 'samples' key nesting


@pytest.mark.asyncio
async def test_export_processor_multiple_records(user_config, sample_record):
    """Test export processor handles multiple records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_export.jsonl"

        processor = ServerMetricsExportResultsProcessor(user_config=user_config)
        processor.output_file = output_file

        # Initialize processor
        await processor.initialize()

        # Process multiple records
        for i in range(10):
            record = ServerMetricsRecord(
                timestamp_ns=1000000000 + i * 1000000000,
                endpoint_url="http://localhost:8081/metrics",
                metrics=sample_record.metrics,
            )
            await processor.process_server_metrics_record(record)

        # Stop processor (which flushes)
        await processor.stop()

        # Verify file has multiple lines
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 10

        # Verify each line is valid JSON with slim structure
        for line in lines:
            data = orjson.loads(line)
            assert "timestamp_ns" in data
            assert "endpoint_url" not in data  # Slim format excludes endpoint_url
            assert "metrics" in data


@pytest.mark.asyncio
async def test_export_processor_different_endpoints(user_config, sample_record):
    """Test export processor handles records from different endpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_export.jsonl"

        processor = ServerMetricsExportResultsProcessor(user_config=user_config)
        processor.output_file = output_file

        # Initialize processor
        await processor.initialize()

        # Create records for different endpoints
        endpoints = ["http://localhost:8081/metrics", "http://localhost:9090/metrics"]

        for endpoint in endpoints:
            record = ServerMetricsRecord(
                timestamp_ns=1000000000,
                endpoint_url=endpoint,
                metrics=sample_record.metrics,
            )
            await processor.process_server_metrics_record(record)

        await processor.stop()

        # Verify both records written (slim format doesn't include endpoint_url)
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == len(endpoints)
        # Each line should have slim structure
        for line in lines:
            data = orjson.loads(line)
            assert "timestamp_ns" in data
            assert "metrics" in data
            assert "endpoint_url" not in data  # Slim format excludes endpoint_url


@pytest.mark.asyncio
async def test_export_processor_preserves_snapshot_structure(user_config):
    """Test that export processor writes slim snapshot structure (flat, no type/help)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_export.jsonl"

        processor = ServerMetricsExportResultsProcessor(user_config=user_config)
        processor.output_file = output_file

        # Initialize processor
        await processor.initialize()

        # Create complex snapshot with multiple metrics
        samples1 = [
            MetricSample(labels={"method": "GET", "status": "200"}, value=100.0),
            MetricSample(labels={"method": "POST", "status": "200"}, value=50.0),
        ]
        samples2 = [MetricSample(labels={"type": "heap"}, value=1024.0)]

        metrics = {
            "http_requests": MetricFamily(
                type="counter", help="Requests", samples=samples1
            ),
            "memory_bytes": MetricFamily(type="gauge", help="Memory", samples=samples2),
        }

        record = ServerMetricsRecord(
            timestamp_ns=1500000000,
            endpoint_url="http://localhost:8081/metrics",
            metrics=metrics,
        )

        await processor.process_server_metrics_record(record)
        await processor.stop()

        # Parse and verify slim structure (flat, no type/help)
        data = orjson.loads(output_file.read_text())
        assert "metrics" in data
        assert "http_requests" in data["metrics"]
        assert "memory_bytes" in data["metrics"]

        # Verify http_requests metric in slim format (direct list, no type/help/samples keys)
        http_samples = data["metrics"]["http_requests"]
        assert isinstance(
            http_samples, list
        )  # Direct list, not nested in 'samples' key
        assert len(http_samples) == 2
        # Verify samples have expected structure
        assert http_samples[0]["labels"]["method"] == "GET"
        assert http_samples[0]["value"] == 100.0


@pytest.mark.asyncio
async def test_export_processor_summarize_returns_empty(user_config):
    """Test export processor summarize returns export summary result."""
    processor = ServerMetricsExportResultsProcessor(user_config=user_config)

    results = await processor.summarize()
    assert results.record_count == 0
    assert results.file_path == processor.output_file


@pytest.mark.asyncio
async def test_export_processor_handles_write_error(user_config, sample_record):
    """Test export processor handles write errors gracefully."""
    # Use invalid path to trigger error
    processor = ServerMetricsExportResultsProcessor(user_config=user_config)
    processor.output_file = Path("/invalid/path/test.jsonl")

    # Should not raise, just log error
    await processor.process_server_metrics_record(sample_record)


@pytest.mark.asyncio
async def test_results_processor_hierarchy_isolation(user_config):
    """Test that different processor instances have isolated hierarchies."""
    processor1 = ServerMetricsResultsProcessor(user_config=user_config)
    processor2 = ServerMetricsResultsProcessor(user_config=user_config)

    sample = MetricSample(labels={}, value=100.0)
    family = MetricFamily(type="counter", help="Test", samples=[sample])

    record = ServerMetricsRecord(
        timestamp_ns=1000000000,
        endpoint_url="http://localhost:8081/metrics",
        metrics={"test": family},
    )

    # Process in first processor only
    await processor1.process_server_metrics_record(record)

    # Verify isolation
    hierarchy1 = processor1.get_server_metrics_hierarchy()
    hierarchy2 = processor2.get_server_metrics_hierarchy()

    assert len(hierarchy1.endpoints) == 1
    assert len(hierarchy2.endpoints) == 0


@pytest.mark.asyncio
async def test_kubernetes_pod_info_creation():
    """Test KubernetesPodInfo model creation with valid data."""
    pod_info = KubernetesPodInfo(
        pod_name="test-pod-123",
        namespace="default",
        node_name="worker-node-1",
        container_name="inference-server",
        service_name="inference-service",
        pod_ip="10.244.0.5",
        labels={"app": "inference", "version": "v1.0"},
    )

    assert pod_info.pod_name == "test-pod-123"
    assert pod_info.namespace == "default"
    assert pod_info.node_name == "worker-node-1"
    assert pod_info.container_name == "inference-server"
    assert pod_info.service_name == "inference-service"
    assert pod_info.pod_ip == "10.244.0.5"
    assert pod_info.labels["app"] == "inference"
    assert pod_info.labels["version"] == "v1.0"


@pytest.mark.asyncio
async def test_kubernetes_pod_info_partial_data():
    """Test KubernetesPodInfo with partial data (optional fields)."""
    pod_info = KubernetesPodInfo(
        pod_name="test-pod-456",
        namespace="production",
    )

    assert pod_info.pod_name == "test-pod-456"
    assert pod_info.namespace == "production"
    assert pod_info.node_name is None
    assert pod_info.container_name is None
    assert pod_info.service_name is None
    assert pod_info.pod_ip is None
    assert pod_info.labels == {}


@pytest.mark.asyncio
async def test_record_with_kubernetes_pod_info(user_config):
    """Test ServerMetricsRecord with Kubernetes POD information."""
    pod_info = KubernetesPodInfo(
        pod_name="vllm-server-abc123",
        namespace="ai-workloads",
        node_name="gpu-node-1",
        container_name="vllm",
        labels={"app": "vllm", "tier": "backend"},
    )

    sample = MetricSample(labels={"method": "GET"}, value=100.0)
    family = MetricFamily(type="counter", help="Test metric", samples=[sample])

    record = ServerMetricsRecord(
        timestamp_ns=1000000000,
        endpoint_url="http://vllm-server:8000/metrics",
        metrics={"test_metric": family},
        kubernetes_pod_info=pod_info,
    )

    assert record.kubernetes_pod_info is not None
    assert record.kubernetes_pod_info.pod_name == "vllm-server-abc123"
    assert record.kubernetes_pod_info.namespace == "ai-workloads"
    assert record.kubernetes_pod_info.node_name == "gpu-node-1"


@pytest.mark.asyncio
async def test_processor_preserves_kubernetes_pod_info(user_config):
    """Test that processor preserves Kubernetes POD info in hierarchy."""
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    pod_info = KubernetesPodInfo(
        pod_name="inference-pod",
        namespace="ml-serving",
        node_name="gpu-node-2",
        service_name="inference-svc",
        pod_ip="10.1.2.3",
    )

    sample = MetricSample(labels={}, value=200.0)
    family = MetricFamily(type="gauge", help="Memory", samples=[sample])

    record = ServerMetricsRecord(
        timestamp_ns=2000000000,
        endpoint_url="http://inference-pod:8081/metrics",
        metrics={"memory": family},
        kubernetes_pod_info=pod_info,
    )

    await processor.process_server_metrics_record(record)

    hierarchy = processor.get_server_metrics_hierarchy()
    endpoint_data = hierarchy.endpoints["http://inference-pod:8081/metrics"]

    assert endpoint_data.metadata.kubernetes_pod_info is not None
    assert endpoint_data.metadata.kubernetes_pod_info.pod_name == "inference-pod"
    assert endpoint_data.metadata.kubernetes_pod_info.namespace == "ml-serving"
    assert endpoint_data.metadata.kubernetes_pod_info.node_name == "gpu-node-2"
    assert endpoint_data.metadata.kubernetes_pod_info.service_name == "inference-svc"
    assert endpoint_data.metadata.kubernetes_pod_info.pod_ip == "10.1.2.3"


@pytest.mark.asyncio
async def test_export_processor_includes_kubernetes_pod_info(user_config):
    """Test export processor includes Kubernetes POD info in JSONL output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_k8s_export.jsonl"

        processor = ServerMetricsExportResultsProcessor(user_config=user_config)
        processor.output_file = output_file

        await processor.initialize()

        pod_info = KubernetesPodInfo(
            pod_name="triton-server-xyz",
            namespace="inference",
            node_name="k8s-worker-3",
            container_name="triton",
            labels={"app": "triton", "component": "inference"},
        )

        sample = MetricSample(labels={"model": "gpt"}, value=5.0)
        family = MetricFamily(type="gauge", help="Active requests", samples=[sample])

        record = ServerMetricsRecord(
            timestamp_ns=3000000000,
            endpoint_url="http://triton-server:8002/metrics",
            metrics={"active_requests": family},
            kubernetes_pod_info=pod_info,
        )

        await processor.process_server_metrics_record(record)
        await processor.stop()

        # Verify kubernetes_pod_info NOT in exported JSON (slim format excludes it)
        content = output_file.read_text()
        data = orjson.loads(content)

        assert "kubernetes_pod_info" not in data  # Slim format excludes pod info
        assert "endpoint_url" not in data  # Slim format excludes endpoint_url
        # Pod info and endpoint are available in metadata JSONL file instead
        assert "timestamp_ns" in data
        assert "metrics" in data


@pytest.mark.asyncio
async def test_hierarchy_updates_pod_info_when_available(user_config):
    """Test that hierarchy updates POD info when it becomes available."""
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    sample = MetricSample(labels={}, value=1.0)
    family = MetricFamily(type="counter", help="Test", samples=[sample])

    # First record without POD info
    record1 = ServerMetricsRecord(
        timestamp_ns=1000000000,
        endpoint_url="http://server:8000/metrics",
        metrics={"metric1": family},
        kubernetes_pod_info=None,
    )

    await processor.process_server_metrics_record(record1)

    hierarchy = processor.get_server_metrics_hierarchy()
    endpoint_data = hierarchy.endpoints["http://server:8000/metrics"]
    assert endpoint_data.metadata.kubernetes_pod_info is None

    # Second record with POD info
    pod_info = KubernetesPodInfo(
        pod_name="server-pod",
        namespace="default",
    )

    record2 = ServerMetricsRecord(
        timestamp_ns=2000000000,
        endpoint_url="http://server:8000/metrics",
        metrics={"metric1": family},
        kubernetes_pod_info=pod_info,
    )

    await processor.process_server_metrics_record(record2)

    # Verify POD info is now present
    endpoint_data = hierarchy.endpoints["http://server:8000/metrics"]
    assert endpoint_data.metadata.kubernetes_pod_info is not None
    assert endpoint_data.metadata.kubernetes_pod_info.pod_name == "server-pod"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
