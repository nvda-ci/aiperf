# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ServerMetricsDataCollector parsing functionality."""

import pytest


class TestMetricParsing:
    """Tests for parsing Prometheus metrics."""

    def test_parse_simple_counter(
        self, server_metrics_collector, sample_prometheus_text
    ):
        """Test parsing simple counter metrics."""
        records = server_metrics_collector._parse_metrics_to_records(
            sample_prometheus_text
        )

        assert len(records) == 1
        record = records[0]
        assert record.endpoint_url == "http://localhost:8081/metrics"
        # Note: prometheus_client parser strips _total suffix from counters
        assert "http_requests" in record.metrics
        assert "process_cpu_seconds" in record.metrics

        # Verify counter metric
        http_requests = record.metrics["http_requests"]
        assert http_requests.type == "counter"
        assert len(http_requests.samples) == 2

        # Verify labels and values
        labels_to_values = {
            tuple(sorted(s.labels.items())): s.value for s in http_requests.samples
        }
        assert labels_to_values[(("method", "GET"), ("status", "200"))] == 1547.0
        assert labels_to_values[(("method", "POST"), ("status", "200"))] == 892.0

    def test_parse_histogram(self, server_metrics_collector, sample_prometheus_text):
        """Test parsing histogram metrics."""
        records = server_metrics_collector._parse_metrics_to_records(
            sample_prometheus_text
        )

        assert len(records) == 1
        record = records[0]
        assert "http_request_duration_seconds" in record.metrics

        # Verify histogram metric
        duration_histogram = record.metrics["http_request_duration_seconds"]
        assert duration_histogram.type == "histogram"
        assert len(duration_histogram.samples) == 1

        # Verify histogram structure
        sample = duration_histogram.samples[0]
        assert sample.labels["method"] == "GET"
        assert sample.histogram is not None
        assert sample.histogram.sum == 125.5
        assert sample.histogram.count == 500.0
        assert sample.histogram.buckets["0.1"] == 100
        assert sample.histogram.buckets["0.5"] == 450
        assert sample.histogram.buckets["+Inf"] == 500

    @pytest.mark.parametrize(
        "empty_input",
        [
            "",
            "   \n  \n  ",
            "\t\t\n",
        ],
    )
    def test_empty_metrics(self, server_metrics_collector, empty_input):
        """Test parsing empty metrics."""
        records = server_metrics_collector._parse_metrics_to_records(empty_input)
        assert len(records) == 0


class TestMetricDeduplication:
    """Tests for metric deduplication logic."""

    def test_duplicate_labels_last_wins(self, server_metrics_collector):
        """Test that duplicate labels are de-duplicated (last wins)."""
        metrics_with_duplicates = """# HELP test_metric Test metric
# TYPE test_metric counter
test_metric{label="a"} 100
test_metric{label="a"} 200
"""

        records = server_metrics_collector._parse_metrics_to_records(
            metrics_with_duplicates
        )
        assert len(records) == 1

        test_metric = records[0].metrics["test_metric"]
        assert len(test_metric.samples) == 1
        assert test_metric.samples[0].value == 200.0  # Last value wins

    @pytest.mark.parametrize(
        "num_duplicates,expected_value",
        [
            (2, 200.0),
            (3, 300.0),
            (5, 500.0),
        ],
    )
    def test_multiple_duplicates(
        self, server_metrics_collector, num_duplicates, expected_value
    ):
        """Test deduplication with varying numbers of duplicates."""
        lines = ["# HELP test_metric Test metric\n# TYPE test_metric counter\n"]
        for i in range(1, num_duplicates + 1):
            lines.append(f'test_metric{{label="a"}} {i * 100.0}\n')

        metrics_text = "".join(lines)
        records = server_metrics_collector._parse_metrics_to_records(metrics_text)

        test_metric = records[0].metrics["test_metric"]
        assert len(test_metric.samples) == 1
        assert test_metric.samples[0].value == expected_value


class TestCreatedMetricsFiltering:
    """Tests for filtering _created metrics."""

    def test_created_metrics_filtered(self, server_metrics_collector):
        """Test that _created metrics are filtered out and not stored separately."""
        metrics_with_created = """# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET"} 1547.0

# HELP http_requests_total_created Total HTTP requests
# TYPE http_requests_total_created gauge
http_requests_total_created{method="GET"} 1.7624927793601315e+09

# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{method="GET",le="0.1"} 100
http_request_duration_seconds_bucket{method="GET",le="+Inf"} 500
http_request_duration_seconds_sum{method="GET"} 125.5
http_request_duration_seconds_count{method="GET"} 500

# HELP http_request_duration_seconds_created HTTP request duration
# TYPE http_request_duration_seconds_created gauge
http_request_duration_seconds_created{method="GET"} 1.7624927793601475e+09

# HELP active_requests Current active requests
# TYPE active_requests gauge
active_requests 42.0
"""

        records = server_metrics_collector._parse_metrics_to_records(
            metrics_with_created
        )
        assert len(records) == 1

        record = records[0]
        # Verify _created metrics are not in the parsed metrics
        assert "http_requests_total_created" not in record.metrics
        assert "http_request_duration_seconds_created" not in record.metrics

        # Verify the actual metrics are still there
        assert (
            "http_requests" in record.metrics
        )  # _total suffix stripped by prometheus_client
        assert "http_request_duration_seconds" in record.metrics
        assert "active_requests" in record.metrics

        # Verify the metrics have correct structure
        http_requests = record.metrics["http_requests"]
        assert http_requests.type == "counter"
        assert len(http_requests.samples) == 1
        assert http_requests.samples[0].value == 1547.0

        duration_histogram = record.metrics["http_request_duration_seconds"]
        assert duration_histogram.type == "histogram"
        assert len(duration_histogram.samples) == 1
        assert duration_histogram.samples[0].histogram is not None

        active = record.metrics["active_requests"]
        assert active.type == "gauge"
        assert len(active.samples) == 1
        assert active.samples[0].value == 42.0

    @pytest.mark.parametrize(
        "metric_name",
        [
            "http_requests_total_created",
            "http_request_duration_seconds_created",
            "rpc_duration_seconds_created",
            "some_metric_created",
        ],
    )
    def test_various_created_metrics_filtered(
        self, server_metrics_collector, metric_name
    ):
        """Test that various _created metrics are filtered."""
        base_name = metric_name.replace("_created", "")
        metrics_text = f"""# HELP {base_name} Test metric
# TYPE {base_name} counter
{base_name}{{method="GET"}} 100.0

# HELP {metric_name} Created timestamp
# TYPE {metric_name} gauge
{metric_name}{{method="GET"}} 1.7624927793601315e+09
"""

        records = server_metrics_collector._parse_metrics_to_records(metrics_text)
        assert len(records) == 1

        record = records[0]
        # Verify _created metric is not in parsed metrics
        assert metric_name not in record.metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
