# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.metric_utils import (
    build_hostname_aware_prometheus_endpoints,
    normalize_metrics_endpoint_url,
)


class TestNormalizeMetricsEndpointUrl:
    """Test URL normalization for metrics endpoints."""

    @pytest.mark.parametrize(
        "input_url,expected",
        [
            ("http://localhost:9400", "http://localhost:9400/metrics"),
            ("http://localhost:9400/", "http://localhost:9400/metrics"),
            ("http://localhost:9400/metrics", "http://localhost:9400/metrics"),
            ("http://localhost:9400/metrics/", "http://localhost:9400/metrics"),
            ("http://node1:8081", "http://node1:8081/metrics"),
            ("https://secure:443", "https://secure:443/metrics"),
            ("http://10.0.0.1:9090", "http://10.0.0.1:9090/metrics"),
        ],
    )  # fmt: skip
    def test_normalize_url_variations(self, input_url: str, expected: str):
        """Test URL normalization handles various input formats correctly."""
        assert normalize_metrics_endpoint_url(input_url) == expected

    def test_normalize_preserves_scheme(self):
        """Test that URL normalization preserves https scheme."""
        result = normalize_metrics_endpoint_url("https://secure:9400")
        assert result.startswith("https://")
        assert result == "https://secure:9400/metrics"

    def test_normalize_removes_trailing_slashes(self):
        """Test that multiple trailing slashes are removed."""
        result = normalize_metrics_endpoint_url("http://localhost:9400///")
        assert result == "http://localhost:9400/metrics"


class TestBuildHostnameAwarePrometheusEndpoints:
    """Test hostname-aware Prometheus endpoint URL generation."""

    def test_basic_endpoint_generation(self):
        """Test generating endpoints from inference URL and default ports."""
        endpoints = build_hostname_aware_prometheus_endpoints(
            "http://localhost:8000/v1/chat", [9400, 9401], include_inference_port=False
        )
        assert len(endpoints) == 2
        assert "http://localhost:9400/metrics" in endpoints
        assert "http://localhost:9401/metrics" in endpoints

    def test_preserves_scheme(self):
        """Test that generated endpoints use the same scheme as inference endpoint."""
        endpoints = build_hostname_aware_prometheus_endpoints(
            "https://secure-server:8000", [8081], include_inference_port=False
        )
        assert len(endpoints) == 1
        assert endpoints[0] == "https://secure-server:8081/metrics"

    def test_extracts_hostname_from_url(self):
        """Test hostname extraction from various URL formats."""
        test_cases = [
            ("http://node1:8000/v1/chat", [8081], "http://node1:8081/metrics"),
            ("http://10.0.0.5:8000", [9090], "http://10.0.0.5:9090/metrics"),
            (
                "https://api.example.com:443/inference",
                [8081],
                "https://api.example.com:8081/metrics",
            ),
        ]
        for inference_url, ports, expected_endpoint in test_cases:
            endpoints = build_hostname_aware_prometheus_endpoints(
                inference_url, ports, include_inference_port=False
            )
            assert expected_endpoint in endpoints

    def test_multiple_ports(self):
        """Test generating multiple endpoints from multiple ports."""
        endpoints = build_hostname_aware_prometheus_endpoints(
            "http://server:8000", [8081, 6880, 9090], include_inference_port=False
        )
        assert len(endpoints) == 3
        assert "http://server:8081/metrics" in endpoints
        assert "http://server:6880/metrics" in endpoints
        assert "http://server:9090/metrics" in endpoints

    def test_empty_ports_list(self):
        """Test behavior with empty ports list and no inference port."""
        endpoints = build_hostname_aware_prometheus_endpoints(
            "http://localhost:8000", [], include_inference_port=False
        )
        assert len(endpoints) == 0

    def test_url_without_port(self):
        """Test handling URLs without explicit port."""
        endpoints = build_hostname_aware_prometheus_endpoints(
            "http://localhost/v1/chat", [8081], include_inference_port=False
        )
        assert len(endpoints) == 1
        assert "http://localhost:8081/metrics" in endpoints

    def test_include_inference_port_with_explicit_port(self):
        """Test including inference endpoint port when explicitly specified."""
        endpoints = build_hostname_aware_prometheus_endpoints(
            "http://localhost:8000/v1/chat", [9400], include_inference_port=True
        )
        assert len(endpoints) == 2
        assert "http://localhost:8000/metrics" in endpoints
        assert "http://localhost:9400/metrics" in endpoints
        # Inference port should be first
        assert endpoints[0] == "http://localhost:8000/metrics"

    def test_include_inference_port_without_explicit_port(self):
        """Test including default port when inference URL has no port."""
        endpoints = build_hostname_aware_prometheus_endpoints(
            "http://localhost/v1/chat", [9400], include_inference_port=True
        )
        assert len(endpoints) == 2
        assert "http://localhost:80/metrics" in endpoints
        assert "http://localhost:9400/metrics" in endpoints

    def test_include_inference_port_https_default(self):
        """Test including HTTPS default port 443."""
        endpoints = build_hostname_aware_prometheus_endpoints(
            "https://secure/v1/chat", [9400], include_inference_port=True
        )
        assert len(endpoints) == 2
        assert "https://secure:443/metrics" in endpoints
        assert "https://secure:9400/metrics" in endpoints
