# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import param

from aiperf.plugin import plugin_registry
from aiperf.plugin.enums import TransportType


class TestPluginRegistryDetectFromUrl:
    """Test suite for plugin_registry.detect_from_url method."""

    @pytest.mark.parametrize(
        "url,expected_transport",
        [
            param("http://api.example.com:8000", TransportType.HTTP.value, id="http_with_port"),
            param("https://api.example.com:8443", TransportType.HTTP.value, id="https_with_port"),
            param("http://localhost:8000", TransportType.HTTP.value, id="http_localhost"),
            param("http://127.0.0.1:8000", TransportType.HTTP.value, id="http_localhost_ip"),
            param("http://[::1]:8000", TransportType.HTTP.value, id="http_ipv6"),
            param("http://api.example.com", TransportType.HTTP.value, id="http_no_port"),
            param("https://api.example.com", TransportType.HTTP.value, id="https_no_port"),
            param("http://localhost:8000/api/v1/chat", TransportType.HTTP.value, id="with_path"),
            param("http://api.example.com?model=gpt-4&key=value", TransportType.HTTP.value, id="with_query"),
            param("http://user:password@api.example.com:8000", TransportType.HTTP.value, id="with_credentials"),
            param("http://api.example.com#section", TransportType.HTTP.value, id="with_fragment"),
            param("http://api.example.com/path/with%20spaces", TransportType.HTTP.value, id="with_encoded_spaces"),
            param("https://api.openai.com/v1/chat/completions", TransportType.HTTP.value, id="openai_api"),
        ],
    )  # fmt: skip
    def test_http_https_detection(self, url, expected_transport):
        """Test detection of HTTP/HTTPS URLs with various components."""
        result = plugin_registry.detect_type_from_url("transport", url)
        assert result is not None
        assert result == expected_transport

    @pytest.mark.parametrize(
        "url",
        [
            param("HTTP://api.example.com", id="uppercase_scheme"),
            param("Http://api.example.com", id="mixed_case_scheme"),
            param("hTTp://api.example.com", id="random_case_scheme"),
        ],
    )
    def test_scheme_case_insensitive(self, url):
        """Test that scheme detection is case-insensitive."""
        result = plugin_registry.detect_type_from_url("transport", url)
        assert result == TransportType.HTTP.value

    @pytest.mark.parametrize(
        "url",
        [
            param("", id="empty_string"),
            param("http://", id="scheme_only"),
            param("api.example.com:8000", id="no_scheme_with_port"),
            param("api.example.com", id="no_scheme_no_port"),
            param("localhost", id="localhost_no_scheme"),
            param("/path/to/file.sock", id="file_path"),  # defaults to http (no scheme)
        ],
    )
    def test_edge_cases_default_to_http_or_raise(self, url):
        """Test edge cases return HTTP or raise ValueError."""
        try:
            result = plugin_registry.detect_type_from_url("transport", url)
            assert result == TransportType.HTTP.value
        except ValueError:
            # No transport found is also acceptable for malformed URLs
            pass

    @pytest.mark.parametrize(
        "url",
        [
            param("unknown://api.example.com", id="unknown_scheme"),
            param("ftp://files.example.com", id="ftp_scheme"),
            param("grpc://localhost:50051", id="grpc_scheme"),
        ],
    )
    def test_unregistered_schemes_raise_error(self, url):
        """Test that unregistered schemes raise ValueError."""
        with pytest.raises(ValueError):
            plugin_registry.detect_type_from_url("transport", url)
