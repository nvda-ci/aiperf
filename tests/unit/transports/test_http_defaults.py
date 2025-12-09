# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from aiperf.transports.http_defaults import AioHttpDefaults


class TestAioHttpDefaults:
    """Tests for AioHttpDefaults including aiohttp version compatibility."""

    def test_filter_supported_kwargs_keeps_supported_params(self):
        """Test that filter keeps parameters that TCPConnector supports."""
        kwargs = {"limit": 100, "force_close": True}
        result = AioHttpDefaults._filter_supported_tcp_connector_kwargs(kwargs)

        assert result["limit"] == 100
        assert result["force_close"] is True

    def test_filter_supported_kwargs_removes_unsupported_params(self):
        """Test that filter removes parameters that TCPConnector doesn't support."""
        kwargs = {"limit": 100, "nonexistent_param": "value"}
        result = AioHttpDefaults._filter_supported_tcp_connector_kwargs(kwargs)

        assert "limit" in result
        assert "nonexistent_param" not in result

    def test_filter_supported_kwargs_returns_empty_for_all_unsupported(self):
        """Test that filter returns empty dict when all params are unsupported."""
        kwargs = {"fake_param_1": "a", "fake_param_2": "b"}
        result = AioHttpDefaults._filter_supported_tcp_connector_kwargs(kwargs)

        assert result == {}

    def test_get_default_kwargs_contains_required_keys(self):
        """Test that get_default_kwargs returns expected connection settings."""
        kwargs = AioHttpDefaults.get_default_kwargs()

        assert "limit" in kwargs
        assert "limit_per_host" in kwargs
        assert "ttl_dns_cache" in kwargs
        assert "use_dns_cache" in kwargs
        assert "enable_cleanup_closed" in kwargs
        assert "force_close" in kwargs
        assert "keepalive_timeout" in kwargs
        assert "family" in kwargs

    def test_get_default_kwargs_includes_happy_eyeballs_when_supported(self):
        """Test happy_eyeballs_delay is included when aiohttp supports it."""
        with patch.object(
            AioHttpDefaults,
            "_filter_supported_tcp_connector_kwargs",
            side_effect=lambda kwargs: kwargs,  # Return all kwargs unfiltered
        ):
            kwargs = AioHttpDefaults.get_default_kwargs()
            assert "happy_eyeballs_delay" in kwargs

    def test_get_default_kwargs_excludes_happy_eyeballs_when_unsupported(self):
        """Test happy_eyeballs_delay is excluded when aiohttp doesn't support it."""

        def filter_out_happy_eyeballs(kwargs):
            return {k: v for k, v in kwargs.items() if k != "happy_eyeballs_delay"}

        with patch.object(
            AioHttpDefaults,
            "_filter_supported_tcp_connector_kwargs",
            side_effect=filter_out_happy_eyeballs,
        ):
            kwargs = AioHttpDefaults.get_default_kwargs()
            assert "happy_eyeballs_delay" not in kwargs

    @pytest.mark.parametrize(
        "input_kwargs,expected_keys",
        [
            ({"limit": 10}, {"limit"}),
            ({"limit": 10, "fake": 1}, {"limit"}),
            ({"fake1": 1, "fake2": 2}, set()),
            ({"limit": 10, "force_close": True, "fake": 1}, {"limit", "force_close"}),
        ],
        ids=["single-valid", "mixed", "all-invalid", "multiple-valid-with-invalid"],
    )  # fmt: skip
    def test_filter_supported_kwargs_various_inputs(self, input_kwargs, expected_keys):
        """Test _filter_supported_tcp_connector_kwargs with various inputs."""
        result = AioHttpDefaults._filter_supported_tcp_connector_kwargs(input_kwargs)
        assert set(result.keys()) == expected_keys

    def test_supports_tcp_connector_param_returns_true_for_known_params(self):
        """Test supports_tcp_connector_param returns True for known params."""
        assert AioHttpDefaults.supports_tcp_connector_param("limit") is True
        assert AioHttpDefaults.supports_tcp_connector_param("force_close") is True

    def test_supports_tcp_connector_param_returns_false_for_unknown_params(self):
        """Test supports_tcp_connector_param returns False for unknown params."""
        assert (
            AioHttpDefaults.supports_tcp_connector_param("nonexistent_param") is False
        )
