# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from pytest import param

from aiperf.common.environment import _ServerMetricsSettings, _ServiceSettings


class TestServiceSettingsUvloopWindows:
    """Test suite for automatic uvloop disabling on Windows."""

    @pytest.mark.parametrize(
        "platform_name,expected_disable_uvloop",
        [
            param("Windows", True, id="windows_auto_disabled"),
            param("Linux", False, id="linux_enabled"),
            param("Darwin", False, id="macos_enabled"),
        ],
    )
    @patch("aiperf.common.environment.platform.system")
    def test_platform_uvloop_detection(
        self, mock_platform, platform_name, expected_disable_uvloop
    ):
        """Test that uvloop is automatically disabled on Windows and enabled elsewhere."""
        mock_platform.return_value = platform_name

        settings = _ServiceSettings()

        assert settings.DISABLE_UVLOOP is expected_disable_uvloop

    @pytest.mark.parametrize(
        "platform_name,manual_setting,expected_result",
        [
            param("Windows", False, True, id="windows_override_attempt"),
            param("Windows", True, True, id="windows_manual_disable"),
            param("Linux", True, True, id="linux_manual_disable"),
            param("Linux", False, False, id="linux_default_enabled"),
            param("Darwin", True, True, id="macos_manual_disable"),
            param("Darwin", False, False, id="macos_default_enabled"),
        ],
    )
    @patch("aiperf.common.environment.platform.system")
    def test_manual_uvloop_settings(
        self, mock_platform, platform_name, manual_setting, expected_result
    ):
        """Test manual DISABLE_UVLOOP settings across platforms."""
        mock_platform.return_value = platform_name

        settings = _ServiceSettings(DISABLE_UVLOOP=manual_setting)

        assert settings.DISABLE_UVLOOP is expected_result


class TestServerMetricsSettings:
    """Test suite for server metrics environment settings."""

    def test_default_server_metrics_settings(self):
        """Test that server metrics settings have correct defaults."""
        settings = _ServerMetricsSettings()

        assert settings.COLLECTION_FLUSH_PERIOD == 2.0
        assert settings.COLLECTION_INTERVAL == 0.1
        assert settings.DEFAULT_BACKEND_PORTS == [8081, 6880]
        assert settings.REACHABILITY_TIMEOUT == 5
        assert settings.SHUTDOWN_DELAY == 5.0

    @pytest.mark.parametrize(
        "interval,expected",
        [
            param(0.1, 0.1, id="min_interval"),
            param(1.0, 1.0, id="default_interval"),
            param(5.0, 5.0, id="custom_interval"),
            param(300.0, 300.0, id="max_interval"),
        ],
    )
    def test_collection_interval_values(self, interval, expected):
        """Test various collection interval values."""
        settings = _ServerMetricsSettings(COLLECTION_INTERVAL=interval)
        assert expected == settings.COLLECTION_INTERVAL

    @pytest.mark.parametrize(
        "ports,expected",
        [
            param(
                "8081",
                [8081],
                id="single_port_string",
            ),
            param(
                "8081,6880",
                [8081, 6880],
                id="csv_ports",
            ),
            param(
                [8081, 6880, 9000],
                [8081, 6880, 9000],
                id="list_ports",
            ),
        ],
    )
    def test_default_backend_ports_parsing(self, ports, expected):
        """Test parsing of default backend ports from various formats."""
        settings = _ServerMetricsSettings(DEFAULT_BACKEND_PORTS=ports)
        assert expected == settings.DEFAULT_BACKEND_PORTS

    @pytest.mark.parametrize(
        "timeout,expected",
        [
            param(1, 1, id="min_timeout"),
            param(5, 5, id="default_timeout"),
            param(30, 30, id="custom_timeout"),
            param(300, 300, id="max_timeout"),
        ],
    )
    def test_reachability_timeout_values(self, timeout, expected):
        """Test various reachability timeout values."""
        settings = _ServerMetricsSettings(REACHABILITY_TIMEOUT=timeout)
        assert expected == settings.REACHABILITY_TIMEOUT

    @pytest.mark.parametrize(
        "delay,expected",
        [
            param(1.0, 1.0, id="min_delay"),
            param(5.0, 5.0, id="default_delay"),
            param(10.0, 10.0, id="custom_delay"),
        ],
    )
    def test_shutdown_delay_values(self, delay, expected):
        """Test various shutdown delay values."""
        settings = _ServerMetricsSettings(SHUTDOWN_DELAY=delay)
        assert expected == settings.SHUTDOWN_DELAY

    @pytest.mark.parametrize(
        "flush_period,expected",
        [
            param(0.0, 0.0, id="no_flush_period"),
            param(1.0, 1.0, id="min_flush_period"),
            param(2.0, 2.0, id="default_flush_period"),
            param(5.0, 5.0, id="custom_flush_period"),
            param(30.0, 30.0, id="max_flush_period"),
        ],
    )
    def test_collection_flush_period_values(self, flush_period, expected):
        """Test various collection flush period values."""
        settings = _ServerMetricsSettings(COLLECTION_FLUSH_PERIOD=flush_period)
        assert expected == settings.COLLECTION_FLUSH_PERIOD
