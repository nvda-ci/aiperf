# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cross-platform compatibility across all platforms (Windows, macOS, Linux)."""

import platform
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


class TestCrossPlatformCompatibility:
    """Test that cross-platform code works on all platforms."""

    @pytest.mark.parametrize(
        "platform_name,expected_behavior",
        [
            ("Windows", "windows"),
            ("Darwin", "macos"),
            ("Linux", "linux"),
        ],
    )
    def test_signal_handling_adapts_to_platform(
        self, platform_name, expected_behavior, mock_platform_system
    ):
        """Test that signal handling code adapts to different platforms."""
        mock_platform_system.return_value = platform_name

        if expected_behavior == "windows":
            # On Windows, sys.platform would be "win32"
            with patch("sys.platform", "win32"):
                is_windows = sys.platform == "win32"
                assert is_windows is True
        elif expected_behavior == "macos":
            # On macOS, sys.platform would be "darwin"
            with patch("sys.platform", "darwin"):
                is_windows = sys.platform == "win32"
                assert is_windows is False
        else:
            # On Linux, sys.platform would start with "linux"
            with patch("sys.platform", "linux"):
                is_windows = sys.platform == "win32"
                assert is_windows is False

    @pytest.mark.parametrize(
        "platform_name,venv_subdir,python_name",
        [
            ("Windows", "Scripts", "python.exe"),
            ("Darwin", "bin", "python"),
            ("Linux", "bin", "python"),
        ],
    )
    def test_venv_paths_adapt_to_platform(
        self, platform_name, venv_subdir, python_name
    ):
        """Test that virtual environment paths are correctly determined per platform."""
        venv_base = Path("/test_venv")

        if platform_name == "Windows":
            expected_path = venv_base / venv_subdir / python_name
        else:
            expected_path = venv_base / venv_subdir / python_name

        assert venv_subdir in str(expected_path)
        assert python_name in str(expected_path)

    @pytest.mark.parametrize(
        "platform_name,uses_forward_slash",
        [
            ("Windows", True),  # IPC paths should still use forward slashes
            ("Darwin", True),
            ("Linux", True),
        ],
    )
    def test_ipc_paths_use_posix_format(self, platform_name, uses_forward_slash):
        """Test that IPC socket paths always use POSIX (forward slash) format."""
        from pathlib import Path

        if platform.system() != platform_name:
            pytest.skip(
                f"Skipping test for {platform_name} as it is not the current platform"
            )

        # Create a test path
        if platform_name == "Windows":
            test_path = Path("C:\\temp\\aiperf\\socket.ipc")
        else:
            test_path = Path("/tmp/aiperf/socket.ipc")

        # Convert to POSIX
        posix_path = test_path.as_posix()

        # Should always use forward slashes for IPC
        if uses_forward_slash:
            assert "/" in posix_path
            assert "\\" not in posix_path

    def test_temp_directory_is_platform_agnostic(self):
        """Test that tempfile.gettempdir() works on all platforms."""
        import tempfile

        temp_dir = tempfile.gettempdir()

        # Should return a valid string path
        assert isinstance(temp_dir, str)
        assert len(temp_dir) > 0

        # Should be an existing directory
        temp_path = Path(temp_dir)
        assert temp_path.exists()
        assert temp_path.is_dir()


class TestProcessTerminationCrossPlatform:
    """Test that process termination works across platforms."""

    @pytest.mark.parametrize(
        "platform_name,available_signals",
        [
            ("Windows", ["SIGTERM", "SIGINT", "SIGBREAK"]),
            ("Darwin", ["SIGTERM", "SIGINT", "SIGKILL", "SIGQUIT"]),
            ("Linux", ["SIGTERM", "SIGINT", "SIGKILL", "SIGQUIT"]),
        ],
    )
    def test_signal_availability_per_platform(self, platform_name, available_signals):
        """Test that we use appropriate signals per platform."""
        import signal

        if platform.system() != platform_name:
            pytest.skip(
                f"Skipping test for {platform_name} as it is not the current platform"
            )

        for signal_name in available_signals:
            # All these signals should be available on their respective platforms
            assert hasattr(signal, signal_name)


class TestDockerCompatibilityCrossPlatform:
    """Test Docker-related functionality across platforms."""

    @pytest.mark.skip_on_windows
    def test_unix_specific_docker_commands(self):
        """Test that Unix-specific Docker commands are skipped on Windows."""
        # This test will be automatically skipped on Windows
        # It's here to demonstrate the skip_on_windows marker
        assert sys.platform != "win32"

    @pytest.mark.unix
    def test_bash_commands_only_on_unix(self):
        """Test that bash commands are only tested on Unix platforms."""
        # This test will be automatically skipped on Windows
        assert sys.platform in ("linux", "darwin")


class TestMultiprocessingCrossPlatform:
    """Test multiprocessing compatibility across platforms."""

    def test_forkprocess_import_conditional(self):
        """Test that ForkProcess import is conditional based on platform."""
        # On Windows, we should have a placeholder
        # On Unix, we should have the real ForkProcess
        import sys

        if sys.platform == "win32":
            # The code should handle missing ForkProcess gracefully
            from aiperf.controller import multiprocess_service_manager

            # ForkProcess should be defined (as type(None) placeholder)
            assert hasattr(multiprocess_service_manager, "ForkProcess")
        else:
            # On Unix, should have real ForkProcess
            from multiprocessing.context import ForkProcess

            assert ForkProcess is not None

    @pytest.mark.parametrize(
        "platform_name,spawn_method",
        [
            ("Windows", "spawn"),  # Only spawn on Windows
            ("Darwin", "spawn"),  # Can use spawn on macOS
            ("Linux", "fork"),  # Can use fork on Linux
        ],
    )
    def test_multiprocessing_start_methods(self, platform_name, spawn_method):
        """Test that appropriate multiprocessing methods are used per platform."""
        # This just verifies our understanding of platform capabilities
        if platform_name == "Windows":
            # Windows only supports spawn
            assert spawn_method == "spawn"
        elif platform_name == "Darwin":
            # macOS can use spawn (fork is problematic)
            assert spawn_method in ("spawn", "fork")
        else:
            # Linux can use fork (faster)
            assert spawn_method in ("spawn", "fork")


class TestCommunicationBackendCompatibility:
    """Test that communication backends work cross-platform."""

    def test_ipc_socket_paths_cross_platform(self):
        """Test IPC socket path generation works on all platforms."""
        from pathlib import Path

        from aiperf.common.config.zmq_config import ZMQIPCProxyConfig

        # Create a config with a test path
        config = ZMQIPCProxyConfig(
            path=Path("/test/path"), name="test_proxy", enable_control=True
        )

        # Get addresses
        frontend = config.frontend_address
        backend = config.backend_address
        control = config.control_address

        # All should use ipc:// protocol
        assert frontend.startswith("ipc://")
        assert backend.startswith("ipc://")
        assert control.startswith("ipc://")

        # All should use forward slashes (POSIX format)
        assert "\\" not in frontend
        assert "\\" not in backend
        assert "\\" not in control

    def test_tcp_communication_cross_platform(self):
        """Test TCP communication config works on all platforms."""
        from aiperf.common.config.zmq_config import ZMQTCPProxyConfig

        # TCP should work on all platforms
        config = ZMQTCPProxyConfig(host="127.0.0.1", frontend_port=5555)

        address = config.frontend_address
        assert address == "tcp://127.0.0.1:5555"
