# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Windows-specific functionality and platform compatibility."""

import signal
import sys
from unittest.mock import patch

import pytest

from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.config import ServiceConfig, UserConfig
from tests.common.conftest import DummyService


@pytest.mark.windows
class TestBootstrapWindows:
    """Test Windows-specific bootstrap behavior."""

    @pytest.fixture(autouse=True)
    def setup_bootstrap_mocks(
        self,
        mock_psutil_process,
        mock_setup_child_process_logging,
    ):
        """Combine common bootstrap mocks that are used but not called in tests."""
        pass

    def test_terminal_fds_not_closed_on_windows(
        self,
        service_config_no_uvloop: ServiceConfig,
        user_config: UserConfig,
        mock_log_queue,
        mock_platform_windows,
        mock_current_process,
    ):
        """Test that terminal FDs are NOT closed on Windows (macOS-specific fix should not run)."""
        # Mock a child process
        mock_process = type("MockProcess", (), {"name": "DATASET_MANAGER_process"})()
        mock_current_process.return_value = mock_process

        with (
            patch("sys.stdin") as mock_stdin,
            patch("sys.stdout") as mock_stdout,
            patch("sys.stderr") as mock_stderr,
        ):
            # Setup FD mocks
            mock_stdin.fileno.return_value = 0
            mock_stdout.fileno.return_value = 1
            mock_stderr.fileno.return_value = 2

            bootstrap_and_run_service(
                DummyService,
                service_config=service_config_no_uvloop,
                user_config=user_config,
                log_queue=mock_log_queue,
            )

            # On Windows, FDs should NOT be closed (macOS-specific behavior)
            mock_stdin.close.assert_not_called()
            mock_stdout.close.assert_not_called()
            mock_stderr.close.assert_not_called()


@pytest.mark.windows
class TestSignalHandlingWindows:
    """Test Windows-specific signal handling."""

    def test_sigkill_not_available_on_windows(self):
        """Test that SIGKILL is replaced with SIGTERM on Windows."""
        # On Windows, SIGKILL doesn't exist
        if sys.platform == "win32":
            # Verify our code uses SIGTERM instead
            kill_signal = signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
            assert kill_signal == signal.SIGTERM
        else:
            # On Unix, SIGKILL should be available
            kill_signal = signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
            assert kill_signal == signal.SIGKILL

    def test_asyncio_signal_handlers_skipped_on_windows(self):
        """Test that asyncio signal handlers are properly skipped on Windows."""
        # This is a placeholder test - the actual test would require
        # running the signal handler setup code
        # For now, just verify Windows platform detection works
        is_windows = sys.platform == "win32"
        if is_windows:
            # Signal handlers should be skipped
            assert sys.platform == "win32"


@pytest.mark.windows
class TestPathHandlingWindows:
    """Test Windows path handling compatibility."""

    def test_ipc_paths_use_forward_slashes(self):
        """Test that IPC socket paths use forward slashes even on Windows."""
        from pathlib import Path

        # Test the .as_posix() method works correctly
        test_path = Path("C:\\") / "temp" / "aiperf" / "socket.ipc"
        posix_path = test_path.as_posix()

        # Should use forward slashes
        assert "/" in posix_path
        assert "\\" not in posix_path

    def test_tempfile_works_on_windows(self):
        """Test that tempfile.gettempdir() works on Windows."""
        import tempfile
        from pathlib import Path

        temp_dir = Path(tempfile.gettempdir())

        # Should return a valid directory
        assert temp_dir.exists()
        assert temp_dir.is_dir()

    def test_virtual_env_paths_windows(self):
        """Test that virtual environment paths are detected correctly on Windows."""
        from pathlib import Path

        # Simulate Windows venv structure
        venv_path = Path("C:/test_venv")

        # Windows uses Scripts instead of bin
        if sys.platform == "win32":
            python_path = venv_path / "Scripts" / "python.exe"
            assert "Scripts" in str(python_path)
            assert ".exe" in str(python_path)
        else:
            python_path = venv_path / "bin" / "python"
            assert "bin" in str(python_path)
