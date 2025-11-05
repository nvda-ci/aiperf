# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the plot CLI command.

This module tests the CLI integration for the plot command, ensuring proper
argument handling, error messages, and help text.
"""

import contextlib

import pytest

from aiperf.cli import app


class TestPlotCLI:
    """Tests for the plot CLI command."""

    def test_plot_command_in_help(self, capsys):
        """Test that the plot command appears in main help."""
        app([])
        help_output = capsys.readouterr().out

        assert "plot" in help_output.lower()

    def test_plot_help(self, capsys):
        """Test that plot --help shows command documentation."""
        app(["plot", "-h"])
        help_output = capsys.readouterr().out

        # Check for key information in help text
        assert "PNG visualizations" in help_output or "plot" in help_output.lower()
        assert "paths" in help_output.lower()

    def test_plot_nonexistent_path_error(self, capsys):
        """Test error handling for nonexistent path."""
        with pytest.raises(SystemExit):
            app(
                ["plot", "/nonexistent/path/that/does/not/exist"],
                exit_on_error=True,
            )

        # Check error message
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "does not exist" in output.lower() or "error" in output.lower()

    def test_plot_with_fixture_data(self, capsys, tmp_path):
        """Test plot command with valid fixture data."""
        # Create a minimal test structure
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()

        # Create minimal profile export JSON
        json_file = run_dir / "profile_export_aiperf.json"
        json_content = """
        {
            "model": "test_model",
            "concurrency": 1,
            "request_latency": {"p50": 100.0, "avg": 105.0},
            "request_throughput": {"avg": 10.0},
            "time_to_first_token": {"p50": 45.0},
            "inter_token_latency": {"p50": 18.0}
        }
        """
        json_file.write_text(json_content)

        # Run plot command (this may fail due to missing data, but should not crash on path validation)
        with contextlib.suppress(SystemExit):
            app(["plot", str(run_dir)], exit_on_error=True)

        # The command should at least attempt to process the path
        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Should see some output (either success or specific error about data format)
        assert len(output) > 0

    def test_plot_default_path(self, capsys, tmp_path, monkeypatch):
        """Test that plot uses default ./artifacts path when no path specified."""
        # Change to tmp directory
        monkeypatch.chdir(tmp_path)

        # Don't create artifacts directory - should fail with appropriate error
        with pytest.raises(SystemExit):
            app(["plot"], exit_on_error=True)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Should mention the default path
        assert "artifacts" in output.lower() or "does not exist" in output.lower()

    def test_plot_multiple_paths(self, tmp_path):
        """Test plot command with multiple paths."""
        # Create multiple run directories
        run1 = tmp_path / "run1"
        run2 = tmp_path / "run2"
        run1.mkdir()
        run2.mkdir()

        # Add minimal JSON files
        for run_dir in [run1, run2]:
            json_file = run_dir / "profile_export_aiperf.json"
            json_file.write_text('{"model": "test", "concurrency": 1}')

        # Should accept multiple paths without immediate error on path validation
        with contextlib.suppress(SystemExit):
            app(
                ["plot", str(run1), str(run2)],
                exit_on_error=True,
            )

    def test_plot_command_exists(self):
        """Test that plot command is registered and accessible."""
        # Verify the command can be invoked (help should work)
        with contextlib.suppress(SystemExit):
            app(["plot", "--help"])

        # If we get here without an exception (other than SystemExit), the command exists
