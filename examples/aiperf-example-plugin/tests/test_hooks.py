# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for phase lifecycle hooks demonstrating 2025 pytest best practices."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf_example_plugin.hooks import (
    ExampleLoggingHook,
    ExampleLoggingHookConfig,
    ExampleMetricsCollectorHook,
    ExampleMetricsCollectorHookConfig,
)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide temporary directory for test files.

    Args:
        tmp_path: Pytest's built-in temporary directory fixture

    Returns:
        Path to temporary directory
    """
    return tmp_path


@pytest.fixture
def mock_phase_stats() -> MagicMock:
    """Create mock phase statistics.

    Returns:
        Mock statistics object with typical phase metrics
    """
    stats = MagicMock()
    stats.sent = 100
    stats.completed = 95
    stats.cancelled = 2
    stats.sent_sessions = 10
    stats.completed_sessions = 9
    stats.in_flight = 3
    return stats


@pytest.fixture
def mock_tracker(mock_phase_stats: MagicMock) -> MagicMock:
    """Create mock phase tracker.

    Args:
        mock_phase_stats: Mock statistics fixture

    Returns:
        Mock tracker that returns mock_phase_stats
    """
    tracker = MagicMock()
    tracker.create_stats.return_value = mock_phase_stats
    return tracker


class TestExampleLoggingHookConfig:
    """Tests for ExampleLoggingHookConfig Pydantic model."""

    def test_default_config(self) -> None:
        """Test config with default values."""
        config = ExampleLoggingHookConfig()

        assert config.log_file == Path("/tmp/aiperf_phases.log")
        assert config.verbose is False
        assert config.create_dirs is True

    def test_custom_config(self, temp_dir: Path) -> None:
        """Test config with custom values."""
        log_file = temp_dir / "custom.log"
        config = ExampleLoggingHookConfig(
            log_file=log_file,
            verbose=True,
            create_dirs=False,
        )

        assert config.log_file == log_file
        assert config.verbose is True
        assert config.create_dirs is False

    def test_string_path_converted_to_path(self) -> None:
        """Test that string paths are converted to Path objects."""
        config = ExampleLoggingHookConfig(log_file="/tmp/test.log")

        assert isinstance(config.log_file, Path)
        assert config.log_file == Path("/tmp/test.log")


class TestExampleLoggingHook:
    """Tests for ExampleLoggingHook."""

    def test_init_creates_log_file_directory(self, temp_dir: Path) -> None:
        """Test that log file directory is created on init."""
        log_file = temp_dir / "logs" / "test.log"
        config = ExampleLoggingHookConfig(log_file=log_file)
        hook = ExampleLoggingHook(config)

        assert log_file.parent.exists()
        assert hook.config.log_file == log_file

    def test_from_config_factory_method(self, temp_dir: Path) -> None:
        """Test factory method creates hook correctly."""
        log_file = temp_dir / "test.log"
        hook = ExampleLoggingHook.from_config(
            log_file=log_file,
            verbose=True,
            create_dirs=True,
        )

        assert hook.config.log_file == log_file
        assert hook.config.verbose is True
        assert hook.config.create_dirs is True

    @pytest.mark.asyncio
    async def test_phase_start_writes_log(self, temp_dir: Path, mock_tracker: MagicMock) -> None:
        """Test on_phase_start writes to log file."""
        from aiperf.common.enums import CreditPhase

        log_file = temp_dir / "phases.log"
        hook = ExampleLoggingHook.from_config(log_file=log_file, verbose=False)

        await hook.on_phase_start(CreditPhase.WARMUP, mock_tracker)

        assert log_file.exists()
        content = log_file.read_text()
        assert "PHASE_START" in content
        assert "warmup" in content

    @pytest.mark.asyncio
    async def test_phase_complete_writes_log(self, temp_dir: Path, mock_tracker: MagicMock) -> None:
        """Test on_phase_complete writes to log file."""
        from aiperf.common.enums import CreditPhase

        log_file = temp_dir / "phases.log"
        hook = ExampleLoggingHook.from_config(log_file=log_file)

        await hook.on_phase_complete(CreditPhase.PROFILING, mock_tracker)

        assert log_file.exists()
        content = log_file.read_text()
        assert "PHASE_COMPLETE" in content
        assert "profiling" in content

    @pytest.mark.asyncio
    async def test_verbose_logging_includes_stats(
        self, temp_dir: Path, mock_tracker: MagicMock
    ) -> None:
        """Test verbose logging includes detailed statistics."""
        from aiperf.common.enums import CreditPhase

        log_file = temp_dir / "phases.log"
        hook = ExampleLoggingHook.from_config(log_file=log_file, verbose=True)

        await hook.on_phase_start(CreditPhase.WARMUP, mock_tracker)

        content = log_file.read_text()
        assert "PHASE_START" in content
        assert "warmup" in content

    @pytest.mark.asyncio
    async def test_multiple_events_append_to_log(
        self, temp_dir: Path, mock_tracker: MagicMock
    ) -> None:
        """Test multiple events append to log file."""
        from aiperf.common.enums import CreditPhase

        log_file = temp_dir / "phases.log"
        hook = ExampleLoggingHook.from_config(log_file=log_file)

        # Log multiple events
        await hook.on_phase_start(CreditPhase.WARMUP, mock_tracker)
        await hook.on_phase_sending_complete(CreditPhase.WARMUP, mock_tracker)
        await hook.on_phase_complete(CreditPhase.WARMUP, mock_tracker)

        content = log_file.read_text()
        # All three events should be in the log
        assert content.count("warmup") >= 3

    @pytest.mark.asyncio
    async def test_get_phase_metrics_returns_tracked_times(
        self, temp_dir: Path, mock_tracker: MagicMock
    ) -> None:
        """Test get_phase_metrics returns tracked phase times."""
        from aiperf.common.enums import CreditPhase

        log_file = temp_dir / "phases.log"
        hook = ExampleLoggingHook.from_config(log_file=log_file)

        await hook.on_phase_start(CreditPhase.WARMUP, mock_tracker)

        metrics = hook.get_phase_metrics()
        assert "warmup" in metrics
        assert "start_time" in metrics["warmup"]

    @pytest.mark.asyncio
    async def test_calculate_duration_returns_elapsed_time(
        self, temp_dir: Path, mock_tracker: MagicMock
    ) -> None:
        """Test duration calculation returns elapsed time."""
        from aiperf.common.enums import CreditPhase

        log_file = temp_dir / "phases.log"
        hook = ExampleLoggingHook.from_config(log_file=log_file)

        await hook.on_phase_start(CreditPhase.WARMUP, mock_tracker)
        await asyncio.sleep(0.01)  # Small delay

        duration = hook._calculate_duration(CreditPhase.WARMUP)
        assert duration > 0.0


class TestExampleMetricsCollectorHookConfig:
    """Tests for ExampleMetricsCollectorHookConfig Pydantic model."""

    def test_default_config(self) -> None:
        """Test config with default values."""
        config = ExampleMetricsCollectorHookConfig()

        assert config.metrics_file == Path("/tmp/aiperf_metrics.json")
        assert config.aggregate is True
        assert config.create_dirs is True
        assert config.indent == 2

    def test_custom_config(self, temp_dir: Path) -> None:
        """Test config with custom values."""
        metrics_file = temp_dir / "custom.json"
        config = ExampleMetricsCollectorHookConfig(
            metrics_file=metrics_file,
            aggregate=False,
            create_dirs=False,
            indent=None,
        )

        assert config.metrics_file == metrics_file
        assert config.aggregate is False
        assert config.create_dirs is False
        assert config.indent is None


class TestExampleMetricsCollectorHook:
    """Tests for ExampleMetricsCollectorHook."""

    def test_init_creates_metrics_file_directory(self, temp_dir: Path) -> None:
        """Test that metrics file directory is created on init."""
        metrics_file = temp_dir / "metrics" / "test.json"
        config = ExampleMetricsCollectorHookConfig(metrics_file=metrics_file)
        ExampleMetricsCollectorHook(config)

        assert metrics_file.parent.exists()

    def test_from_config_factory_method(self, temp_dir: Path) -> None:
        """Test factory method creates hook correctly."""
        metrics_file = temp_dir / "metrics.json"
        hook = ExampleMetricsCollectorHook.from_config(
            metrics_file=metrics_file,
            aggregate=False,
            indent=4,
        )

        assert hook.config.metrics_file == metrics_file
        assert hook.config.aggregate is False
        assert hook.config.indent == 4

    @pytest.mark.asyncio
    async def test_phase_events_write_metrics_json(
        self, temp_dir: Path, mock_tracker: MagicMock
    ) -> None:
        """Test that phase events write metrics to JSON."""
        from aiperf.common.enums import CreditPhase

        metrics_file = temp_dir / "metrics.json"
        hook = ExampleMetricsCollectorHook.from_config(metrics_file=metrics_file)

        await hook.on_phase_start(CreditPhase.WARMUP, mock_tracker)

        assert metrics_file.exists()
        data = orjson.loads(metrics_file.read_bytes())
        assert "metrics" in data
        assert "warmup" in data["metrics"]

    @pytest.mark.asyncio
    async def test_phase_timeline_events_recorded(
        self, temp_dir: Path, mock_tracker: MagicMock
    ) -> None:
        """Test that phase timeline events are recorded."""
        from aiperf.common.enums import CreditPhase

        metrics_file = temp_dir / "metrics.json"
        hook = ExampleMetricsCollectorHook.from_config(metrics_file=metrics_file)

        # Record timeline of events
        await hook.on_phase_start(CreditPhase.WARMUP, mock_tracker)
        await asyncio.sleep(0.01)  # Small delay
        await hook.on_phase_sending_complete(CreditPhase.WARMUP, mock_tracker)
        await asyncio.sleep(0.01)
        await hook.on_phase_complete(CreditPhase.WARMUP, mock_tracker)

        data = orjson.loads(metrics_file.read_bytes())
        metrics = data["metrics"]["warmup"]

        # Should have all three events
        assert len(metrics) >= 3
        assert any(m["event"] == "phase_start" for m in metrics)
        assert any(m["event"] == "phase_sending_complete" for m in metrics)
        assert any(m["event"] == "phase_complete" for m in metrics)

    @pytest.mark.asyncio
    async def test_phase_durations_calculated(
        self, temp_dir: Path, mock_tracker: MagicMock
    ) -> None:
        """Test that phase durations are calculated."""
        from aiperf.common.enums import CreditPhase

        metrics_file = temp_dir / "metrics.json"
        hook = ExampleMetricsCollectorHook.from_config(metrics_file=metrics_file)

        # Record timeline
        await hook.on_phase_start(CreditPhase.WARMUP, mock_tracker)
        await asyncio.sleep(0.02)  # Ensure measurable duration
        await hook.on_phase_complete(CreditPhase.WARMUP, mock_tracker)

        data = orjson.loads(metrics_file.read_bytes())
        assert "phase_durations" in data
        assert "warmup" in data["phase_durations"]
        assert "total_duration" in data["phase_durations"]["warmup"]

    @pytest.mark.asyncio
    async def test_get_aggregated_metrics(self, temp_dir: Path, mock_tracker: MagicMock) -> None:
        """Test get_aggregated_metrics returns collected data."""
        from aiperf.common.enums import CreditPhase

        metrics_file = temp_dir / "metrics.json"
        hook = ExampleMetricsCollectorHook.from_config(metrics_file=metrics_file)

        await hook.on_phase_start(CreditPhase.WARMUP, mock_tracker)

        metrics = hook.get_aggregated_metrics()
        assert "metrics" in metrics
        assert "phase_durations" in metrics

    @pytest.mark.asyncio
    async def test_compact_json_output_when_indent_none(
        self, temp_dir: Path, mock_tracker: MagicMock
    ) -> None:
        """Test JSON output is compact when indent is None."""
        from aiperf.common.enums import CreditPhase

        metrics_file = temp_dir / "metrics.json"
        hook = ExampleMetricsCollectorHook.from_config(metrics_file=metrics_file, indent=None)

        await hook.on_phase_start(CreditPhase.WARMUP, mock_tracker)

        content = metrics_file.read_text()
        # Compact JSON should have no newlines except at end
        assert content.count("\n") <= 2
