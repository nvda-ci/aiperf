# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example phase lifecycle hooks demonstrating 2025 Python best practices.

This module demonstrates:
- Pydantic configuration models with validation
- Modern type hints with Protocol and type aliases
- Async I/O with pathlib
- Comprehensive docstrings with examples
- Context managers for resource management

Phase Event Flow:
    Phase Start → on_phase_start()
         ↓
    Sending Credits → (credits issued)
         ↓
    Sending Complete → on_phase_sending_complete()
         ↓
    Returning Credits → (credits return)
         ↓
    Phase Complete → on_phase_complete()
         ↓
    (If timeout) → on_phase_timeout()

Example:
    >>> from aiperf_example_plugin.hooks import ExampleLoggingHook
    >>> hook = ExampleLoggingHook.from_config(
    ...     log_file="/var/log/aiperf/phases.log",
    ...     verbose=True
    ... )
    >>> orchestrator.register_hook(hook)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson
from aiperf.timing.phase_lifecycle_hooks import BasePhaseLifecycleHook
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from aiperf.common.enums import CreditPhase

# Type aliases for better readability
PhaseKey = str
Timestamp = float
Metrics = dict[str, Any]

__all__ = [
    "ExampleLoggingHookConfig",
    "ExampleLoggingHook",
    "ExampleMetricsCollectorHookConfig",
    "ExampleMetricsCollectorHook",
]


class ExampleLoggingHookConfig(BaseModel):
    """Configuration for ExampleLoggingHook.

    This demonstrates Pydantic-based configuration with validation.

    Attributes:
        log_file: Path where phase events will be logged
        verbose: Whether to include detailed statistics in logs
        create_dirs: Whether to create parent directories if they don't exist
    """

    log_file: Path = Field(
        default=Path("/tmp/aiperf_phases.log"),
        description="Path to write phase event logs",
    )
    verbose: bool = Field(
        default=False,
        description="Enable detailed statistics in log entries",
    )
    create_dirs: bool = Field(
        default=True,
        description="Create parent directories if they don't exist",
    )

    @field_validator("log_file", mode="before")
    @classmethod
    def validate_log_file(cls, v: str | Path) -> Path:
        """Ensure log_file is a Path object.

        Args:
            v: Log file path as string or Path

        Returns:
            Path object for log file
        """
        return Path(v) if isinstance(v, str) else v

    model_config = {"frozen": False}


class ExampleLoggingHook(BasePhaseLifecycleHook):
    """Example hook that logs phase transitions to file.

    This hook demonstrates:
    - Pydantic configuration with validation
    - Async file I/O with pathlib
    - Type hints throughout
    - Resource management best practices
    - Comprehensive docstrings

    Thread Safety:
        File writes use append mode with automatic flushing.
        Safe for concurrent use with multiple async tasks.

    Example:
        >>> # Using config object
        >>> config = ExampleLoggingHookConfig(
        ...     log_file="/var/log/aiperf/phases.log",
        ...     verbose=True
        ... )
        >>> hook = ExampleLoggingHook(config)
        >>> orchestrator.register_hook(hook)

        >>> # Using factory method
        >>> hook = ExampleLoggingHook.from_config(
        ...     log_file="/var/log/aiperf/phases.log",
        ...     verbose=True
        ... )

    Note:
        All I/O is performed synchronously in a thread-safe manner.
        For high-throughput scenarios, consider using async file I/O.
    """

    def __init__(self, config: ExampleLoggingHookConfig) -> None:
        """Initialize logging hook.

        Args:
            config: Hook configuration
        """
        self.config = config
        self._phase_metrics: dict[PhaseKey, dict[str, Timestamp]] = {}

        # Create parent directory if requested
        if self.config.create_dirs:
            self.config.log_file.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(
        cls,
        log_file: str | Path = "/tmp/aiperf_phases.log",
        verbose: bool = False,
        create_dirs: bool = True,
    ) -> ExampleLoggingHook:
        """Factory method to create hook from parameters.

        Args:
            log_file: Path to write phase event logs
            verbose: Enable detailed statistics
            create_dirs: Create parent directories

        Returns:
            Configured ExampleLoggingHook instance

        Example:
            >>> hook = ExampleLoggingHook.from_config(
            ...     log_file="/tmp/phases.log",
            ...     verbose=True
            ... )
        """
        config = ExampleLoggingHookConfig(
            log_file=log_file,
            verbose=verbose,
            create_dirs=create_dirs,
        )
        return cls(config)

    async def on_phase_start(self, phase: CreditPhase, tracker: Any) -> None:
        """Log phase start event.

        Args:
            phase: Phase that is starting
            tracker: Phase statistics tracker
        """
        stats = tracker.create_stats() if hasattr(tracker, "create_stats") else None
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        message = f"[{timestamp}] PHASE_START: {phase}"
        if self.config.verbose and stats:
            message += f" | Stats: {stats}"

        await self._write_log(message)

        # Track metric for this phase
        phase_key = str(phase)
        self._phase_metrics[phase_key] = {"start_time": time.time()}

    async def on_phase_sending_complete(self, phase: CreditPhase, tracker: Any) -> None:
        """Log when phase completes sending all credits.

        Args:
            phase: Phase that finished sending
            tracker: Phase statistics tracker
        """
        stats = tracker.create_stats() if hasattr(tracker, "create_stats") else None
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        message = f"[{timestamp}] PHASE_SENDING_COMPLETE: {phase}"
        if self.config.verbose and stats:
            duration = self._calculate_duration(phase)
            message += (
                f" | Sent: {stats.sent}, Sessions: {stats.sent_sessions}, Duration: {duration:.2f}s"
            )

        await self._write_log(message)

    async def on_phase_complete(self, phase: CreditPhase, tracker: Any) -> None:
        """Log phase completion (all credits returned).

        Args:
            phase: Phase that completed
            tracker: Final phase statistics
        """
        stats = tracker.create_stats() if hasattr(tracker, "create_stats") else None
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        message = f"[{timestamp}] PHASE_COMPLETE: {phase}"
        if self.config.verbose and stats:
            duration = self._calculate_duration(phase)
            message += (
                f" | Completed: {stats.completed}, "
                f"Cancelled: {stats.cancelled}, "
                f"Duration: {duration:.2f}s"
            )

        await self._write_log(message)

    async def on_phase_timeout(self, phase: CreditPhase, tracker: Any) -> None:
        """Log phase timeout event.

        Args:
            phase: Phase that timed out
            tracker: Phase statistics at timeout
        """
        stats = tracker.create_stats() if hasattr(tracker, "create_stats") else None
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        message = f"[{timestamp}] PHASE_TIMEOUT: {phase}"
        if self.config.verbose and stats:
            duration = self._calculate_duration(phase)
            message += f" | In-Flight: {stats.in_flight}, Duration: {duration:.2f}s"

        await self._write_log(message)

    async def _write_log(self, message: str) -> None:
        """Write message to log file.

        Uses append mode with automatic flushing for thread safety.

        Args:
            message: Log message to write
        """
        try:
            with self.config.log_file.open("a") as f:
                f.write(message + "\n")
                f.flush()
        except OSError as e:
            print(f"Warning: Failed to write to log file {self.config.log_file}: {e}")

    def _calculate_duration(self, phase: CreditPhase) -> float:
        """Calculate duration since phase start.

        Args:
            phase: Phase to calculate duration for

        Returns:
            Duration in seconds, or 0.0 if phase not tracked
        """
        phase_key = str(phase)
        if phase_key in self._phase_metrics:
            return time.time() - self._phase_metrics[phase_key]["start_time"]
        return 0.0

    def get_phase_metrics(self) -> dict[PhaseKey, dict[str, Timestamp]]:
        """Get collected phase metrics.

        Returns:
            Dictionary of phase metrics with timestamps
        """
        return self._phase_metrics.copy()


class ExampleMetricsCollectorHookConfig(BaseModel):
    """Configuration for ExampleMetricsCollectorHook.

    Attributes:
        metrics_file: Path where metrics JSON will be written
        aggregate: Whether to aggregate metrics across phases
        create_dirs: Whether to create parent directories
        indent: JSON indentation level (None for compact output)
    """

    metrics_file: Path = Field(
        default=Path("/tmp/aiperf_metrics.json"),
        description="Path to write metrics JSON",
    )
    aggregate: bool = Field(
        default=True,
        description="Aggregate metrics across all phases",
    )
    create_dirs: bool = Field(
        default=True,
        description="Create parent directories if they don't exist",
    )
    indent: int | None = Field(
        default=2,
        description="JSON indentation level (None for compact)",
    )

    @field_validator("metrics_file", mode="before")
    @classmethod
    def validate_metrics_file(cls, v: str | Path) -> Path:
        """Ensure metrics_file is a Path object.

        Args:
            v: Metrics file path as string or Path

        Returns:
            Path object for metrics file
        """
        return Path(v) if isinstance(v, str) else v

    model_config = {"frozen": False}


class ExampleMetricsCollectorHook(BasePhaseLifecycleHook):
    """Example hook that collects and aggregates phase metrics.

    This hook demonstrates:
    - Real-time metric collection with JSON serialization
    - Phase duration tracking
    - Automatic checkpoint writing
    - orjson for high-performance JSON serialization

    The hook writes metrics incrementally to disk, allowing real-time
    monitoring of phase execution progress.

    Example:
        >>> config = ExampleMetricsCollectorHookConfig(
        ...     metrics_file="/tmp/metrics.json",
        ...     aggregate=True
        ... )
        >>> hook = ExampleMetricsCollectorHook(config)
        >>> orchestrator.register_hook(hook)
        >>> # After execution:
        >>> metrics = hook.get_aggregated_metrics()
        >>> print(f"Total phases: {len(metrics['metrics'])}")

    Note:
        Uses orjson for fast JSON serialization with proper UTF-8 handling.
    """

    def __init__(self, config: ExampleMetricsCollectorHookConfig) -> None:
        """Initialize metrics collector hook.

        Args:
            config: Hook configuration
        """
        self.config = config
        self._metrics: dict[PhaseKey, list[Metrics]] = {}
        self._phase_times: dict[PhaseKey, dict[str, Timestamp]] = {}

        # Create parent directory if requested
        if self.config.create_dirs:
            self.config.metrics_file.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(
        cls,
        metrics_file: str | Path = "/tmp/aiperf_metrics.json",
        aggregate: bool = True,
        create_dirs: bool = True,
        indent: int | None = 2,
    ) -> ExampleMetricsCollectorHook:
        """Factory method to create hook from parameters.

        Args:
            metrics_file: Path to write metrics JSON
            aggregate: Aggregate metrics across phases
            create_dirs: Create parent directories
            indent: JSON indentation (None for compact)

        Returns:
            Configured ExampleMetricsCollectorHook instance
        """
        config = ExampleMetricsCollectorHookConfig(
            metrics_file=metrics_file,
            aggregate=aggregate,
            create_dirs=create_dirs,
            indent=indent,
        )
        return cls(config)

    async def on_phase_start(self, phase: CreditPhase, tracker: Any) -> None:
        """Record phase start metrics.

        Args:
            phase: Phase starting
            tracker: Phase statistics tracker
        """
        phase_key = str(phase)

        if phase_key not in self._metrics:
            self._metrics[phase_key] = []

        if phase_key not in self._phase_times:
            self._phase_times[phase_key] = {}

        self._phase_times[phase_key]["start"] = time.time()

        metric_entry: Metrics = {
            "event": "phase_start",
            "timestamp": time.time(),
            "phase": phase_key,
        }

        self._metrics[phase_key].append(metric_entry)
        await self._write_metrics_checkpoint()

    async def on_phase_sending_complete(self, phase: CreditPhase, tracker: Any) -> None:
        """Record phase sending complete metrics.

        Args:
            phase: Phase that finished sending
            tracker: Phase statistics tracker
        """
        stats = tracker.create_stats() if hasattr(tracker, "create_stats") else None
        phase_key = str(phase)

        self._phase_times[phase_key]["sending_complete"] = time.time()

        metric_entry: Metrics = {
            "event": "phase_sending_complete",
            "timestamp": time.time(),
            "phase": phase_key,
            "sent": stats.sent if stats else None,
            "sent_sessions": stats.sent_sessions if stats else None,
        }

        self._metrics[phase_key].append(metric_entry)
        await self._write_metrics_checkpoint()

    async def on_phase_complete(self, phase: CreditPhase, tracker: Any) -> None:
        """Record phase completion metrics.

        Args:
            phase: Phase that completed
            tracker: Final phase statistics
        """
        stats = tracker.create_stats() if hasattr(tracker, "create_stats") else None
        phase_key = str(phase)

        self._phase_times[phase_key]["complete"] = time.time()

        metric_entry: Metrics = {
            "event": "phase_complete",
            "timestamp": time.time(),
            "phase": phase_key,
            "completed": stats.completed if stats else None,
            "cancelled": stats.cancelled if stats else None,
            "in_flight": stats.in_flight if stats else None,
        }

        self._metrics[phase_key].append(metric_entry)
        await self._write_metrics_checkpoint()

    async def on_phase_timeout(self, phase: CreditPhase, tracker: Any) -> None:
        """Record phase timeout metrics.

        Args:
            phase: Phase that timed out
            tracker: Phase statistics at timeout
        """
        stats = tracker.create_stats() if hasattr(tracker, "create_stats") else None
        phase_key = str(phase)

        metric_entry: Metrics = {
            "event": "phase_timeout",
            "timestamp": time.time(),
            "phase": phase_key,
            "in_flight": stats.in_flight if stats else None,
        }

        self._metrics[phase_key].append(metric_entry)
        await self._write_metrics_checkpoint()

    async def _write_metrics_checkpoint(self) -> None:
        """Write current metrics to JSON file.

        Uses orjson for fast, correct JSON serialization.
        """
        try:
            metrics_data: Metrics = {
                "metrics": self._metrics,
                "phase_durations": self._calculate_phase_durations(),
                "last_updated": time.time(),
            }

            # Use orjson for fast JSON serialization
            json_bytes = orjson.dumps(
                metrics_data,
                option=orjson.OPT_INDENT_2 if self.config.indent else 0,
            )

            self.config.metrics_file.write_bytes(json_bytes)
        except OSError as e:
            print(f"Warning: Failed to write metrics file {self.config.metrics_file}: {e}")

    def _calculate_phase_durations(self) -> dict[PhaseKey, dict[str, float]]:
        """Calculate phase durations.

        Returns:
            Dictionary of phase durations with timing breakdowns
        """
        durations: dict[PhaseKey, dict[str, float]] = {}

        for phase_key, times in self._phase_times.items():
            durations[phase_key] = {}

            if "start" in times and "sending_complete" in times:
                durations[phase_key]["sending_duration"] = (
                    times["sending_complete"] - times["start"]
                )

            if "start" in times and "complete" in times:
                durations[phase_key]["total_duration"] = times["complete"] - times["start"]

        return durations

    def get_aggregated_metrics(self) -> Metrics:
        """Get aggregated metrics across all phases.

        Returns:
            Aggregated metrics dictionary with phase durations
        """
        return {
            "metrics": self._metrics.copy(),
            "phase_durations": self._calculate_phase_durations(),
        }
