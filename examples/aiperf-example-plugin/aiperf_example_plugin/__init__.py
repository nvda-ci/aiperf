# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIPerf Example Plugin demonstrating 2025 Python best practices.

This plugin provides reference implementations of:
- Phase lifecycle hooks with Pydantic configuration
- Custom metrics processing with type safety
- Registry-based plugin loading
- Modern async patterns

Example:
    >>> from aiperf_example_plugin.hooks import ExampleLoggingHook
    >>> from aiperf_example_plugin.processors import ExampleMetricsProcessor
    >>>
    >>> # Using factory methods for clean initialization
    >>> hook = ExampleLoggingHook.from_config(
    ...     log_file="/tmp/aiperf_phases.log",
    ...     verbose=True
    ... )
    >>> orchestrator.register_hook(hook)
    >>>
    >>> # Type-safe processor configuration
    >>> processor = ExampleMetricsProcessor.from_config(
    ...     output_file="/tmp/metrics.json",
    ...     percentiles=[50, 90, 95, 99]
    ... )
    >>> result = await processor.process(raw_results)
    >>> print(f"Processed {result.record_count} records")
"""

from __future__ import annotations

from pathlib import Path

__version__ = "1.0.0"
__author__ = "NVIDIA"


def get_registry_path() -> str:
    """Return the path to this plugin's registry.yaml.

    This function is called by the AIPerf plugin discovery system
    via the entry point defined in pyproject.toml.

    Returns:
        Absolute path to the registry.yaml file.
    """
    return str(Path(__file__).parent / "registry.yaml")

from aiperf_example_plugin.hooks import (
    ExampleLoggingHook,
    ExampleLoggingHookConfig,
    ExampleMetricsCollectorHook,
    ExampleMetricsCollectorHookConfig,
)
from aiperf_example_plugin.processors import (
    ExampleMetricsProcessor,
    ExampleMetricsProcessorConfig,
    ExampleResultsAggregator,
    ProcessingResult,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Plugin discovery
    "get_registry_path",
    # Hooks
    "ExampleLoggingHook",
    "ExampleLoggingHookConfig",
    "ExampleMetricsCollectorHook",
    "ExampleMetricsCollectorHookConfig",
    # Processors
    "ExampleMetricsProcessor",
    "ExampleMetricsProcessorConfig",
    "ExampleResultsAggregator",
    "ProcessingResult",
]
