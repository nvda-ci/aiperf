# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory for GPU telemetry collectors."""

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiperf.common.enums import GPUTelemetryCollectorType
from aiperf.common.factories import AIPerfFactory

if TYPE_CHECKING:
    from aiperf.common.models import ErrorDetails, TelemetryRecord


@runtime_checkable
class GPUTelemetryCollectorProtocol(Protocol):
    """Protocol for GPU telemetry collectors.

    Defines the interface for collectors that gather GPU metrics from various sources
    (DCGM HTTP endpoints, pynvml library, etc.) and deliver them via callbacks.
    """

    @property
    def id(self) -> str:
        """Get the collector's unique identifier."""
        ...

    @property
    def endpoint_url(self) -> str:
        """Get the source identifier (URL for DCGM, 'pynvml://localhost' for pynvml)."""
        ...

    async def initialize(self) -> None:
        """Initialize the collector resources."""
        ...

    async def start(self) -> None:
        """Start the background collection task."""
        ...

    async def stop(self) -> None:
        """Stop the collector and clean up resources."""
        ...

    async def is_url_reachable(self) -> bool:
        """Check if the collector source is available.

        For DCGM: Tests HTTP endpoint reachability.
        For pynvml: Tests NVML library initialization.

        Returns:
            True if the source is available and ready for collection.
        """
        ...


# Type aliases for callbacks
TRecordCallback = Callable[[list["TelemetryRecord"], str], Awaitable[None]]
TErrorCallback = Callable[["ErrorDetails", str], Awaitable[None]]


class GPUTelemetryCollectorFactory(
    AIPerfFactory[GPUTelemetryCollectorType, GPUTelemetryCollectorProtocol]
):
    """Factory for creating GPU telemetry collector instances.

    Supports multiple collector implementations:
    - DCGM: HTTP-based collection from DCGM Prometheus exporter
    - PYNVML: Direct collection using pynvml Python library

    Example:
        collector = GPUTelemetryCollectorFactory.create_instance(
            GPUTelemetryCollectorType.DCGM,
            dcgm_url="http://localhost:9400/metrics",
            collection_interval=0.333,
            record_callback=my_callback,
        )
    """

    pass
