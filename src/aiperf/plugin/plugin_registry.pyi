# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Type stubs for plugin_registry module.

This file is AUTO-GENERATED from categories.yaml.
Run `python tools/generate_plugin_overloads.py` to regenerate.

These stubs provide IDE autocomplete and type checking for the
get_class() function with category-specific return types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Literal, TypedDict, overload

from aiperf.common.protocols import (
    AIPerfUIProtocol,
    CommunicationClientProtocol,
    CommunicationProtocol,
    ConsoleExporterProtocol,
    DataExporterProtocol,
    DatasetBackingStoreProtocol,
    DatasetClientStoreProtocol,
    DatasetSamplingStrategyProtocol,
    EndpointProtocol,
    RecordProcessorProtocol,
    ServiceManagerProtocol,
    ServiceProtocol,
    TransportProtocol,
)
from aiperf.dataset.composer import BaseDatasetComposer
from aiperf.dataset.loader import CustomDatasetLoaderProtocol
from aiperf.plot.core.plot_type_handlers import PlotTypeHandlerProtocol
from aiperf.plugin.enums import PluginCategory
from aiperf.post_processors import BaseMetricsProcessor
from aiperf.timing.intervals import IntervalGeneratorProtocol
from aiperf.timing.ramping import RampStrategyProtocol
from aiperf.timing.strategies.core import TimingStrategyProtocol
from aiperf.zmq import BaseZMQProxy

# ==============================================================================
# Constants
# ==============================================================================

SUPPORTED_SCHEMA_VERSIONS: Final[tuple[str, ...]]
DEFAULT_SCHEMA_VERSION: Final[str]
DEFAULT_ENTRY_POINT_GROUP: Final[str]

# ==============================================================================
# Type Definitions
# ==============================================================================

class PackageMetadata(TypedDict, total=False):
    """Package metadata from YAML manifest."""

    name: str
    version: str
    description: str
    author: str
    license: str
    homepage: str
    builtin: bool

class ManifestData(TypedDict, total=False):
    """YAML manifest structure."""

    schema_version: str
    plugin: dict[str, Any]

# ==============================================================================
# Custom Exceptions
# ==============================================================================

class PluginError(Exception):
    """Base exception for plugin system errors."""

class TypeNotFoundError(PluginError):
    """Type not found in category. Includes available types in error message."""

    category: str
    type_name: str
    available: list[str]

    def __init__(self, category: str, type_name: str, available: list[str]) -> None: ...

# ==============================================================================
# TypeEntry Dataclass
# ==============================================================================

class TypeEntry:
    """Lazy-loading type entry with metadata. Call load() to import the class."""

    category: str
    type_name: str
    package_name: str
    class_path: str
    priority: int
    description: str
    metadata: PackageMetadata
    loaded_class: type | None
    is_builtin: bool

    def __init__(
        self,
        category: str,
        type_name: str,
        package_name: str,
        class_path: str,
        priority: int = ...,
        description: str = ...,
        metadata: PackageMetadata = ...,
        loaded_class: type | None = ...,
        is_builtin: bool = ...,
    ) -> None: ...
    def load(self) -> type: ...
    def validate(self, check_class: bool = ...) -> tuple[bool, str | None]: ...

# ==============================================================================
# PluginRegistry Class
# ==============================================================================

class PluginRegistry:
    """Plugin registry singleton with discovery and lazy loading."""

    def __init__(self) -> None: ...
    def load_registry(self, registry_path: Path | str) -> None: ...
    def discover_plugins(self, entry_point_group: str = ...) -> None: ...
    def get_class(self, category: str, name_or_class_path: str) -> type: ...
    def list_types(self, category: str) -> list[TypeEntry]: ...
    def validate_all(
        self, check_class: bool = ...
    ) -> dict[str, list[tuple[str, str]]]: ...
    def list_packages(self, builtin_only: bool = ...) -> list[str]: ...
    def find_registered_name(self, category: str, cls: type) -> str | None: ...

# ==============================================================================
# get_class() Overloads
# ==============================================================================

# Timing strategies control request scheduling and credit issuance.
@overload
def get_class(
    category: Literal[PluginCategory.TIMING_STRATEGY], name_or_class_path: str
) -> type[TimingStrategyProtocol]: ...

# Interval generators determine inter-arrival times for request rate strategy.
@overload
def get_class(
    category: Literal[PluginCategory.ARRIVAL_PATTERN], name_or_class_path: str
) -> type[IntervalGeneratorProtocol]: ...

# Ramp strategies control how values are gradually transitioned over time.
@overload
def get_class(
    category: Literal[PluginCategory.RAMP], name_or_class_path: str
) -> type[RampStrategyProtocol]: ...

# Dataset backing stores manage conversation data on the DatasetManager side.
@overload
def get_class(
    category: Literal[PluginCategory.DATASET_BACKING_STORE], name_or_class_path: str
) -> type[DatasetBackingStoreProtocol]: ...

# Dataset client stores read conversation data on the Worker side.
@overload
def get_class(
    category: Literal[PluginCategory.DATASET_CLIENT_STORE], name_or_class_path: str
) -> type[DatasetClientStoreProtocol]: ...

# Dataset samplers control how conversations are selected from the dataset.
@overload
def get_class(
    category: Literal[PluginCategory.DATASET_SAMPLER], name_or_class_path: str
) -> type[DatasetSamplingStrategyProtocol]: ...

# Dataset composers create conversation datasets from various sources.
@overload
def get_class(
    category: Literal[PluginCategory.DATASET_COMPOSER], name_or_class_path: str
) -> type[BaseDatasetComposer]: ...

# Custom dataset loaders parse different JSONL file formats into conversations.
@overload
def get_class(
    category: Literal[PluginCategory.CUSTOM_DATASET_LOADER], name_or_class_path: str
) -> type[CustomDatasetLoaderProtocol]: ...

# Endpoints define how to format requests and parse responses for different APIs.
@overload
def get_class(
    category: Literal[PluginCategory.ENDPOINT], name_or_class_path: str
) -> type[EndpointProtocol]: ...

# Transports handle the network layer for sending requests to inference servers.
@overload
def get_class(
    category: Literal[PluginCategory.TRANSPORT], name_or_class_path: str
) -> type[TransportProtocol]: ...

# Record processors stream records and compute metrics in a distributed manner.
@overload
def get_class(
    category: Literal[PluginCategory.RECORD_PROCESSOR], name_or_class_path: str
) -> type[RecordProcessorProtocol]: ...

# Results processors aggregate results from record processors and compute derived metrics.
@overload
def get_class(
    category: Literal[PluginCategory.RESULTS_PROCESSOR], name_or_class_path: str
) -> type[BaseMetricsProcessor]: ...

# Data exporters write benchmark results to files in various formats.
@overload
def get_class(
    category: Literal[PluginCategory.DATA_EXPORTER], name_or_class_path: str
) -> type[DataExporterProtocol]: ...

# Console exporters display benchmark results and diagnostics to stdout.
@overload
def get_class(
    category: Literal[PluginCategory.CONSOLE_EXPORTER], name_or_class_path: str
) -> type[ConsoleExporterProtocol]: ...

# UI components provide progress tracking and visualization during benchmark execution.
@overload
def get_class(
    category: Literal[PluginCategory.UI], name_or_class_path: str
) -> type[AIPerfUIProtocol]: ...

# Services are the core processes that make up the AIPerf distributed system.
@overload
def get_class(
    category: Literal[PluginCategory.SERVICE], name_or_class_path: str
) -> type[ServiceProtocol]: ...

# Service managers orchestrate how services are launched and managed.
@overload
def get_class(
    category: Literal[PluginCategory.SERVICE_MANAGER], name_or_class_path: str
) -> type[ServiceManagerProtocol]: ...

# Communication backends provide the underlying transport for inter-service messaging.
@overload
def get_class(
    category: Literal[PluginCategory.COMMUNICATION], name_or_class_path: str
) -> type[CommunicationProtocol]: ...

# Communication clients implement different ZMQ socket patterns for messaging.
@overload
def get_class(
    category: Literal[PluginCategory.COMMUNICATION_CLIENT], name_or_class_path: str
) -> type[CommunicationClientProtocol]: ...

# ZMQ proxies provide message routing between different socket patterns.
@overload
def get_class(
    category: Literal[PluginCategory.ZMQ_PROXY], name_or_class_path: str
) -> type[BaseZMQProxy]: ...

# Plot handlers create different types of visualizations from benchmark data.
@overload
def get_class(
    category: Literal[PluginCategory.PLOT], name_or_class_path: str
) -> type[PlotTypeHandlerProtocol]: ...

# Fallback for unknown categories
@overload
def get_class(category: PluginCategory, name_or_class_path: str) -> type: ...

# ==============================================================================
# Module-Level Functions
# ==============================================================================

def get_class(category: PluginCategory, name_or_class_path: str) -> type:
    """Get type class by name or fully qualified class path (lazy-loaded, cached)."""
    ...

def list_types(category: PluginCategory) -> list[TypeEntry]:
    """List all TypeEntry objects for a category (sorted alphabetically)."""
    ...

def validate_all(check_class: bool = ...) -> dict[str, list[tuple[str, str]]]:
    """Validate all registered types without loading. Returns {category: [(type, error)]}."""
    ...

def find_registered_name(category: str, cls: type) -> str | None:
    """Reverse lookup: find registered name for a class, or None if not found."""
    ...

def load_registry(registry_path: str | Path) -> None:
    """Load plugin types from a YAML registry manifest."""
    ...

def list_packages(builtin_only: bool = ...) -> list[str]:
    """List all loaded plugin package names."""
    ...

def get_package_metadata(package_name: str) -> PackageMetadata:
    """Get metadata for a loaded plugin package. Raises KeyError if not found."""
    ...

def list_categories() -> list[str]:
    """List all registered category names (sorted alphabetically)."""
    ...

def reset() -> None:
    """Reset registry to empty state and reload built-in plugins (for testing)."""
    ...

def register(
    category: str,
    type_name: str,
    cls: type,
    *,
    priority: int = ...,
    is_builtin: bool = ...,
) -> None:
    """Register a class programmatically (for dynamic classes or test overrides)."""
    ...

def create_enum(category: str, enum_name: str) -> type:
    """Create an ExtensibleStrEnum from registered types in a category."""
    ...

def detect_type_from_url(category: str, url: str) -> str:
    """Detect the plugin type from a URL by matching its scheme."""
    ...
