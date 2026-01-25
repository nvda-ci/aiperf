# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type stubs for plugin_registry module.

Provides complete type hints for IDE autocomplete and static type checking.
Uses TYPE_CHECKING to avoid circular imports and runtime overhead.

This stub file enables perfect IDE support for the unified package registry:
- Autocomplete for all 17 category types
- Type-safe overloads for get_class()
- No circular import issues
- Compatible with pyright/mypy

Usage:
    from aiperf.common import plugin_registry

    # Full type inference
    EndpointClass = plugin_registry.get_class('endpoint', 'openai')
    # Type: type[EndpointProtocol]

    endpoint = EndpointClass(model_endpoint=config)
    # Type: EndpointProtocol
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict, overload

from aiperf.timing.strategies.core import TimingStrategyProtocol

# ==============================================================================
# Module Exports
# ==============================================================================

__all__ = [
    # Main API
    "get_class",
    "list_types",
    "list_categories",
    "create_enum",
    "register",
    "find_registered_name",
    "detect_type_from_url",
    # Utilities
    "list_packages",
    "reset",
    # Types
    "TypeEntry",
    "PluginRegistry",
    "PackageMetadata",
    "TypeSpec",
    "CategoryType",
    # Exceptions
    "PluginError",
    "TypeNotFoundError",
    "TypeLoadError",
    "TypeConflictError",
]

# ==============================================================================
# Conditional Imports (Type Checking Only)
# ==============================================================================

if TYPE_CHECKING:
    # Protocol imports
    from aiperf.common.protocols import (
        AIPerfUIProtocol,
        CommunicationClientProtocol,
        CommunicationProtocol,
        DataExporterProtocol,
        DatasetSamplingStrategyProtocol,
        EndpointProtocol,
        IntervalGeneratorProtocol,
        RecordProcessorProtocol,
        ServiceManagerProtocol,
        ServiceProtocol,
        TransportProtocol,
    )

    # Base class imports
    from aiperf.dataset.composer.base import BaseDatasetComposer
    from aiperf.dataset.loader.base_loader import BaseDatasetLoader
    from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor
    from aiperf.zmq.zmq_proxy_base import BaseZMQProxy

# ==============================================================================
# Exception Classes
# ==============================================================================

class PluginError(Exception):
    """Base exception for package system errors."""

    ...

class TypeNotFoundError(PluginError):
    """Type not found in category.

    Attributes:
        category: Category name
        type_name: Type name
        available: List of available types
    """

    category: str
    type_name: str
    available: list[str]

    def __init__(self, category: str, type_name: str, available: list[str]) -> None: ...

class TypeLoadError(PluginError):
    """Failed to load type.

    Attributes:
        class_path: Fully qualified class path
        category: Category name
        type_name: Type name
        cause: Original exception
    """

    class_path: str
    category: str
    type_name: str
    cause: Exception

    def __init__(
        self,
        class_path: str,
        category: str,
        type_name: str,
        cause: Exception,
    ) -> None: ...

class TypeConflictError(PluginError):
    """Multiple packages provide the same type.

    Attributes:
        category: Category name
        type_name: Type name
        existing_package: Name of existing package
        new_package: Name of new package
    """

    category: str
    type_name: str
    existing_package: str
    new_package: str

    def __init__(
        self,
        category: str,
        type_name: str,
        existing_package: str,
        new_package: str,
    ) -> None: ...

# ==============================================================================
# Core Classes (Re-exports)
# ==============================================================================

class TypeEntry:
    """Lazy-loading type entry.

    Attributes:
        category: Category name (e.g., 'endpoint', 'timing_strategy')
        type_name: Type name (e.g., 'openai', 'fixed_schedule')
        package_name: Package name (e.g., 'aiperf', 'aiperf-custom-plugin')
        class_path: Full class path (e.g., 'aiperf.endpoints.openai_chat:ChatEndpoint')
        priority: Priority for conflict resolution (higher = preferred, default: 0)
        description: Human-readable description
        metadata: Package metadata from installed package (version, author, etc.)
        loaded_class: Cached class after first load (None until loaded)
        is_builtin: Whether this is a built-in type
    """

    category: str
    type_name: str
    package_name: str
    class_path: str
    priority: int
    description: str
    metadata: dict[str, Any]
    loaded_class: type | None
    is_builtin: bool

    def load(self) -> type:
        """Load the type class.

        Returns:
            The loaded class (not instantiated)

        Raises:
            ValueError: If class_path format is invalid
            TypeLoadError: If module or class cannot be imported
        """
        ...

class PluginRegistry:
    """Simplified unified package registry with lazy loading.

    Methods:
        get(category, name_or_path): Get type class by name or class path
        list_types(category): List all types for category
        load_builtins(registry_path): Load built-in registry from YAML
        discover_plugins(entry_point_group): Discover package registries
        list_packages(builtin_only): List all loaded packages
    """

    def get(self, category: str, name_or_class_path: str) -> type:
        """Get type class by name or fully qualified class path.

        Args:
            category: Category identifier
            name_or_class_path: Either short name or full class path

        Returns:
            The type class (not instantiated)

        Raises:
            KeyError: If category or type not found
            ValueError: If class path category doesn't match requested category
            TypeLoadError: If class cannot be imported
            TypeNotFoundError: If type not found
        """
        ...

    def list_types(self, category: str) -> list[TypeEntry]:
        """List all types for a category.

        Args:
            category: Category name

        Returns:
            List of TypeEntry objects (sorted alphabetically)
        """
        ...

    def load_builtins(self, registry_path: str | None = None) -> None:
        """Load built-in registry from YAML file.

        Args:
            registry_path: Optional path to registry.yaml

        Raises:
            FileNotFoundError: If registry file not found at specified path
            RuntimeError: If built-in registry.yaml not found in package
        """
        ...

    def discover_plugins(self, entry_point_group: str = "aiperf.plugins") -> None:
        """Discover and load package registries via entry points.

        Args:
            entry_point_group: Entry point group name for package discovery
        """
        ...

    def list_packages(self, builtin_only: bool = False) -> list[str]:
        """List all loaded packages.

        Args:
            builtin_only: If True, only return built-in packages

        Returns:
            List of package names
        """
        ...

# ==============================================================================
# TypedDict Definitions (Manifest Structure)
# ==============================================================================

class PackageMetadata(TypedDict, total=False):
    """Package metadata from YAML manifest."""

    name: str
    version: str
    description: str
    author: str
    builtin: bool

class TypeSpec(TypedDict, total=False):
    """Type specification in YAML manifest."""

    class_: str  # 'class' is keyword, represented as 'class_' in dict
    description: str
    priority: int

# ==============================================================================
# Category Type Literal
# ==============================================================================

CategoryType = Literal[
    "timing_strategy",
    "arrival_pattern",
    "endpoint",
    "dataset_sampler",
    "dataset_composer",
    "custom_dataset_loader",
    "record_processor",
    "results_processor",
    "data_exporter",
    "console_exporter",
    "transport",
    "ui",
    "service",
    "service_manager",
    "communication",
    "communication_client",
    "zmq_proxy",
]

# ==============================================================================
# Module-Level get_class() Function Overloads
# ==============================================================================

if TYPE_CHECKING:
    # Timing strategies
    @overload
    def get_class(
        category: Literal["timing_strategy"], name_or_class_path: str
    ) -> type[TimingStrategyProtocol]: ...

    # Request rate generators
    @overload
    def get_class(
        category: Literal["arrival_pattern"], name_or_class_path: str
    ) -> type[IntervalGeneratorProtocol]: ...

    # Endpoints
    @overload
    def get_class(
        category: Literal["endpoint"], name_or_class_path: str
    ) -> type[EndpointProtocol]: ...

    # Dataset samplers
    @overload
    def get_class(
        category: Literal["dataset_sampler"], name_or_class_path: str
    ) -> type[DatasetSamplingStrategyProtocol]: ...

    # Dataset composers
    @overload
    def get_class(
        category: Literal["dataset_composer"], name_or_class_path: str
    ) -> type[BaseDatasetComposer]: ...

    # Custom dataset loaders
    @overload
    def get_class(
        category: Literal["custom_dataset_loader"], name_or_class_path: str
    ) -> type[BaseDatasetLoader]: ...

    # Record processors
    @overload
    def get_class(
        category: Literal["record_processor"], name_or_class_path: str
    ) -> type[RecordProcessorProtocol]: ...

    # Results processors
    @overload
    def get_class(
        category: Literal["results_processor"], name_or_class_path: str
    ) -> type[BaseMetricsProcessor]: ...

    # Data exporters
    @overload
    def get_class(
        category: Literal["data_exporter"], name_or_class_path: str
    ) -> type[DataExporterProtocol]: ...

    # Console exporters (no base class, returns raw type)
    @overload
    def get_class(
        category: Literal["console_exporter"], name_or_class_path: str
    ) -> type: ...

    # Transports
    @overload
    def get_class(
        category: Literal["transport"], name_or_class_path: str
    ) -> type[TransportProtocol]: ...

    # UI
    @overload
    def get_class(
        category: Literal["ui"], name_or_class_path: str
    ) -> type[AIPerfUIProtocol]: ...

    # Services
    @overload
    def get_class(
        category: Literal["service"], name_or_class_path: str
    ) -> type[ServiceProtocol]: ...

    # Service managers
    @overload
    def get_class(
        category: Literal["service_manager"], name_or_class_path: str
    ) -> type[ServiceManagerProtocol]: ...

    # Communication
    @overload
    def get_class(
        category: Literal["communication"], name_or_class_path: str
    ) -> type[CommunicationProtocol]: ...

    # Communication clients
    @overload
    def get_class(
        category: Literal["communication_client"], name_or_class_path: str
    ) -> type[CommunicationClientProtocol]: ...

    # ZMQ proxies
    @overload
    def get_class(
        category: Literal["zmq_proxy"], name_or_class_path: str
    ) -> type[BaseZMQProxy]: ...

    # Fallback for unknown categories
    @overload
    def get_class(category: str, name_or_class_path: str) -> type: ...

# Implementation signature (no TYPE_CHECKING guard needed)
def get_class(category: str, name_or_class_path: str) -> type: ...

# ==============================================================================
# Other Module-Level Functions
# ==============================================================================

def list_types(category: str) -> list[TypeEntry]:
    """List all types for category.

    Returns TypeEntry objects (NOT loaded classes).

    Args:
        category: Category name

    Returns:
        List of TypeEntry objects (sorted alphabetically)
    """
    ...

def list_categories() -> list[str]:
    """List all registered categories.

    Returns:
        List of category names (sorted alphabetically)
    """
    ...

def load_builtins(registry_path: str | None = None) -> None:
    """Load built-in registry from YAML.

    Args:
        registry_path: Optional path to registry.yaml
    """
    ...

def discover_plugins(entry_point_group: str = "aiperf.plugins") -> None:
    """Discover package registries via entry points.

    Args:
        entry_point_group: Entry point group to scan
    """
    ...

def list_packages(builtin_only: bool = False) -> list[str]:
    """List all loaded packages.

    Args:
        builtin_only: Only return built-in packages

    Returns:
        List of package names
    """
    ...

def reset() -> None:
    """Reset registry (for testing).

    Creates a fresh registry instance.
    """
    ...

def register(
    category: str,
    type_name: str,
    class_path: str,
    description: str = "",
    priority: int = 0,
    is_builtin: bool = False,
) -> None:
    """Register a new type programmatically.

    Args:
        category: Category name (e.g., 'endpoint', 'timing_strategy')
        type_name: Type name
        class_path: Fully qualified class path (module:ClassName)
        description: Human-readable description
        priority: Priority for conflict resolution (higher wins)
        is_builtin: Whether this is a built-in type
    """
    ...

def find_registered_name(category: str, cls: type) -> str | None:
    """Find the registered name for a class within a category.

    Args:
        category: Category name
        cls: Class to look up

    Returns:
        Registered type name, or None if not found
    """
    ...

def create_enum(category: str, enum_name: str) -> type:
    """Create an ExtensibleStrEnum from registered types in a category.

    Creates a dynamic enum that works with Pydantic validation and cyclopts CLI.
    Member names are UPPER_SNAKE_CASE versions of the type names.

    Args:
        category: Category name (e.g., 'endpoint', 'ui')
        enum_name: Name for the enum class (e.g., 'EndpointType')

    Returns:
        An ExtensibleStrEnum subclass with all registered types as members

    Raises:
        KeyError: If category has no registered types

    Example:
        >>> EndpointType = plugin_registry.create_enum('endpoint', 'EndpointType')
        >>> EndpointType.CHAT  # Access member
        >>> list(EndpointType)  # Iterate for CLI choices
    """
    ...

def detect_type_from_url(category: str, url: str) -> str:
    """Detect the type from a URL.

    Args:
        category: Category name (e.g., 'transport')
        url: URL to detect transport for

    Returns:
        Type name that can handle this URL

    Raises:
        ValueError: If no type can handle the URL
    """
    ...
