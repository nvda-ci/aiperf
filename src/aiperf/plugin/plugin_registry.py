# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin registry singleton with lazy loading and priority-based conflict resolution.

Usage:
    from aiperf.plugin import plugin_registry
    from aiperf.plugin.enums import PluginCategory

    EndpointClass = plugin_registry.get_class(PluginCategory.ENDPOINT, 'openai')
    for impl in plugin_registry.list_types(PluginCategory.ENDPOINT):
        print(f"{impl.type_name}: {impl.description}")

Conflict resolution: higher priority wins; equal priority: external beats built-in.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.plugin.enums import PluginCategory

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.plugin._plugin_registry import PluginRegistry
from aiperf.plugin.types import PackageMetadata, TypeEntry

_logger = AIPerfLogger(__name__)

# ==============================================================================
# Constants
# ==============================================================================


# ==============================================================================
# Module-Level Singleton
# ==============================================================================
# This pattern follows the random_generator module design.
# Usage:
#   from aiperf.plugin import plugin_registry
#   from aiperf.plugin.enums import PluginCategory
#   EndpointClass = plugin_registry.get_class(PluginCategory.ENDPOINT, 'openai')
# ==============================================================================

# Create singleton instance at module load
_registry = PluginRegistry()


# ==============================================================================
# Public API: Module-Level Functions
# ==============================================================================
# Type stubs with get_class() overloads are in plugin_registry.pyi
# ==============================================================================


def get_class(category: PluginCategory, name_or_class_path: str) -> type:
    """Get type class by name or fully qualified class path.

    Args:
        category: Plugin category (e.g., PluginCategory.ENDPOINT).
        name_or_class_path: Either a short type name (e.g., 'chat') or
            a fully qualified class path (e.g., 'aiperf.endpoints:ChatEndpoint').

    Returns:
        The plugin class (lazy-loaded, cached after first access).

    Raises:
        TypeNotFoundError: If the type name is not found in the category.
        KeyError: If the category or class path is not registered.
        ValueError: If using class path and category doesn't match.

    Example:
        >>> from aiperf.plugin import plugin_registry
        >>> from aiperf.plugin.enums import PluginCategory
        >>> EndpointClass = plugin_registry.get_class(PluginCategory.ENDPOINT, 'chat')
    """
    return _registry.get_class(category, name_or_class_path)


def list_types(category: PluginCategory) -> list[TypeEntry]:
    """List all TypeEntry objects for a category (sorted alphabetically).

    Args:
        category: Plugin category to list types for.

    Returns:
        List of TypeEntry objects with metadata (type_name, description, priority, etc.).
        Returns empty list if category doesn't exist.

    Example:
        >>> for impl in plugin_registry.list_types(PluginCategory.ENDPOINT):
        ...     print(f"{impl.type_name}: {impl.description}")
    """
    return _registry.list_types(category)


def validate_all(check_class: bool = False) -> dict[str, list[tuple[str, str]]]:
    """Validate all registered types without loading them.

    Checks that modules are importable (and optionally that classes exist)
    without actually executing any import statements.

    Args:
        check_class: If True, also verify class exists via AST parsing.

    Returns:
        Dict mapping category names to lists of (type_name, error_message) tuples.
        Empty dict means all types are valid.

    Example:
        >>> errors = plugin_registry.validate_all(check_class=True)
        >>> if errors:
        ...     for category, type_errors in errors.items():
        ...         for type_name, error in type_errors:
        ...             print(f"{category}:{type_name}: {error}")
    """
    return _registry.validate_all(check_class=check_class)


def find_registered_name(category: str, cls: type) -> str | None:
    """Reverse lookup: find the registered name for a class.

    Searches by class identity first (for loaded classes), then by class path
    (for classes not loaded via registry).

    Args:
        category: Plugin category to search in.
        cls: The class to find the registered name for.

    Returns:
        The registered type name, or None if not found.

    Example:
        >>> from aiperf.endpoints import ChatEndpoint
        >>> name = plugin_registry.find_registered_name('endpoint', ChatEndpoint)
        >>> print(name)  # 'chat'
    """
    return _registry.find_registered_name(category, cls)


def load_registry(registry_path: str | Path) -> None:
    """Load plugin types from a YAML registry manifest.

    Parses the YAML file, validates the schema, and registers all types
    with priority-based conflict resolution.

    Args:
        registry_path: Path to the registry YAML file.

    Raises:
        FileNotFoundError: If the registry file doesn't exist.
        ValueError: If the path is a directory or schema is invalid.
        RuntimeError: If the file cannot be read.

    Example:
        >>> plugin_registry.load_registry('/path/to/custom/plugins.yaml')
    """
    _registry.load_registry(registry_path)


def list_packages(builtin_only: bool = False) -> list[str]:
    """List all loaded plugin package names.

    Args:
        builtin_only: If True, only return built-in packages (aiperf core).

    Returns:
        List of package names that have been loaded into the registry.

    Example:
        >>> plugin_registry.list_packages()
        ['aiperf', 'my-custom-plugin']
        >>> plugin_registry.list_packages(builtin_only=True)
        ['aiperf']
    """
    return _registry.list_packages(builtin_only)


def get_package_metadata(package_name: str) -> PackageMetadata:
    """Get metadata for a loaded plugin package.

    Args:
        package_name: Name of the plugin package.

    Returns:
        PackageMetadata dict with name, version, description, author, etc.

    Raises:
        KeyError: If the package has not been loaded.

    Example:
        >>> meta = plugin_registry.get_package_metadata('aiperf')
        >>> print(meta['version'])
    """
    if package_name not in _registry._loaded_plugins:
        raise KeyError(f"Package '{package_name}' not found in loaded plugins")
    return _registry._loaded_plugins[package_name]


def list_categories() -> list[str]:
    """List all registered category names (sorted alphabetically).

    Returns:
        Sorted list of category names (e.g., ['endpoint', 'transport', ...]).

    Example:
        >>> categories = plugin_registry.list_categories()
        >>> print(categories)
        ['arrival_pattern', 'communication', 'endpoint', ...]
    """
    return sorted(_registry._types.keys())


def reset() -> None:
    """Reset registry to empty state and reload built-in plugins.

    Intended for testing only. Clears all registered types and reloads
    the built-in registry manifest.

    Warning:
        This will invalidate any cached class references.
    """
    global _registry
    PluginRegistry._reset_singleton()
    _registry = PluginRegistry()
    _logger.debug("Registry reset")


def register(
    category: str,
    type_name: str,
    cls: type,
    *,
    priority: int = 0,
    is_builtin: bool = True,
) -> None:
    """Register a class programmatically (for dynamic classes or test overrides).

    Useful for registering classes created at runtime or overriding built-in
    types in tests. Uses the same priority-based conflict resolution as YAML.

    Args:
        category: Plugin category to register under.
        type_name: Short name for the type (can be an enum value).
        cls: The class to register.
        priority: Conflict resolution priority (higher wins). Default: 0.
        is_builtin: Whether this is a built-in type. Default: True.

    Example:
        >>> class MyCustomEndpoint:
        ...     pass
        >>> plugin_registry.register('endpoint', 'custom', MyCustomEndpoint, priority=10)
    """
    # Convert enum to string if needed
    name = type_name.value if hasattr(type_name, "value") else str(type_name)

    # Create a TypeEntry with the pre-loaded class
    lazy_type = TypeEntry(
        category=category,
        type_name=name,
        package_name="aiperf-test" if not is_builtin else "aiperf",
        class_path=f"{cls.__module__}:{cls.__name__}",
        priority=priority,
        description=cls.__doc__ or "",
        metadata=PackageMetadata(name="aiperf", builtin=is_builtin),
        loaded_class=cls,
        is_builtin=is_builtin,
    )

    # Ensure category exists
    if category not in _registry._types:
        _registry._types[category] = {}

    # Use conflict resolution to handle priority-based overrides
    _registry._resolve_conflict_and_register(category, name, lazy_type)

    _logger.debug(
        lambda: f"Registered dynamic type {category}:{name} -> {cls.__name__} (priority={priority})"
    )


def create_enum(category: str, enum_name: str) -> type:
    """Create an ExtensibleStrEnum from registered types in a category.

    Dynamically generates an enum class with members for each registered type.
    Member names are UPPER_SNAKE_CASE, values are the original type names.

    Args:
        category: Plugin category to create enum from.
        enum_name: Name for the generated enum class.

    Returns:
        A new ExtensibleStrEnum subclass.

    Raises:
        KeyError: If no types are registered for the category.

    Example:
        >>> EndpointEnum = plugin_registry.create_enum('endpoint', 'EndpointType')
        >>> print(EndpointEnum.CHAT)  # 'chat'
    """
    from aiperf.common.enums import create_enum as _create_enum

    types = _registry.list_types(category)
    if not types:
        raise KeyError(
            f"No types registered for category '{category}'. "
            f"Available categories: {sorted(_registry._types.keys())}"
        )

    # Create members dict: UPPER_SNAKE_CASE name -> string value
    members = {
        impl.type_name.replace("-", "_").upper(): impl.type_name for impl in types
    }

    return _create_enum(enum_name, members)


def detect_type_from_url(category: str, url: str) -> str:
    """Detect the plugin type from a URL by matching its scheme.

    Iterates through registered types and checks if their metadata declares
    support for the URL's scheme (e.g., 'http', 'https', 'grpc').

    Args:
        category: Plugin category to search.
        url: URL to extract scheme from.

    Returns:
        The type name that supports the URL scheme.

    Raises:
        ValueError: If no type supports the URL scheme.

    Example:
        >>> transport = plugin_registry.detect_type_from_url('transport', 'https://api.example.com')
        >>> print(transport)  # 'http'
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    # urlparse mishandles URLs without schemes (e.g., 'localhost:8765')
    # by treating the host as the scheme. Detect this by checking if netloc is empty.
    if parsed.scheme and not parsed.netloc:
        # Re-parse with http:// prefix
        parsed = urlparse(f"http://{url}")
    scheme = parsed.scheme.lower() if parsed.scheme else "http"

    for impl in list_types(category):
        try:
            cls = _registry._load_entry(impl)
            if hasattr(cls, "metadata"):
                metadata = cls.metadata()
                if hasattr(metadata, "url_schemes") and scheme in metadata.url_schemes:
                    return impl.type_name
        except Exception:
            continue

    raise ValueError(
        f"No {category} type found for URL scheme '{scheme}' in URL: {url}"
    )
