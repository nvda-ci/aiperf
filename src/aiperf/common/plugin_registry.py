# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin registry singleton with lazy loading and priority-based conflict resolution.

This module provides a modern, extensible plugin system for AIPerf that supports:
- Lazy class loading for optimal startup performance (Python 3.10+ features)
- Priority-based conflict resolution for package overrides
- Entry point discovery for external packages
- Type-safe APIs with comprehensive type hints and modern typing features
- Rich error messages with actionable feedback
- Pydantic validation for registry data
- Structural pattern matching for elegant parsing

Basic Usage
-----------
    from aiperf.common import plugin_registry

    # Get type class by name
    EndpointClass = plugin_registry.get_class('endpoint', 'openai')
    endpoint = EndpointClass(model_endpoint=config)

    # Or by fully qualified class path
    EndpointClass = plugin_registry.get_class(
        'endpoint',
        'aiperf.endpoints.openai:OpenAIEndpoint'
    )

    # List available types
    for impl in plugin_registry.list_types('endpoint'):
        print(f"{impl.type_name}: {impl.description}")

    # Get class and instantiate
    EndpointClass = plugin_registry.get_class('endpoint', 'openai')
    endpoint = EndpointClass(model_endpoint=config)

Advanced Usage
--------------
    # Conditional loading based on metadata
    for impl in plugin_registry.list_types('endpoint'):
        if impl.priority > 50 and not impl.is_builtin:
            EndpointClass = impl.load()
            endpoint = EndpointClass(...)

    # Testing with isolated registry
    plugin_registry.reset()
    plugin_registry.load_registry(test_registry_path)

Priority System
---------------
- Built-ins: priority = 0 (default, don't specify in YAML)
- External plugins: priority = 0 (default, can override)

Conflict Resolution:
1. If priorities differ → Higher priority wins
2. If priorities equal → External plugin ALWAYS wins over built-in
3. If both external plugins → First registered wins (warning issued)

YAML Schema
-----------
schema_version: "1.0"

plugin:
  name: my-plugin
  version: 1.0.0
  description: Custom plugin
  author: Your Name

endpoint:
  custom_endpoint:
    class: my_plugin.endpoints:CustomEndpoint
    description: Custom endpoint type
    priority: 100  # Optional, default: 0

timing_strategy:
  custom_strategy:
    class: my_plugin.timing:CustomStrategy
    description: Custom timing strategy
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypeAlias, TypedDict

import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.singleton import Singleton

if TYPE_CHECKING:
    from types import TraversableType

# ==============================================================================
# Type Aliases for Clarity
# ==============================================================================

CategoryName: TypeAlias = str
TypeName: TypeAlias = str
ClassPath: TypeAlias = str
PackageName: TypeAlias = str
Priority: TypeAlias = int

__all__ = [
    # Main API
    "get_class",
    "list_types",
    "list_categories",
    "create_enum",
    "register",
    "detect_type_from_url",
    # Startup
    "load_registry",
    # Utilities
    "list_packages",
    "get_package_metadata",
    "reset",
    "clear_singleton",
    "clear_all_singletons",
    # Types
    "TypeEntry",
    "PluginRegistry",
    # Exceptions
    "PluginError",
    "TypeNotFoundError",
    "TypeLoadError",
    "TypeConflictError",
    "SchemaVersionError",
]

_logger = AIPerfLogger(__name__)

# ==============================================================================
# Constants
# ==============================================================================

# Supported schema versions for registry manifests
SUPPORTED_SCHEMA_VERSIONS: Final[tuple[str, ...]] = ("1.0",)

# Default schema version when not specified
DEFAULT_SCHEMA_VERSION: Final[str] = "1.0"

# Entry point group for package discovery
DEFAULT_ENTRY_POINT_GROUP: Final[str] = "aiperf.plugins"

# ==============================================================================
# Type Definitions
# ==============================================================================


class PackageMetadata(TypedDict, total=False):
    """Package metadata from YAML manifest or package metadata.

    Attributes:
        name: Package name
        version: Version string
        description: Human-readable description
        author: Package author
        license: License identifier
        homepage: Package homepage URL
        builtin: Whether this is a built-in package
    """

    name: str
    version: str
    description: str
    author: str
    license: str
    homepage: str
    builtin: bool


class ManifestData(TypedDict, total=False):
    """YAML manifest structure.

    Attributes:
        schema_version: Schema version (e.g., "1.0")
        plugin: Plugin metadata section
    """

    schema_version: str
    plugin: dict[str, Any]


class TypeSpec(TypedDict, total=False):
    """Type specification from YAML.

    Can be either:
    - Simple format: "module:Class"
    - Full format: {"class": "module:Class", "description": "...", "priority": 100}

    Note: The YAML uses "class" as a key, but Python requires a workaround.
    We use dict access (spec.get("class")) instead of attribute access.

    Attributes:
        description: Human-readable description
        priority: Conflict resolution priority
    """

    description: str
    priority: int


# ==============================================================================
# Custom Exceptions
# ==============================================================================


class PluginError(Exception):
    """Base exception for plugin system errors.

    All plugin-related exceptions inherit from this base class, allowing
    callers to catch all plugin errors with a single except clause.

    Example:
        >>> try:
        ...     plugin_registry.get('endpoint', 'unknown')
        ... except PluginError as e:
        ...     print(f"Plugin error: {e}")
    """


class TypeNotFoundError(PluginError):
    """Type not found in category.

    Attributes:
        category: Category name
        type_name: Type name
        available: List of available types
    """

    def __init__(self, category: str, type_name: str, available: list[str]) -> None:
        """Initialize with rich error message.

        Args:
            category: Category name
            type_name: Type name that was not found
            available: List of available type names
        """
        self.category = category
        self.type_name = type_name
        self.available = available

        available_str = "\n".join(f"  • {name}" for name in sorted(available))
        super().__init__(
            f"Type '{type_name}' not found for category '{category}'.\n"
            f"Available types:\n{available_str}"
        )


class TypeLoadError(PluginError):
    """Failed to load type.

    This exception preserves the original exception type (ImportError or AttributeError)
    in the `cause` attribute for backward compatibility. Tests that check exception
    types should check isinstance(e.cause, ImportError) or isinstance(e, TypeLoadError).

    Attributes:
        class_path: Fully qualified class path
        category: Category name
        type_name: Type name
        cause: Original exception (ImportError or AttributeError)
    """

    def __init__(
        self,
        class_path: str,
        category: str,
        type_name: str,
        cause: Exception,
    ) -> None:
        """Initialize with rich error message.

        Args:
            class_path: Fully qualified class path
            category: Category name
            type_name: Type name
            cause: Original exception that caused the load failure
        """
        self.class_path = class_path
        self.category = category
        self.type_name = type_name
        self.cause = cause

        # Construct rich error message
        # Use specific terminology for backward compatibility with existing tests
        if isinstance(cause, ImportError):
            message = (
                f"Failed to import module for {category}:{type_name} from '{class_path}'\n"
                f"Reason: {cause!r}\n"
                f"Tip: Check that the module is installed and importable"
            )
        elif isinstance(cause, AttributeError):
            # Extract class name from class_path for error message
            _, class_name = (
                class_path.rsplit(":", 1) if ":" in class_path else ("", "Unknown")
            )
            message = (
                f"Class '{class_name}' not found for {category}:{type_name} from '{class_path}'\n"
                f"Reason: {cause!r}\n"
                f"Tip: Check that the class name is spelled correctly and exported from the module"
            )
        else:
            message = (
                f"Failed to load {category}:{type_name} from '{class_path}'\n"
                f"Reason: {cause!r}"
            )

        super().__init__(message)


class TypeConflictError(PluginError):
    """Multiple plugins provide the same type.

    Attributes:
        category: Category name
        type_name: Type name
        existing_package: Name of existing package
        new_package: Name of new package
    """

    def __init__(
        self,
        category: str,
        type_name: str,
        existing_package: str,
        new_package: str,
    ) -> None:
        """Initialize with rich error message.

        Args:
            category: Category name
            type_name: Type name
            existing_package: Name of package that registered first
            new_package: Name of package attempting to register
        """
        self.category = category
        self.type_name = type_name
        self.existing_package = existing_package
        self.new_package = new_package

        super().__init__(
            f"Conflict for {category}:{type_name}\n"
            f"Both '{existing_package}' and '{new_package}' provide this type."
        )


class SchemaVersionError(PluginError):
    """Incompatible schema version in registry manifest.

    Attributes:
        schema_version: The unsupported schema version
        supported_versions: List of supported schema versions
    """

    def __init__(self, schema_version: str, supported_versions: Sequence[str]) -> None:
        """Initialize with rich error message.

        Args:
            schema_version: The incompatible schema version
            supported_versions: List of supported schema versions
        """
        self.schema_version = schema_version
        self.supported_versions = list(supported_versions)

        supported_str = ", ".join(f"'{v}'" for v in supported_versions)
        super().__init__(
            f"Unsupported registry schema version: '{schema_version}'\n"
            f"Supported versions: {supported_str}\n"
            f"Tip: Update your package registry to use a supported schema version"
        )


# ==============================================================================
# Implementation Classes
# ==============================================================================


@dataclass(frozen=True, slots=True)
class TypeEntry:
    """Lazy-loading type entry with metadata.

    This class represents a type that hasn't been loaded yet.
    The actual class is only imported when load() is called, enabling fast
    startup and minimal memory usage.

    Architecture:
    ┌──────────────────────────────┐
    │ TypeEntry                     │
    │ ────────────────             │
    │ • type_name: str             │
    │ • class_path: str            │
    │ • priority: int              │
    │ • description: str           │
    │ • loaded_class: type | None  │
    │                              │
    │ load() → class (lazy!)       │
    └──────────────────────────────┘

    Attributes:
        category: Category name (e.g., 'endpoint', 'timing_strategy')
        type_name: Type name (e.g., 'openai', 'fixed_schedule')
        package_name: Package name (e.g., 'aiperf', 'aiperf-custom-plugin')
        class_path: Full class path (e.g., 'aiperf.endpoints.openai:OpenAIEndpoint')
        priority: Priority for conflict resolution (higher = preferred, default: 0)
        description: Human-readable description of type
        metadata: Package metadata from installed package (version, author, etc.)
        is_builtin: Whether this is a built-in type

    Thread Safety:
        This class is immutable (frozen=True) except for the cached loaded_class.
        The load() method uses object.__setattr__ for thread-safe caching.
    """

    category: str = field(metadata={"description": "Category identifier"})
    type_name: str = field(metadata={"description": "Type name"})
    package_name: str = field(metadata={"description": "Package providing this type"})
    class_path: str = field(
        metadata={"description": "Fully qualified class path (module:Class)"}
    )
    priority: int = field(
        default=0, metadata={"description": "Conflict resolution priority"}
    )
    description: str = field(
        default="", metadata={"description": "Human-readable description"}
    )
    metadata: PackageMetadata = field(
        default_factory=dict, metadata={"description": "Package metadata"}
    )
    loaded_class: type | None = field(
        default=None, metadata={"description": "Cached class after loading"}
    )
    is_builtin: bool = field(
        default=False, metadata={"description": "Whether this is built-in"}
    )

    def load(self) -> type:
        """Load the type class with lazy caching.

        This method imports the module and retrieves the class on first call,
        then caches the result for subsequent calls. The caching is thread-safe
        due to the frozen dataclass and object.__setattr__ usage.

        Returns:
            The loaded class (not instantiated - caller must instantiate)

        Raises:
            ValueError: If class_path format is invalid
            TypeLoadError: If module or class cannot be imported

        Example:
            >>> lazy_type = TypeEntry(
            ...     category='endpoint',
            ...     type_name='openai',
            ...     package_name='aiperf',
            ...     class_path='aiperf.endpoints.openai:OpenAIEndpoint'
            ... )
            >>> EndpointClass = lazy_type.load()
            >>> endpoint = EndpointClass(model_endpoint=config)
        """
        # Return cached class if already loaded
        if self.loaded_class is not None:
            return self.loaded_class

        # Validate and parse class path using structural pattern matching
        match self.class_path.split(":"):
            case [module_path, class_name] if module_path and class_name:
                # Valid format: "module.path:ClassName"
                pass
            case _:
                # Invalid format
                raise ValueError(
                    f"Invalid class_path format: {self.class_path}\n"
                    f"Expected format: 'module.path:ClassName'\n"
                    f"Example: 'aiperf.endpoints.openai:OpenAIEndpoint'"
                )

        # Import and cache the class
        try:
            cls = _import_class_cached(module_path, class_name)

            # Set registration metadata on class for reverse lookup
            # This allows classes to know their registered name (e.g., for service_type)
            cls._registered_name = self.type_name

            # Cache for future calls (thread-safe with frozen dataclass)
            object.__setattr__(self, "loaded_class", cls)

            _logger.debug(
                lambda: f"Loaded {self.category}:{self.type_name} from {self.class_path}"
            )

            return cls

        except ImportError as e:
            # Raise enriched ImportError for backward compatibility
            raise ImportError(
                f"Failed to import module for {self.category}:{self.type_name} from '{self.class_path}'\n"
                f"Reason: {e!r}\n"
                f"Tip: Check that the module is installed and importable"
            ) from e
        except AttributeError as e:
            # Raise enriched AttributeError for backward compatibility
            raise AttributeError(
                f"Class '{class_name}' not found for {self.category}:{self.type_name} from '{self.class_path}'\n"
                f"Reason: {e!r}\n"
                f"Tip: Check that the class name is spelled correctly and exported from the module"
            ) from e


# ==============================================================================
# Registry Class
# ==============================================================================


class PluginRegistry(Singleton):
    """Plugin registry singleton with discovery and lazy loading.

    This class manages the complete lifecycle of plugins:
    - Discovery from built-in registry and external entry points
    - Lazy loading of type classes
    - Priority-based conflict resolution
    - Metadata tracking for debugging and introspection

    Thread Safety:
        This class is NOT thread-safe. It should only be modified during
        application startup before concurrent access begins.
    """

    def __init__(self) -> None:
        super().__init__()
        _logger.debug("Initialized plugin registry singleton")
        # Nested dict: category -> type_name -> TypeEntry
        self._types: dict[str, dict[str, TypeEntry]] = {}
        # Reverse lookup: class_path -> TypeEntry
        self._by_class_path: dict[str, TypeEntry] = {}
        # Loaded plugin metadata: plugin_name -> metadata
        self._loaded_plugins: dict[str, PackageMetadata] = {}

        # Load the builtin registry manifest and discover plugins once on startup
        self.load_registry(_get_builtins_path())
        self.discover_plugins()

    @classmethod
    def _reset_singleton(cls) -> None:
        """Reset the singleton instance (for testing only).

        This clears the singleton instance, allowing a fresh one to be
        created on the next instantiation. This is primarily useful for
        testing scenarios where you need an isolated registry.
        """
        import os

        from aiperf.common.singleton import SingletonMeta

        # Remove from the metaclass's instances dict
        key = (cls, os.getpid())
        SingletonMeta._instances.pop(key, None)

    def load_registry(self, registry_path: Path | str | TraversableType) -> None:
        """Load built-in registry from YAML file.

        This method loads the main registry.yaml file that defines all built-in
        types. It uses importlib.resources for proper package resource
        access when installed.

        Args:
            registry_path: Path to registry.yaml. If None, uses the built-in
                registry from the aiperf package.

        Raises:
            FileNotFoundError: If registry file not found at specified path
            RuntimeError: If built-in registry.yaml not found in package
            yaml.YAMLError: If YAML parsing fails

        Example:
            # Load built-in registry
            >>> registry = PluginRegistry()
            >>> registry.load_builtins()

            # Load custom registry for testing
            >>> registry.load_builtins("/path/to/test_registry.yaml")
        """
        # Load YAML content
        yaml_content = self._read_registry_file(registry_path)

        # Parse YAML
        data: ManifestData = yaml.safe_load(yaml_content)

        if not data:
            _logger.warning(f"Empty registry YAML: {registry_path}")
            return

        # Validate schema
        self._validate_manifest_schema(data)

        # Extract plugin info
        plugin_info = data.get("plugin", {})
        package_name = plugin_info.get("name", "aiperf")
        is_builtin = plugin_info.get("builtin", True)

        _logger.info(
            f"Loading registry: {package_name} (builtin={is_builtin}, schema={data.get('schema_version', '1.0')})"
        )

        # Register types from manifest
        self._register_manifest(package_name, data, is_builtin)

        _logger.info(
            f"Loaded registry: {package_name} with {len([k for k in data if k not in ('plugin', 'schema_version')])} categories"
        )

    def discover_plugins(
        self, entry_point_group: str = DEFAULT_ENTRY_POINT_GROUP
    ) -> None:
        """Discover and load plugin registries via entry points.

        This method scans for external plugins that have registered themselves
        via setuptools entry points. Each entry point should return a path to
        a registry.yaml file.

        Args:
            entry_point_group: Entry point group name for plugin discovery.
                Plugins register themselves in setup.py/pyproject.toml:
                [project.entry-points."aiperf.plugins"]
                my_plugin = "my_plugin:get_registry_path"

        Example:
            # In plugin's setup.py or pyproject.toml:
            entry_points={
                "aiperf.plugins": [
                    "my_plugin = my_plugin:get_registry_path"
                ]
            }

            # In plugin's __init__.py:
            def get_registry_path():
                return str(Path(__file__).parent / "registry.yaml")

            # In AIPerf:
            >>> registry.discover_plugins()
            # Automatically discovers and loads all registered plugins

        Raises:
            yaml.YAMLError: If plugin YAML parsing fails (logged, not raised)
        """
        _logger.debug(lambda: f"Discovering plugins in {entry_point_group}")

        # Discover entry points (Python 3.10+ API)
        eps = entry_points(group=entry_point_group)

        # Handle both Python 3.9 and 3.10+ APIs
        plugin_eps = list(eps) if hasattr(eps, "__iter__") else []

        for ep in plugin_eps:
            try:
                # Load entry point (should return path to registry.yaml)
                registry_path = ep.load()

                if not isinstance(registry_path, (str, Path)):
                    _logger.warning(
                        f"Invalid entry point return type for {ep.name}: {type(registry_path).__name__}"
                    )
                    continue

                _logger.info(f"Loading plugin: {ep.name}")

                # Load plugin registry
                self.load_registry(registry_path)

            except Exception as e:
                # Graceful failure for bad plugins
                _logger.exception(f"Failed to load plugin {ep.name}: {e!r}")

        _logger.info(f"Plugin discovery complete: {len(plugin_eps)} plugins found")

    def get(self, category: str, name_or_class_path: str) -> type:
        """Get type class by name or fully qualified class path.

        This is the primary method for retrieving plugin types.
        It supports both short names ('openai') and full class paths
        ('aiperf.endpoints.openai:OpenAIEndpoint').

        The class is lazy-loaded on first access and cached for subsequent calls.

        Args:
            category: Category name identifier (e.g., 'endpoint', 'timing_strategy').
                See registry.yaml for available categories.
            name_or_class_path: Either:
                - Short name: 'openai', 'anthropic', 'fixed_schedule'
                - Full class path: 'aiperf.endpoints.openai:OpenAIEndpoint'

        Returns:
            The type class (not instantiated). Caller must instantiate
            with appropriate constructor arguments.

        Raises:
            KeyError: If category or type not found
            ValueError: If class path category doesn't match requested category
            TypeLoadError: If class cannot be imported
            TypeNotFoundError: If type not found (with suggestions)

        Example:
            Get by name:
            >>> EndpointClass = plugin_registry.get('endpoint', 'openai')
            >>> endpoint = EndpointClass(model_endpoint=config)

            Get by class path:
            >>> EndpointClass = plugin_registry.get(
            ...     'endpoint',
            ...     'aiperf.endpoints.openai:OpenAIEndpoint'
            ... )
            >>> endpoint = EndpointClass(model_endpoint=config)

            Handle errors:
            >>> try:
            ...     cls = plugin_registry.get('endpoint', 'unknown')
            ... except TypeNotFoundError as e:
            ...     print(f"Available: {e.available}")
        """
        # Check if it's a class path (contains ':')
        if ":" in name_or_class_path:
            return self._get_by_class_path(category, name_or_class_path)
        else:
            return self._get_by_name(category, name_or_class_path)

    def list_types(self, category: str) -> list[TypeEntry]:
        """List all types for a category.

        Returns TypeEntry objects (NOT loaded classes). This allows
        inspecting metadata without triggering imports. Caller can load classes
        manually using impl.load().

        Args:
            category: Category name (e.g., 'endpoint', 'timing_strategy')

        Returns:
            List of TypeEntry objects, sorted alphabetically by name.
            Returns empty list if category not found.

        Example:
            Inspect metadata:
            >>> for impl in plugin_registry.list_types('endpoint'):
            ...     print(f"{impl.type_name}: {impl.description}")
            ...     print(f"  Priority: {impl.priority}")
            ...     print(f"  Plugin: {impl.package_name}")

            Conditional loading:
            >>> for impl in plugin_registry.list_types('endpoint'):
            ...     if impl.priority > 50 and not impl.is_builtin:
            ...         EndpointClass = impl.load()
            ...         endpoint = EndpointClass(...)

            Get all names:
            >>> names = [impl.type_name for impl in plugin_registry.list_types('endpoint')]
        """
        if category not in self._types:
            _logger.debug(
                lambda cat=category: f"Category not found in list_types: {cat}"
            )
            return []

        impls = list(self._types[category].values())
        impls.sort(key=lambda x: x.type_name)  # Alphabetical order

        _logger.debug(lambda cat=category, i=impls: f"Listed {len(i)} types for {cat}")

        return impls

    def list_packages(self, builtin_only: bool = False) -> list[str]:
        """List all loaded plugin packages.

        Args:
            builtin_only: If True, only return built-in packages

        Returns:
            List of package names

        Example:
            >>> packages = registry.list_packages()
            >>> print(f"Loaded packages: {', '.join(packages)}")

            >>> builtins = registry.list_packages(builtin_only=True)
        """
        if builtin_only:
            return [
                name
                for name, meta in self._loaded_plugins.items()
                if meta.get("builtin", False)
            ]
        return list(self._loaded_plugins.keys())

    # --------------------------------------------------------------------------
    # Private: Class Path Operations
    # --------------------------------------------------------------------------

    def _get_by_class_path(self, category: str, class_path: str) -> type:
        """Get type by class path with category validation.

        Args:
            category: Expected category
            class_path: Fully qualified class path

        Returns:
            Loaded class

        Raises:
            KeyError: If class path not registered
            ValueError: If category mismatch
        """
        if class_path not in self._by_class_path:
            raise KeyError(
                f"No type with class path: {class_path}\n"
                f"Hint: Class path must be registered in a registry.yaml"
            )

        lazy_type = self._by_class_path[class_path]

        # Verify category matches
        if lazy_type.category != category:
            raise ValueError(
                f"Category mismatch: {class_path} is registered for category "
                f"'{lazy_type.category}', not '{category}'"
            )

        return lazy_type.load()

    def _get_by_name(self, category: str, type_name: str) -> type:
        """Get type by short name.

        Args:
            category: Category name
            type_name: Implementation name

        Returns:
            Loaded class

        Raises:
            KeyError: If category not found
            TypeNotFoundError: If type not found
        """
        if category not in self._types:
            raise KeyError(
                f"Unknown category: {category}\n"
                f"Available categories: {sorted(self._types.keys())}"
            )

        if type_name not in self._types[category]:
            available = list(self._types[category].keys())
            raise TypeNotFoundError(category, type_name, available)

        lazy_type = self._types[category][type_name]
        return lazy_type.load()

    def find_registered_name(self, category: str, cls: type) -> str | None:
        """Find the registered name for a class within a category.

        This performs a reverse lookup to find what name a class is registered
        under. Useful when a class is instantiated directly (not via the registry)
        and needs to know its registered identity.

        Args:
            category: Category name to search in (e.g., 'service', 'endpoint')
            cls: The class to look up

        Returns:
            The registered type name, or None if not found

        Example:
            >>> name = registry.find_registered_name('service', DatasetManager)
            >>> print(name)  # 'dataset_manager'
        """
        if category not in self._types:
            return None

        # Build class path for the class we're looking for
        target_class_path = f"{cls.__module__}:{cls.__name__}"

        # Search through types
        for type_name, lazy_type in self._types[category].items():
            if lazy_type.class_path == target_class_path:
                return type_name

        return None

    # --------------------------------------------------------------------------
    # Private: Registry Loading
    # --------------------------------------------------------------------------

    def _read_registry_file(self, registry_path: Path | str | TraversableType) -> str:
        """Read registry YAML file content with robust error handling.

        This method handles different path types (Path, str, Traversable) and provides
        clear error messages when files cannot be read.

        Args:
            registry_path: Path to registry file (Path, str, or Traversable)

        Returns:
            YAML content as string

        Raises:
            FileNotFoundError: If file not found (with helpful message)
            RuntimeError: If built-in registry not found (critical error)
            OSError: If file cannot be read due to permissions or I/O error

        Example:
            >>> content = registry._read_registry_file(Path("registry.yaml"))
            >>> content = registry._read_registry_file("/path/to/registry.yaml")
        """
        try:
            if hasattr(registry_path, "read_text"):
                # Traversable from importlib.resources
                return registry_path.read_text(encoding="utf-8")
            else:
                # Regular Path (convert str to Path)
                path = (
                    Path(registry_path)
                    if isinstance(registry_path, str)
                    else registry_path
                )

                if not path.exists():
                    raise FileNotFoundError(
                        f"Registry file not found: {path.absolute()}\n"
                        f"Please ensure the registry.yaml file exists at this location.\n"
                        f"Tip: Check your package installation or path configuration"
                    )

                if not path.is_file():
                    raise ValueError(
                        f"Registry path is not a file: {path.absolute()}\n"
                        f"Expected a YAML file, got a directory or special file"
                    )

                return path.read_text(encoding="utf-8")

        except FileNotFoundError:
            # Re-raise with context for debugging
            raise RuntimeError(
                f"Built-in registry.yaml not found at {registry_path}.\n"
                "This is a critical error - the package system cannot function without it.\n"
                "Tip: Reinstall the aiperf package or check your installation"
            ) from None
        except OSError as e:
            # Handle permission errors, I/O errors, etc.
            raise RuntimeError(
                f"Failed to read registry file: {registry_path}\n"
                f"Reason: {e}\n"
                f"Tip: Check file permissions and disk status"
            ) from e

    def _validate_manifest_schema(self, manifest_data: ManifestData) -> None:
        """Validate manifest structure and schema version.

        Args:
            manifest_data: Parsed YAML data

        Raises:
            ValueError: If schema is invalid or incompatible
            SchemaVersionError: If schema version is unsupported
        """
        # Validate schema_version field
        schema_version = manifest_data.get("schema_version")

        # Handle missing schema version
        if not schema_version:
            _logger.warning(
                f"Missing schema_version in manifest, assuming {DEFAULT_SCHEMA_VERSION}"
            )
            return

        # Validate type
        if not isinstance(schema_version, str):
            raise ValueError(
                f"schema_version must be string, got {type(schema_version).__name__}"
            )

        # Check for supported versions using pattern matching
        match schema_version:
            case version if version in SUPPORTED_SCHEMA_VERSIONS:
                # Supported version - continue
                _logger.debug(lambda v=version: f"Using schema version {v}")
            case _:
                # Unsupported version - warn but continue for forward compatibility
                _logger.warning(
                    f"Unknown schema version {schema_version}, supported: {list(SUPPORTED_SCHEMA_VERSIONS)}"
                )

    def _register_manifest(
        self, package_name: str, manifest_data: ManifestData, is_builtin: bool
    ) -> None:
        """Register types from manifest with conflict resolution.

        This method processes a YAML manifest and registers all types.
        It handles conflicts using priority-based resolution:
        1. Higher priority wins
        2. Equal priority: package beats built-in
        3. Both plugins: first registered wins (with warning)

        Args:
            package_name: Plugin name from manifest
            manifest_data: Parsed YAML data
            is_builtin: Whether this is a built-in plugin

        Example Manifest:
            schema_version: "1.0"
            plugin:
              name: my-plugin
              version: 1.0.0
            endpoint:
              openai:
                class: my_plugin.endpoints:OpenAI
                description: Custom OpenAI endpoint
                priority: 100
        """
        # Load package metadata from installed package
        package_metadata = self._load_package_metadata(package_name, is_builtin)
        self._loaded_plugins[package_name] = package_metadata

        # Process each category
        for category_name, types_dict in manifest_data.items():
            # Skip metadata sections
            if category_name in ("plugin", "schema_version"):
                continue

            if not isinstance(types_dict, dict):
                _logger.warning(
                    f"Invalid category section type for {category_name}: {type(types_dict).__name__}"
                )
                continue

            # Ensure category exists in registry
            if category_name not in self._types:
                self._types[category_name] = {}

            # Register each type
            for type_name, type_data in types_dict.items():
                self._register_type(
                    category_name=category_name,
                    type_name=type_name,
                    type_data=type_data,
                    package_name=package_name,
                    package_metadata=package_metadata,
                    is_builtin=is_builtin,
                )

    def _register_type(
        self,
        category_name: str,
        type_name: str,
        type_data: TypeSpec | str,
        package_name: str,
        package_metadata: PackageMetadata,
        is_builtin: bool,
    ) -> None:
        """Register a single type with conflict resolution.

        Args:
            category_name: Category name
            type_name: Type name
            type_data: Type spec (string or dict)
            package_name: Package providing this type
            package_metadata: Package metadata
            is_builtin: Whether this is built-in
        """
        # Normalize type data
        match type_data:
            case dict() | TypeSpec():
                # Full format with metadata
                spec = type_data  # type: ignore[assignment]
            case str():
                # Simple format: "module:Class"
                spec: TypeSpec = {"class": type_data}
            case _:
                _logger.warning(
                    f"Invalid type format for {category_name}:{type_name}: {type(type_data).__name__}"
                )
                return

        # Extract required class path
        class_path = spec.get("class")
        if not class_path:
            _logger.warning(f"Missing class path for {category_name}:{type_name}")
            return

        # Extract optional fields
        priority = spec.get("priority")
        if priority is None:
            priority = 0  # Default priority

        description = spec.get("description", "")

        # Create lazy type
        lazy_type = TypeEntry(
            category=category_name,
            type_name=type_name,
            package_name=package_name,
            class_path=class_path,
            priority=priority,
            description=description,
            metadata=package_metadata,
            is_builtin=is_builtin,
        )

        # Handle conflicts with smart resolution
        self._resolve_conflict_and_register(category_name, type_name, lazy_type)

    def _resolve_conflict_and_register(
        self,
        category_name: str,
        type_name: str,
        lazy_type: TypeEntry,
    ) -> None:
        """Resolve conflicts and register type.

        Conflict Resolution Rules:
        1. Higher priority wins
        2. Equal priority: package beats built-in
        3. Both plugins: first registered wins (warn)

        Args:
            category_name: Category name
            type_name: Implementation name
            lazy_type: New type to register
        """
        existing = self._types[category_name].get(type_name)

        if existing is None:
            # No conflict - register directly
            self._types[category_name][type_name] = lazy_type
            self._by_class_path[lazy_type.class_path] = lazy_type

            _logger.debug(
                lambda cat=category_name,
                t=type_name,
                lt=lazy_type: f"Registered {cat}:{t} from {lt.package_name} (priority={lt.priority})"
            )
            return

        # Conflict exists - resolve based on priority and type
        winner, reason = self._resolve_conflict(existing, lazy_type)

        if winner is lazy_type:
            # New type wins
            self._types[category_name][type_name] = lazy_type
            self._by_class_path[lazy_type.class_path] = lazy_type

            _logger.info(
                f"Override registered {category_name}:{type_name}: {lazy_type.package_name} beats {existing.package_name} ({reason})"
            )
        else:
            # Existing type wins
            _logger.debug(
                lambda cat=category_name,
                t=type_name,
                ex=existing,
                lt=lazy_type,
                r=reason: f"Override rejected {cat}:{t}: {ex.package_name} beats {lt.package_name} ({r})"
            )

    def _resolve_conflict(
        self,
        existing: TypeEntry,
        new: TypeEntry,
    ) -> tuple[TypeEntry, str]:
        """Resolve conflict between existing and new type.

        Args:
            existing: Currently registered type
            new: New type attempting to register

        Returns:
            Tuple of (winner, reason)
        """
        # Rule 1: Higher priority wins
        if new.priority > existing.priority:
            return new, f"priority {new.priority} > {existing.priority}"
        elif new.priority < existing.priority:
            return existing, f"priority {existing.priority} > {new.priority}"

        # Rule 2: Equal priority - package beats built-in
        if not new.is_builtin and existing.is_builtin:
            return new, "package overrides built-in (equal priority)"
        elif new.is_builtin and not existing.is_builtin:
            return existing, "package overrides built-in (equal priority)"

        # Rule 3: Both same type - first wins (warn)
        _logger.warning(
            f"Plugin conflict for {new.category}:{new.type_name}: {existing.package_name} vs {new.package_name} (priority={new.priority})"
        )

        return existing, "first registered wins (both same type)"

    def _load_package_metadata(
        self, package_name: str, is_builtin: bool
    ) -> PackageMetadata:
        """Load package metadata from installed package.

        Args:
            package_name: Package name
            is_builtin: Whether this is built-in

        Returns:
            Metadata dict with version, author, etc.
        """
        if is_builtin:
            return PackageMetadata(name="aiperf", builtin=True)

        try:
            import importlib.metadata

            pkg_metadata = importlib.metadata.metadata(package_name)

            return PackageMetadata(
                name=pkg_metadata.get("Name", package_name),
                version=pkg_metadata.get("Version", "unknown"),
                description=pkg_metadata.get("Summary", ""),
                author=pkg_metadata.get("Author", ""),
                license=pkg_metadata.get("License", ""),
                homepage=pkg_metadata.get("Home-page", ""),
                builtin=False,
            )
        except Exception as e:
            _logger.warning(
                f"Failed to load package metadata for {package_name}: {e!r}"
            )
            return PackageMetadata(name=package_name, builtin=False)


# ==============================================================================
# Module-Level Singleton
# ==============================================================================
# This pattern follows the random_generator module design.
# Usage:
#   from aiperf.common import plugin_registry
#   EndpointClass = plugin_registry.get('endpoint', 'openai')
# ==============================================================================


def _get_builtins_path() -> Path | TraversableType:
    """Get path to built-in registry.yaml.

    Returns:
        Path to registry.yaml in aiperf package

    Raises:
        RuntimeError: If registry.yaml not found
    """
    try:
        from importlib.resources import files

        return files("aiperf") / "registry.yaml"
    except Exception as e:
        # Fallback to relative path if running from source
        fallback = Path(__file__).parent.parent / "registry.yaml"
        if not fallback.exists():
            raise RuntimeError(
                "Built-in registry.yaml not found in aiperf package.\n"
                "This is a critical error - the package system cannot function without it."
            ) from e
        return fallback


# Create singleton instance at module load
_registry = PluginRegistry()


# ==============================================================================
# Public API: Module-Level Functions
# ==============================================================================


def get_class(category: str, name_or_class_path: str) -> type:
    """Get type class by name or fully qualified class path.

    This is the primary API for retrieving plugin types.
    See PluginRegistry.get() for full documentation.

    Args:
        category: Category name (e.g., 'endpoint', 'timing_strategy')
        name_or_class_path: Implementation name or full class path

    Returns:
        Loaded class (not instantiated)

    Raises:
        KeyError: If category or type not found
        TypeNotFoundError: If type not found (with suggestions)
        TypeLoadError: If class cannot be imported

    Example:
        >>> EndpointClass = plugin_registry.get_class('endpoint', 'openai')
        >>> endpoint = EndpointClass(model_endpoint=config)
    """
    return _registry.get(category, name_or_class_path)


def list_types(category: str) -> list[TypeEntry]:
    """List all types for a category.

    See PluginRegistry.list_types() for full documentation.

    Args:
        category: Category name

    Returns:
        List of TypeEntry objects (sorted alphabetically)

    Example:
        >>> for impl in plugin_registry.list_types('endpoint'):
        ...     print(f"{impl.type_name}: {impl.description}")
    """
    return _registry.list_types(category)


def find_registered_name(category: str, cls: type) -> str | None:
    """Find the registered name for a class within a category.

    Performs a reverse lookup to find what name a class is registered under.
    Useful when a class is instantiated directly (not via the registry) and
    needs to know its registered identity.

    Args:
        category: Category name to search in (e.g., 'service', 'endpoint')
        cls: The class to look up

    Returns:
        The registered type name, or None if not found

    Example:
        >>> from aiperf.dataset import DatasetManager
        >>> name = plugin_registry.find_registered_name('service', DatasetManager)
        >>> print(name)  # 'dataset_manager'
    """
    return _registry.find_registered_name(category, cls)


def load_registry(registry_path: str | Path) -> None:
    """Load built-in registry from YAML file.

    See PluginRegistry.load_builtins() for full documentation.

    Args:
        registry_path: Path to registry.yaml

    Example:
        >>> plugin_registry.load_builtins("/path/to/custom.yaml")
    """
    _registry.load_registry(registry_path)


def list_packages(builtin_only: bool = False) -> list[str]:
    """List all loaded plugin packages.

    Args:
        builtin_only: If True, only return built-in packages

    Returns:
        List of package names

    Example:
        >>> packages = plugin_registry.list_packages()
        >>> print(f"Loaded: {', '.join(packages)}")
    """
    return _registry.list_packages(builtin_only)


def get_package_metadata(package_name: str) -> PackageMetadata:
    """Get metadata for a loaded plugin package.

    Args:
        package_name: Name of the package to get metadata for

    Returns:
        PackageMetadata dict with name, version, builtin flag, etc.

    Raises:
        KeyError: If package not found

    Example:
        >>> metadata = plugin_registry.get_package_metadata("aiperf")
        >>> print(f"Built-in: {metadata.get('builtin', False)}")
    """
    if package_name not in _registry._loaded_plugins:
        raise KeyError(f"Package '{package_name}' not found in loaded plugins")
    return _registry._loaded_plugins[package_name]


def list_categories() -> list[str]:
    """List all registered category names.

    Returns a list of all category names that have at least one
    registered type. This is useful for dynamic enum generation
    and introspection.

    Returns:
        List of category names sorted alphabetically

    Example:
        >>> categories = plugin_registry.list_categories()
        >>> print(categories)
        ['arrival_pattern', 'communication', 'endpoint', ...]
    """
    return sorted(_registry._types.keys())


def reset() -> None:
    """Reset registry to empty state (for testing).

    Clears the singleton and creates a fresh registry instance.
    This is primarily useful for testing scenarios where you need an isolated
    registry.

    Example:
        >>> plugin_registry.reset()
        >>> plugin_registry.load_builtins(test_registry_path)
    """
    global _registry
    PluginRegistry._reset_singleton()
    _registry = PluginRegistry()
    _logger.debug("Registry reset")


def clear_all_singletons() -> None:
    """Clear all singleton instances cached by SingletonMeta.

    This is useful for test cleanup to ensure singleton communication
    backends and other singleton instances don't leak between tests.
    """
    import os

    from aiperf.common.singleton import SingletonMeta

    pid = os.getpid()
    keys_to_remove = [key for key in SingletonMeta._instances if key[1] == pid]
    for key in keys_to_remove:
        SingletonMeta._instances.pop(key, None)
    _logger.debug(f"Cleared {len(keys_to_remove)} singleton instances")


def clear_singleton(category: str, type_name: str) -> None:
    """Clear a specific singleton instance by category and type.

    Args:
        category: Category name (e.g., 'communication')
        type_name: Type name (e.g., 'zmq_ipc')
    """
    import os

    from aiperf.common.singleton import SingletonMeta

    try:
        cls = _registry.get(category, type_name)
        key = (cls, os.getpid())
        if key in SingletonMeta._instances:
            SingletonMeta._instances.pop(key, None)
            _logger.debug(f"Cleared singleton for {category}:{type_name}")
    except TypeNotFoundError:
        pass  # Type doesn't exist, nothing to clear


def register(
    category: str,
    type_name: str,
    cls: type,
    *,
    priority: int = 0,
    is_builtin: bool = True,
) -> None:
    """Register a class for a category programmatically.

    This is useful for dynamically generated classes that cannot be
    registered via registry.yaml, and for test overrides.

    Args:
        category: Category name (e.g., 'zmq_proxy')
        type_name: Implementation name (can be enum or string)
        cls: The class to register
        priority: Override priority (higher wins). Use sys.maxsize for test overrides.
        is_builtin: Whether this is a built-in type (False for test fakes)

    Example:
        >>> plugin_registry.register('zmq_proxy', ZMQProxyType.XPUB_XSUB, MyProxyClass)
        >>> # Test override with high priority:
        >>> plugin_registry.register('communication', 'zmq_ipc', FakeComm, priority=sys.maxsize)
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
        metadata={"name": "aiperf", "builtin": is_builtin},
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

    This creates a dynamic enum that:
    - Works with Pydantic validation (it's a str subclass)
    - Works with cyclopts CLI (has __iter__ for choices)
    - Supports case-insensitive lookups
    - Can be extended at runtime if new types are registered

    The enum is created from the currently registered types in the category.
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
        >>> EndpointType.CHAT
        EndpointType.CHAT
        >>> EndpointType('chat')  # case-insensitive
        EndpointType.CHAT
        >>> list(EndpointType)  # cyclopts uses this for --help
        [EndpointType.CHAT, EndpointType.COMPLETIONS, ...]

        # Use with Pydantic
        >>> class Config(BaseModel):
        ...     endpoint: EndpointType

        # Use with cyclopts
        >>> @app.command
        ... def run(endpoint: EndpointType): ...
    """
    from aiperf.common.enums.base_enums import create_enum as _create_enum

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
    """Detect the type from a URL.

    Queries all types for the category and returns the first
    one whose metadata indicates it can handle the URL scheme.

    Args:
        category: Category name (e.g., 'transport')
        url: URL to detect transport for

    Returns:
        Implementation name that can handle this URL

    Raises:
        ValueError: If no type can handle the URL

    Example:
        >>> transport_type = plugin_registry.detect_type_from_url('transport', 'http://example.com')
        >>> # Returns 'http' or similar
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
            cls = impl.load()
            if hasattr(cls, "metadata"):
                metadata = cls.metadata()
                if hasattr(metadata, "url_schemes") and scheme in metadata.url_schemes:
                    return impl.type_name
        except Exception:
            continue

    raise ValueError(
        f"No {category} type found for URL scheme '{scheme}' in URL: {url}"
    )


# ==============================================================================
# Private Module Functions
# ==============================================================================


@cache
def _import_class_cached(module_path: str, class_name: str) -> type:
    """Import class with LRU caching for optimal performance.

    This function caches imported classes to avoid redundant imports when
    the same class is loaded multiple times. The unbounded cache (maxsize=None)
    is safe because:
    1. The number of unique class paths is bounded by package count
    2. Class objects are lightweight references (not instances)
    3. Classes persist in sys.modules anyway

    Performance Benefits:
    - Avoids repeated module imports and attribute lookups
    - Critical for test suites that reload plugins frequently
    - Negligible memory overhead (typically < 100 entries)

    Args:
        module_path: Module path (e.g., 'aiperf.endpoints.openai')
        class_name: Class name (e.g., 'OpenAIEndpoint')

    Returns:
        Imported class (not instantiated)

    Raises:
        ImportError: If module cannot be imported
        AttributeError: If class not found in module

    Example:
        >>> EndpointClass = _import_class_cached('aiperf.endpoints.openai', 'OpenAIEndpoint')
        >>> # Subsequent calls return cached class instantly
        >>> SameClass = _import_class_cached('aiperf.endpoints.openai', 'OpenAIEndpoint')
        >>> assert EndpointClass is SameClass
    """
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
