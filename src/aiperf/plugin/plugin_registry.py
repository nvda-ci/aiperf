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

import ast
import importlib
import importlib.util
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypedDict
from weakref import WeakKeyDictionary

from ruamel.yaml import YAML

if TYPE_CHECKING:
    from importlib.abc import Traversable

    from aiperf.plugin.enums import PluginCategory

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.singleton import Singleton

_logger = AIPerfLogger(__name__)
_yaml = YAML(typ="safe")

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


# TypeSpec uses functional form because 'class' is a Python keyword.
TypeSpec = TypedDict(
    "TypeSpec",
    {
        "class": str,  # Fully qualified class path (module:Class)
        "description": str,
        "priority": int,
    },
    total=False,
)


# ==============================================================================
# Custom Exceptions
# ==============================================================================


class PluginError(Exception):
    """Base exception for plugin system errors."""


class TypeNotFoundError(PluginError):
    """Type not found in category. Includes available types in error message."""

    def __init__(self, category: str, type_name: str, available: list[str]) -> None:
        self.category = category
        self.type_name = type_name
        self.available = available

        available_str = "\n".join(f"  • {name}" for name in sorted(available))
        super().__init__(
            f"Type '{type_name}' not found for category '{category}'.\n"
            f"Available types:\n{available_str}"
        )


# ==============================================================================
# Implementation Classes
# ==============================================================================


@dataclass(frozen=True, slots=True)
class TypeEntry:
    """Lazy-loading type entry with metadata. Call load() to import the class."""

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
        """Import and return the class (cached after first call)."""
        # Return cached class if already loaded
        if self.loaded_class is not None:
            return self.loaded_class

        # Validate and parse class path using structural pattern matching
        module_path, _, class_name = self.class_path.rpartition(":")
        if not module_path or not class_name:
            raise ValueError(
                f"Invalid class_path format: {self.class_path}\n"
                f"Expected format: 'module.path:ClassName'\n"
                f"Example: 'aiperf.endpoints.openai:OpenAIEndpoint'"
            )

        # Import and cache the class
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

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

    def validate(self, check_class: bool = False) -> tuple[bool, str | None]:
        """Validate class is loadable without importing. Returns (is_valid, error_message)."""
        # Already loaded means it's valid
        if self.loaded_class is not None:
            return True, None

        # Validate class_path format
        parts = self.class_path.split(":")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return (
                False,
                f"Invalid class_path format: {self.class_path} (expected 'module:ClassName')",
            )

        module_path, class_name = parts

        # Check if module exists without importing it
        try:
            spec = importlib.util.find_spec(module_path)
            if spec is None:
                return False, f"Module not found: {module_path}"
        except ModuleNotFoundError as e:
            return False, f"Module not found: {module_path} ({e})"
        except Exception as e:
            return False, f"Error checking module {module_path}: {e}"

        # Optionally verify class exists via AST (no code execution)
        if check_class and spec is not None and spec.origin is not None:
            try:
                source_path = Path(spec.origin)
                if source_path.suffix == ".py" and source_path.exists():
                    source = source_path.read_text(encoding="utf-8")
                    tree = ast.parse(source)

                    # Look for class definition or import/assignment
                    class_found = False
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and node.name == class_name:
                            class_found = True
                            break
                        # Also check for imports that might bring in the class
                        if isinstance(node, ast.ImportFrom) and node.names:
                            for alias in node.names:
                                if (
                                    alias.name == class_name
                                    or alias.asname == class_name
                                ):
                                    class_found = True
                                    break

                    if not class_found:
                        return False, f"Class '{class_name}' not found in {module_path}"
            except SyntaxError as e:
                return False, f"Syntax error in {module_path}: {e}"
            except Exception as e:
                # AST parsing failed, but module exists - don't fail validation
                _logger.debug(lambda err=e: f"Could not verify class via AST: {err}")

        return True, None


# ==============================================================================
# Registry Class
# ==============================================================================


class PluginRegistry(Singleton):
    """Plugin registry singleton with discovery and lazy loading."""

    def __init__(self) -> None:
        super().__init__()
        _logger.debug("Initialized plugin registry singleton")
        # Nested dict: category -> type_name -> TypeEntry
        self._types: dict[str, dict[str, TypeEntry]] = {}
        # Reverse lookup: class_path -> TypeEntry
        self._type_entries_by_class_path: dict[str, TypeEntry] = {}
        # Loaded plugin metadata: plugin_name -> metadata
        self._loaded_plugins: dict[str, PackageMetadata] = {}
        # Reverse mapping from class to registered name (for find_registered_name)
        self._class_to_name: WeakKeyDictionary[type, str] = WeakKeyDictionary()

        # Load the builtin registry manifest and discover plugins once on startup
        self.load_registry(_get_builtins_path())
        self.discover_plugins()

    def load_registry(self, registry_path: Path | str | Traversable) -> None:
        """Load registry from YAML file."""
        # Load YAML content
        yaml_content = self._read_registry_file(registry_path)

        # Parse YAML
        data: ManifestData = _yaml.load(yaml_content)

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
        self._register_types_from_manifest(package_name, data, is_builtin)

        _logger.info(
            f"Loaded registry: {package_name} with {len([k for k in data if k not in ('plugin', 'schema_version')])} categories"
        )

    def discover_plugins(
        self, entry_point_group: str = DEFAULT_ENTRY_POINT_GROUP
    ) -> None:
        """Discover and load plugin registries via setuptools entry points."""
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

    def get_class(self, category: str, name_or_class_path: str) -> type:
        """Get type class by name or fully qualified class path (lazy-loaded, cached)."""
        # Check if it's a class path (contains ':')
        if ":" in name_or_class_path:
            return self._get_class_by_class_path(category, name_or_class_path)
        else:
            return self._get_class_by_name(category, name_or_class_path)

    def list_types(self, category: str) -> list[TypeEntry]:
        """List all TypeEntry objects for a category (sorted alphabetically)."""
        if category not in self._types:
            _logger.debug(
                lambda cat=category: f"Category not found in list_types: {cat}"
            )
            return []

        impls = list(self._types[category].values())
        impls.sort(key=lambda x: x.type_name)  # Alphabetical order

        _logger.debug(lambda cat=category, i=impls: f"Listed {len(i)} types for {cat}")

        return impls

    def validate_all(
        self, check_class: bool = False
    ) -> dict[str, list[tuple[str, str]]]:
        """Validate all registered types without loading them. Returns {category: [(type, error)]}."""
        errors: dict[str, list[tuple[str, str]]] = {}

        for category, types in self._types.items():
            for type_name, entry in types.items():
                valid, error = entry.validate(check_class=check_class)
                if not valid and error:
                    if category not in errors:
                        errors[category] = []
                    errors[category].append((type_name, error))

        return errors

    def list_packages(self, builtin_only: bool = False) -> list[str]:
        """List all loaded plugin package names."""
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

    def _get_class_by_class_path(self, category: str, class_path: str) -> type:
        """Get type by class path with category validation."""
        if class_path not in self._type_entries_by_class_path:
            raise KeyError(
                f"No type with class path: {class_path}\n"
                f"Hint: Class path must be registered in a registry.yaml"
            )

        lazy_type = self._type_entries_by_class_path[class_path]

        # Verify category matches
        if lazy_type.category != category:
            raise ValueError(
                f"Category mismatch: {class_path} is registered for category "
                f"'{lazy_type.category}', not '{category}'"
            )

        return self._load_entry(lazy_type)

    def _get_class_by_name(self, category: str, type_name: str) -> type:
        """Get type by short name."""
        if category not in self._types:
            raise KeyError(
                f"Unknown category: {category}\n"
                f"Available categories: {sorted(self._types.keys())}"
            )

        if type_name not in self._types[category]:
            available = list(self._types[category].keys())
            raise TypeNotFoundError(category, type_name, available)

        lazy_type = self._types[category][type_name]
        return self._load_entry(lazy_type)

    def _load_entry(self, entry: TypeEntry) -> type:
        """Load a TypeEntry and update the reverse class-to-name mapping."""
        cls = entry.load()
        self._class_to_name[cls] = entry.type_name
        return cls

    def find_registered_name(self, category: str, cls: type) -> str | None:
        """Reverse lookup: find registered name for a class, or None if not found."""
        if category not in self._types:
            return None

        # Fast path: check reverse mapping for already-loaded classes
        if cls in self._class_to_name:
            name = self._class_to_name[cls]
            # Verify it's in the requested category
            if name in self._types[category]:
                return name

        # Slow path: search by class path for classes not loaded via registry
        target_class_path = f"{cls.__module__}:{cls.__name__}"

        for type_name, lazy_type in self._types[category].items():
            if lazy_type.class_path == target_class_path:
                return type_name

        return None

    # --------------------------------------------------------------------------
    # Private: Registry Loading
    # --------------------------------------------------------------------------

    def _read_registry_file(self, registry_path: Path | str | Traversable) -> str:
        """Read registry YAML file content."""
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
        """Validate manifest structure and schema version."""
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

        # Check for supported versions
        if schema_version in SUPPORTED_SCHEMA_VERSIONS:
            _logger.debug(lambda: f"Using schema version {schema_version}")
        else:
            _logger.warning(
                f"Unknown schema version {schema_version}, supported: {list(SUPPORTED_SCHEMA_VERSIONS)}"
            )

    def _register_types_from_manifest(
        self, package_name: str, manifest_data: ManifestData, is_builtin: bool
    ) -> None:
        """Register types from manifest with conflict resolution."""
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
        """Register a single type with conflict resolution."""
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
        """Resolve conflicts and register type."""
        existing = self._types[category_name].get(type_name)

        if existing is None:
            # No conflict - register directly
            self._types[category_name][type_name] = lazy_type
            self._type_entries_by_class_path[lazy_type.class_path] = lazy_type

            _logger.debug(
                lambda: f"Registered {category_name}:{type_name} from {lazy_type.package_name} (priority={lazy_type.priority})"
            )
            return

        # Conflict exists - resolve based on priority and type
        winner, reason = self._resolve_conflict(existing, lazy_type)

        if winner is lazy_type:
            # New type wins
            self._types[category_name][type_name] = lazy_type
            self._type_entries_by_class_path[lazy_type.class_path] = lazy_type

            _logger.info(
                f"Override registered {category_name}:{type_name}: {lazy_type.package_name} beats {existing.package_name} ({reason})"
            )
        else:
            # Existing type wins
            _logger.debug(
                lambda: f"Override rejected {category_name}:{type_name}: {existing.package_name} beats {lazy_type.package_name} ({reason})"
            )

    def _resolve_conflict(
        self,
        existing: TypeEntry,
        new: TypeEntry,
    ) -> tuple[TypeEntry, str]:
        """Resolve conflict between existing and new type. Returns (winner, reason)."""
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
        """Load package metadata from installed package."""
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
#   from aiperf.plugin import plugin_registry
#   from aiperf.plugin.enums import PluginCategory
#   EndpointClass = plugin_registry.get_class(PluginCategory.ENDPOINT, 'openai')
# ==============================================================================


def _get_builtins_path() -> Path | Traversable:
    """Get path to built-in registry.yaml."""
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
# Type stubs with get_class() overloads are in plugin_registry.pyi
# ==============================================================================


def get_class(category: PluginCategory, name_or_class_path: str) -> type:
    """Get type class by name or class path. See PluginRegistry.get_class()."""
    return _registry.get_class(category, name_or_class_path)


def list_types(category: PluginCategory) -> list[TypeEntry]:
    """List all TypeEntry objects for a category. See PluginRegistry.list_types()."""
    return _registry.list_types(category)


def validate_all(check_class: bool = False) -> dict[str, list[tuple[str, str]]]:
    """Validate all registered types without loading. See PluginRegistry.validate_all()."""
    return _registry.validate_all(check_class=check_class)


def find_registered_name(category: str, cls: type) -> str | None:
    """Reverse lookup: find registered name for a class. See PluginRegistry.find_registered_name()."""
    return _registry.find_registered_name(category, cls)


def load_registry(registry_path: str | Path) -> None:
    """Load registry from YAML file. See PluginRegistry.load_registry()."""
    _registry.load_registry(registry_path)


def list_packages(builtin_only: bool = False) -> list[str]:
    """List all loaded plugin package names. See PluginRegistry.list_packages()."""
    return _registry.list_packages(builtin_only)


def get_package_metadata(package_name: str) -> PackageMetadata:
    """Get metadata for a loaded plugin package. Raises KeyError if not found."""
    if package_name not in _registry._loaded_plugins:
        raise KeyError(f"Package '{package_name}' not found in loaded plugins")
    return _registry._loaded_plugins[package_name]


def list_categories() -> list[str]:
    """List all registered category names (sorted alphabetically)."""
    return sorted(_registry._types.keys())


def reset() -> None:
    """Reset registry to empty state (for testing)."""
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
    """Register a class programmatically (for dynamic classes or test overrides)."""
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
    """Create an ExtensibleStrEnum from registered types in a category."""
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
    """Detect the type from a URL by matching URL scheme to type metadata."""
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
