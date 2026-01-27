# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plugin registry singleton with lazy loading and priority-based conflict resolution.

Usage:
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    EndpointClass = plugins.get_class(PluginType.ENDPOINT, 'openai')
    for impl in plugins.list_types(PluginType.ENDPOINT):
        print(f"{impl.name}: {impl.description}")

Conflict resolution: higher priority wins; equal priority: external beats built-in.
"""

from __future__ import annotations

import importlib
import importlib.util
from importlib.metadata import Distribution, entry_points
from importlib.resources.abc import Traversable
from pathlib import Path
from weakref import WeakKeyDictionary

from pydantic import ValidationError
from ruamel.yaml import YAML

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.plugin.constants import (
    DEFAULT_ENTRY_POINT_GROUP,
    SUPPORTED_SCHEMA_VERSIONS,
)
from aiperf.plugin.schema import PackageInfo, PluginsManifest, PluginSpec
from aiperf.plugin.types import (
    PluginEntry,
    TypeNotFoundError,
)

_logger = AIPerfLogger(__name__)
_yaml = YAML(typ="safe")


# ==============================================================================
# Registry Class
# ==============================================================================


class _PluginRegistry:
    """Plugin registry with discovery and lazy loading."""

    def __init__(self) -> None:
        _logger.debug("Initializing plugin registry")
        # Nested dict: category -> name -> TypeEntry
        self._types: dict[str, dict[str, PluginEntry]] = {}
        # Reverse lookup: class_path -> TypeEntry
        self._type_entries_by_class_path: dict[str, PluginEntry] = {}
        # Loaded plugin metadata: plugin_name -> metadata
        self._loaded_plugins: dict[str, PackageInfo] = {}
        # Reverse mapping from class to registered name (for find_registered_name)
        self._class_to_name: WeakKeyDictionary[type, str] = WeakKeyDictionary()
        # Category metadata cache (loaded lazily from categories.yaml)
        self._category_metadata: dict[str, dict] | None = None

        # Load the builtin registry manifest and discover plugins once on startup
        self.discover_plugins()

    def reset(self) -> None:
        """Reset registry to empty state and reload built-in plugins.

        Intended for testing only. Clears all registered types and reloads
        the built-in registry manifest.
        """
        self._types.clear()
        self._type_entries_by_class_path.clear()
        self._loaded_plugins.clear()
        self._class_to_name.clear()
        self._category_metadata = None
        self.discover_plugins()
        _logger.debug("Registry reset")

    def load_registry(
        self,
        registry_path: Path | str | Traversable,
        *,
        plugin_name: str | None = None,
        dist: Distribution | None = None,
    ) -> None:
        """Load plugin types from a YAML registry manifest.

        Parses the YAML file, validates the schema, and registers all types
        with priority-based conflict resolution.

        Args:
            registry_path: Path to the registry YAML file.
            plugin_name: Optional plugin name override.
            dist: Optional distribution for metadata lookup.

        Raises:
            FileNotFoundError: If the registry file doesn't exist.
            ValueError: If the path is a directory or schema is invalid.
            RuntimeError: If the file cannot be read.
        """
        if isinstance(registry_path, str) and ":" in registry_path:
            package, _, path = registry_path.rpartition(":")
            try:
                registry_path = importlib.resources.files(package) / path
            except Exception as e:
                raise ValueError(
                    f"Invalid registry path: {registry_path}\nReason: {e!r}"
                ) from e

        # Load YAML content
        yaml_content = self._read_registry_file(registry_path)

        # Parse YAML
        raw_data = _yaml.load(yaml_content)

        if not raw_data:
            _logger.warning(f"Empty registry YAML: {registry_path}")
            return

        # Validate and parse using Pydantic model
        try:
            plugins_file = PluginsManifest.model_validate(raw_data)
        except ValidationError as e:
            raise ValueError(
                f"Invalid plugins.yaml schema at {registry_path}:\n{e}"
            ) from e

        # Check schema version
        if plugins_file.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            _logger.warning(
                f"Unknown schema version {plugins_file.schema_version}, "
                f"supported: {list(SUPPORTED_SCHEMA_VERSIONS)}"
            )

        # Extract plugin info from validated model (fallback to plugin_name or "unknown")
        package_name = (
            plugins_file.package.name
            if plugins_file.package
            else (plugin_name or "unknown")
        )

        _logger.info(
            f"Loading registry: {package_name} (schema={plugins_file.schema_version})"
        )

        # Register types from manifest (use model_extra for category data)
        self._register_types_from_manifest(package_name, plugins_file, dist=dist)

        # Count categories (fields in model_extra)
        category_count = (
            len(plugins_file.model_extra) if plugins_file.model_extra else 0
        )
        _logger.info(
            f"Loaded registry: {package_name} with {category_count} categories"
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
                # Skip already-loaded plugins (e.g., builtin aiperf loaded in __init__)
                if ep.name in self._loaded_plugins:
                    _logger.debug(
                        lambda name=ep.name: f"Skipping already-loaded plugin: {name}"
                    )
                    continue

                # Load entry point (should return path to plugins.yaml)
                module_name, _, filename = ep.value.rpartition(":")
                spec = importlib.util.find_spec(module_name)
                if not spec or not spec.submodule_search_locations:
                    _logger.warning(
                        f"Could not locate module for plugin {ep.name}: {module_name}"
                    )
                    continue
                registry_path = Path(spec.submodule_search_locations[0]) / filename

                _logger.info(f"Loading plugin: {ep.name}")

                # Load plugin registry (pass dist for metadata lookup)
                self.load_registry(registry_path, plugin_name=ep.name, dist=ep.dist)

            except Exception as e:
                # Graceful failure for bad plugins
                _logger.exception(f"Failed to load plugin {ep.name}: {e!r}")

        _logger.info(f"Plugin discovery complete: {len(plugin_eps)} plugins found")

    def get_class(self, category: str, name_or_class_path: str) -> type:
        """Get type class by name or fully qualified class path.

        Args:
            category: Plugin category (e.g., PluginType.ENDPOINT).
            name_or_class_path: Either a short type name (e.g., 'chat') or
                a fully qualified class path (e.g., 'aiperf.endpoints:ChatEndpoint').

        Returns:
            The plugin class (lazy-loaded, cached after first access).

        Raises:
            TypeNotFoundError: If the type name is not found in the category.
            KeyError: If the category or class path is not registered.
            ValueError: If using class path and category doesn't match.
        """
        # Handle enum values by extracting their string value
        if hasattr(name_or_class_path, "value"):
            name_or_class_path = name_or_class_path.value

        # Check if it's a class path (contains ':')
        if ":" in name_or_class_path:
            return self._get_class_by_class_path(category, name_or_class_path)
        else:
            return self._get_class_by_name(category, name_or_class_path)

    def list_types(self, category: str) -> list[PluginEntry]:
        """List all PluginEntry objects for a category (sorted alphabetically).

        Args:
            category: Plugin category to list types for.

        Returns:
            List of PluginEntry objects with metadata (name, description, priority, etc.).
            Returns empty list if category doesn't exist.
        """
        if category not in self._types:
            _logger.debug(
                lambda cat=category: f"Category not found in list_types: {cat}"
            )
            return []

        impls = list(self._types[category].values())
        impls.sort(key=lambda x: x.name)  # Alphabetical order

        _logger.debug(lambda cat=category, i=impls: f"Listed {len(i)} types for {cat}")

        return impls

    def validate_all(
        self, check_class: bool = False
    ) -> dict[str, list[tuple[str, str]]]:
        """Validate all registered types without loading them.

        Checks that modules are importable (and optionally that classes exist)
        without actually executing any import statements.

        Args:
            check_class: If True, also verify class exists via AST parsing.

        Returns:
            Dict mapping category names to lists of (name, error_message) tuples.
            Empty dict means all types are valid.
        """
        errors: dict[str, list[tuple[str, str]]] = {}

        for category, types in self._types.items():
            for name, entry in types.items():
                valid, error = entry.validate(check_class=check_class)
                if not valid and error:
                    if category not in errors:
                        errors[category] = []
                    errors[category].append((name, error))

        return errors

    def list_packages(self, builtin_only: bool = False) -> list[str]:
        """List all loaded plugin package names.

        Args:
            builtin_only: If True, only return built-in packages (aiperf core).

        Returns:
            List of package names that have been loaded into the registry.
        """
        if builtin_only:
            return [
                name for name, meta in self._loaded_plugins.items() if meta.is_builtin
            ]
        return list(self._loaded_plugins.keys())

    def get_categories(self) -> list[str]:
        """Return sorted list of all registered category names."""
        return sorted(self._types.keys())

    def has_category(self, category: str) -> bool:
        """Check if a category exists."""
        return category in self._types

    def ensure_category(self, category: str) -> None:
        """Create category if it doesn't exist."""
        if category not in self._types:
            self._types[category] = {}

    def get_package_metadata(self, package_name: str) -> PackageInfo:
        """Get metadata for a loaded plugin package.

        Args:
            package_name: Name of the loaded plugin package.

        Returns:
            PackageInfo with version, description, etc.

        Raises:
            KeyError: If package not found in loaded plugins.
        """
        if package_name not in self._loaded_plugins:
            raise KeyError(f"Package '{package_name}' not found in loaded plugins")
        return self._loaded_plugins[package_name]

    def get_category_metadata(self, category: str) -> dict | None:
        """Get metadata for a plugin category from categories.yaml.

        Args:
            category: Category name to get metadata for.

        Returns:
            Category metadata dict or None if not found.
        """
        if self._category_metadata is None:
            self._load_category_metadata()
        return self._category_metadata.get(category)

    def register_type(self, entry: PluginEntry) -> None:
        """Register a type entry with conflict resolution.

        Args:
            entry: TypeEntry to register. Must have category and name set.
        """
        self.ensure_category(entry.category)
        self._resolve_conflict_and_register(entry.category, entry.name, entry)

    def register(
        self,
        category: str,
        name: str,
        cls: type,
        *,
        priority: int = 0,
    ) -> None:
        """Register a class programmatically (for dynamic classes or test overrides).

        Useful for registering classes created at runtime or overriding built-in
        types in tests. Uses the same priority-based conflict resolution as YAML.

        Args:
            category: Plugin category to register under.
            name: Short name for the type (can be an enum value).
            cls: The class to register.
            priority: Conflict resolution priority (higher wins). Default: 0.
        """
        # Convert enum to string if needed
        name = name.value if hasattr(name, "value") else str(name)

        # Create a PluginEntry with the pre-loaded class
        entry = PluginEntry(
            category=category,
            name=name,
            package=cls.__module__,
            class_path=f"{cls.__module__}:{cls.__name__}",
            priority=priority,
            description=cls.__doc__ or "",
            metadata={},
            loaded_class=cls,
        )

        # Register with conflict resolution
        self.register_type(entry)

        _logger.debug(
            lambda: f"Registered dynamic type {category}:{name} -> {cls.__name__} (priority={priority})"
        )

    def list_categories(self, *, include_internal: bool = True) -> list[str]:
        """List all registered category names (sorted alphabetically).

        Args:
            include_internal: If False, exclude internal categories (default: True).

        Returns:
            Sorted list of category names (e.g., ['endpoint', 'transport', ...]).
        """
        categories = self.get_categories()
        if not include_internal:
            categories = [c for c in categories if not self.is_internal_category(c)]
        return categories

    def is_internal_category(self, category: str) -> bool:
        """Check if a category is internal (not user-facing).

        Args:
            category: Category name to check.

        Returns:
            True if the category is marked as internal, False otherwise.
        """
        meta = self.get_category_metadata(category)
        if meta is None:
            return False
        return meta.get("internal", False)

    def create_enum(self, category: str, enum_name: str) -> type:
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
        """
        import sys

        from aiperf.common.enums import create_enum as _create_enum

        types = self.list_types(category)
        if not types:
            raise KeyError(
                f"No types registered for category '{category}'. "
                f"Available categories: {self.get_categories()}"
            )

        # Create members dict: UPPER_SNAKE_CASE name -> string value
        members = {impl.name.replace("-", "_").upper(): impl.name for impl in types}

        # Get the caller's module so pickle can find the enum
        frame = sys._getframe(2)  # 2 because we're called via wrapper
        module = frame.f_globals.get("__name__", __name__)

        return _create_enum(enum_name, members, module=module)

    def detect_type_from_url(self, category: str, url: str) -> str:
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
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        # urlparse mishandles URLs without schemes (e.g., 'localhost:8765')
        # by treating the host as the scheme. Detect this by checking if netloc is empty.
        if parsed.scheme and not parsed.netloc:
            # Re-parse with http:// prefix
            parsed = urlparse(f"http://{url}")
        scheme = parsed.scheme.lower() if parsed.scheme else "http"

        for impl in self.list_types(category):
            try:
                cls = impl.load()
                if hasattr(cls, "metadata"):
                    metadata = cls.metadata()
                    if (
                        hasattr(metadata, "url_schemes")
                        and scheme in metadata.url_schemes
                    ):
                        return impl.name
            except (ImportError, AttributeError) as e:
                _logger.debug(
                    lambda n=impl.name,
                    err=e: f"Skipping {n} during URL detection: {err}"
                )
                continue

        raise ValueError(
            f"No {category} type found for URL scheme '{scheme}' in URL: {url}"
        )

    def _load_category_metadata(self) -> None:
        """Load category metadata from categories.yaml (lazy, cached)."""
        try:
            categories_path = (
                importlib.resources.files("aiperf.plugin") / "categories.yaml"
            )
            content = categories_path.read_text(encoding="utf-8")
        except Exception:
            # Fallback to relative path
            fallback = Path(__file__).parent / "categories.yaml"
            if not fallback.exists():
                _logger.warning("categories.yaml not found")
                self._category_metadata = {}
                return
            content = fallback.read_text(encoding="utf-8")

        data = _yaml.load(content) or {}

        # Filter out non-category keys
        self._category_metadata = {
            k: v
            for k, v in data.items()
            if k not in ("schema_version",) and isinstance(v, dict)
        }

    # --------------------------------------------------------------------------
    # Private: Class Path Operations
    # --------------------------------------------------------------------------

    def _get_class_by_class_path(self, category: str, class_path: str) -> type:
        """Get type by class path with category validation."""
        if class_path not in self._type_entries_by_class_path:
            raise KeyError(
                f"No type with class path: {class_path}\n"
                f"Hint: Class path must be registered in a plugins.yaml"
            )

        lazy_type = self._type_entries_by_class_path[class_path]

        # Verify category matches
        if lazy_type.category != category:
            raise ValueError(
                f"Category mismatch: {class_path} is registered for category "
                f"'{lazy_type.category}', not '{category}'"
            )

        return self._load_entry(lazy_type)

    def _get_class_by_name(self, category: str, name: str) -> type:
        """Get type by short name."""
        if category not in self._types:
            raise KeyError(
                f"Unknown category: {category}\n"
                f"Available categories: {sorted(self._types.keys())}"
            )

        if name not in self._types[category]:
            available = list(self._types[category].keys())
            raise TypeNotFoundError(category, name, available)

        lazy_type = self._types[category][name]
        return self._load_entry(lazy_type)

    def _load_entry(self, entry: PluginEntry) -> type:
        """Load a TypeEntry and update the reverse class-to-name mapping."""
        cls = entry.load()
        self._class_to_name[cls] = entry.name
        return cls

    def find_registered_name(self, category: str, cls: type) -> str | None:
        """Reverse lookup: find the registered name for a class.

        Searches by class identity first (for loaded classes), then by class path
        (for classes not loaded via registry).

        Args:
            category: Plugin category to search in.
            cls: The class to find the registered name for.

        Returns:
            The registered type name, or None if not found.
        """
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

        for _, entry in self._types[category].items():
            if entry.class_path == target_class_path:
                return entry.name

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
                        f"Please ensure the plugins.yaml file exists at this location.\n"
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
                f"Built-in plugins.yaml not found at {registry_path}.\n"
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

    def _register_types_from_manifest(
        self,
        package: str,
        plugins_file: PluginsManifest,
        *,
        dist: Distribution | None = None,
    ) -> None:
        """Register types from manifest with conflict resolution."""
        # Load package metadata from distribution or installed package
        package_metadata = self._load_package_metadata(package, dist=dist)
        self._loaded_plugins[package] = package_metadata

        # Process each category from model_extra (where Pydantic stores extra fields)
        categories = plugins_file.model_extra or {}
        for category_name, types_dict in categories.items():
            if not isinstance(types_dict, dict):
                _logger.warning(
                    f"Invalid category section type for {category_name}: {type(types_dict).__name__}"
                )
                continue

            # Ensure category exists in registry
            if category_name not in self._types:
                self._types[category_name] = {}

            # Register each type
            for name, type_spec_data in types_dict.items():
                # Convert raw dict to TypeSpec (model_extra stores raw dicts)
                if isinstance(type_spec_data, dict):
                    try:
                        type_spec = PluginSpec.model_validate(type_spec_data)
                    except ValidationError as e:
                        _logger.warning(
                            f"Invalid type spec for {category_name}:{name}: {e}"
                        )
                        continue
                else:
                    _logger.warning(
                        f"Invalid type spec format for {category_name}:{name}: "
                        f"expected dict, got {type(type_spec_data).__name__}"
                    )
                    continue

                self._register_type(
                    category_name=category_name,
                    name=name,
                    type_spec=type_spec,
                    package=package,
                )

    def _register_type(
        self,
        category_name: str,
        name: str,
        type_spec: PluginSpec,
        package: str,
    ) -> None:
        """Register a single type with conflict resolution."""
        # Validate required class path
        if not type_spec.class_:
            raise ValueError(f"Missing 'class' field for {category_name}:{name}")

        entry = PluginEntry.from_type_spec(type_spec, package, category_name, name)

        # Handle conflicts with smart resolution
        self._resolve_conflict_and_register(category_name, name, entry)

    def _resolve_conflict_and_register(
        self,
        category_name: str,
        name: str,
        entry: PluginEntry,
    ) -> None:
        """Resolve conflicts and register type."""
        existing = self._types[category_name].get(name)

        if existing is None:
            # No conflict - register directly
            self._types[category_name][name] = entry
            self._type_entries_by_class_path[entry.class_path] = entry

            _logger.debug(
                lambda cat=category_name,
                n=name,
                e=entry: f"Registered {cat}:{n} from {e.package} (priority={e.priority})"
            )
            return

        # Conflict exists - resolve based on priority and type
        winner, reason = self._resolve_conflict(existing, entry)

        # Always register by class_path so ALL plugins remain accessible via fully-qualified path
        self._type_entries_by_class_path[entry.class_path] = entry

        if winner is entry:
            # New type wins - update the name-based lookup
            self._types[category_name][name] = entry

            _logger.info(
                f"Override registered {category_name}:{name}: {entry.package} beats {existing.package} ({reason})"
            )
        else:
            # Existing type wins - name lookup unchanged, but class_path still accessible
            _logger.debug(
                lambda cat=category_name,
                n=name,
                ex=existing,
                e=entry,
                r=reason: f"Override rejected {cat}:{n}: {ex.package} beats {e.package} ({r})"
            )

    def _resolve_conflict(
        self,
        existing: PluginEntry,
        new: PluginEntry,
    ) -> tuple[PluginEntry, str]:
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
            f"Plugin conflict for {new.category}:{new.name}: {existing.package} vs {new.package} (priority={new.priority})"
        )

        return existing, "first registered wins (both same type)"

    def _load_package_metadata(
        self, package: str, *, dist: Distribution | None = None
    ) -> PackageInfo:
        """Load package metadata from distribution or installed package.

        If dist is provided, uses it directly. Otherwise falls back to looking up
        the package by name.
        """
        # Use distribution directly if provided (from entry point)
        if dist is not None:
            pkg_metadata = dist.metadata
        else:
            # Fallback: look up by package name
            try:
                import importlib.metadata

                pkg_metadata = importlib.metadata.metadata(package)
            except importlib.metadata.PackageNotFoundError:
                _logger.warning(f"Failed to load package metadata for {package}")
                return PackageInfo(name=package)

        return PackageInfo(
            name=package,
            version=pkg_metadata.get("Version", "unknown"),
            description=pkg_metadata.get("Summary", ""),
            author=pkg_metadata.get("Author", ""),
            license=pkg_metadata.get("License", ""),
            homepage=pkg_metadata.get("Home-page", ""),
        )


# ==============================================================================
# Module-Level Singleton
# ==============================================================================
# This pattern follows the random_generator module design.
# Usage:
#   from aiperf.plugin import plugins
#   from aiperf.plugin.enums import PluginType
#   EndpointClass = plugins.get_class(PluginType.ENDPOINT, 'openai')
#   endpoint = EndpointClass(...)
# ==============================================================================

# Create singleton instance at module load
_registry = _PluginRegistry()


# ==============================================================================
# Public API: Module-Level Functions
# ==============================================================================

get_class = _registry.get_class
list_types = _registry.list_types
validate_all = _registry.validate_all
find_registered_name = _registry.find_registered_name
load_registry = _registry.load_registry
list_packages = _registry.list_packages
get_package_metadata = _registry.get_package_metadata
list_categories = _registry.list_categories
get_category_metadata = _registry.get_category_metadata
is_internal_category = _registry.is_internal_category
create_enum = _registry.create_enum
detect_type_from_url = _registry.detect_type_from_url
reset = _registry.reset
register = _registry.register
register_type = _registry.register_type
