# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from importlib.abc import Traversable
from importlib.metadata import entry_points
from pathlib import Path
from weakref import WeakKeyDictionary

from ruamel.yaml import YAML

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.singleton import Singleton
from aiperf.plugin.constants import (
    DEFAULT_ENTRY_POINT_GROUP,
    DEFAULT_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
)
from aiperf.plugin.types import (
    ManifestData,
    PackageMetadata,
    TypeEntry,
    TypeNotFoundError,
    TypeSpec,
)

_logger = AIPerfLogger(__name__)
_yaml = YAML(typ="safe")


def _get_builtins_path() -> Path | Traversable:
    """Get path to built-in plugins.yaml."""
    try:
        from importlib.resources import files

        return files("aiperf.plugin") / "plugins.yaml"
    except Exception as e:
        # Fallback to relative path if running from source
        fallback = Path(__file__).parent / "plugins.yaml"
        if not fallback.exists():
            raise RuntimeError(
                "Built-in plugins.yaml not found in aiperf.plugin package.\n"
                "This is a critical error - the package system cannot function without it."
            ) from e
        return fallback


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
        """See module-level load_registry() for details."""
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
                # Load entry point (should return path to plugins.yaml)
                registry_path = ep.load()

                if not isinstance(registry_path, str | Path):
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
        """See module-level get_class() for details."""
        # Check if it's a class path (contains ':')
        if ":" in name_or_class_path:
            return self._get_class_by_class_path(category, name_or_class_path)
        else:
            return self._get_class_by_name(category, name_or_class_path)

    def list_types(self, category: str) -> list[TypeEntry]:
        """See module-level list_types() for details."""
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
        """See module-level validate_all() for details."""
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
        """See module-level list_packages() for details."""
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
        """See module-level find_registered_name() for details."""
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
            case dict():
                # Full format with metadata (TypeSpec is a TypedDict, which is a dict at runtime)
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
