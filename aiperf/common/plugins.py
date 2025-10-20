# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
from importlib.metadata import PackageNotFoundError, entry_points, metadata
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from aiperf.common.exceptions import PluginNotFoundError


class AIPerfPluginMetadata(BaseModel):
    """Metadata for a plugin.

    This is used to identify the plugin in the plugin factory.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(..., description="The name of the plugin.", min_length=1)
    version: str = Field(..., description="The version of the plugin.", min_length=1)
    description: str | None = Field(
        default=None, description="The description of the plugin.", min_length=1
    )
    author: str | None = Field(
        default=None, description="The author of the plugin.", min_length=1
    )
    author_email: str | None = Field(
        default=None, description="The email of the author of the plugin.", min_length=1
    )
    url: str | None = Field(
        default=None, description="The URL of the plugin.", min_length=1
    )


class AIPerfPluginMapping(BaseModel):
    """Mapping information for a plugin.

    Used to store the plugin type, name, class type, entry point, and metadata for the PluginManager.
    """

    plugin_type: type[Protocol] = Field(
        ...,
        description="The type of the plugin. This is the plugin type as defined in the pyproject.toml file.",
    )
    name: str = Field(
        ...,
        description="The name of the plugin. This is the name of the plugin as defined in the pyproject.toml file.",
        min_length=1,
    )
    package_name: str = Field(
        ...,
        description="The name of the package that provides the plugin. This is the name of the package as defined in the pyproject.toml file.",
        min_length=1,
    )
    built_in: bool = Field(
        default=False,
        description="Whether the plugin is a built-in plugin. This is used to indicate that the plugin is a built-in plugin that is included in the AIPerf package.",
    )
    class_type: type[Any] | None = Field(
        default=None,
        description="The class type of the plugin. This is lazy loaded when needed.",
    )
    entry_point: Any = Field(
        ...,
        description="The entry point of the plugin. This is the importlib.metadata.EntryPoint object.",
    )
    metadata: AIPerfPluginMetadata = Field(
        ...,
        description="The metadata of the plugin. This is the metadata of the plugin as defined in the pyproject.toml file.",
    )


class AIPerfPluginManager:
    """Factory for managing plugin mappings and lazy loading plugin classes."""

    _instance_lock: threading.Lock = threading.Lock()
    _logger: logging.Logger = logging.getLogger(__name__)

    def __new__(cls, *args, **kwargs) -> Self:
        """Create a new plugin manager."""
        if not hasattr(cls, "_instance"):
            with cls._instance_lock:
                if not hasattr(cls, "_instance"):
                    cls._instance = super().__new__(cls, *args, **kwargs)
                    cls._instance._init_singleton()
        return cls._instance

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the plugin manager."""
        super().__init__(*args, **kwargs)

    def _init_singleton(self) -> None:
        """Initialize the plugin manager singleton."""
        self._logger.info("Initializing plugin factory singleton.")
        self._plugin_mappings: dict[type[Protocol], dict[str, AIPerfPluginMapping]] = {}
        self._register_all()

    def _extract_package_metadata(self, ep) -> AIPerfPluginMetadata:
        """Extract metadata from the package that provides the entry point."""
        package_name = ep.dist.name if ep.dist else None

        if not package_name:
            self._logger.warning(
                f"Could not determine package for entry point {ep.name}"
            )
            return AIPerfPluginMetadata(name=ep.name, version="unknown")

        try:
            pkg_meta = metadata(package_name)
            author_field = pkg_meta.get("Author-Email", "")
            author = None
            author_email = None

            if author_field and "<" in author_field and ">" in author_field:
                author = author_field.split("<")[0].strip()
                author_email = author_field.split("<")[1].split(">")[0].strip()

            return AIPerfPluginMetadata(
                name=ep.name,
                version=pkg_meta["Version"],
                description=pkg_meta.get("Summary"),
                author=author,
                author_email=author_email,
                url=pkg_meta.get("Home-Page"),
            )
        except PackageNotFoundError:
            self._logger.warning(f"Package metadata not found for {package_name}")
            return AIPerfPluginMetadata(name=ep.name, version="unknown")

    def _register_all(self) -> None:
        """Register all plugins without loading them (lazy loading)."""
        self._logger.debug("Registering all plugins.")

        # TODO: Add more plugin types here as they are added.
        for plugin_type in self.PLUGIN_TYPES:
            # Initialize the plugin_type key if it doesn't exist
            if plugin_type not in self._plugin_mappings:
                self._plugin_mappings[plugin_type] = {}

            registered = 0
            self._logger.debug(f"Registering plugins for {plugin_type}.")

            all_eps = list(entry_points(group=f"aiperf.plugins.{plugin_type.__name__}"))

            # Sort so built-in 'aiperf' package comes first (lowest precedence)
            # Third-party plugins come last and can override built-ins
            sorted_eps = sorted(
                all_eps, key=lambda ep: (ep.dist.name != "aiperf" if ep.dist else True)
            )

            for ep in sorted_eps:
                pkg_name = ep.dist.name if ep.dist else "unknown"

                if ep.name in self._plugin_mappings[plugin_type]:
                    existing_pkg = self._plugin_mappings[plugin_type][
                        ep.name
                    ].entry_point.dist.name
                    self._logger.warning(
                        f"Plugin '{ep.name}' from '{pkg_name}' is overriding "
                        f"existing plugin from '{existing_pkg}'. "
                        f"Using '{pkg_name}' (plugins can override built-ins)."
                    )

                self._logger.debug(
                    f"Discovering plugin {ep.name} for {plugin_type.__name__} "
                    f"from package '{pkg_name}' (lazy load - not importing yet)."
                )

                pkg_metadata = self._extract_package_metadata(ep)
                self._logger.debug(f"Plugin metadata: {pkg_metadata}.")

                self._plugin_mappings[plugin_type][ep.name] = AIPerfPluginMapping(
                    plugin_type=plugin_type,
                    name=ep.name,
                    package_name=pkg_name,
                    built_in=pkg_name == "aiperf",
                    class_type=None,
                    entry_point=ep,
                    metadata=pkg_metadata,
                )
                registered += 1
            self._logger.debug(
                f"Registered {registered} plugins for {plugin_type.__name__}."
            )

    def register_lazy(
        self,
        plugin_type: type[Protocol],
        name: str,
        module_path: str,
        class_name: str,
        built_in: bool = False,
    ) -> None:
        """Register a plugin lazily."""
        package_name = module_path.split(".", 1)[0]
        id = name.replace("-", "_").upper()
        self._plugin_mappings[plugin_type][id] = AIPerfPluginMapping(
            plugin_type=plugin_type,
            name=id,
            package_name=package_name,
            built_in=built_in,
            class_type=None,
            entry_point=None,
            metadata=AIPerfPluginMetadata(
                name=id,
                version="unknown",
                description="Unknown",
                author=None,
                author_email=None,
                url=None,
            ),
        )

    def list_plugin_names(self, plugin_type: type[Protocol]) -> list[str]:
        """List all plugins of the given plugin type."""
        return list(self._plugin_mappings[plugin_type].keys())

    def list_plugin_mappings(
        self, plugin_type: type[Protocol]
    ) -> dict[str, AIPerfPluginMapping]:
        """List all plugin mappings of the given plugin type."""
        return self._plugin_mappings[plugin_type]

    def get_metadata(
        self, plugin_type: type[Protocol], name: str
    ) -> AIPerfPluginMetadata:
        """Get the metadata for the given plugin type and name.

        Raises:
            PluginNotFoundError: If the plugin metadata is not found.
        """
        plugin = self._plugin_mappings[plugin_type].get(name)
        if not plugin:
            raise PluginNotFoundError(
                f"Plugin metadata for '{name}' not found for plugin type '{plugin_type}'"
            )
        return plugin.metadata

    def get_plugin_class(self, plugin_type: type[Protocol], name: str) -> type[Any]:
        """Get the plugin class for the given plugin type and name (lazy loads if needed).

        Raises:
            PluginNotFoundError: If the plugin class is not found.
            TypeError: If the plugin class is not a class.
        """
        plugin = self._plugin_mappings[plugin_type].get(name)
        if not plugin:
            raise PluginNotFoundError(
                f"Plugin class for '{name}' not found for plugin type '{plugin_type}'"
            )

        if plugin.class_type:
            self._logger.debug(
                f"Plugin class '{name}' already loaded for plugin type '{plugin_type}'."
            )
            return plugin.class_type

        self._logger.debug(
            f"Lazy loading plugin class '{name}' for plugin type '{plugin_type}'."
        )
        plugin.class_type = plugin.entry_point.load()
        plugin.class_type.tag = plugin.name
        plugin.class_type.service_type = plugin.name

        if not isinstance(plugin.class_type, type):
            raise TypeError(
                f"Plugin entry point '{name}' must be a class, got {type(plugin.class_type)}"
            )

        self._logger.debug(
            f"Plugin class '{name}' loaded for plugin type '{plugin_type}': {plugin.class_type!r}."
        )
        return plugin.class_type
