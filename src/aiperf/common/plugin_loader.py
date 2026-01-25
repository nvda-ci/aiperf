# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modern plugin loader with rich console output and comprehensive validation.

This module provides a clean, type-safe interface for discovering and loading
plugins at application startup. It integrates with the PluginRegistry
for lazy loading and includes beautiful console feedback.

Architecture:
    ┌──────────────────────────────────────────┐
    │ PluginLoader                             │
    │ ────────────────                         │
    │ 1. Discover via entry points             │
    │ 2. Load registry.yaml from each package  │
    │ 3. Register with PluginRegistry         │
    │ 4. Load configured plugins               │
    │ 5. Call plugin.register_hooks()          │
    └──────────────────────────────────────────┘

Flow:
    Application startup
         ↓
    initialize_plugins(config)
         ↓
    PluginLoader.initialize_plugin_system()
         ↓
    1. Load built-in registry
    2. Discover external plugins
    3. Load configured plugins (lazy)
         ↓
    Plugins register hooks with PhaseHookRegistry, etc.

Usage:
    Basic initialization:
    >>> from aiperf.common.plugin_loader import initialize_plugins
    >>> config = load_config()
    >>> plugin_loader = initialize_plugins(config)

    With plugin configuration:
    >>> config = {
    ...     'plugins': {
    ...         'phase_hooks': [
    ...             {'name': 'datadog_reporter', 'config': {'api_key': '...'}},
    ...         ]
    ...     }
    ... }
    >>> loader = initialize_plugins(config)
    >>> loaded = loader.get_loaded_plugins()

Example:
    Loading plugins with rich console output:
    >>> loader = PluginLoader()
    >>> plugins = loader.initialize_plugin_system({
    ...     'phase_hooks': [
    ...         {'name': 'example_logging_hook', 'config': {'verbose': True}}
    ...     ]
    ... })
    ✓ Loaded phase hook: example_logging_hook

    Plugin system initialized: 1 plugin(s) loaded

Note:
    This module uses structured logging via Python's logging module.
    Rich console output is used for user-facing messages when available,
    with graceful fallback to plain text logging.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError, field_validator

from aiperf.common import plugin_registry
from aiperf.common.exceptions import AIPerfError

# Rich is available (checked in pyproject.toml)
try:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration Models
# ==============================================================================


class PluginConfig(BaseModel):
    """Plugin configuration with validation.

    This model validates plugin configuration from user config files,
    ensuring required fields are present and properly formatted.

    Attributes:
        name: Plugin implementation name (e.g., 'example_logging_hook').
              Must be registered in the plugin registry.
        config: Plugin initialization config passed to plugin __init__.
                Structure depends on the specific plugin implementation.
        enabled: Whether this plugin should be loaded. Default: True.
                 Set to False to disable a plugin without removing config.

    Examples:
        Valid configuration:
        >>> config = PluginConfig(
        ...     name='datadog_reporter',
        ...     config={'api_key': 'xxx', 'tags': ['env:prod']},
        ...     enabled=True
        ... )

        Minimal configuration:
        >>> config = PluginConfig(name='simple_logger')

        Disabled plugin:
        >>> config = PluginConfig(name='debug_hook', enabled=False)

    Raises:
        ValidationError: If name is empty or whitespace-only.
    """

    model_config = {"extra": "forbid", "frozen": True}

    name: str = Field(
        ...,
        description="Plugin implementation name registered in the registry",
        min_length=1,
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific initialization configuration",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this plugin should be loaded",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate plugin name is non-empty after stripping whitespace.

        Args:
            v: Plugin name from config

        Returns:
            Stripped plugin name

        Raises:
            ValueError: If name is empty or whitespace-only
        """
        stripped = v.strip()
        if not stripped:
            msg = "Plugin name cannot be empty or whitespace"
            raise ValueError(msg)
        return stripped


class PluginInitializationError(AIPerfError):
    """Exception raised when plugin initialization fails critically.

    This exception is raised for critical plugin initialization failures that
    should halt plugin loading. Non-critical failures are logged but don't
    raise exceptions.

    Attributes:
        plugin_name: Name of the plugin that failed to initialize
        original_error: The underlying exception that caused the failure

    Examples:
        Basic error:
        >>> raise PluginInitializationError(
        ...     "Failed to import module 'foo.bar'",
        ...     plugin_name="my_plugin"
        ... )

        With original error:
        >>> try:
        ...     import nonexistent_module
        ... except ImportError as e:
        ...     raise PluginInitializationError(
        ...         "Module import failed",
        ...         plugin_name="my_plugin",
        ...         original_error=e
        ...     )
    """

    def __init__(
        self,
        message: str,
        plugin_name: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize plugin initialization error.

        Args:
            message: Human-readable error message
            plugin_name: Name of plugin that failed (optional)
            original_error: Original exception that caused failure (optional)
        """
        self.plugin_name = plugin_name
        self.original_error = original_error

        # Build rich error message
        parts = []
        if plugin_name:
            parts.append(f"Plugin '{plugin_name}' failed to initialize")
        else:
            parts.append("Plugin initialization failed")

        parts.append(f"Reason: {message}")

        if original_error:
            parts.append(f"Caused by: {original_error!r}")

        super().__init__("\n  ".join(parts))


# ==============================================================================
# Plugin Protocol
# ==============================================================================


@runtime_checkable
class AIPerfPlugin(Protocol):
    """Protocol for AIPerf plugins.

    All plugins must implement this interface to be compatible with the
    plugin loading system. The protocol uses runtime checking via isinstance()
    to verify plugins conform to this interface.

    Architecture:
        ┌─────────────────────────────┐
        │ AIPerfPlugin                │
        │ ────────────────            │
        │ __init__(config: dict)      │
        │ register_hooks() → None     │
        └─────────────────────────────┘

    Flow:
        PluginLoader creates instance
             ↓
        Calls register_hooks()
             ↓
        Plugin registers with PhaseHookRegistry, etc.

    Methods:
        __init__: Initialize plugin with configuration dict
        register_hooks: Register callbacks with global registries

    Examples:
        Implementing a phase hook plugin:
        >>> class MyPhaseHook:
        ...     def __init__(self, verbose: bool = False):
        ...         self.verbose = verbose
        ...
        ...     def register_hooks(self) -> None:
        ...         from aiperf.timing.phase_hooks import PhaseHookRegistry
        ...         PhaseHookRegistry.register_phase_complete(self.on_phase_complete)
        ...
        ...     async def on_phase_complete(self, phase, tracker):
        ...         if self.verbose:
        ...             print(f"Phase {phase.name} complete")

        Implementing a custom plugin:
        >>> class MetricsCollector:
        ...     def __init__(self, output_file: str):
        ...         self.output_file = output_file
        ...         self.metrics = []
        ...
        ...     def register_hooks(self) -> None:
        ...         # Register with appropriate registry
        ...         ...

    Note:
        Plugins can also implement register_with_global_registry() for backward
        compatibility. The loader will try this method first before falling
        back to register_hooks().
    """

    def __init__(self, config: dict) -> None:
        """Initialize plugin with configuration.

        Args:
            config: Plugin-specific configuration dictionary
        """
        ...

    def register_hooks(self) -> None:
        """Register hooks with global registries.

        This method is called after plugin instantiation and should register
        any callbacks, hooks, or handlers with the appropriate global registries.

        For example:
        - Phase hooks: Register with PhaseHookRegistry
        - Post processors: Register with PostProcessorRegistry
        - Custom hooks: Register with appropriate registry

        Note:
            This method should NOT perform long-running operations or I/O.
            It should only register callbacks with registries.
        """
        ...


# ==============================================================================
# Plugin Loader
# ==============================================================================


class PluginLoader:
    """Loads and initializes plugins at application startup.

    This class handles plugin discovery, validation, and loading. It integrates
    with the PluginRegistry for lazy loading and provides rich console
    feedback for user-facing operations.

    Architecture:
        ┌──────────────────────────────────────────┐
        │ PluginLoader                             │
        │ ────────────────                         │
        │ 1. Discover via entry points             │
        │ 2. Load registry.yaml from each package  │
        │ 3. Register with PluginRegistry         │
        │ 4. Load configured plugins               │
        │ 5. Call plugin.register_hooks()          │
        └──────────────────────────────────────────┘

    Plugins are lazy-loaded based on configuration. Only plugins explicitly
    enabled in config are instantiated.

    Attributes:
        _loaded_plugins: List of successfully loaded plugin instances
        _plugin_metadata: Metadata for discovered plugins (name -> metadata)
        _failed_plugins: List of (name, error) tuples for failed loads

    Examples:
        Basic usage:
        >>> loader = PluginLoader()
        >>> plugins = loader.initialize_plugin_system()

        With configuration:
        >>> config = {
        ...     'phase_hooks': [
        ...         {'name': 'example_logging_hook', 'config': {'verbose': True}}
        ...     ]
        ... }
        >>> loader = PluginLoader()
        >>> plugins = loader.initialize_plugin_system(config)

        Getting loaded plugins:
        >>> loaded = loader.get_loaded_plugins()
        >>> for plugin in loaded:
        ...     print(f"Loaded: {plugin.__class__.__name__}")
    """

    def __init__(self) -> None:
        """Initialize plugin loader with empty state."""
        self._loaded_plugins: list[Any] = []
        self._plugin_metadata: dict[str, dict[str, Any]] = {}
        self._failed_plugins: list[tuple[str, Exception]] = []

    def initialize_plugin_system(
        self, plugin_config: dict[str, Any] | None = None
    ) -> list[Any]:
        """Initialize complete plugin system.

        This is the main entry point for plugin loading. It performs the
        following steps:
        1. Load built-in registry
        2. Discover external plugins
        3. Load configured plugins
        4. Display results

        Args:
            plugin_config: Plugin configurations from config file.
                Expected structure:
                ```yaml
                phase_hooks:
                  - name: hook_name
                    config: {key: value}
                    enabled: true
                ```

        Returns:
            List of successfully loaded plugin instances

        Examples:
            Load without configuration:
            >>> loader = PluginLoader()
            >>> plugins = loader.initialize_plugin_system()

            Load with configuration:
            >>> config = {
            ...     'phase_hooks': [
            ...         {'name': 'example_logging_hook', 'config': {'verbose': True}}
            ...     ]
            ... }
            >>> plugins = loader.initialize_plugin_system(config)

        Note:
            This method should be called exactly once at application startup.
            Multiple calls will reload plugins unnecessarily.
        """
        logger.info("Initializing plugin system...")

        # 1. Load built-in registry (already done via plugin_registry module initialization)
        logger.debug("Built-in registry loaded")

        # 2. Discover external plugins (already done via plugin_registry module initialization)
        logger.debug("Plugin discovery complete")

        # 3. Log what was discovered
        self._log_discovered_plugins()

        # 4. Load configured plugins (if any)
        if plugin_config:
            self._load_configured_plugins(plugin_config)

        logger.info(
            f"Plugin system initialized: {len(self._loaded_plugins)} plugin(s) loaded",
        )

        return self._loaded_plugins

    def get_loaded_plugins(self) -> list[Any]:
        """Get list of successfully loaded plugins.

        Returns a copy of the loaded plugins list to prevent external
        modification of internal state.

        Returns:
            Copy of loaded plugins list

        Examples:
            >>> loader = PluginLoader()
            >>> loader.initialize_plugin_system(...)
            >>> plugins = loader.get_loaded_plugins()
            >>> len(plugins)
            2
        """
        return self._loaded_plugins.copy()

    # ==========================================================================
    # Private Methods - Discovery and Logging
    # ==========================================================================

    def _log_discovered_plugins(self) -> None:
        """Log discovered types.

        Queries the plugin registry to determine what types were
        discovered during registry initialization and logs them appropriately.

        This method provides feedback about:
        - Total number of types discovered
        - Breakdown by category
        - Warning if no types found
        """
        # Get all categories that have types
        categories: dict[str, list[str]] = {}
        for category in ["phase_hook", "post_processor", "results_processor"]:
            lazy_types = plugin_registry.list_types(category)
            if lazy_types:
                categories[category] = [lt.type_name for lt in lazy_types]

        if categories:
            total_count = sum(len(v) for v in categories.values())
            logger.info("Discovered %d type(s)", total_count)

            # Log details at debug level
            for category, type_names in categories.items():
                logger.debug("  %s: %s", category, ", ".join(type_names))
        else:
            logger.warning("No types discovered")

    def _display_results(self) -> None:
        """Display plugin loading results with rich console output.

        Shows a beautiful summary of:
        - Successfully loaded plugins
        - Failed plugins (if any)
        - Available types

        Uses Rich library for formatted output when available, falls back to
        plain text logging otherwise.
        """
        if not RICH_AVAILABLE or console is None:
            # Fallback to plain logging
            self._display_results_plain()
            return

        # Build status message
        lines = ["[bold green]Plugin System Initialized[/bold green]"]
        lines.append(f"Loaded {len(self._loaded_plugins)} plugin(s)")

        if self._failed_plugins:
            lines.append(
                f"[yellow]Failed: {len(self._failed_plugins)} plugin(s)[/yellow]"
            )

        # Get available categories
        categories = []
        for category in ["phase_hook", "post_processor", "results_processor"]:
            lazy_types = plugin_registry.list_types(category)
            if lazy_types:
                categories.append(category)

        if categories:
            lines.append(f"Available categories: {', '.join(categories)}")

        # Display panel
        with suppress(Exception):  # Graceful degradation
            console.print(
                Panel.fit(
                    "\n".join(lines),
                    title="[bold]✓ Ready[/bold]",
                    border_style="green",
                )
            )

    def _display_results_plain(self) -> None:
        """Display results using plain text logging (fallback).

        Used when Rich is not available or when console output should be
        minimal (e.g., in scripts or tests).
        """
        logger.info(
            "Plugin system ready: %d loaded, %d failed",
            len(self._loaded_plugins),
            len(self._failed_plugins),
        )

    # ==========================================================================
    # Private Methods - Plugin Loading
    # ==========================================================================

    # Mapping from config key (plural) to registry category (singular)
    _CONFIG_TO_CATEGORY: dict[str, str] = {
        "phase_hooks": "phase_hook",
        "post_processors": "post_processor",
        "results_processors": "results_processor",
    }

    def _load_configured_plugins(self, plugin_config: dict[str, Any]) -> None:
        """Load plugins specified in configuration.

        Parses the plugin configuration and loads each enabled plugin type.
        Supports:
        - phase_hooks: Phase lifecycle hooks
        - post_processors: Post-processing plugins
        - results_processors: Results processing plugins

        Args:
            plugin_config: Plugin configurations with structure:
                ```yaml
                phase_hooks:
                  - name: hook_name
                    config: {...}
                    enabled: true
                post_processors:
                  - name: processor_name
                    config: {...}
                ```

        Examples:
            >>> plugin_config = {
            ...     'phase_hooks': [
            ...         {'name': 'example_logging_hook', 'config': {'verbose': True}},
            ...     ],
            ...     'post_processors': [
            ...         {'name': 'example_metrics_processor', 'config': {...}}
            ...     ]
            ... }
            >>> loader._load_configured_plugins(plugin_config)
        """
        for config_key, category in self._CONFIG_TO_CATEGORY.items():
            if config_key in plugin_config:
                self._load_plugins_by_category(
                    category=category,
                    plugin_configs=plugin_config[config_key],
                    config_key=config_key,
                )

    def _load_plugins_by_category(
        self,
        category: str,
        plugin_configs: list[dict[str, Any]],
        config_key: str,
    ) -> None:
        """Load plugins for a specific category with validation and error handling.

        Each plugin config is validated, loaded from the registry, instantiated,
        and registered with the global registry. Failures are logged but don't
        stop processing of remaining plugins.

        Args:
            category: Registry category name (e.g., "phase_hook", "post_processor")
            plugin_configs: List of plugin configurations. Each should contain:
                - name: Plugin implementation name (required)
                - config: Plugin initialization config (optional)
                - enabled: Whether to load this plugin (optional, default: True)
            config_key: Config key name for logging (e.g., "phase_hooks")

        Examples:
            >>> configs = [
            ...     {'name': 'example_logging_hook', 'config': {'verbose': True}},
            ... ]
            >>> loader._load_plugins_by_category("phase_hook", configs, "phase_hooks")
        """
        for plugin_config_dict in plugin_configs:
            try:
                # Validate configuration
                plugin_config = PluginConfig(**plugin_config_dict)

                # Skip disabled plugins
                if not plugin_config.enabled:
                    logger.debug(
                        "Skipping disabled %s: %s", category, plugin_config.name
                    )
                    continue

                # Load and instantiate plugin
                self._load_single_plugin(category, plugin_config)

            except ValidationError as e:
                logger.error("Invalid %s configuration: %s", config_key, e)
                self._failed_plugins.append(
                    (plugin_config_dict.get("name", "unknown"), e)
                )

            except Exception as e:
                logger.error(
                    "Unexpected error processing %s config: %r",
                    config_key,
                    e,
                    exc_info=True,
                )
                self._failed_plugins.append(
                    (plugin_config_dict.get("name", "unknown"), e)
                )

    def _load_phase_hooks(self, hook_configs: list[dict[str, Any]]) -> None:
        """Load configured phase hooks (convenience wrapper).

        Args:
            hook_configs: List of hook configurations
        """
        self._load_plugins_by_category("phase_hook", hook_configs, "phase_hooks")

    def _load_single_plugin(self, category: str, plugin_config: PluginConfig) -> None:
        """Load a single plugin with comprehensive error handling.

        This method handles the full lifecycle of loading a single plugin:
        1. Get plugin class from registry (lazy load)
        2. Instantiate with config
        3. Register with global registry (if applicable)
        4. Add to loaded plugins list

        Args:
            category: Registry category name
            plugin_config: Validated plugin configuration

        Raises:
            KeyError: If plugin not found in registry (caught by caller)
            Exception: Any error during loading (caught by caller)
        """
        name = plugin_config.name
        init_config = plugin_config.config

        try:
            # Get plugin class via registry (LAZY LOAD!)
            logger.debug(
                "Loading %s: %s with config keys: %s",
                category,
                name,
                list(init_config.keys()),
            )
            PluginClass = plugin_registry.get_class(category, name)

            # Manually instantiate the plugin
            try:
                plugin = PluginClass(**init_config)
            except TypeError as e:
                # Invalid config parameters
                msg = f"Invalid configuration parameters: {e}"
                raise PluginInitializationError(
                    msg, plugin_name=name, original_error=e
                ) from e

            # Let plugin register its callbacks (if it has registration methods)
            self._register_plugin_callbacks(plugin, name, category)

            # Add to loaded plugins
            self._loaded_plugins.append(plugin)
            logger.info("✓ Loaded %s: %s", category, name)

        except KeyError as e:
            logger.error(
                "✗ Failed to load %s '%s': not found in registry", category, name
            )
            self._failed_plugins.append((name, e))

        except PluginInitializationError:
            # Re-raise our own exceptions
            raise

        except Exception as e:
            logger.error(
                "✗ Failed to load %s '%s': %r",
                category,
                name,
                e,
                exc_info=True,
            )
            self._failed_plugins.append((name, e))

    def _register_plugin_callbacks(
        self, plugin: Any, name: str, category: str
    ) -> None:
        """Register plugin callbacks with global registry.

        Tries multiple registration methods in order of preference:
        1. register_with_global_registry() - New standard method
        2. register_hooks() - Backward compatibility method

        Args:
            plugin: Plugin instance to register
            name: Plugin name for logging
            category: Category name for logging

        Note:
            If plugin has no registration method, a warning is logged but
            the plugin is still added to loaded plugins list.
        """
        if hasattr(plugin, "register_with_global_registry"):
            plugin.register_with_global_registry()
        elif hasattr(plugin, "register_hooks"):
            plugin.register_hooks()
        else:
            logger.warning(
                "%s '%s' has no register_with_global_registry() or "
                "register_hooks() method, skipping registration",
                category,
                name,
            )

    # Backward compatibility alias
    _register_hook_callbacks = _register_plugin_callbacks


# ==============================================================================
# Public API
# ==============================================================================


def initialize_plugins(config: dict[str, Any] | None = None) -> PluginLoader:
    """Initialize plugin system (convenience function).

    This is the main entry point for plugin system initialization. Call this
    early in application startup before any plugin implementations are needed.

    Args:
        config: Application configuration containing plugin settings.
            Expected structure:
            ```yaml
            plugins:
              phase_hooks:
                - name: hook_name
                  config: {key: value}
                  enabled: true
            ```

    Returns:
        Initialized PluginLoader instance with loaded plugins

    Examples:
        Basic usage:
        >>> from aiperf.common.plugin_loader import initialize_plugins
        >>> loader = initialize_plugins()
        >>> plugins = loader.get_loaded_plugins()

        With configuration:
        >>> config = {
        ...     'plugins': {
        ...         'phase_hooks': [
        ...             {'name': 'example_logging_hook', 'config': {'verbose': True}}
        ...         ]
        ...     }
        ... }
        >>> loader = initialize_plugins(config)

        Using in application bootstrap:
        >>> def bootstrap_application():
        ...     config = load_application_config()
        ...     plugin_loader = initialize_plugins(config)
        ...     # Continue with application startup...

    Note:
        This function should be called exactly once at application startup.
        Multiple calls will create separate loader instances and reload plugins.
    """
    loader = PluginLoader()
    plugin_config = config.get("plugins", {}) if config else {}
    loader.initialize_plugin_system(plugin_config)
    return loader
