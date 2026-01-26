# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin container with dependency injection via Lagom.

This module provides a dependency injection container that integrates with the
AIPerf plugin system, enabling automatic constructor injection based on type hints.

Usage:
    from aiperf.plugin.container import PluginContainer

    # Create container and register dependencies
    container = PluginContainer()
    container.register(ModelEndpointInfo, model_endpoint)
    container.register(UserConfig, user_config)

    # Create plugin instances with auto-injected dependencies
    endpoint = container.create("endpoint", "chat")
    transport = container.create("transport", "aiohttp")

    # With explicit overrides
    endpoint = container.create("endpoint", "chat", extra_headers={"X-Custom": "value"})

    # Scoped containers for isolation
    with container.scope() as worker_scope:
        worker_scope.register(str, "worker-1")
        worker = worker_scope.create("service", "worker")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from lagom import Container, Singleton

from aiperf.plugin import plugins

if TYPE_CHECKING:
    from collections.abc import Callable

    from aiperf.plugin.enums import PluginCategory

T = TypeVar("T")


class PluginContainer:
    """DI container for plugin instantiation with auto-wiring.

    Wraps Lagom's Container and integrates with the plugin registry
    to enable string-based plugin lookup with automatic dependency injection.

    The container resolves dependencies by matching constructor parameter
    type hints against registered types. This follows the Protocol pattern
    used throughout AIPerf where all plugins in a category share the same
    constructor signature.

    Attributes:
        _container: The underlying Lagom container instance.

    Example:
        >>> container = PluginContainer()
        >>> container.register(ModelEndpointInfo, model_endpoint)
        >>> container.register(UserConfig, user_config)
        >>>
        >>> # Create plugins with auto-injected dependencies
        >>> endpoint = container.create("endpoint", "chat")
        >>> transport = container.create("transport", "aiohttp")
    """

    def __init__(self, parent: PluginContainer | None = None) -> None:
        """Initialize container, optionally inheriting from parent.

        Args:
            parent: Parent container for scoped resolution. Child containers
                inherit all registrations from the parent and can override them.
        """
        if parent is not None:
            # Create child container that inherits from parent
            self._container = parent._container.clone()
        else:
            self._container = Container()

    def register(
        self,
        type_: type[T],
        instance: T | None = None,
        *,
        factory: Callable[[], T] | None = None,
        singleton: bool = True,
    ) -> None:
        """Register a dependency for injection.

        Dependencies are matched against constructor parameter type hints
        when creating plugin instances. You can register either a concrete
        instance or a factory callable.

        Args:
            type_: The type to register (matched against constructor type hints).
            instance: Concrete instance to use (always singleton behavior).
            factory: Callable that creates instances.
            singleton: If True with factory, cache the result after first call.
                If False, factory is called each time the type is resolved.

        Raises:
            ValueError: If neither instance nor factory is provided.

        Example:
            >>> # Register a singleton instance
            >>> container.register(ModelEndpointInfo, model_endpoint)
            >>>
            >>> # Register a factory (new instance each time)
            >>> container.register(Connection, factory=lambda: connect(url), singleton=False)
            >>>
            >>> # Register a singleton factory (created once, cached)
            >>> container.register(ExpensiveService, factory=create_service, singleton=True)
        """
        if instance is not None:
            self._container[type_] = instance
        elif factory is not None:
            if singleton:
                self._container[type_] = Singleton(factory)
            else:
                self._container[type_] = factory
        else:
            raise ValueError("Must provide either instance or factory")

    def create(
        self,
        category: str | PluginCategory,
        name: str,
        **overrides: Any,
    ) -> Any:
        """Create a plugin instance with auto-injected dependencies.

        Gets the plugin class from the registry, then uses Lagom to
        resolve constructor dependencies from registered types. Any
        parameters not resolved from the container use their defaults
        or must be provided via overrides.

        Args:
            category: Plugin category (e.g., 'endpoint', PluginCategory.ENDPOINT).
            name: Plugin name (e.g., 'chat').
            **overrides: Explicit values that override auto-injection.
                These are passed directly to the constructor.

        Returns:
            Instantiated plugin with dependencies injected.

        Raises:
            TypeNotFoundError: If the plugin name is not found in the category.
            Exception: If required dependencies cannot be resolved.

        Example:
            >>> endpoint = container.create("endpoint", "chat")
            >>> # With override
            >>> endpoint = container.create("endpoint", "chat",
            ...     extra_headers={"X-Custom": "value"})
        """
        # Get the class from plugin registry
        cls = plugins.get_class(category, name)

        if overrides:
            # Use magic_partial for auto-wiring, then call with overrides
            return self._container.magic_partial(cls)(**overrides)
        else:
            # Pure auto-wiring
            return self._container[cls]

    def get(self, type_: type[T]) -> T:
        """Get an instance of a type with auto-wiring.

        Resolves the type from the container, creating it if necessary
        by auto-wiring its constructor dependencies.

        Args:
            type_: The type to resolve.

        Returns:
            Instance of the requested type.

        Example:
            >>> connection = container.get(Connection)
        """
        return self._container[type_]

    def has(self, type_: type) -> bool:
        """Check if a type is explicitly registered in the container.

        Note: This only checks for explicit registrations, not whether
        Lagom could auto-wire the type.

        Args:
            type_: The type to check.

        Returns:
            True if the type is explicitly registered.
        """
        # Check Lagom's internal definitions dict
        return type_ in self._container.defined_types

    def scope(self) -> PluginContainer:
        """Create a child scope that inherits from this container.

        Child scopes can register additional dependencies or override
        parent registrations without affecting the parent. Useful for
        per-request or per-worker isolation.

        Returns:
            A new PluginContainer that inherits from this one.

        Example:
            >>> root = PluginContainer()
            >>> root.register(UserConfig, user_config)
            >>>
            >>> with root.scope() as worker_scope:
            ...     worker_scope.register(str, "worker-1")
            ...     worker = worker_scope.create("service", "worker")
        """
        return PluginContainer(parent=self)

    def __enter__(self) -> PluginContainer:
        """Enter context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager."""
        pass  # No cleanup needed currently

    def __contains__(self, type_: type) -> bool:
        """Check if a type is explicitly registered (supports 'in' operator)."""
        return self.has(type_)
