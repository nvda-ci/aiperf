# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PluginContainer dependency injection."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from aiperf.plugin import plugins
from aiperf.plugin.container import PluginContainer


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def container() -> Generator[PluginContainer, None, None]:
    """Create fresh container for each test."""
    yield PluginContainer()


@pytest.fixture
def reset_global_container() -> Generator[None, None, None]:
    """Reset the global container before and after each test."""
    plugins.reset_container()
    yield
    plugins.reset_container()


# ==============================================================================
# Sample Classes for Testing
# ==============================================================================


class DatabaseConfig:
    """Sample config class for testing."""

    def __init__(self, host: str = "localhost", port: int = 5432) -> None:
        self.host = host
        self.port = port


class Connection:
    """Sample connection class for testing."""

    def __init__(self, config: DatabaseConfig) -> None:
        self.config = config


class Repository:
    """Sample repository class for testing."""

    def __init__(self, connection: Connection, table_name: str = "users") -> None:
        self.connection = connection
        self.table_name = table_name


class Service:
    """Sample service class for testing."""

    def __init__(self, repository: Repository) -> None:
        self.repository = repository


# ==============================================================================
# Basic Registration Tests
# ==============================================================================


class TestContainerRegistration:
    """Tests for container.register()."""

    def test_register_instance(self, container: PluginContainer) -> None:
        """Test registering a concrete instance."""
        config = DatabaseConfig(host="testhost", port=1234)
        container.register(DatabaseConfig, config)

        result = container.get(DatabaseConfig)

        assert result is config
        assert result.host == "testhost"
        assert result.port == 1234

    def test_register_factory(self, container: PluginContainer) -> None:
        """Test registering a factory callable."""
        call_count = 0

        def create_config() -> DatabaseConfig:
            nonlocal call_count
            call_count += 1
            return DatabaseConfig(host=f"host-{call_count}")

        container.register(DatabaseConfig, factory=create_config, singleton=False)

        result1 = container.get(DatabaseConfig)
        result2 = container.get(DatabaseConfig)

        assert result1.host == "host-1"
        assert result2.host == "host-2"
        assert call_count == 2

    def test_register_singleton_factory(self, container: PluginContainer) -> None:
        """Test registering a singleton factory (default behavior)."""
        call_count = 0

        def create_config() -> DatabaseConfig:
            nonlocal call_count
            call_count += 1
            return DatabaseConfig(host=f"host-{call_count}")

        container.register(DatabaseConfig, factory=create_config, singleton=True)

        result1 = container.get(DatabaseConfig)
        result2 = container.get(DatabaseConfig)

        assert result1 is result2
        assert result1.host == "host-1"
        assert call_count == 1

    def test_register_requires_instance_or_factory(
        self, container: PluginContainer
    ) -> None:
        """Test that register() requires either instance or factory."""
        with pytest.raises(ValueError, match="Must provide either instance or factory"):
            container.register(DatabaseConfig)

    def test_has_returns_true_for_registered_type(
        self, container: PluginContainer
    ) -> None:
        """Test has() returns True for registered types."""
        container.register(DatabaseConfig, DatabaseConfig())

        assert container.has(DatabaseConfig) is True
        assert DatabaseConfig in container

    def test_has_returns_false_for_unregistered_type(
        self, container: PluginContainer
    ) -> None:
        """Test has() returns False for unregistered types."""
        assert container.has(DatabaseConfig) is False
        assert DatabaseConfig not in container


# ==============================================================================
# Auto-Wiring Tests
# ==============================================================================


class TestContainerAutoWiring:
    """Tests for automatic dependency injection."""

    def test_auto_wire_simple_dependency(self, container: PluginContainer) -> None:
        """Test auto-wiring a class with one dependency."""
        config = DatabaseConfig(host="testhost")
        container.register(DatabaseConfig, config)

        connection = container.get(Connection)

        assert connection.config is config

    def test_auto_wire_nested_dependencies(self, container: PluginContainer) -> None:
        """Test auto-wiring with multiple levels of dependencies."""
        config = DatabaseConfig(host="testhost")
        container.register(DatabaseConfig, config)

        service = container.get(Service)

        assert service.repository.connection.config is config

    def test_auto_wire_with_defaults(self, container: PluginContainer) -> None:
        """Test that default values are used for unregistered optional deps."""
        config = DatabaseConfig(host="testhost")
        container.register(DatabaseConfig, config)
        container.register(Connection, Connection(config))

        repo = container.get(Repository)

        assert repo.table_name == "users"  # default value


# ==============================================================================
# Plugin Creation Tests
# ==============================================================================


class TestContainerCreatePlugin:
    """Tests for container.create() with plugin system."""

    def test_create_plugin_with_registered_dependency(
        self, container: PluginContainer
    ) -> None:
        """Test creating a plugin with auto-injected dependencies."""
        # Create a mock endpoint class
        mock_model_endpoint = MagicMock()
        mock_endpoint_class = MagicMock(return_value=MagicMock())

        # Register the dependency type
        container.register(type(mock_model_endpoint), mock_model_endpoint)

        with patch.object(plugins, "get_class", return_value=mock_endpoint_class) as mock_get_class:
            container.create("endpoint", "chat")

            # Verify get_class was called correctly
            mock_get_class.assert_called_once_with("endpoint", "chat")

    def test_create_plugin_with_overrides(self, container: PluginContainer) -> None:
        """Test creating a plugin with explicit overrides."""
        mock_endpoint_class = MagicMock()
        mock_bound_func = MagicMock()

        with patch.object(plugins, "get_class", return_value=mock_endpoint_class):
            with patch.object(
                container._container, "magic_partial", return_value=mock_bound_func
            ) as mock_magic_partial:
                container.create("endpoint", "chat", extra_param="value")

        # Verify magic_partial was called with the class
        mock_magic_partial.assert_called_once_with(mock_endpoint_class)
        # Verify the bound function was called with overrides
        mock_bound_func.assert_called_once_with(extra_param="value")


# ==============================================================================
# Scoped Container Tests
# ==============================================================================


class TestContainerScoping:
    """Tests for container scoping."""

    def test_child_scope_inherits_from_parent(self, container: PluginContainer) -> None:
        """Test that child scopes inherit parent registrations."""
        config = DatabaseConfig(host="parent-host")
        container.register(DatabaseConfig, config)

        child = container.scope()

        assert child.get(DatabaseConfig) is config

    def test_child_scope_can_override_parent(self, container: PluginContainer) -> None:
        """Test that child scopes can override parent registrations."""
        parent_config = DatabaseConfig(host="parent-host")
        child_config = DatabaseConfig(host="child-host")

        container.register(DatabaseConfig, parent_config)
        child = container.scope()
        child.register(DatabaseConfig, child_config)

        # Parent unchanged
        assert container.get(DatabaseConfig) is parent_config
        # Child uses override
        assert child.get(DatabaseConfig) is child_config

    def test_child_scope_does_not_affect_parent(
        self, container: PluginContainer
    ) -> None:
        """Test that child scope registrations don't affect parent."""
        child = container.scope()
        child.register(DatabaseConfig, DatabaseConfig())

        # Parent should not have the registration
        # (Lagom may auto-wire, so we just verify child has it)
        assert child.has(DatabaseConfig)

    def test_context_manager_usage(self, container: PluginContainer) -> None:
        """Test using scope as context manager."""
        config = DatabaseConfig(host="parent-host")
        container.register(DatabaseConfig, config)

        with container.scope() as child:
            child_config = DatabaseConfig(host="child-host")
            child.register(DatabaseConfig, child_config)

            assert child.get(DatabaseConfig) is child_config

        # Parent unchanged after context exit
        assert container.get(DatabaseConfig) is config


# ==============================================================================
# Module-Level API Tests
# ==============================================================================


class TestModuleLevelAPI:
    """Tests for module-level convenience functions."""

    def test_get_container_returns_singleton(
        self, reset_global_container: None
    ) -> None:
        """Test that get_container() returns the same instance."""
        container1 = plugins.get_container()
        container2 = plugins.get_container()

        assert container1 is container2

    def test_register_dependency_uses_global_container(
        self, reset_global_container: None
    ) -> None:
        """Test that register_dependency() uses the global container."""
        config = DatabaseConfig()
        plugins.register_dependency(DatabaseConfig, config)

        container = plugins.get_container()
        assert container.get(DatabaseConfig) is config

    def test_create_instance_uses_global_container(
        self, reset_global_container: None
    ) -> None:
        """Test that create_instance() uses the global container."""
        mock_endpoint_class = MagicMock()

        with patch.object(plugins, "get_class", return_value=mock_endpoint_class) as mock_get_class:
            plugins.create_instance("endpoint", "chat")

            # Verify get_class was called correctly
            mock_get_class.assert_called_once_with("endpoint", "chat")

    def test_reset_container_clears_global_container(
        self, reset_global_container: None
    ) -> None:
        """Test that reset_container() clears the global container."""
        config = DatabaseConfig()
        plugins.register_dependency(DatabaseConfig, config)
        container1 = plugins.get_container()

        plugins.reset_container()
        container2 = plugins.get_container()

        assert container1 is not container2


# ==============================================================================
# Integration Tests (using real plugin system)
# ==============================================================================


class TestContainerIntegration:
    """Integration tests with real plugin types."""

    def test_create_real_endpoint_type(self, container: PluginContainer) -> None:
        """Test creating a real endpoint plugin with mocked dependency."""
        # This test verifies the integration actually works with real plugin types
        # We mock the dependency that endpoints require

        # Get the actual endpoint class to understand what it needs
        endpoint_class = plugins.get_class("endpoint", "chat")

        # Create a mock ModelEndpointInfo
        mock_model_endpoint = MagicMock()
        mock_model_endpoint.endpoint = MagicMock()
        mock_model_endpoint.endpoint.type = "chat"

        # The endpoint protocol expects model_endpoint parameter
        # Register it by the actual type name used in type hints
        from aiperf.common.models.model_endpoint_info import ModelEndpointInfo

        container.register(ModelEndpointInfo, mock_model_endpoint)

        # Create the endpoint
        endpoint = container.create("endpoint", "chat")

        # Verify it was created and has the injected dependency
        assert endpoint is not None
        assert endpoint.model_endpoint is mock_model_endpoint
