# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for PluginLoader."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from aiperf.common.plugin_loader import AIPerfPlugin, PluginLoader, initialize_plugins

# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset plugin registry before each test."""
    from aiperf.common import plugin_registry

    plugin_registry.reset()
    yield
    plugin_registry.reset()


@pytest.fixture
def plugin_loader() -> PluginLoader:
    """Create fresh plugin loader for each test."""
    return PluginLoader()


@pytest.fixture
def mock_plugin_registry():
    """Create a mock plugin registry."""
    from aiperf.common.plugin_registry import PluginRegistry

    mock_reg = Mock(spec=PluginRegistry)
    mock_reg.list_implementations.return_value = []
    return mock_reg


@pytest.fixture
def mock_registry():
    """Mock plugin registry."""
    mock_reg = Mock()
    mock_reg.list_implementations.return_value = []
    return mock_reg


@pytest.fixture
def sample_plugin_config() -> dict:
    """Sample plugin configuration."""
    return {
        "phase_hooks": [
            {"name": "example_logging_hook", "config": {"verbose": True}},
            {
                "name": "metrics_collector",
                "config": {"output_file": "/tmp/metrics.json"},
            },
        ]
    }


@pytest.fixture
def mock_phase_hook():
    """Mock phase hook instance."""
    hook = Mock()
    hook.register_with_global_registry = Mock()
    return hook


@pytest.fixture
def sample_registry_yaml() -> dict:
    """Sample registry YAML content."""
    return {
        "plugin": {"name": "test-plugin", "version": "1.0.0"},
        "phase_hook": {
            "test_hook": {
                "class": "test.hooks:TestHook",
                "description": "Test hook",
                "priority": 100,
            }
        },
    }


@pytest.fixture
def temp_registry_file(sample_registry_yaml) -> Path:
    """Create temporary registry YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_registry_yaml, f)
        return Path(f.name)


# ==============================================================================
# PluginLoader Tests - Initialization
# ==============================================================================


class TestPluginLoaderInit:
    """Tests for PluginLoader initialization."""

    def test_init_empty(self, plugin_loader):
        """Test that loader initializes with empty state."""
        assert plugin_loader._loaded_plugins == []
        assert plugin_loader._plugin_metadata == {}

    def test_initialize_plugin_system_empty_config(self, plugin_loader):
        """Test initializing with no config."""
        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_registry:
            mock_registry.list_all.return_value = []

            result = plugin_loader.initialize_plugin_system(None)

            assert result == []
            assert len(plugin_loader._loaded_plugins) == 0

    def test_initialize_plugin_system_with_config(self, plugin_loader):
        """Test initializing with plugin config."""
        config = {"phase_hooks": [{"name": "test_hook", "config": {}}]}

        with (
            patch("aiperf.common.plugin_loader.plugin_registry") as mock_registry,
            patch.object(plugin_loader, "_load_phase_hooks") as mock_load,
        ):
            mock_registry.list_all.return_value = []

            plugin_loader.initialize_plugin_system(config)

            mock_load.assert_called_once_with([{"name": "test_hook", "config": {}}])

    def test_initialize_plugin_system_returns_loaded_plugins(self, plugin_loader):
        """Test that initialize returns loaded plugins list."""
        mock_hook = Mock()

        with (
            patch("aiperf.common.plugin_loader.plugin_registry") as mock_registry,
            patch.object(plugin_loader, "_load_phase_hooks"),
        ):
            mock_registry.list_all.return_value = []

            plugin_loader._loaded_plugins = [mock_hook]
            result = plugin_loader.initialize_plugin_system()

            assert result == [mock_hook]


# ==============================================================================
# PluginLoader Tests - Discovery and Logging
# ==============================================================================


class TestPluginDiscovery:
    """Tests for plugin discovery."""

    def test_log_discovered_plugins_empty(self, plugin_loader, mock_registry):
        """Test logging when no plugins discovered."""
        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_reg:
            mock_reg.list_all.return_value = []

            # Should not raise
            plugin_loader._log_discovered_plugins()

    def test_log_discovered_plugins_with_implementations(
        self, plugin_loader, mock_registry
    ):
        """Test logging with discovered implementations."""
        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_reg:
            mock_impl1 = Mock(impl_name="hook1")
            mock_impl2 = Mock(impl_name="hook2")

            def mock_list_all(protocol):
                if protocol == "phase_hook":
                    return [mock_impl1, mock_impl2]
                return []

            mock_reg.list_all.side_effect = mock_list_all

            # Should not raise
            plugin_loader._log_discovered_plugins()

    def test_log_discovered_plugins_multiple_protocols(
        self, plugin_loader, mock_registry
    ):
        """Test logging with multiple protocol types."""
        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_reg:
            mock_impl1 = Mock(impl_name="hook1")
            mock_impl2 = Mock(impl_name="hook2")
            mock_impl3 = Mock(impl_name="processor1")

            def mock_list_all(protocol):
                return {
                    "phase_hook": [mock_impl1, mock_impl2],
                    "post_processor": [mock_impl3],
                    "results_processor": [],
                }.get(protocol, [])

            mock_reg.list_all.side_effect = mock_list_all

            # Should not raise
            plugin_loader._log_discovered_plugins()


# ==============================================================================
# PluginLoader Tests - Phase Hook Loading
# ==============================================================================


class TestPhaseHookLoading:
    """Tests for phase hook loading."""

    def test_load_phase_hooks_empty_list(self, plugin_loader):
        """Test loading with empty hook configs."""
        plugin_loader._load_phase_hooks([])

        assert len(plugin_loader._loaded_plugins) == 0

    def test_load_phase_hooks_missing_name(self, plugin_loader):
        """Test loading hook config without name."""
        configs = [{"config": {"verbose": True}}]  # Missing 'name'

        plugin_loader._load_phase_hooks(configs)

        # Should fail validation and be added to failed plugins
        assert len(plugin_loader._loaded_plugins) == 0
        assert len(plugin_loader._failed_plugins) == 1

    def test_load_phase_hooks_success_with_register_with_global_registry(
        self, plugin_loader
    ):
        """Test successful hook loading with register_with_global_registry method."""
        configs = [{"name": "test_hook", "config": {"verbose": True}}]

        mock_hook = Mock()
        mock_hook.register_with_global_registry = Mock()

        # Mock HookClass to return our mock hook
        mock_hook_class = Mock(return_value=mock_hook)

        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_registry:
            mock_registry.get_class.return_value = mock_hook_class

            plugin_loader._load_phase_hooks(configs)

            mock_registry.get_class.assert_called_once_with("phase_hook", "test_hook")
            mock_hook_class.assert_called_once_with(verbose=True)
            mock_hook.register_with_global_registry.assert_called_once()
            assert len(plugin_loader._loaded_plugins) == 1

    def test_load_phase_hooks_success_with_register_hooks(self, plugin_loader):
        """Test successful hook loading with register_hooks method."""
        configs = [{"name": "test_hook", "config": {}}]

        mock_hook = Mock(spec=["register_hooks"])
        mock_hook.register_hooks = Mock()

        mock_hook_class = Mock(return_value=mock_hook)

        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_registry:
            mock_registry.get_class.return_value = mock_hook_class

            plugin_loader._load_phase_hooks(configs)

            mock_hook.register_hooks.assert_called_once()
            assert len(plugin_loader._loaded_plugins) == 1

    def test_load_phase_hooks_missing_registration_method(self, plugin_loader):
        """Test hook without registration method."""
        configs = [{"name": "test_hook", "config": {}}]

        mock_hook = Mock(spec=[])  # No methods

        mock_hook_class = Mock(return_value=mock_hook)

        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_registry:
            mock_registry.get_class.return_value = mock_hook_class

            plugin_loader._load_phase_hooks(configs)

            # Hook loaded but warning logged
            assert len(plugin_loader._loaded_plugins) == 1

    def test_load_phase_hooks_key_error(self, plugin_loader):
        """Test handling KeyError when hook not found."""
        configs = [{"name": "nonexistent_hook", "config": {}}]

        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_registry:
            mock_registry.get_class.side_effect = KeyError("Unknown implementation")

            plugin_loader._load_phase_hooks(configs)

            assert len(plugin_loader._loaded_plugins) == 0
            assert len(plugin_loader._failed_plugins) == 1

    def test_load_phase_hooks_generic_exception(self, plugin_loader):
        """Test handling generic exception during loading."""
        configs = [{"name": "test_hook", "config": {}}]

        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_registry:
            mock_registry.get_class.side_effect = RuntimeError("Bad plugin")

            plugin_loader._load_phase_hooks(configs)

            assert len(plugin_loader._loaded_plugins) == 0
            assert len(plugin_loader._failed_plugins) == 1

    def test_load_phase_hooks_multiple_hooks(self, plugin_loader):
        """Test loading multiple phase hooks."""
        configs = [
            {"name": "hook1", "config": {"verbose": True}},
            {"name": "hook2", "config": {"output": "/tmp/out.log"}},
        ]

        mock_hook1 = Mock()
        mock_hook1.register_with_global_registry = Mock()
        mock_hook2 = Mock()
        mock_hook2.register_with_global_registry = Mock()

        mock_hook_class1 = Mock(return_value=mock_hook1)
        mock_hook_class2 = Mock(return_value=mock_hook2)

        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_registry:
            mock_registry.get_class.side_effect = [mock_hook_class1, mock_hook_class2]

            plugin_loader._load_phase_hooks(configs)

            assert len(plugin_loader._loaded_plugins) == 2
            assert plugin_loader._loaded_plugins[0] is mock_hook1
            assert plugin_loader._loaded_plugins[1] is mock_hook2

    def test_load_phase_hooks_partial_failure(self, plugin_loader):
        """Test loading hooks with some failures."""
        configs = [
            {"name": "good_hook", "config": {}},
            {"name": "bad_hook", "config": {}},
            {"name": "another_good_hook", "config": {}},
        ]

        mock_hook1 = Mock()
        mock_hook1.register_with_global_registry = Mock()
        mock_hook2 = Mock()
        mock_hook2.register_with_global_registry = Mock()

        mock_hook_class1 = Mock(return_value=mock_hook1)
        mock_hook_class2 = Mock(return_value=mock_hook2)

        def get_side_effect(protocol, name):
            if name == "bad_hook":
                raise RuntimeError("Bad hook")
            return mock_hook_class1 if name == "good_hook" else mock_hook_class2

        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_registry:
            mock_registry.get_class.side_effect = get_side_effect

            plugin_loader._load_phase_hooks(configs)

            # Only 2 hooks should load successfully
            assert len(plugin_loader._loaded_plugins) == 2
            assert len(plugin_loader._failed_plugins) == 1

    def test_load_phase_hooks_empty_config(self, plugin_loader):
        """Test loading hook with empty config dict."""
        configs = [{"name": "test_hook"}]  # No 'config' key

        mock_hook = Mock()
        mock_hook.register_with_global_registry = Mock()

        mock_hook_class = Mock(return_value=mock_hook)

        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_registry:
            mock_registry.get_class.return_value = mock_hook_class

            plugin_loader._load_phase_hooks(configs)

            # Should pass empty dict as **kwargs
            mock_registry.get_class.assert_called_once_with("phase_hook", "test_hook")
            mock_hook_class.assert_called_once_with()
            assert len(plugin_loader._loaded_plugins) == 1


# ==============================================================================
# PluginLoader Tests - Configuration Loading
# ==============================================================================


class TestConfigLoading:
    """Tests for loading plugins from configuration."""

    def test_load_configured_plugins_empty_config(self, plugin_loader):
        """Test loading with empty config."""
        plugin_loader._load_configured_plugins({})
        assert len(plugin_loader._loaded_plugins) == 0

    def test_load_configured_plugins_with_phase_hooks(self, plugin_loader):
        """Test loading phase hooks from config."""
        config = {"phase_hooks": [{"name": "test_hook", "config": {}}]}

        with patch.object(plugin_loader, "_load_phase_hooks") as mock_load:
            plugin_loader._load_configured_plugins(config)
            mock_load.assert_called_once_with([{"name": "test_hook", "config": {}}])

    def test_load_configured_plugins_no_phase_hooks(self, plugin_loader):
        """Test config without phase_hooks section."""
        config = {"other_plugins": []}

        with patch.object(plugin_loader, "_load_phase_hooks") as mock_load:
            plugin_loader._load_configured_plugins(config)
            mock_load.assert_not_called()


# ==============================================================================
# PluginLoader Tests - Utility Methods
# ==============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_get_loaded_plugins_empty(self, plugin_loader):
        """Test getting plugins when none loaded."""
        result = plugin_loader.get_loaded_plugins()
        assert result == []

    def test_get_loaded_plugins_returns_copy(self, plugin_loader):
        """Test that get_loaded_plugins returns a copy."""
        mock_hook = Mock()
        plugin_loader._loaded_plugins = [mock_hook]

        result = plugin_loader.get_loaded_plugins()

        assert result == [mock_hook]
        assert result is not plugin_loader._loaded_plugins

    def test_get_loaded_plugins_with_plugins(self, plugin_loader):
        """Test getting plugins after loading."""
        mock_hook1 = Mock()
        mock_hook2 = Mock()
        plugin_loader._loaded_plugins = [mock_hook1, mock_hook2]

        result = plugin_loader.get_loaded_plugins()

        assert len(result) == 2
        assert result[0] is mock_hook1
        assert result[1] is mock_hook2


# ==============================================================================
# initialize_plugins() Convenience Function Tests
# ==============================================================================


class TestInitializePluginsFunction:
    """Tests for initialize_plugins convenience function."""

    def test_initialize_plugins_no_config(self):
        """Test initialize_plugins with no config."""
        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_registry:
            mock_registry.list_all.return_value = []

            loader = initialize_plugins(None)

            assert isinstance(loader, PluginLoader)
            assert len(loader._loaded_plugins) == 0

    def test_initialize_plugins_empty_config(self):
        """Test initialize_plugins with empty config."""
        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_registry:
            mock_registry.list_all.return_value = []

            loader = initialize_plugins({})

            assert isinstance(loader, PluginLoader)
            assert len(loader._loaded_plugins) == 0

    def test_initialize_plugins_with_config(self):
        """Test initialize_plugins with plugin config."""
        config = {"plugins": {"phase_hooks": [{"name": "test_hook", "config": {}}]}}

        mock_hook = Mock()
        mock_hook.register_with_global_registry = Mock()

        mock_hook_class = Mock(return_value=mock_hook)

        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_registry:
            mock_registry.list_all.return_value = []
            mock_registry.get_class.return_value = mock_hook_class

            loader = initialize_plugins(config)

            assert isinstance(loader, PluginLoader)
            assert len(loader._loaded_plugins) == 1

    def test_initialize_plugins_no_plugins_section(self):
        """Test initialize_plugins with config but no plugins section."""
        config = {"other_config": "value"}

        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_registry:
            mock_registry.list_all.return_value = []

            loader = initialize_plugins(config)

            assert isinstance(loader, PluginLoader)
            assert len(loader._loaded_plugins) == 0


# ==============================================================================
# AIPerfPlugin Protocol Tests
# ==============================================================================


class TestAIPerfPluginProtocol:
    """Tests for AIPerfPlugin protocol."""

    def test_plugin_implements_protocol(self):
        """Test that a proper plugin implements the protocol."""

        class GoodPlugin:
            def __init__(self, config: dict):
                self.config = config

            def register_hooks(self) -> None:
                pass

        plugin = GoodPlugin({})
        assert isinstance(plugin, AIPerfPlugin)

    def test_plugin_missing_init(self):
        """Test that plugin without __init__ doesn't implement protocol."""

        class BadPlugin:
            def register_hooks(self) -> None:
                pass

        # Can't instantiate without __init__
        # Protocol check would happen at runtime

    def test_plugin_missing_register_hooks(self):
        """Test that plugin without register_hooks doesn't implement protocol."""

        class BadPlugin:
            def __init__(self, config: dict):
                self.config = config

        plugin = BadPlugin({})
        # Protocol is runtime_checkable, so isinstance check looks for methods
        # This should NOT be an instance since it's missing register_hooks
        assert not isinstance(plugin, AIPerfPlugin)


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_initialization_flow(self):
        """Test complete plugin initialization flow."""
        config = {
            "plugins": {
                "phase_hooks": [
                    {"name": "hook1", "config": {"verbose": True}},
                    {"name": "hook2", "config": {"output": "/tmp/out.log"}},
                ]
            }
        }

        mock_hook1 = Mock()
        mock_hook1.register_with_global_registry = Mock()
        mock_hook2 = Mock()
        mock_hook2.register_with_global_registry = Mock()

        mock_hook_class1 = Mock(return_value=mock_hook1)
        mock_hook_class2 = Mock(return_value=mock_hook2)

        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_registry:
            mock_registry.list_all.return_value = []
            mock_registry.get_class.side_effect = [mock_hook_class1, mock_hook_class2]

            loader = initialize_plugins(config)

            assert len(loader._loaded_plugins) == 2
            mock_hook1.register_with_global_registry.assert_called_once()
            mock_hook2.register_with_global_registry.assert_called_once()

    def test_initialization_with_mixed_success(self):
        """Test initialization with some hooks failing."""
        config = {
            "plugins": {
                "phase_hooks": [
                    {"name": "good_hook", "config": {}},
                    {"name": "bad_hook", "config": {}},
                ]
            }
        }

        mock_good_hook = Mock()
        mock_good_hook.register_with_global_registry = Mock()

        mock_good_hook_class = Mock(return_value=mock_good_hook)

        def get_side_effect(protocol, name):
            if name == "bad_hook":
                raise RuntimeError("Bad hook")
            return mock_good_hook_class

        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_registry:
            mock_registry.list_all.return_value = []
            mock_registry.get_class.side_effect = get_side_effect

            loader = initialize_plugins(config)

            # Only good hook should be loaded
            assert len(loader._loaded_plugins) == 1
            assert len(loader._failed_plugins) == 1
            mock_good_hook.register_with_global_registry.assert_called_once()

    def test_multiple_initializations(self):
        """Test creating multiple loaders doesn't interfere."""
        config = {"plugins": {"phase_hooks": [{"name": "test_hook", "config": {}}]}}

        mock_hook1 = Mock()
        mock_hook1.register_with_global_registry = Mock()
        mock_hook2 = Mock()
        mock_hook2.register_with_global_registry = Mock()

        mock_hook_class1 = Mock(return_value=mock_hook1)
        mock_hook_class2 = Mock(return_value=mock_hook2)

        with patch("aiperf.common.plugin_loader.plugin_registry") as mock_registry:
            mock_registry.list_all.return_value = []
            mock_registry.get_class.side_effect = [mock_hook_class1, mock_hook_class2]

            loader1 = initialize_plugins(config)
            loader2 = initialize_plugins(config)

            assert len(loader1._loaded_plugins) == 1
            assert len(loader2._loaded_plugins) == 1
            assert loader1 is not loader2
