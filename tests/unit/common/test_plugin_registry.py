# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for simplified PluginRegistry."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from aiperf.common import plugin_registry
from aiperf.common.plugin_registry import (
    PluginRegistry,
    TypeEntry,
    TypeNotFoundError,
    _import_class_cached,
)

# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture(autouse=True)
def clear_import_cache():
    """Clear the import cache before each test to ensure mock patches work."""
    _import_class_cached.cache_clear()
    yield
    _import_class_cached.cache_clear()


@pytest.fixture
def sample_yaml_content() -> dict:
    """Sample YAML content for testing (simplified format)."""
    return {
        "plugin": {"name": "test-plugin", "version": "1.0.0", "builtin": False},
        "endpoint": {
            "test_endpoint": {
                "class": "test.module:TestEndpoint",
                "description": "Test endpoint",
            },
            "low_priority_endpoint": {
                "class": "test.module:LowPriorityEndpoint",
                "description": "Low priority endpoint",
            },
        },
        "post_processor": {
            "test_processor": {
                "class": "test.module:TestProcessor",
                "description": "Test processor",
            },
            "optional_processor": {
                "class": "test.module:OptionalProcessor",
                "description": "Optional processor",
            },
        },
    }


@pytest.fixture
def temp_registry_file(sample_yaml_content) -> Path:
    """Create temporary registry YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_yaml_content, f)
        return Path(f.name)


@pytest.fixture
def registry() -> Generator[PluginRegistry, None, None]:
    """Create fresh registry for each test.

    Creates a new PluginRegistry and resets its state to empty,
    giving tests a clean slate without builtins preloaded.
    Restores the registry to normal state after the test.
    """
    reg = PluginRegistry()
    # Reset to empty state for clean test isolation
    reg._types = {}
    reg._loaded_plugins = {}
    reg._class_path_to_type = {}
    yield reg
    # Restore registry to normal state with builtins loaded
    plugin_registry.reset()


@pytest.fixture
def mock_class():
    """Mock class for testing lazy loading."""
    return type("MockClass", (), {"__init__": lambda self, **kwargs: None})


# ==============================================================================
# TypeEntry Tests
# ==============================================================================


class TestTypeEntry:
    """Tests for TypeEntry class."""

    def test_init_with_all_fields(self):
        """Test that TypeEntry stores all fields correctly."""
        lazy = TypeEntry(
            category="test_category",
            type_name="test_type",
            package_name="test_package",
            class_path="test:TestClass",
            priority=75,
            description="Test description",
            metadata={"version": "1.0.0"},
            is_builtin=False,
        )

        assert lazy.category == "test_category"
        assert lazy.type_name == "test_type"
        assert lazy.package_name == "test_package"
        assert lazy.class_path == "test:TestClass"
        assert lazy.priority == 75
        assert lazy.description == "Test description"
        assert lazy.metadata == {"version": "1.0.0"}
        assert lazy.is_builtin is False

    def test_init_uses_defaults(self):
        """Test that defaults are used for optional fields."""
        lazy = TypeEntry(
            category="test_category",
            type_name="test_type",
            package_name="test_package",
            class_path="test:TestClass",
        )

        assert lazy.priority == 0  # Default is now 0
        assert lazy.description == ""  # Default
        assert lazy.metadata == {}  # Default
        assert lazy.is_builtin is False  # Default

    def test_load_caches_class(self, mock_class):
        """Test that load() caches the class after first load."""
        lazy = TypeEntry(
            category="test_category",
            type_name="test_type",
            package_name="test_package",
            class_path="test.module:TestClass",
            metadata={},
        )

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.TestClass = mock_class
            mock_import.return_value = mock_module

            # First load
            cls1 = lazy.load()
            assert cls1 is mock_class
            assert lazy.loaded_class is mock_class
            assert mock_import.call_count == 1

            # Second load (should use cache)
            cls2 = lazy.load()
            assert cls2 is mock_class
            assert mock_import.call_count == 1  # Not called again

    def test_load_invalid_class_path(self):
        """Test that load() raises ValueError for invalid class_path."""
        lazy = TypeEntry(
            category="test_category",
            type_name="test_type",
            package_name="test_package",
            class_path="invalid_path_without_colon",
            metadata={},
        )

        with pytest.raises(ValueError, match="Invalid class_path format"):
            lazy.load()

    def test_load_module_not_found(self):
        """Test that load() raises ImportError when module not found."""
        lazy = TypeEntry(
            category="test_category",
            type_name="test_type",
            package_name="test_package",
            class_path="nonexistent.module:TestClass",
            metadata={},
        )

        with pytest.raises(ImportError, match="Failed to import module"):
            lazy.load()

    def test_load_class_not_found(self):
        """Test that load() raises AttributeError when class not found."""
        lazy = TypeEntry(
            category="test_category",
            type_name="test_type",
            package_name="test_package",
            class_path="test.module:NonexistentClass",
        )

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock(spec=[])  # No attributes
            mock_import.return_value = mock_module

            with pytest.raises(AttributeError, match="Class .* not found"):
                lazy.load()


# ==============================================================================
# PluginRegistry Tests - Basic Operations
# ==============================================================================


class TestPluginRegistryBasics:
    """Tests for basic registry operations."""

    def test_init_loads_builtins(self):
        """Test that PluginRegistry initializes with builtins loaded.

        Note: This test creates its own PluginRegistry to verify auto-loading,
        since the `registry` fixture resets state for test isolation.
        """
        # Create a fresh registry without using the fixture (which resets state)
        fresh_registry = PluginRegistry()
        # Registry auto-loads builtins in __init__
        assert fresh_registry._types != {}
        assert fresh_registry._loaded_plugins != {}
        # Should have core categories loaded
        assert "endpoint" in fresh_registry._types
        assert "service" in fresh_registry._types

    def test_load_registry(self, registry, temp_registry_file):
        """Test loading built-in registry from YAML."""
        registry.load_registry(temp_registry_file)

        assert "test-plugin" in registry._loaded_plugins
        assert "endpoint" in registry._types
        assert "post_processor" in registry._types
        assert "test_endpoint" in registry._types["endpoint"]
        assert "test_processor" in registry._types["post_processor"]

    def test_load_registry_nonexistent_file(self, registry):
        """Test that loading nonexistent file raises RuntimeError."""
        # Should raise RuntimeError for missing registry
        with pytest.raises(RuntimeError, match="Built-in registry.yaml not found"):
            registry.load_registry("/nonexistent/path/registry.yaml")

    def test_load_registry_priority_conflict(self, registry):
        """Test that higher priority wins in conflicts."""
        # Create YAML with conflicting types
        yaml_content = {
            "plugin": {"name": "plugin1", "version": "1.0.0"},
            "endpoint": {
                "openai": {
                    "class": "plugin1:OpenAI",
                    "priority": 100,
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path1 = Path(f.name)

        # Load first registry
        registry.load_registry(path1)
        lazy_type1 = registry._types["endpoint"]["openai"]
        assert lazy_type1.priority == 100
        assert lazy_type1.class_path == "plugin1:OpenAI"

        # Load second registry with higher priority
        yaml_content2 = {
            "plugin": {"name": "plugin2", "version": "1.0.0"},
            "endpoint": {
                "openai": {
                    "class": "plugin2:OpenAI",
                    "priority": 110,  # Higher priority
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content2, f)
            path2 = Path(f.name)

        # Load second registry
        registry.load_registry(path2)
        lazy_type2 = registry._types["endpoint"]["openai"]
        assert lazy_type2.priority == 110
        assert lazy_type2.class_path == "plugin2:OpenAI"

    def test_discover_plugins(self, registry, temp_registry_file):
        """Test discovering plugin registries via entry points."""
        mock_ep = Mock()
        mock_ep.name = "test-plugin"
        mock_ep.load.return_value = str(temp_registry_file)

        with patch(
            "aiperf.common.plugin_registry.entry_points", return_value=[mock_ep]
        ):
            registry.discover_plugins()

        assert "test-plugin" in registry._loaded_plugins
        assert "endpoint" in registry._types

    def test_discover_plugins_handles_errors(self, registry):
        """Test that bad plugins don't crash discovery."""
        mock_ep = Mock()
        mock_ep.name = "bad-plugin"
        mock_ep.load.side_effect = ImportError("Bad plugin")

        with patch(
            "aiperf.common.plugin_registry.entry_points", return_value=[mock_ep]
        ):
            # Should not raise
            registry.discover_plugins()

        # Bad plugin should not have been loaded
        assert "bad-plugin" not in registry._loaded_plugins
        # Registry should still be empty (fixture resets state, and bad plugin didn't add anything)
        assert registry._types == {}


# ==============================================================================
# PluginRegistry Tests - get() method
# ==============================================================================


class TestGet:
    """Tests for get() method."""

    def test_get_by_name_success(self, registry, temp_registry_file, mock_class):
        """Test getting class by name successfully."""
        registry.load_registry(temp_registry_file)

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.TestEndpoint = mock_class
            mock_import.return_value = mock_module

            cls = registry.get("endpoint", "test_endpoint")

            assert cls is mock_class
            # User manually instantiates
            cls(arg1="value1")

    def test_get_by_class_path_success(self, registry, temp_registry_file, mock_class):
        """Test getting class by class path successfully."""
        registry.load_registry(temp_registry_file)

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.TestEndpoint = mock_class
            mock_import.return_value = mock_module

            cls = registry.get("endpoint", "test.module:TestEndpoint")

            assert cls is mock_class

    def test_get_unknown_category(self, registry):
        """Test getting with unknown category."""
        with pytest.raises(KeyError, match="Unknown category"):
            registry.get("nonexistent_category", "type")

    def test_get_unknown_type(self, registry, temp_registry_file):
        """Test getting with unknown type."""
        registry.load_registry(temp_registry_file)

        with pytest.raises(TypeNotFoundError, match="not found for category"):
            registry.get("endpoint", "nonexistent_type")

    def test_get_class_path_category_mismatch(self, registry, temp_registry_file):
        """Test getting by class path with category mismatch."""
        registry.load_registry(temp_registry_file)

        with pytest.raises(ValueError, match="registered for category"):
            registry.get("post_processor", "test.module:TestEndpoint")


# ==============================================================================
# PluginRegistry Tests - list_all() method
# ==============================================================================


class TestListTypes:
    """Tests for list_types() method."""

    def test_list_types_returns_lazy_types(self, registry, temp_registry_file):
        """Test that list_all returns TypeEntry objects."""
        registry.load_registry(temp_registry_file)

        lazy_types = registry.list_types("endpoint")

        assert len(lazy_types) == 2
        assert all(isinstance(lt, TypeEntry) for lt in lazy_types)

    def test_list_types_sorted_alphabetically(self, registry, temp_registry_file):
        """Test that results are sorted alphabetically by name."""
        registry.load_registry(temp_registry_file)

        lazy_types = registry.list_types("endpoint")

        # Should be sorted: low_priority_endpoint, test_endpoint
        assert lazy_types[0].type_name == "low_priority_endpoint"
        assert lazy_types[1].type_name == "test_endpoint"

    def test_list_types_can_inspect_metadata(self, registry, temp_registry_file):
        """Test that user can inspect metadata without loading."""
        registry.load_registry(temp_registry_file)

        lazy_types = registry.list_types("endpoint")

        # User can inspect metadata
        for lazy_type in lazy_types:
            assert lazy_type.type_name
            assert lazy_type.description
            assert lazy_type.priority >= 0  # Can be 0 (default) or higher
            assert lazy_type.class_path
            # NOT loaded yet
            assert lazy_type.loaded_class is None

    def test_list_types_user_can_load_manually(
        self, registry, temp_registry_file, mock_class
    ):
        """Test that user can manually load classes."""
        registry.load_registry(temp_registry_file)

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.TestEndpoint = mock_class
            mock_module.LowPriorityEndpoint = mock_class
            mock_import.return_value = mock_module

            lazy_types = registry.list_types("endpoint")

            # User manually loads
            for lazy_type in lazy_types:
                if lazy_type.type_name == "test_endpoint":
                    EndpointClass = lazy_type.load()
                    assert EndpointClass is mock_class
                    # User manually instantiates
                    EndpointClass(arg1="value1")

    def test_list_types_unknown_category(self, registry):
        """Test that unknown category returns empty list."""
        lazy_types = registry.list_types("nonexistent_category")
        assert lazy_types == []


# ==============================================================================
# PluginRegistry Tests - Priority Resolution
# ==============================================================================


class TestPriorityResolution:
    """Tests for priority-based conflict resolution."""

    def test_priority_resolves_conflicts(self, registry):
        """Test priority resolves conflicts when same type_name."""
        # Register type with priority 50
        yaml_content = {
            "plugin": {"name": "builtin", "version": "1.0.0", "builtin": True},
            "endpoint": {
                "openai": {
                    "class": "builtin:OpenAI",
                    "priority": 50,
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path1 = Path(f.name)

        registry.load_registry(path1)

        # Register same type_name with priority 100
        yaml_content2 = {
            "plugin": {"name": "plugin", "version": "1.0.0"},
            "endpoint": {
                "openai": {  # SAME NAME!
                    "class": "plugin:EnhancedOpenAI",
                    "priority": 100,  # Higher!
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content2, f)
            path2 = Path(f.name)

        registry.load_registry(path2)

        # Plugin should win
        lazy_type = registry._types["endpoint"]["openai"]
        assert lazy_type.package_name == "plugin"
        assert lazy_type.priority == 100
        assert lazy_type.class_path == "plugin:EnhancedOpenAI"

    def test_lower_priority_loses(self, registry):
        """Test lower priority loses conflict."""
        # Register with priority 100
        yaml_content = {
            "plugin": {"name": "plugin_a", "version": "1.0.0"},
            "endpoint": {"custom": {"class": "a:Custom", "priority": 100}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path1 = Path(f.name)

        registry.load_registry(path1)

        # Try to register same name with priority 50
        yaml_content2 = {
            "plugin": {"name": "plugin_b", "version": "1.0.0"},
            "endpoint": {
                "custom": {"class": "b:Custom", "priority": 50}  # Lower!
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content2, f)
            path2 = Path(f.name)

        registry.load_registry(path2)

        # Plugin A should win
        lazy_type = registry._types["endpoint"]["custom"]
        assert lazy_type.package_name == "plugin_a"
        assert lazy_type.priority == 100

    def test_no_conflict_both_registered(self, registry):
        """Test different names both registered (priority irrelevant)."""
        yaml_content = {
            "plugin": {"name": "builtin", "version": "1.0.0"},
            "endpoint": {
                "openai": {"class": "builtin:OpenAI", "priority": 50},
                "anthropic": {"class": "builtin:Anthropic", "priority": 100},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        registry.load_registry(path)

        # Both registered (different names, no conflict)
        assert "openai" in registry._types["endpoint"]
        assert "anthropic" in registry._types["endpoint"]
        # Priority didn't affect anything - both exist


# ==============================================================================
# Implicit Priority System Tests
# ==============================================================================


class TestImplicitPriority:
    """Tests for the implicit priority system with smart conflict resolution."""

    def test_builtin_auto_priority_zero(self, registry):
        """Test built-ins automatically get priority 0."""
        yaml_content = {
            "plugin": {"name": "aiperf", "version": "1.0.0", "builtin": True},
            "endpoint": {
                "openai": {
                    "class": "aiperf.endpoints:OpenAI"
                    # No priority specified!
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        registry.load_registry(path)

        lazy_type = registry._types["endpoint"]["openai"]
        assert lazy_type.priority == 0
        assert lazy_type.is_builtin is True

    def test_package_auto_priority_zero(self, registry):
        """Test packages default to priority 0."""
        yaml_content = {
            "plugin": {"name": "my_plugin", "version": "1.0.0", "builtin": False},
            "endpoint": {
                "custom": {
                    "class": "my_plugin:Custom"
                    # No priority specified!
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        registry.load_registry(path)

        lazy_type = registry._types["endpoint"]["custom"]
        assert lazy_type.priority == 0
        assert lazy_type.is_builtin is False

    def test_equal_priority_package_beats_builtin(self, registry):
        """Test package beats built-in when priorities equal (both 0)."""
        # Built-in with priority 0 (implicit)
        builtin_yaml = {
            "plugin": {"name": "aiperf", "version": "1.0.0", "builtin": True},
            "endpoint": {
                "openai": {
                    "class": "aiperf:OpenAI"
                    # No priority - defaults to 0
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(builtin_yaml, f)
            builtin_path = Path(f.name)

        # Package with priority 0 (same!)
        package_yaml = {
            "plugin": {"name": "plugin", "version": "1.0.0", "builtin": False},
            "endpoint": {
                "openai": {
                    "class": "plugin:EnhancedOpenAI"
                    # No priority - defaults to 0
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(package_yaml, f)
            package_path = Path(f.name)

        registry.load_registry(builtin_path)
        registry.load_registry(package_path)

        # Package should win
        lazy_type = registry._types["endpoint"]["openai"]
        assert lazy_type.package_name == "plugin"
        assert lazy_type.class_path == "plugin:EnhancedOpenAI"

    def test_higher_priority_package_wins_over_builtin(self, registry):
        """Test explicit priority can override built-in."""
        builtin_yaml = {
            "plugin": {"name": "aiperf", "version": "1.0.0", "builtin": True},
            "endpoint": {
                "openai": {
                    "class": "aiperf:OpenAI"
                    # No priority - defaults to 0
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(builtin_yaml, f)
            builtin_path = Path(f.name)

        package_yaml = {
            "plugin": {"name": "custom_plugin", "version": "1.0.0"},
            "endpoint": {
                "openai": {
                    "class": "custom:OpenAI",
                    "priority": 100,  # Explicit higher priority
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(package_yaml, f)
            package_path = Path(f.name)

        registry.load_registry(builtin_path)
        registry.load_registry(package_path)

        # Package wins (higher priority)
        lazy_type = registry._types["endpoint"]["openai"]
        assert lazy_type.package_name == "custom_plugin"
        assert lazy_type.priority == 100

    def test_equal_priority_builtin_loses_to_package(self, registry):
        """Test package beats built-in even with equal priority."""
        builtin_yaml = {
            "plugin": {"name": "aiperf", "version": "1.0.0", "builtin": True},
            "endpoint": {
                "custom": {
                    "class": "aiperf:Custom",
                    "priority": 10,  # Explicit priority
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(builtin_yaml, f)
            builtin_path = Path(f.name)

        package_yaml = {
            "plugin": {"name": "override_plugin", "version": "1.0.0", "builtin": False},
            "endpoint": {
                "custom": {
                    "class": "override:Custom",
                    "priority": 10,  # SAME priority
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(package_yaml, f)
            package_path = Path(f.name)

        registry.load_registry(builtin_path)
        registry.load_registry(package_path)

        # Package wins (equal priority, but package beats built-in)
        lazy_type = registry._types["endpoint"]["custom"]
        assert lazy_type.package_name == "override_plugin"
        assert lazy_type.priority == 10

    def test_lower_priority_package_loses(self, registry):
        """Test lower priority package loses to higher priority."""
        package_a_yaml = {
            "plugin": {"name": "plugin_a", "version": "1.0.0"},
            "endpoint": {
                "custom": {
                    "class": "a:Custom",
                    "priority": 100,
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(package_a_yaml, f)
            path_a = Path(f.name)

        package_b_yaml = {
            "plugin": {"name": "plugin_b", "version": "1.0.0"},
            "endpoint": {
                "custom": {
                    "class": "b:Custom",
                    "priority": 50,  # Lower!
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(package_b_yaml, f)
            path_b = Path(f.name)

        registry.load_registry(path_a)
        registry.load_registry(path_b)

        # Package A wins (higher priority)
        lazy_type = registry._types["endpoint"]["custom"]
        assert lazy_type.package_name == "plugin_a"
        assert lazy_type.priority == 100


# ==============================================================================
# Global Registry Tests
# ==============================================================================


class TestGlobalRegistry:
    """Tests for global registry singleton."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the global singleton before and after each test."""
        plugin_registry.reset()
        yield
        plugin_registry.reset()

    def test_module_level_reset(self):
        """Test that plugin_registry.reset() clears custom types."""
        # Load a custom type
        registry_data = {
            "plugin": {"name": "test", "version": "1.0.0"},
            "endpoint": {
                "custom_test_type_xyz": {
                    "class": "test.module:TestClass",
                    "description": "Test",
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(registry_data, f)
            path = Path(f.name)

        plugin_registry._registry.load_registry(path)
        # Verify custom type was loaded
        assert "custom_test_type_xyz" in plugin_registry._registry._types.get(
            "endpoint", {}
        )

        # Now reset
        plugin_registry.reset()

        # After reset, custom type should be gone (builtins are reloaded)
        assert "custom_test_type_xyz" not in plugin_registry._registry._types.get(
            "endpoint", {}
        )

    def test_module_level_get_class(self):
        """Test that plugin_registry.get_class() works at module level."""
        # Register a test type first
        registry_data = {
            "plugin": {"name": "test", "version": "1.0.0"},
            "endpoint": {
                "test_type": {
                    "class": "test.module:TestClass",
                    "description": "Test",
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(registry_data, f)
            path = Path(f.name)

        plugin_registry._registry.load_registry(path)

        # Now test module-level get_class (it should work)
        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.TestClass = type("TestClass", (), {})
            mock_import.return_value = mock_module

            cls = plugin_registry.get_class("endpoint", "test_type")
            assert cls is not None

    def test_module_level_list_types(self):
        """Test that plugin_registry.list_types() works at module level."""
        registry_data = {
            "plugin": {"name": "test", "version": "1.0.0"},
            "endpoint": {
                "test_type": {"class": "test.module:TestClass", "description": "Test"}
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(registry_data, f)
            path = Path(f.name)

        plugin_registry._registry.load_registry(path)

        # Test module-level list_types
        lazy_types = plugin_registry.list_types("endpoint")
        assert len(lazy_types) > 0


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_builtin_and_package_merge(self, registry):
        """Test merging built-in and package registries."""
        # Create built-in registry
        builtin_yaml = {
            "plugin": {"name": "aiperf", "version": "1.0.0", "builtin": True},
            "endpoint": {
                "openai": {
                    "class": "aiperf.endpoints:OpenAI",
                    "priority": 100,
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(builtin_yaml, f)
            builtin_path = Path(f.name)

        # Create package registry (overrides with higher priority)
        package_yaml = {
            "plugin": {"name": "custom-plugin", "version": "1.0.0"},
            "endpoint": {
                "openai": {
                    "class": "custom.endpoints:CustomOpenAI",
                    "priority": 110,  # Higher priority
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(package_yaml, f)
            package_path = Path(f.name)

        # Load both
        registry.load_registry(builtin_path)
        registry.load_registry(package_path)

        # Package should override (higher priority)
        lazy_type = registry._types["endpoint"]["openai"]
        assert lazy_type.package_name == "custom-plugin"
        assert lazy_type.priority == 110
        assert lazy_type.class_path == "custom.endpoints:CustomOpenAI"

    def test_lazy_loading_only_on_demand(
        self, registry, temp_registry_file, mock_class
    ):
        """Test that classes are only imported on demand."""
        registry.load_registry(temp_registry_file)

        # list_all should NOT import
        with patch("importlib.import_module") as mock_import:
            lazy_types = registry.list_types("endpoint")
            assert len(lazy_types) == 2
            mock_import.assert_not_called()

        # get() SHOULD import
        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.TestEndpoint = mock_class
            mock_import.return_value = mock_module

            cls = registry.get("endpoint", "test_endpoint")
            assert cls is mock_class
            mock_import.assert_called_once()

    def test_user_manual_instantiation(self, registry, temp_registry_file, mock_class):
        """Test that user manually instantiates classes."""
        registry.load_registry(temp_registry_file)

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_instance = Mock()
            mock_class_with_init = Mock(return_value=mock_instance)
            mock_module.TestEndpoint = mock_class_with_init
            mock_import.return_value = mock_module

            # Get class
            EndpointClass = registry.get("endpoint", "test_endpoint")

            # User manually instantiates
            endpoint = EndpointClass(arg1="value1", arg2="value2")

            assert endpoint is mock_instance
            mock_class_with_init.assert_called_once_with(arg1="value1", arg2="value2")
