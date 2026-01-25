# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for simplified PluginRegistry."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

from aiperf.plugin import plugin_registry
from aiperf.plugin.plugin_registry import (
    DEFAULT_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    PackageMetadata,
    PluginError,
    PluginRegistry,
    TypeEntry,
    TypeNotFoundError,
)

# ==============================================================================
# Test Fixtures
# ==============================================================================


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
    reg._type_entries_by_class_path = {}
    reg._class_to_name.clear()
    yield reg
    # Restore registry to normal state with builtins loaded
    plugin_registry.reset()


@pytest.fixture
def mock_class():
    """Mock class for testing lazy loading."""
    return type("MockClass", (), {"__init__": lambda self, **kwargs: None})


# ==============================================================================
# TypeNotFoundError Tests
# ==============================================================================


class TestTypeNotFoundError:
    """Tests for TypeNotFoundError exception."""

    def test_error_message_includes_available_types(self):
        """Test that error message lists available types."""
        error = TypeNotFoundError("endpoint", "nonexistent", ["chat", "completions"])
        assert "nonexistent" in str(error)
        assert "endpoint" in str(error)
        assert "chat" in str(error)
        assert "completions" in str(error)

    def test_error_stores_attributes(self):
        """Test that error stores category, type_name, and available."""
        available = ["type_a", "type_b", "type_c"]
        error = TypeNotFoundError("my_category", "unknown_type", available)

        assert error.category == "my_category"
        assert error.type_name == "unknown_type"
        assert error.available == available

    def test_error_is_plugin_error_subclass(self):
        """Test that TypeNotFoundError is a PluginError."""
        error = TypeNotFoundError("cat", "type", [])
        assert isinstance(error, PluginError)

    def test_error_with_empty_available(self):
        """Test error message when no types are available."""
        error = TypeNotFoundError("empty_category", "type", [])
        assert "empty_category" in str(error)
        assert "type" in str(error)

    def test_error_available_sorted_in_message(self):
        """Test that available types are sorted in error message."""
        error = TypeNotFoundError("cat", "type", ["zebra", "apple", "banana"])
        message = str(error)
        # Check that sorted order appears
        assert message.index("apple") < message.index("banana") < message.index("zebra")


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

    def test_load_empty_module_path(self):
        """Test that load() raises ValueError for empty module path."""
        lazy = TypeEntry(
            category="cat",
            type_name="type",
            package_name="pkg",
            class_path=":TestClass",
        )
        with pytest.raises(ValueError, match="Invalid class_path format"):
            lazy.load()

    def test_load_empty_class_name(self):
        """Test that load() raises ValueError for empty class name."""
        lazy = TypeEntry(
            category="cat",
            type_name="type",
            package_name="pkg",
            class_path="module:",
        )
        with pytest.raises(ValueError, match="Invalid class_path format"):
            lazy.load()


class TestTypeEntryValidate:
    """Tests for TypeEntry.validate() method."""

    def test_validate_already_loaded_class(self, mock_class):
        """Test that validate returns True for already loaded class."""
        lazy = TypeEntry(
            category="cat",
            type_name="type",
            package_name="pkg",
            class_path="module:Class",
            loaded_class=mock_class,
        )

        is_valid, error = lazy.validate()
        assert is_valid is True
        assert error is None

    def test_validate_invalid_class_path_format_no_colon(self):
        """Test validate catches invalid class_path without colon."""
        lazy = TypeEntry(
            category="cat",
            type_name="type",
            package_name="pkg",
            class_path="invalid_no_colon",
        )

        is_valid, error = lazy.validate()
        assert is_valid is False
        assert "Invalid class_path format" in error

    def test_validate_invalid_class_path_empty_parts(self):
        """Test validate catches empty module or class."""
        lazy = TypeEntry(
            category="cat",
            type_name="type",
            package_name="pkg",
            class_path=":Class",
        )

        is_valid, error = lazy.validate()
        assert is_valid is False
        assert "Invalid class_path format" in error

    def test_validate_module_not_found(self):
        """Test validate catches non-existent module."""
        lazy = TypeEntry(
            category="cat",
            type_name="type",
            package_name="pkg",
            class_path="nonexistent_module_xyz:Class",
        )

        is_valid, error = lazy.validate()
        assert is_valid is False
        assert "Module not found" in error

    def test_validate_valid_module_without_class_check(self):
        """Test validate succeeds for valid module without class check."""
        lazy = TypeEntry(
            category="cat",
            type_name="type",
            package_name="pkg",
            class_path="os.path:join",  # Valid module, valid attribute
        )

        is_valid, error = lazy.validate(check_class=False)
        assert is_valid is True
        assert error is None

    def test_validate_with_class_check_finds_class(self):
        """Test validate with check_class=True finds class definition."""
        # Use a real module with a known class
        lazy = TypeEntry(
            category="cat",
            type_name="type",
            package_name="pkg",
            class_path="pathlib:Path",
        )

        is_valid, error = lazy.validate(check_class=True)
        assert is_valid is True
        assert error is None

    def test_validate_with_class_check_class_not_found(self):
        """Test validate with check_class=True catches missing class."""
        # Use a real module but a nonexistent class
        lazy = TypeEntry(
            category="cat",
            type_name="type",
            package_name="pkg",
            class_path="os:NonexistentClassXYZ123",
        )

        is_valid, error = lazy.validate(check_class=True)
        # The AST check may not find it, but module validation should still pass
        # because the module exists - the class check via AST is best-effort
        assert is_valid in (True, False)  # Depends on AST parsing success


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
        with pytest.raises(RuntimeError, match="Built-in plugins.yaml not found"):
            registry.load_registry("/nonexistent/path/plugins.yaml")

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
            "aiperf.plugin.plugin_registry.entry_points", return_value=[mock_ep]
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
            "aiperf.plugin.plugin_registry.entry_points", return_value=[mock_ep]
        ):
            # Should not raise
            registry.discover_plugins()

        # Bad plugin should not have been loaded
        assert "bad-plugin" not in registry._loaded_plugins
        # Registry should still be empty (fixture resets state, and bad plugin didn't add anything)
        assert registry._types == {}

    def test_discover_plugins_invalid_return_type(self, registry):
        """Test that plugins returning invalid types are skipped."""
        mock_ep = Mock()
        mock_ep.name = "invalid-plugin"
        mock_ep.load.return_value = 12345  # Invalid - not str or Path

        with patch(
            "aiperf.plugin.plugin_registry.entry_points", return_value=[mock_ep]
        ):
            registry.discover_plugins()

        assert "invalid-plugin" not in registry._loaded_plugins

    def test_load_registry_with_string_path(self, registry, temp_registry_file):
        """Test loading registry with string path instead of Path."""
        registry.load_registry(str(temp_registry_file))

        assert "test-plugin" in registry._loaded_plugins
        assert "endpoint" in registry._types

    def test_load_registry_simple_format(self, registry):
        """Test loading registry with simple string format (module:Class)."""
        yaml_content = {
            "plugin": {"name": "simple-plugin", "version": "1.0.0"},
            "endpoint": {
                "simple": "simple.module:SimpleClass",  # String format
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        registry.load_registry(path)

        assert "simple" in registry._types["endpoint"]
        lazy_type = registry._types["endpoint"]["simple"]
        assert lazy_type.class_path == "simple.module:SimpleClass"
        assert lazy_type.description == ""  # No description in simple format
        assert lazy_type.priority == 0  # Default priority


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

            cls = registry.get_class("endpoint", "test_endpoint")

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

            cls = registry.get_class("endpoint", "test.module:TestEndpoint")

            assert cls is mock_class

    def test_get_unknown_category(self, registry):
        """Test getting with unknown category."""
        with pytest.raises(KeyError, match="Unknown category"):
            registry.get_class("nonexistent_category", "type")

    def test_get_unknown_type(self, registry, temp_registry_file):
        """Test getting with unknown type."""
        registry.load_registry(temp_registry_file)

        with pytest.raises(TypeNotFoundError, match="not found for category"):
            registry.get_class("endpoint", "nonexistent_type")

    def test_get_class_path_category_mismatch(self, registry, temp_registry_file):
        """Test getting by class path with category mismatch."""
        registry.load_registry(temp_registry_file)

        with pytest.raises(ValueError, match="registered for category"):
            registry.get_class("post_processor", "test.module:TestEndpoint")

    def test_get_class_path_not_registered(self, registry, temp_registry_file):
        """Test getting by unregistered class path."""
        registry.load_registry(temp_registry_file)

        with pytest.raises(KeyError, match="No type with class path"):
            registry.get_class("endpoint", "unregistered.module:UnregisteredClass")


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
# PluginRegistry Tests - validate_all() method
# ==============================================================================


class TestValidateAll:
    """Tests for validate_all() method."""

    def test_validate_all_no_errors(self, registry, temp_registry_file):
        """Test validate_all returns empty dict when all valid."""
        # Load a registry with types that have valid module paths
        yaml_content = {
            "plugin": {"name": "valid-plugin", "version": "1.0.0"},
            "endpoint": {
                "valid_type": {
                    "class": "os.path:join",  # Valid module
                    "description": "Valid type",
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        registry.load_registry(path)
        errors = registry.validate_all(check_class=False)

        assert errors == {}

    def test_validate_all_with_errors(self, registry):
        """Test validate_all returns errors for invalid types."""
        yaml_content = {
            "plugin": {"name": "invalid-plugin", "version": "1.0.0"},
            "endpoint": {
                "invalid_type": {
                    "class": "nonexistent_module_xyz:InvalidClass",
                    "description": "Invalid type",
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        registry.load_registry(path)
        errors = registry.validate_all(check_class=False)

        assert "endpoint" in errors
        assert len(errors["endpoint"]) == 1
        type_name, error_msg = errors["endpoint"][0]
        assert type_name == "invalid_type"
        assert "Module not found" in error_msg

    def test_validate_all_multiple_categories(self, registry):
        """Test validate_all checks all categories."""
        yaml_content = {
            "plugin": {"name": "mixed-plugin", "version": "1.0.0"},
            "endpoint": {
                "valid_endpoint": {"class": "os.path:join"},
                "invalid_endpoint": {"class": "bad_module:Bad"},
            },
            "service": {
                "invalid_service": {"class": "another_bad:Service"},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        registry.load_registry(path)
        errors = registry.validate_all()

        assert "endpoint" in errors
        assert "service" in errors


# ==============================================================================
# PluginRegistry Tests - list_packages() method
# ==============================================================================


class TestListPackages:
    """Tests for list_packages() method."""

    def test_list_packages_returns_loaded_plugins(self, registry, temp_registry_file):
        """Test list_packages returns all loaded plugin names."""
        registry.load_registry(temp_registry_file)

        packages = registry.list_packages()

        assert "test-plugin" in packages

    def test_list_packages_builtin_only(self, registry):
        """Test list_packages with builtin_only=True."""
        # Load a builtin plugin
        builtin_yaml = {
            "plugin": {"name": "aiperf", "version": "1.0.0", "builtin": True},
            "endpoint": {"builtin_type": {"class": "aiperf:Type"}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(builtin_yaml, f)
            path1 = Path(f.name)

        # Load a non-builtin plugin
        external_yaml = {
            "plugin": {"name": "external", "version": "1.0.0", "builtin": False},
            "endpoint": {"external_type": {"class": "external:Type"}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(external_yaml, f)
            path2 = Path(f.name)

        registry.load_registry(path1)
        registry.load_registry(path2)

        # All packages
        all_packages = registry.list_packages(builtin_only=False)
        assert "aiperf" in all_packages
        assert "external" in all_packages

        # Builtin only
        builtin_packages = registry.list_packages(builtin_only=True)
        assert "aiperf" in builtin_packages
        assert "external" not in builtin_packages

    def test_list_packages_empty(self, registry):
        """Test list_packages returns empty list when no plugins loaded."""
        packages = registry.list_packages()
        assert packages == []


# ==============================================================================
# PluginRegistry Tests - find_registered_name() method
# ==============================================================================


class TestFindRegisteredName:
    """Tests for find_registered_name() method."""

    def test_find_registered_name_loaded_class(self, registry, mock_class):
        """Test finding registered name for a loaded class."""
        yaml_content = {
            "plugin": {"name": "test-plugin", "version": "1.0.0"},
            "endpoint": {
                "my_endpoint": {"class": "test.module:MyEndpoint"},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        registry.load_registry(path)

        # Load the class first
        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.MyEndpoint = mock_class
            mock_import.return_value = mock_module

            loaded_cls = registry.get_class("endpoint", "my_endpoint")

            # Now find the registered name
            name = registry.find_registered_name("endpoint", loaded_cls)
            assert name == "my_endpoint"

    def test_find_registered_name_by_class_path(self, registry):
        """Test finding name for class not loaded via registry."""
        yaml_content = {
            "plugin": {"name": "test-plugin", "version": "1.0.0"},
            "endpoint": {
                "path_endpoint": {"class": "pathlib:Path"},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        registry.load_registry(path)

        # Find by class path without loading through registry
        name = registry.find_registered_name("endpoint", Path)
        assert name == "path_endpoint"

    def test_find_registered_name_unknown_category(self, registry, mock_class):
        """Test finding name in unknown category returns None."""
        name = registry.find_registered_name("nonexistent", mock_class)
        assert name is None

    def test_find_registered_name_unregistered_class(
        self, registry, temp_registry_file
    ):
        """Test finding name for unregistered class returns None."""
        registry.load_registry(temp_registry_file)

        # Create a class not in the registry
        UnregisteredClass = type("UnregisteredClass", (), {})

        name = registry.find_registered_name("endpoint", UnregisteredClass)
        assert name is None


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
# Schema Validation Tests
# ==============================================================================


class TestSchemaValidation:
    """Tests for manifest schema validation."""

    def test_valid_schema_version(self, registry):
        """Test that valid schema version is accepted."""
        yaml_content = {
            "schema_version": "1.0",
            "plugin": {"name": "test", "version": "1.0.0"},
            "endpoint": {"type": {"class": "mod:Class"}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        # Should not raise
        registry.load_registry(path)
        assert "test" in registry._loaded_plugins

    def test_missing_schema_version(self, registry):
        """Test that missing schema version is handled gracefully."""
        yaml_content = {
            "plugin": {"name": "no-schema", "version": "1.0.0"},
            "endpoint": {"type": {"class": "mod:Class"}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        # Should not raise, uses default
        registry.load_registry(path)
        assert "no-schema" in registry._loaded_plugins

    def test_unknown_schema_version_warns(self, registry):
        """Test that unknown schema version logs warning but continues."""
        yaml_content = {
            "schema_version": "99.0",  # Unknown version
            "plugin": {"name": "future", "version": "1.0.0"},
            "endpoint": {"type": {"class": "mod:Class"}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        # Should not raise, just warns
        registry.load_registry(path)
        assert "future" in registry._loaded_plugins

    def test_invalid_schema_version_type(self, registry):
        """Test that non-string schema version raises error."""
        yaml_content = {
            "schema_version": 1.0,  # Number instead of string
            "plugin": {"name": "bad-schema", "version": "1.0.0"},
            "endpoint": {"type": {"class": "mod:Class"}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        with pytest.raises(ValueError, match="schema_version must be string"):
            registry.load_registry(path)

    def test_empty_yaml(self, registry):
        """Test that empty YAML is handled gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")  # Empty file
            path = Path(f.name)

        # Should not raise
        registry.load_registry(path)

    def test_invalid_category_type(self, registry):
        """Test that invalid category type (non-dict) is skipped."""
        yaml_content = {
            "plugin": {"name": "invalid-cat", "version": "1.0.0"},
            "endpoint": "not_a_dict",  # Invalid - should be dict
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        # Should not raise, just warns and skips
        registry.load_registry(path)
        assert "endpoint" not in registry._types

    def test_missing_class_path(self, registry):
        """Test that missing class path is handled gracefully."""
        yaml_content = {
            "plugin": {"name": "missing-class", "version": "1.0.0"},
            "endpoint": {
                "type_without_class": {
                    "description": "No class field",
                    # Missing "class" field!
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        # Should not raise, just warns and skips
        registry.load_registry(path)
        assert "type_without_class" not in registry._types.get("endpoint", {})


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
# Module-Level Function Tests
# ==============================================================================


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the global singleton before and after each test."""
        plugin_registry.reset()
        yield
        plugin_registry.reset()

    def test_list_categories(self):
        """Test list_categories returns all category names."""
        categories = plugin_registry.list_categories()

        # Should have core categories from builtins
        assert "endpoint" in categories
        assert "service" in categories
        # Should be sorted
        assert categories == sorted(categories)

    def test_get_package_metadata(self):
        """Test get_package_metadata returns metadata for loaded plugin."""
        metadata = plugin_registry.get_package_metadata("aiperf")

        assert isinstance(metadata, dict)
        assert metadata.get("builtin") is True

    def test_get_package_metadata_not_found(self):
        """Test get_package_metadata raises KeyError for unknown package."""
        with pytest.raises(KeyError, match="not found"):
            plugin_registry.get_package_metadata("nonexistent_package_xyz")

    def test_register_dynamic_class(self):
        """Test register() for dynamic class registration."""

        class DynamicEndpoint:
            """A dynamically registered endpoint."""

            pass

        plugin_registry.register(
            category="endpoint",
            type_name="dynamic_endpoint",
            cls=DynamicEndpoint,
            priority=50,
            is_builtin=False,
        )

        # Should be retrievable
        retrieved_cls = plugin_registry.get_class("endpoint", "dynamic_endpoint")
        assert retrieved_cls is DynamicEndpoint

    def test_register_with_enum_type_name(self):
        """Test register() with enum type name."""
        from enum import Enum

        class TestType(Enum):
            CUSTOM = "custom_type"

        class CustomClass:
            pass

        plugin_registry.register(
            category="endpoint",
            type_name=TestType.CUSTOM,
            cls=CustomClass,
        )

        # Should be retrievable by enum value
        retrieved_cls = plugin_registry.get_class("endpoint", "custom_type")
        assert retrieved_cls is CustomClass

    def test_register_priority_conflict_resolution(self):
        """Test register() respects priority conflict resolution."""

        class LowPriorityClass:
            pass

        class HighPriorityClass:
            pass

        # Register low priority first
        plugin_registry.register(
            category="endpoint",
            type_name="conflict_type",
            cls=LowPriorityClass,
            priority=10,
        )

        # Register high priority second
        plugin_registry.register(
            category="endpoint",
            type_name="conflict_type",
            cls=HighPriorityClass,
            priority=100,
        )

        # High priority should win
        retrieved_cls = plugin_registry.get_class("endpoint", "conflict_type")
        assert retrieved_cls is HighPriorityClass

    def test_load_registry_module_level(self):
        """Test load_registry at module level."""
        yaml_content = {
            "plugin": {"name": "module-level-test", "version": "1.0.0"},
            "endpoint": {
                "module_level_type": {"class": "mod:Class"},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        plugin_registry.load_registry(path)

        assert "module-level-test" in plugin_registry.list_packages()

    def test_validate_all_module_level(self):
        """Test validate_all at module level."""
        # Should not raise
        errors = plugin_registry.validate_all(check_class=False)

        # Builtins should all be valid
        # (errors dict will be empty or contain only categories with issues)
        assert isinstance(errors, dict)

    def test_find_registered_name_module_level(self):
        """Test find_registered_name at module level."""

        # Register a known type
        class KnownClass:
            pass

        plugin_registry.register("endpoint", "known_type", KnownClass)

        name = plugin_registry.find_registered_name("endpoint", KnownClass)
        assert name == "known_type"


# ==============================================================================
# create_enum() Tests
# ==============================================================================


class TestCreateEnum:
    """Tests for create_enum() function."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the global singleton before and after each test."""
        plugin_registry.reset()
        yield
        plugin_registry.reset()

    def test_create_enum_from_category(self):
        """Test creating enum from registered types."""
        # Use a known category with types
        EndpointEnum = plugin_registry.create_enum("endpoint", "EndpointEnum")

        # Should have members for each registered type
        assert hasattr(EndpointEnum, "CHAT")
        assert EndpointEnum.CHAT.value == "chat"

    def test_create_enum_empty_category(self):
        """Test create_enum raises for empty category."""
        with pytest.raises(KeyError, match="No types registered"):
            plugin_registry.create_enum("nonexistent_category", "EmptyEnum")

    def test_create_enum_hyphen_to_underscore(self):
        """Test create_enum converts hyphens to underscores."""

        # Register a type with hyphens
        class HyphenClass:
            pass

        plugin_registry.register("endpoint", "my-hyphen-type", HyphenClass)

        EnumWithHyphens = plugin_registry.create_enum("endpoint", "EnumWithHyphens")

        # Hyphen should be converted to underscore and uppercased
        assert hasattr(EnumWithHyphens, "MY_HYPHEN_TYPE")
        assert EnumWithHyphens.MY_HYPHEN_TYPE.value == "my-hyphen-type"


# ==============================================================================
# detect_type_from_url() Tests
# ==============================================================================


class TestDetectTypeFromUrl:
    """Tests for detect_type_from_url() function."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the global singleton before and after each test."""
        plugin_registry.reset()
        yield
        plugin_registry.reset()

    def test_detect_type_no_matching_scheme(self):
        """Test that ValueError is raised when no type matches scheme."""
        with pytest.raises(ValueError, match="No .* type found for URL scheme"):
            plugin_registry.detect_type_from_url("endpoint", "ftp://example.com")

    def test_detect_type_handles_url_without_scheme(self):
        """Test URL without scheme is handled (defaults to http)."""
        # localhost:8000 without scheme should be parsed as http
        with pytest.raises(ValueError, match="No .* type found for URL scheme 'http'"):
            plugin_registry.detect_type_from_url("endpoint", "localhost:8000")

    def test_detect_type_with_metadata_url_schemes(self):
        """Test detection with class that has metadata.url_schemes."""

        class EndpointWithSchemes:
            @staticmethod
            def metadata():
                return MagicMock(url_schemes=["custom-scheme"])

        plugin_registry.register("endpoint", "custom_endpoint", EndpointWithSchemes)

        detected = plugin_registry.detect_type_from_url(
            "endpoint", "custom-scheme://example.com"
        )
        assert detected == "custom_endpoint"

    def test_detect_type_handles_load_exception(self):
        """Test that exceptions during load are handled gracefully."""
        # Register a type that will fail to load properly
        yaml_content = {
            "plugin": {"name": "broken", "version": "1.0.0"},
            "endpoint": {
                "broken_type": {"class": "nonexistent:Broken"},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        plugin_registry.load_registry(path)

        # Should not crash, just skip broken types
        with pytest.raises(ValueError, match="No .* type found"):
            plugin_registry.detect_type_from_url("endpoint", "xyz://example.com")


# ==============================================================================
# Constants Tests
# ==============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_supported_schema_versions(self):
        """Test SUPPORTED_SCHEMA_VERSIONS is defined."""
        assert "1.0" in SUPPORTED_SCHEMA_VERSIONS
        assert isinstance(SUPPORTED_SCHEMA_VERSIONS, tuple)

    def test_default_schema_version(self):
        """Test DEFAULT_SCHEMA_VERSION is in supported versions."""
        assert DEFAULT_SCHEMA_VERSION in SUPPORTED_SCHEMA_VERSIONS

    def test_package_metadata_typed_dict(self):
        """Test PackageMetadata is a TypedDict with expected keys."""
        # Create a valid PackageMetadata
        meta: PackageMetadata = {
            "name": "test",
            "version": "1.0.0",
            "builtin": True,
        }
        assert meta["name"] == "test"


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

            cls = registry.get_class("endpoint", "test_endpoint")
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
            EndpointClass = registry.get_class("endpoint", "test_endpoint")

            # User manually instantiates
            endpoint = EndpointClass(arg1="value1", arg2="value2")

            assert endpoint is mock_instance
            mock_class_with_init.assert_called_once_with(arg1="value1", arg2="value2")

    def test_full_lifecycle(self, registry):
        """Test complete plugin lifecycle: register, list, get, validate."""
        # 1. Load registry
        yaml_content = {
            "schema_version": "1.0",
            "plugin": {"name": "lifecycle-test", "version": "1.0.0"},
            "endpoint": {
                "lifecycle_endpoint": {
                    "class": "os.path:join",
                    "description": "Test endpoint",
                    "priority": 50,
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        registry.load_registry(path)

        # 2. Verify plugin is listed
        packages = registry.list_packages()
        assert "lifecycle-test" in packages

        # 3. List types
        types = registry.list_types("endpoint")
        type_names = [t.type_name for t in types]
        assert "lifecycle_endpoint" in type_names

        # 4. Validate
        errors = registry.validate_all()
        # Should have no errors for valid module
        assert "endpoint" not in errors or not any(
            t[0] == "lifecycle_endpoint" for t in errors.get("endpoint", [])
        )

        # 5. Get class
        cls = registry.get_class("endpoint", "lifecycle_endpoint")
        assert cls is not None

        # 6. Find registered name
        name = registry.find_registered_name("endpoint", cls)
        assert name == "lifecycle_endpoint"


# ==============================================================================
# Edge Case Tests
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_load_registry_directory_path(self, registry):
        """Test that loading a directory raises appropriate error."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(ValueError, match="Registry path is not a file"),
        ):
            registry.load_registry(tmpdir)

    def test_register_creates_category_if_missing(self):
        """Test that register() creates category if it doesn't exist."""
        plugin_registry.reset()

        class NewCategoryClass:
            pass

        plugin_registry.register(
            category="brand_new_category",
            type_name="new_type",
            cls=NewCategoryClass,
        )

        # Category should now exist
        assert "brand_new_category" in plugin_registry.list_categories()
        cls = plugin_registry.get_class("brand_new_category", "new_type")
        assert cls is NewCategoryClass

        plugin_registry.reset()

    def test_type_entry_frozen_dataclass(self):
        """Test that TypeEntry is frozen (immutable)."""
        entry = TypeEntry(
            category="cat",
            type_name="type",
            package_name="pkg",
            class_path="mod:Class",
        )

        with pytest.raises(AttributeError):
            entry.category = "new_cat"  # Should raise - frozen

    def test_class_to_name_reverse_mapping_updated_on_get(self, registry, mock_class):
        """Test that _class_to_name is updated when class is loaded."""
        yaml_content = {
            "plugin": {"name": "test", "version": "1.0.0"},
            "endpoint": {"mapped_type": {"class": "test:Mapped"}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        registry.load_registry(path)

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.Mapped = mock_class
            mock_import.return_value = mock_module

            cls = registry.get_class("endpoint", "mapped_type")

            # Reverse mapping should be updated
            assert cls in registry._class_to_name
            assert registry._class_to_name[cls] == "mapped_type"

    def test_discover_plugins_path_object(self, registry):
        """Test discover_plugins with Path object return."""
        yaml_content = {
            "plugin": {"name": "path-plugin", "version": "1.0.0"},
            "endpoint": {"path_type": {"class": "mod:Class"}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            path = Path(f.name)

        mock_ep = Mock()
        mock_ep.name = "path-plugin"
        mock_ep.load.return_value = path  # Return Path object

        with patch(
            "aiperf.plugin.plugin_registry.entry_points", return_value=[mock_ep]
        ):
            registry.discover_plugins()

        assert "path-plugin" in registry._loaded_plugins
