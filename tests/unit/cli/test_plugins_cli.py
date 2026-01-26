# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for plugin management CLI commands."""

from unittest.mock import MagicMock, patch

import pytest

from aiperf.cli_commands.plugins_cli import (
    ensure_registry_loaded,
    get_all_categories,
    show_category_types,
    show_overview,
    show_packages,
    show_type_details,
)
from aiperf.cli_commands.plugins_cli import (
    plugins as plugins_cmd,
)
from aiperf.plugin import plugins as plugin_registry


@pytest.fixture
def mock_console() -> MagicMock:
    """Mock console for testing output."""
    with patch("aiperf.cli_commands.plugins_cli.console") as mock:
        yield mock


@pytest.fixture(autouse=True)
def setup_registry() -> None:
    """Setup registry for each test."""
    plugin_registry.reset()
    _ = plugin_registry.list_categories()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_all_categories(self) -> None:
        """Test getting all category types."""
        categories = get_all_categories()
        assert isinstance(categories, list)
        assert len(categories) > 0

    def test_ensure_registry_loaded(self) -> None:
        """Test ensuring registry is loaded."""
        plugin_registry.reset()
        ensure_registry_loaded()
        assert len(get_all_categories()) > 0


class TestShowOverview:
    """Tests for overview display."""

    def test_show_overview(self, mock_console: MagicMock) -> None:
        """Test showing category overview."""
        show_overview()
        assert mock_console.print.call_count >= 2


class TestShowCategoryTypes:
    """Tests for category types display."""

    def test_valid_category(self, mock_console: MagicMock) -> None:
        """Test listing types for valid category."""
        categories = get_all_categories()
        if categories:
            show_category_types(categories[0])
            assert mock_console.print.call_count >= 2

    def test_invalid_category(self, mock_console: MagicMock) -> None:
        """Test listing types for invalid category."""
        show_category_types("nonexistent")
        calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("unknown" in c.lower() for c in calls)


class TestShowTypeDetails:
    """Tests for type details display."""

    def test_valid_type(self, mock_console: MagicMock) -> None:
        """Test showing details for valid type."""
        categories = get_all_categories()
        if categories:
            types = plugin_registry.list_types(categories[0])
            if types:
                show_type_details(categories[0], types[0].name)
                assert mock_console.print.call_count >= 1

    def test_invalid_type(self, mock_console: MagicMock) -> None:
        """Test showing details for invalid type."""
        categories = get_all_categories()
        if categories:
            show_type_details(categories[0], "nonexistent")
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("not found" in c.lower() for c in calls)


class TestShowPackages:
    """Tests for package listing."""

    def test_show_packages(self, mock_console: MagicMock) -> None:
        """Test listing packages."""
        show_packages()
        assert mock_console.print.call_count >= 1


class TestMainCommand:
    """Tests for main plugins command."""

    def test_no_args_shows_overview(self, mock_console: MagicMock) -> None:
        """Test that no args shows overview."""
        plugins_cmd(category=None, name=None, packages=False, validate=False)
        calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("categor" in c.lower() for c in calls)

    def test_category_arg_shows_types(self, mock_console: MagicMock) -> None:
        """Test that category arg shows types."""
        categories = get_all_categories()
        if categories:
            plugins_cmd(
                category=categories[0], name=None, packages=False, validate=False
            )
            assert mock_console.print.call_count >= 2

    def test_category_and_type_shows_details(self, mock_console: MagicMock) -> None:
        """Test that category+type shows details."""
        categories = get_all_categories()
        if categories:
            types = plugin_registry.list_types(categories[0])
            if types:
                plugins_cmd(
                    category=categories[0],
                    name=types[0].name,
                    packages=False,
                    validate=False,
                )
                assert mock_console.print.call_count >= 1

    def test_packages_flag(self, mock_console: MagicMock) -> None:
        """Test --packages flag prints something."""
        plugins_cmd(category=None, name=None, packages=True, validate=False)
        # Should print at least once (either table or "No plugins found")
        assert mock_console.print.call_count >= 1


class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self, mock_console: MagicMock) -> None:
        """Test exploring plugins -> category -> type."""
        # 1. Show overview
        plugins_cmd(category=None, name=None, packages=False, validate=False)
        assert mock_console.print.call_count >= 2
        mock_console.reset_mock()

        # 2. Pick a category
        categories = get_all_categories()
        if categories:
            plugins_cmd(
                category=categories[0], name=None, packages=False, validate=False
            )
            assert mock_console.print.call_count >= 2
            mock_console.reset_mock()

            # 3. Pick a type
            types = plugin_registry.list_types(categories[0])
            if types:
                plugins_cmd(
                    category=categories[0],
                    name=types[0].name,
                    packages=False,
                    validate=False,
                )
                assert mock_console.print.call_count >= 1
