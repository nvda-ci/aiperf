#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate plugin_registry.pyi type stub from categories.yaml.

This script reads the plugin category definitions from categories.yaml and generates
a complete type stub file with get_class() overloads and all public API stubs.

Usage:
    python tools/generate_plugin_overloads.py [--check] [--output FILE]

Options:
    --check     Check if the current stubs are up-to-date (exit 1 if not)
    --output    Write to specified file instead of default (plugin_registry.pyi)
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# Paths relative to repo root
REPO_ROOT = Path(__file__).parent.parent
CATEGORIES_YAML = REPO_ROOT / "src" / "aiperf" / "plugin" / "categories.yaml"
PLUGIN_REGISTRY_PYI = REPO_ROOT / "src" / "aiperf" / "plugin" / "plugin_registry.pyi"


def load_categories(yaml_path: Path) -> dict:
    """Load categories from YAML file."""
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Filter out non-category keys
    return {
        k: v
        for k, v in data.items()
        if k not in ("schema_version",) and isinstance(v, dict)
    }


def parse_class_path(class_path: str) -> tuple[str, str]:
    """Parse 'module.path:ClassName' into (module_path, class_name)."""
    if ":" not in class_path:
        raise ValueError(
            f"Invalid class path format: {class_path} (expected 'module:Class')"
        )
    module_path, class_name = class_path.split(":", 1)
    return module_path, class_name


def generate_imports(categories: dict) -> str:
    """Generate import statements grouped by module."""
    # Group protocols by their import module
    imports_by_module: dict[str, list[str]] = defaultdict(list)

    for category_data in categories.values():
        protocol_path = category_data.get("protocol", "")
        if not protocol_path:
            continue

        module_path, class_name = parse_class_path(protocol_path)
        imports_by_module[module_path].append(class_name)

    # Sort and deduplicate
    for module in imports_by_module:
        imports_by_module[module] = sorted(set(imports_by_module[module]))

    # Generate import lines
    lines = []

    # Standard library imports
    lines.append("from pathlib import Path")
    lines.append("from typing import Any, Final, Literal, TypedDict, overload")
    lines.append("")

    # Protocol imports grouped by module, sorted
    for module_path in sorted(imports_by_module.keys()):
        class_names = imports_by_module[module_path]
        if len(class_names) == 1:
            lines.append(f"from {module_path} import {class_names[0]}")
        else:
            lines.append(f"from {module_path} import (")
            for name in class_names:
                lines.append(f"    {name},")
            lines.append(")")

    # Always import PluginCategory from enums
    lines.append("from aiperf.plugin.enums import PluginCategory")

    return "\n".join(lines)


def generate_overloads(categories: dict) -> str:
    """Generate @overload decorated function stubs."""
    lines = []

    for category_name, category_data in categories.items():
        protocol_path = category_data.get("protocol", "")
        description = (
            category_data.get("description", "").strip().split("\n")[0]
        )  # First line only

        # Convert category_name to enum member (e.g., timing_strategy -> TIMING_STRATEGY)
        enum_member = category_name.upper()

        # Parse protocol
        if protocol_path:
            _, class_name = parse_class_path(protocol_path)
            return_type = f"type[{class_name}]"
        else:
            return_type = "type"

        # Generate comment from first line of description
        if description:
            lines.append(f"# {description}")

        # Generate overload
        lines.append("@overload")
        lines.append("def get_class(")
        lines.append(
            f"    category: Literal[PluginCategory.{enum_member}], name_or_class_path: str"
        )
        lines.append(f") -> {return_type}: ...")
        lines.append("")

    # Add fallback overload
    lines.append("# Fallback for unknown categories")
    lines.append("@overload")
    lines.append(
        "def get_class(category: PluginCategory, name_or_class_path: str) -> type: ..."
    )

    return "\n".join(lines)


def generate_full_stub(categories: dict) -> str:
    """Generate the complete .pyi stub file."""
    lines = [
        "# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        '"""',
        "Type stubs for plugin_registry module.",
        "",
        "This file is AUTO-GENERATED from categories.yaml.",
        "Run `python tools/generate_plugin_overloads.py` to regenerate.",
        "",
        "These stubs provide IDE autocomplete and type checking for the",
        "get_class() function with category-specific return types.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        generate_imports(categories),
        "",
        "# ==============================================================================",
        "# Constants",
        "# ==============================================================================",
        "",
        "SUPPORTED_SCHEMA_VERSIONS: Final[tuple[str, ...]]",
        "DEFAULT_SCHEMA_VERSION: Final[str]",
        "DEFAULT_ENTRY_POINT_GROUP: Final[str]",
        "",
        "# ==============================================================================",
        "# Type Definitions",
        "# ==============================================================================",
        "",
        "class PackageMetadata(TypedDict, total=False):",
        '    """Package metadata from YAML manifest."""',
        "",
        "    name: str",
        "    version: str",
        "    description: str",
        "    author: str",
        "    license: str",
        "    homepage: str",
        "    builtin: bool",
        "",
        "",
        "class ManifestData(TypedDict, total=False):",
        '    """YAML manifest structure."""',
        "",
        "    schema_version: str",
        "    plugin: dict[str, Any]",
        "",
        "",
        "# ==============================================================================",
        "# Custom Exceptions",
        "# ==============================================================================",
        "",
        "",
        "class PluginError(Exception):",
        '    """Base exception for plugin system errors."""',
        "",
        "",
        "class TypeNotFoundError(PluginError):",
        '    """Type not found in category. Includes available types in error message."""',
        "",
        "    category: str",
        "    type_name: str",
        "    available: list[str]",
        "",
        "    def __init__(self, category: str, type_name: str, available: list[str]) -> None: ...",
        "",
        "",
        "# ==============================================================================",
        "# TypeEntry Dataclass",
        "# ==============================================================================",
        "",
        "",
        "class TypeEntry:",
        '    """Lazy-loading type entry with metadata. Call load() to import the class."""',
        "",
        "    category: str",
        "    type_name: str",
        "    package_name: str",
        "    class_path: str",
        "    priority: int",
        "    description: str",
        "    metadata: PackageMetadata",
        "    loaded_class: type | None",
        "    is_builtin: bool",
        "",
        "    def __init__(",
        "        self,",
        "        category: str,",
        "        type_name: str,",
        "        package_name: str,",
        "        class_path: str,",
        "        priority: int = ...,",
        "        description: str = ...,",
        "        metadata: PackageMetadata = ...,",
        "        loaded_class: type | None = ...,",
        "        is_builtin: bool = ...,",
        "    ) -> None: ...",
        "    def load(self) -> type: ...",
        "    def validate(self, check_class: bool = ...) -> tuple[bool, str | None]: ...",
        "",
        "",
        "# ==============================================================================",
        "# PluginRegistry Class",
        "# ==============================================================================",
        "",
        "",
        "class PluginRegistry:",
        '    """Plugin registry singleton with discovery and lazy loading."""',
        "",
        "    def __init__(self) -> None: ...",
        "    def load_registry(self, registry_path: Path | str) -> None: ...",
        "    def discover_plugins(self, entry_point_group: str = ...) -> None: ...",
        "    def get_class(self, category: str, name_or_class_path: str) -> type: ...",
        "    def list_types(self, category: str) -> list[TypeEntry]: ...",
        "    def validate_all(",
        "        self, check_class: bool = ...",
        "    ) -> dict[str, list[tuple[str, str]]]: ...",
        "    def list_packages(self, builtin_only: bool = ...) -> list[str]: ...",
        "    def find_registered_name(self, category: str, cls: type) -> str | None: ...",
        "",
        "",
        "# ==============================================================================",
        "# get_class() Overloads",
        "# ==============================================================================",
        "",
        generate_overloads(categories),
        "",
        "",
        "# ==============================================================================",
        "# Module-Level Functions",
        "# ==============================================================================",
        "",
        "",
        "def get_class(category: PluginCategory, name_or_class_path: str) -> type:",
        '    """Get type class by name or class path. See PluginRegistry.get_class()."""',
        "    ...",
        "",
        "",
        "def list_types(category: PluginCategory) -> list[TypeEntry]:",
        '    """List all TypeEntry objects for a category. See PluginRegistry.list_types()."""',
        "    ...",
        "",
        "",
        "def validate_all(check_class: bool = ...) -> dict[str, list[tuple[str, str]]]:",
        '    """Validate all registered types without loading. See PluginRegistry.validate_all()."""',
        "    ...",
        "",
        "",
        "def find_registered_name(category: str, cls: type) -> str | None:",
        '    """Reverse lookup: find registered name for a class. See PluginRegistry.find_registered_name()."""',
        "    ...",
        "",
        "",
        "def load_registry(registry_path: str | Path) -> None:",
        '    """Load registry from YAML file. See PluginRegistry.load_registry()."""',
        "    ...",
        "",
        "",
        "def list_packages(builtin_only: bool = ...) -> list[str]:",
        '    """List all loaded plugin package names. See PluginRegistry.list_packages()."""',
        "    ...",
        "",
        "",
        "def get_package_metadata(package_name: str) -> PackageMetadata:",
        '    """Get metadata for a loaded plugin package. Raises KeyError if not found."""',
        "    ...",
        "",
        "",
        "def list_categories() -> list[str]:",
        '    """List all registered category names (sorted alphabetically)."""',
        "    ...",
        "",
        "",
        "def reset() -> None:",
        '    """Reset registry to empty state (for testing)."""',
        "    ...",
        "",
        "",
        "def register(",
        "    category: str,",
        "    type_name: str,",
        "    cls: type,",
        "    *,",
        "    priority: int = ...,",
        "    is_builtin: bool = ...,",
        ") -> None:",
        '    """Register a class programmatically (for dynamic classes or test overrides)."""',
        "    ...",
        "",
        "",
        "def create_enum(category: str, enum_name: str) -> type:",
        '    """Create an ExtensibleStrEnum from registered types in a category."""',
        "    ...",
        "",
        "",
        "def detect_type_from_url(category: str, url: str) -> str:",
        '    """Detect the type from a URL by matching URL scheme to type metadata."""',
        "    ...",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate plugin_registry.pyi stub from categories.yaml"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check if stubs are up-to-date"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PLUGIN_REGISTRY_PYI,
        help="Output file (default: src/aiperf/plugin/plugin_registry.pyi)",
    )
    args = parser.parse_args()

    # Load categories
    if not CATEGORIES_YAML.exists():
        print(f"Error: {CATEGORIES_YAML} not found", file=sys.stderr)
        return 1

    categories = load_categories(CATEGORIES_YAML)

    # Generate the stub
    generated = generate_full_stub(categories)

    if args.check:
        # Compare with current content
        if not args.output.exists():
            print(f"Error: {args.output} not found", file=sys.stderr)
            return 1

        current = args.output.read_text(encoding="utf-8")

        # Normalize whitespace for comparison
        current_normalized = "\n".join(
            line.rstrip() for line in current.strip().split("\n")
        )
        generated_normalized = "\n".join(
            line.rstrip() for line in generated.strip().split("\n")
        )

        if current_normalized != generated_normalized:
            print(
                "Stubs are out of date. Run: python tools/generate_plugin_overloads.py"
            )
            return 1

        print("Stubs are up-to-date")
        return 0

    # Write the stub file
    args.output.write_text(generated, encoding="utf-8")
    print(f"Generated {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
