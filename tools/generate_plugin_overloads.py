#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate get_class() overloads directly in plugins.py from categories.yaml.

This script reads the plugin category definitions from categories.yaml and injects
type overloads directly into plugins.py between marker comments.

Features:
- Literal string category names for autocomplete (e.g., "endpoint")
- Literal string plugin names for autocomplete (e.g., "chat")
- Enum category support (e.g., PluginType.ENDPOINT)
- Type-safe return types based on category

Usage:
    python tools/generate_plugin_overloads.py [--check]

Options:
    --check     Check if the overloads are up-to-date (exit 1 if not)
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# =============================================================================
# Paths
# =============================================================================

REPO_ROOT = Path(__file__).parent.parent
PLUGIN_DIR = REPO_ROOT / "src" / "aiperf" / "plugin"
CATEGORIES_YAML = PLUGIN_DIR / "categories.yaml"
PLUGINS_PY = PLUGIN_DIR / "plugins.py"

# Markers in plugins.py
IMPORTS_START = "    # <generated-imports>"
IMPORTS_END = "    # </generated-imports>"
OVERLOADS_START = "# <generated-overloads>"
OVERLOADS_END = "# </generated-overloads>"


# =============================================================================
# YAML Loading
# =============================================================================


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


def load_plugins_from_registry() -> dict[str, list[str]]:
    """Load plugin names from the runtime registry.

    Returns:
        Dict mapping category name to list of plugin names.
    """
    # Add src to path for imports
    sys.path.insert(0, str(REPO_ROOT / "src"))

    try:
        from aiperf.plugin import plugins

        result = {}
        for category in plugins.list_categories():
            types = plugins.list_types(category)
            if types:
                result[category] = sorted(t.name for t in types)
        return result
    except Exception as e:
        print(f"Warning: Could not load plugins from registry: {e}", file=sys.stderr)
        return {}


def parse_class_path(class_path: str) -> tuple[str, str]:
    """Parse 'module.path:ClassName' into (module_path, class_name)."""
    if ":" not in class_path:
        raise ValueError(
            f"Invalid class path format: {class_path} (expected 'module:Class')"
        )
    module_path, class_name = class_path.split(":", 1)
    return module_path, class_name


# =============================================================================
# Code Generation Helpers
# =============================================================================


def get_literal_name(category_name: str) -> str:
    """Get the Literal type alias name for a category.

    Must match the naming convention in generate_plugin_enums.py.
    Example: endpoint -> EndpointName
    """
    return "".join(word.title() for word in category_name.split("_")) + "Name"


def generate_imports(
    categories: dict, plugins_by_category: dict[str, list[str]]
) -> str:
    """Generate import statements for protocols (under TYPE_CHECKING)."""
    # Group protocols by their import module
    imports_by_module: dict[str, list[str]] = defaultdict(list)

    # Always include Literal (stdlib)
    imports_by_module["typing"].append("Literal")

    # Import plugin name Literals from enums
    literal_names = []
    for category_name in sorted(categories.keys()):
        if plugins_by_category.get(category_name):
            literal_names.append(get_literal_name(category_name))
    if literal_names:
        imports_by_module["aiperf.plugin.enums"].extend(literal_names)

    for category_data in categories.values():
        protocol_path = category_data.get("protocol", "")
        if not protocol_path:
            continue

        module_path, class_name = parse_class_path(protocol_path)
        imports_by_module[module_path].append(class_name)

    # Sort and deduplicate
    for module in imports_by_module:
        imports_by_module[module] = sorted(set(imports_by_module[module]))

    # Generate import lines - stdlib first, then third-party/local sorted
    lines = []

    # Stdlib imports first
    stdlib_modules = ["typing"]
    for module_path in stdlib_modules:
        if module_path in imports_by_module:
            class_names = imports_by_module.pop(module_path)
            lines.append(f"    from {module_path} import {', '.join(class_names)}")

    if lines:
        lines.append("")

    # Local imports sorted by module path
    for module_path in sorted(imports_by_module.keys()):
        class_names = imports_by_module[module_path]
        if len(class_names) == 1:
            lines.append(f"    from {module_path} import {class_names[0]}")
        else:
            lines.append(f"    from {module_path} import (")
            for name in class_names:
                lines.append(f"        {name},")
            lines.append("    )")

    return "\n".join(lines)


def generate_overloads(
    categories: dict, plugins_by_category: dict[str, list[str]]
) -> str:
    """Generate @overload decorated function stubs."""
    lines = []

    # Generate overloads for each category
    for category_name, category_data in categories.items():
        protocol_path = category_data.get("protocol", "")
        # Convert category_name to enum member (e.g., timing_strategy -> TIMING_STRATEGY)
        enum_member = category_name.upper()

        # Parse protocol
        if protocol_path:
            _, class_name = parse_class_path(protocol_path)
            return_type = f"type[{class_name}]"
        else:
            return_type = "type"

        # Get plugin names for this category
        plugin_names = plugins_by_category.get(category_name, [])

        # Determine the name parameter type
        if plugin_names:
            # Use Literal from enums.pyi for autocomplete + str for extensibility
            literal_name = get_literal_name(category_name)
            name_type = f"{literal_name} | str"
        else:
            name_type = "str"

        # Generate TWO overloads per category:
        # 1. Literal string category (e.g., "endpoint")
        # 2. Enum category (e.g., PluginType.ENDPOINT)

        # Overload 1: Literal string category
        lines.append("@overload")
        lines.append("def get_class(")
        lines.append(f'    category: Literal["{category_name}"],')
        lines.append(f"    name_or_class_path: {name_type},")
        lines.append(f") -> {return_type}: ...")
        lines.append("")
        lines.append("")

        # Overload 2: Enum category
        lines.append("@overload")
        lines.append("def get_class(")
        lines.append(f"    category: Literal[PluginType.{enum_member}],")
        lines.append(f"    name_or_class_path: {name_type},")
        lines.append(f") -> {return_type}: ...")
        lines.append("")
        lines.append("")

    # Add fallback overloads
    lines.append("# Fallback for unknown categories")
    lines.append("@overload")
    lines.append(
        "def get_class(category: PluginType, name_or_class_path: str) -> type: ..."
    )
    lines.append("")
    lines.append("")
    lines.append("@overload")
    lines.append("def get_class(category: str, name_or_class_path: str) -> type: ...")
    lines.append("")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# File Manipulation
# =============================================================================


def replace_between_markers(
    content: str, start_marker: str, end_marker: str, replacement: str
) -> str:
    """Replace content between markers (exclusive of markers)."""
    # Match markers with optional content between them (handles empty or populated)
    pattern = re.compile(
        rf"({re.escape(start_marker)})\n(.*?)({re.escape(end_marker)})",
        re.DOTALL,
    )

    if not pattern.search(content):
        raise ValueError(f"Markers not found: {start_marker} ... {end_marker}")

    # If replacement is empty, just keep markers with newline between
    if not replacement.strip():
        return pattern.sub(r"\1\n\3", content)

    return pattern.sub(rf"\1\n{replacement}\n\3", content)


def inject_generated_code(
    content: str, categories: dict, plugins_by_category: dict[str, list[str]]
) -> str:
    """Inject generated imports and overloads into plugins.py content."""
    # Generate sections
    imports_code = generate_imports(categories, plugins_by_category)
    overloads_code = generate_overloads(categories, plugins_by_category)

    # Replace imports section
    content = replace_between_markers(content, IMPORTS_START, IMPORTS_END, imports_code)

    # Replace overloads section
    content = replace_between_markers(
        content, OVERLOADS_START, OVERLOADS_END, overloads_code
    )

    return content


# =============================================================================
# Main Entry Point
# =============================================================================


def normalize_content(content: str) -> str:
    """Normalize content for comparison."""
    return "\n".join(line.rstrip() for line in content.strip().split("\n"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate get_class() overloads in plugins.py from categories.yaml"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check if overloads are up-to-date"
    )
    args = parser.parse_args()

    # Load categories
    if not CATEGORIES_YAML.exists():
        print(f"Error: {CATEGORIES_YAML} not found", file=sys.stderr)
        return 1

    if not PLUGINS_PY.exists():
        print(f"Error: {PLUGINS_PY} not found", file=sys.stderr)
        return 1

    categories = load_categories(CATEGORIES_YAML)
    plugins_by_category = load_plugins_from_registry()
    current_content = PLUGINS_PY.read_text(encoding="utf-8")

    # Generate the updated content
    try:
        generated_content = inject_generated_code(
            current_content, categories, plugins_by_category
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Check if content changed
    is_up_to_date = normalize_content(current_content) == normalize_content(
        generated_content
    )

    if args.check:
        if not is_up_to_date:
            print(
                "Overloads are out of date. Run: python tools/generate_plugin_overloads.py"
            )
            return 1

        print("Overloads are up-to-date")
        return 0

    # Only write if content changed
    if is_up_to_date:
        print("Overloads are up-to-date, no changes needed")
        return 0

    PLUGINS_PY.write_text(generated_content, encoding="utf-8")
    print(f"Updated {PLUGINS_PY}")
    print("\nGenerated overloads now support:")
    print('  - Literal string categories: plugins.get_class("endpoint", ...)')
    print("  - Enum categories: plugins.get_class(PluginType.ENDPOINT, ...)")
    print('  - Literal plugin names: plugins.get_class("endpoint", "chat")')
    print('  - 3rd party names (| str): plugins.get_class("endpoint", "custom")')

    return 0


if __name__ == "__main__":
    sys.exit(main())
