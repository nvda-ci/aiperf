#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Generate enums.py and enums.pyi for the plugin system.

This script generates:
1. enums.py - Dynamic enum definitions using plugins.create_enum() (from YAML)
2. enums.pyi - Type stubs for IDE autocomplete (from runtime plugin registry)

The enums.pyi file is NOT committed to the repository (.gitignore). Run this generator
after cloning or when adding 3rd-party plugins.

Section groupings in enums.py are extracted from YAML comments in categories.yaml.
Section names are transformed by replacing "Categories/Category" with "Types".

Usage:
    python tools/generate_plugin_enums.py [--py-only] [--pyi-only]

Options:
    --py-only   Only generate enums.py
    --pyi-only  Only generate enums.pyi
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from ruamel.yaml import YAML

# Use round-trip mode to preserve comments
yaml_rt = YAML(typ="rt")
yaml_safe = YAML(typ="safe")

# Paths
REPO_ROOT = Path(__file__).parent.parent
PLUGIN_DIR = REPO_ROOT / "src" / "aiperf" / "plugin"
CATEGORIES_YAML = PLUGIN_DIR / "categories.yaml"
ENUMS_PY = PLUGIN_DIR / "enums.py"
ENUMS_PYI = PLUGIN_DIR / "enums.pyi"

# Metadata keys to skip in categories.yaml
METADATA_KEYS = {"schema_version", "package"}


def load_categories() -> dict:
    """Load the plugin categories from categories.yaml (safe mode)."""
    if not CATEGORIES_YAML.exists():
        raise FileNotFoundError(f"Categories file not found: {CATEGORIES_YAML}")

    with open(CATEGORIES_YAML) as f:
        return yaml_safe.load(f)


def load_categories_with_sections() -> tuple[dict, dict[str, str]]:
    """Load categories and extract section groupings from YAML comments.

    Parses the raw YAML file to find section header comments (# === Section ===)
    that precede each category definition.

    Returns:
        Tuple of (categories_dict, category_to_section_map)
        Section names have "Categories/Category" replaced with "Types".
    """
    if not CATEGORIES_YAML.exists():
        raise FileNotFoundError(f"Categories file not found: {CATEGORIES_YAML}")

    # Load YAML data
    with open(CATEGORIES_YAML) as f:
        data = yaml_safe.load(f)

    # Parse raw file to extract section comments before each top-level key
    raw_content = CATEGORIES_YAML.read_text()

    # Find all section headers and the category keys that follow them
    # Pattern: section comment block followed by a top-level key (no leading whitespace)
    section_pattern = re.compile(
        r"#\s*=+\s*\n"  # Opening ===
        r"#\s+(.+?)\s*\n"  # Section name
        r"#\s*=+\s*\n"  # Closing ===
        r"(?:.*?\n)*?"  # Any content until...
        r"^(\w[\w_-]*):",  # Top-level key (no indent)
        re.MULTILINE,
    )

    category_to_section: dict[str, str] = {}
    current_section: str | None = None

    # Find all section headers and their following keys
    for match in section_pattern.finditer(raw_content):
        section_name = match.group(1).strip()

        # Replace "Categories" or "Category" with "Types"
        section_name = re.sub(r"\s*Categor(y|ies)\s*$", " Types", section_name)
        current_section = section_name

    # Simpler approach: scan line by line
    category_to_section = {}
    current_section = None
    lines = raw_content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for section header (3-line pattern)
        if re.match(r"^#\s*=+\s*$", line) and i + 2 < len(lines):
            name_line = lines[i + 1]
            close_line = lines[i + 2]
            if re.match(r"^#\s+\S", name_line) and re.match(r"^#\s*=+\s*$", close_line):
                # Extract section name
                section_name = re.sub(r"^#\s+", "", name_line).strip()
                # Replace "Categories" or "Category" with "Types"
                section_name = re.sub(r"\s*Categor(y|ies)\s*$", " Types", section_name)
                current_section = section_name
                i += 3
                continue

        # Check for top-level key (category definition)
        key_match = re.match(r"^(\w[\w_-]*):\s*$", line)
        if key_match and current_section:
            key = key_match.group(1)
            if key not in METADATA_KEYS:
                category_to_section[key] = current_section

        i += 1

    return dict(data), category_to_section


def load_plugins(*, builtin_only: bool = True) -> dict:
    """Load plugins from the runtime registry.

    Args:
        builtin_only: If True, only include built-in plugins (aiperf*).

    Returns:
        Dict mapping category names to dicts of plugin names to plugin specs.
    """
    from aiperf.plugin import plugins

    merged: dict = {}

    for category in plugins.list_categories():
        types = plugins.list_types(category)
        if builtin_only:
            types = [t for t in types if t.is_builtin]
        if types:
            merged[category] = {t.name: {"class": t.class_path} for t in types}

    return merged


def type_to_member_name(name: str) -> str:
    """Convert type name to enum member name (UPPER_CASE)."""
    return name.replace("-", "_").upper()


def get_description(type_spec: str | dict) -> str | None:
    """Extract description from type spec."""
    if isinstance(type_spec, dict):
        return type_spec.get("description")
    return None


def get_category_names(categories: dict, preserve_order: bool = False) -> list[str]:
    """Get list of category names excluding metadata keys.

    Args:
        categories: The categories dict
        preserve_order: If True, preserve YAML order. If False, sort alphabetically.
    """
    names = [k for k in categories if k not in METADATA_KEYS]
    return names if preserve_order else sorted(names)


def get_enum_names(categories: dict, plugins: dict) -> list[str]:
    """Get sorted list of all enum names."""
    category_names = get_category_names(categories)
    enum_names = ["PluginType"]
    for cat in category_names:
        cat_info = categories.get(cat, {})
        if isinstance(cat_info, dict) and cat_info.get("enum") and plugins.get(cat):
            enum_names.append(cat_info["enum"])
    return sorted(set(enum_names))


# =============================================================================
# enums.py Generation (Dynamic)
# =============================================================================


def generate_enums_py(categories: dict, plugins: dict) -> str:
    """Generate the content of enums.py with dynamic plugins.create_enum() calls.

    Args:
        categories: Category definitions from categories.yaml
        plugins: Plugin definitions from plugins.yaml
    """
    # Load categories with section info from comments
    _, category_to_section = load_categories_with_sections()

    enum_names = get_enum_names(categories, plugins)

    header = """\
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# AUTO-GENERATED FILE - DO NOT EDIT
# ============================================================================
# Generated by: tools/generate_plugin_enums.py
# Sources: Runtime plugin registry (built-in plugins only)
#
# To regenerate: make generate-plugin-enums
# ============================================================================
\"\"\"
Plugin Type Enums

Type enums for all plugin categories, generated dynamically from the plugin registry.
These enums work with pydantic validation, cyclopts CLI parsing, and support
compile-time checks (e.g., `if ui_type == UIType.DASHBOARD`).

New plugins registered at runtime automatically extend the appropriate enum.

Usage:
    from aiperf.plugin.enums import EndpointType, UIType
    # or
    from aiperf.plugin import EndpointType, UIType
\"\"\"

from typing import TYPE_CHECKING

from aiperf.common.enums import create_enum
from aiperf.plugin import plugins
"""
    lines = header.split("\n")

    # Generate PluginType with TYPE_CHECKING block
    lines.extend(_generate_plugin_category_dynamic())

    # Group categories by section (preserving order from YAML)
    current_section: str | None = None
    category_names = get_category_names(categories, preserve_order=True)

    for category in category_names:
        category_info = categories.get(category, {})
        if not isinstance(category_info, dict):
            continue

        enum_name = category_info.get("enum")
        if not enum_name:
            continue

        types = plugins.get(category, {})
        if not types:
            continue

        # Check if we need a new section header
        section = category_to_section.get(category)
        if section and section != current_section:
            section_header = f"""\
# ============================================================================
# {section}
# ============================================================================
"""
            lines.extend(section_header.split("\n"))
            current_section = section

        # Generate the dynamic enum assignment
        lines.extend(_generate_dynamic_enum(category, category_info, enum_name, types))

    # Add __all__ export
    lines.append("__all__ = [")
    for name in enum_names:
        lines.append(f'    "{name}",')
    lines.append("]")
    lines.append("")

    return "\n".join(lines)


def _generate_plugin_category_dynamic() -> list[str]:
    """Generate the PluginType with TYPE_CHECKING block."""
    code = """\
# ============================================================================
# Plugin Protocol Categories
# ============================================================================

if TYPE_CHECKING:
    # Import the enum from the stubs file
    from aiperf.plugin.enums import PluginType
else:
    # Create runtime enum with all plugin category names from the registry
    _all_plugin_categories = plugins.list_categories()
    PluginType = create_enum(
        "PluginType",
        {
            category.replace("-", "_").upper(): category
            for category in _all_plugin_categories
        },
    )
    \"\"\"
    Dynamic enum for plugin categories.

    Members are auto-generated from registered plugin categories in plugins.yaml.
    Example: PluginType.ENDPOINT, PluginType.UI, PluginType.TRANSPORT, etc.
    \"\"\"
"""
    return code.split("\n")


def _generate_dynamic_enum(
    category: str, category_info: dict, enum_name: str, types: dict
) -> list[str]:
    """Generate a dynamic enum assignment with plugins.create_enum()."""
    category_member = category.replace("-", "_").upper()
    category_display = category.replace("_", " ")

    # Build example string from type names
    type_names = sorted(types.keys())
    examples = [type_to_member_name(t) for t in type_names[:3]]
    example_str = ", ".join(f"{enum_name}.{e}" for e in examples)

    # Check line length - wrap if over 88 chars (ruff default)
    single_line = f'{enum_name} = plugins.create_enum(PluginType.{category_member}, "{enum_name}")'
    if len(single_line) <= 88:
        assignment = single_line
    else:
        assignment = f"""{enum_name} = plugins.create_enum(
    PluginType.{category_member}, "{enum_name}"
)"""

    code = f"""\
{assignment}
\"\"\"
Dynamic enum for {category_display} implementations.

Members are auto-generated from registered {category_display} plugins.
Example: {example_str}
\"\"\"
"""
    return code.split("\n")


# =============================================================================
# enums.pyi Generation (from loaded plugins at runtime)
# =============================================================================


def generate_enums_pyi() -> str:
    """Generate enums.pyi from actual loaded plugins module.

    This imports the real plugins module to get all registered plugins,
    including any 3rd-party plugins that have been loaded.
    Descriptions are obtained from TypeEntry objects in the registry.
    """
    # Import the actual plugins module
    from aiperf.plugin import plugins

    categories = load_categories()
    category_names = get_category_names(categories)

    header = """\
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# AUTO-GENERATED FILE - DO NOT EDIT
# ============================================================================
# This file is automatically generated by tools/generate_plugin_enums.py --runtime
# It includes all loaded plugins (built-in + 3rd-party) for IDE autocomplete.
#
# This file is NOT committed to the repository (.gitignore).
# To regenerate: python tools/generate_plugin_enums.py --runtime
# ============================================================================
\"\"\"
Type stubs for dynamically generated plugin enums.

This file includes all currently loaded plugins, including 3rd-party plugins.
\"\"\"

from typing import Literal

from aiperf.common.enums import ExtensibleStrEnum

class PluginType(ExtensibleStrEnum):
    \"\"\"
    Dynamic enum for plugin categories.

    Each category represents a type of plugin that can be registered
    and used within the AIPerf framework.
    \"\"\"
"""
    lines = header.split("\n")

    # Get all categories from the runtime registry
    runtime_categories = plugins.list_categories()
    for category in sorted(runtime_categories):
        member_name = category.replace("-", "_").upper()
        lines.append(f'    {member_name} = "{category}"')

        # Add description from categories.yaml if available
        category_info = categories.get(category, {})
        if isinstance(category_info, dict) and category_info.get("description"):
            desc = category_info["description"].strip()
            desc_lines = desc.split("\n")
            lines.append(f'    """{desc_lines[0].strip()}')
            for desc_line in desc_lines[1:]:
                lines.append(f"    {desc_line.rstrip()}")
            lines.append('    """')
        lines.append("")

    lines.append("")

    # Build list of enum names
    enum_names = ["PluginType"]

    # Generate enums for each category from runtime registry
    for category in category_names:
        category_info = categories.get(category, {})
        if not isinstance(category_info, dict):
            continue

        enum_name = category_info.get("enum")
        if not enum_name:
            continue

        # Get types from runtime registry
        try:
            type_entries = plugins.list_types(category)
        except Exception:
            type_entries = []

        if not type_entries:
            continue

        enum_names.append(enum_name)
        lines.append(f"class {enum_name}(ExtensibleStrEnum):")

        description = category_info.get("description", "").strip()
        if description:
            lines.append('    """')
            for desc_line in description.split("\n"):
                lines.append(f"    {desc_line.rstrip()}")
            lines.append('    """')
        else:
            lines.append(f'    """Dynamic enum for {category} plugin types."""')
        lines.append("")

        # Add enum members with descriptions from TypeEntry
        for entry in sorted(type_entries, key=lambda e: e.name):
            member_name = type_to_member_name(entry.name)
            lines.append(f'    {member_name} = "{entry.name}"')
            if entry.description:
                desc_lines = entry.description.strip().split("\n")
                if len(desc_lines) == 1:
                    lines.append(f'    """{desc_lines[0]}"""')
                else:
                    lines.append(f'    """{desc_lines[0]}')
                    for desc_line in desc_lines[1:]:
                        lines.append(f"    {desc_line.rstrip()}")
                    lines.append('    """')
            lines.append("")

        lines.append("")

    # Generate Literal type aliases for plugin names (for autocomplete)
    lines.append(
        "# ============================================================================"
    )
    lines.append("# Plugin Name Literals (for autocomplete)")
    lines.append(
        "# ============================================================================"
    )
    lines.append("")

    literal_names = []
    for category in category_names:
        category_info = categories.get(category, {})
        if not isinstance(category_info, dict):
            continue

        # Get types from runtime registry
        try:
            type_entries = plugins.list_types(category)
        except Exception:
            type_entries = []

        if not type_entries:
            continue

        # Create Literal type alias name: endpoint -> EndpointName
        literal_name = "".join(word.title() for word in category.split("_")) + "Name"
        literal_names.append(literal_name)

        # Get sorted plugin names
        plugin_names = sorted(e.name for e in type_entries)

        # Format the literal
        if len(plugin_names) <= 3:
            values = ", ".join(f'"{n}"' for n in plugin_names)
            lines.append(f"{literal_name} = Literal[{values}]")
        else:
            lines.append(f"{literal_name} = Literal[")
            for name in plugin_names:
                lines.append(f'    "{name}",')
            lines.append("]")
        lines.append("")

    # Add __all__ export
    all_exports = sorted(set(enum_names + literal_names))
    lines.append("__all__ = [")
    for name in all_exports:
        lines.append(f'    "{name}",')
    lines.append("]")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def normalize_content(content: str) -> str:
    """Normalize content for comparison."""
    return "\n".join(line.rstrip() for line in content.strip().split("\n"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate enums.py and enums.pyi for the plugin system"
    )
    parser.add_argument("--py-only", action="store_true", help="Only generate enums.py")
    parser.add_argument(
        "--pyi-only", action="store_true", help="Only generate enums.pyi"
    )
    args = parser.parse_args()

    # Determine which files to generate
    generate_py = not args.pyi_only
    generate_pyi = not args.py_only

    # Load YAML data for enums.py generation
    try:
        categories = load_categories()
        plugins = load_plugins()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Generate enums.py content from YAML
    py_content = generate_enums_py(categories, plugins) if generate_py else None

    # Write files only if they changed
    files_written = 0

    if generate_py:
        current = ENUMS_PY.read_text(encoding="utf-8") if ENUMS_PY.exists() else ""
        if normalize_content(current) != normalize_content(py_content):
            ENUMS_PY.write_text(py_content, encoding="utf-8")
            print(f"Generated: {ENUMS_PY}")
            files_written += 1

    if generate_pyi:
        try:
            pyi_content = generate_enums_pyi()
            current = (
                ENUMS_PYI.read_text(encoding="utf-8") if ENUMS_PYI.exists() else ""
            )
            if normalize_content(current) != normalize_content(pyi_content):
                ENUMS_PYI.write_text(pyi_content, encoding="utf-8")
                print(f"Generated: {ENUMS_PYI}")
                files_written += 1
        except Exception as e:
            print(f"Error generating pyi stubs: {e}", file=sys.stderr)
            return 1

    if files_written == 0:
        print("Enum files are already up-to-date")
    else:
        print("IDEs will now provide autocomplete for plugin enum members!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
