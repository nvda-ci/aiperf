#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Generate enums.py and enums.pyi from categories.yaml and registry.yaml.

This script generates:
1. enums.py - Dynamic enum definitions using plugin_registry.create_enum()
2. enums.pyi - Static type stubs for IDE autocomplete

Section groupings in enums.py are extracted from YAML comments in categories.yaml.
Section names are transformed by replacing "Categories/Category" with "Types".

Usage:
    python tools/generate_plugin_enums.py [--check] [--py-only] [--pyi-only]

Options:
    --check     Check if files are up-to-date (exit 1 if not)
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
REGISTRY_YAML = REPO_ROOT / "src" / "aiperf" / "registry.yaml"
ENUMS_PY = PLUGIN_DIR / "enums.py"
ENUMS_PYI = PLUGIN_DIR / "enums.pyi"

# Metadata keys to skip in categories.yaml
METADATA_KEYS = {"schema_version"}


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


def load_registry() -> dict:
    """Load the plugin registry from registry.yaml."""
    if not REGISTRY_YAML.exists():
        raise FileNotFoundError(f"Registry file not found: {REGISTRY_YAML}")

    with open(REGISTRY_YAML) as f:
        return yaml_safe.load(f)


def type_to_member_name(type_name: str) -> str:
    """Convert type name to enum member name (UPPER_CASE)."""
    return type_name.replace("-", "_").upper()


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


def get_enum_names(categories: dict, registry: dict) -> list[str]:
    """Get sorted list of all enum names."""
    category_names = get_category_names(categories)
    enum_names = ["PluginCategory"]
    for cat in category_names:
        cat_info = categories.get(cat, {})
        if isinstance(cat_info, dict) and cat_info.get("enum") and registry.get(cat):
            enum_names.append(cat_info["enum"])
    return sorted(set(enum_names))


# =============================================================================
# enums.py Generation (Dynamic)
# =============================================================================


def generate_enums_py(categories: dict, registry: dict) -> str:
    """Generate the content of enums.py with dynamic plugin_registry.create_enum() calls.

    Args:
        categories: Category definitions from categories.yaml
        registry: Plugin registry from registry.yaml
    """
    # Load categories with section info from comments
    _, category_to_section = load_categories_with_sections()

    enum_names = get_enum_names(categories, registry)

    header = """\
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# AUTO-GENERATED FILE - DO NOT EDIT
# ============================================================================
# This file is automatically generated by tools/generate_plugin_enums.py
# Any manual changes will be overwritten on the next generation.
#
# To regenerate: python tools/generate_plugin_enums.py
# ============================================================================
\"\"\"
Dynamic Plugin Enums

All plugin-based type enums are generated here dynamically from the PluginRegistry.
This ensures:
- No circular imports (plugin module loads registry before any plugin modules)
- Enums loaded AFTER all types are registered
- Works with pydantic validation
- Works with cyclopts CLI parsing
- Supports hardcoded checks (e.g., ui_type == UIType.DASHBOARD)
- Can be extended at runtime if new plugins are registered

Import these types from aiperf.plugin.enums or aiperf.plugin.
\"\"\"

from typing import TYPE_CHECKING

from aiperf.common.enums import create_enum
from aiperf.plugin import plugin_registry
"""
    lines = header.split("\n")

    # Generate PluginCategory with TYPE_CHECKING block
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

        types = registry.get(category, {})
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
    """Generate the PluginCategory with TYPE_CHECKING block."""
    code = """\
# ============================================================================
# Plugin Protocol Categories
# ============================================================================

if TYPE_CHECKING:
    # Import the enum from the stubs file
    from aiperf.plugin.enums import PluginCategory
else:
    # Create runtime enum with all plugin category names from the registry
    _all_plugin_categories = plugin_registry.list_categories()
    PluginCategory = create_enum(
        "PluginCategory",
        {
            category.replace("-", "_").upper(): category
            for category in _all_plugin_categories
        },
    )
    \"\"\"
    Dynamic enum for plugin categories.

    Members are auto-generated from registered plugin categories in registry.yaml.
    Example: PluginCategory.ENDPOINT, PluginCategory.UI, PluginCategory.TRANSPORT, etc.
    \"\"\"
"""
    return code.split("\n")


def _generate_dynamic_enum(
    category: str, category_info: dict, enum_name: str, types: dict
) -> list[str]:
    """Generate a dynamic enum assignment with plugin_registry.create_enum()."""
    category_member = category.replace("-", "_").upper()
    category_display = category.replace("_", " ")

    # Build example string from type names
    type_names = sorted(types.keys())
    examples = [type_to_member_name(t) for t in type_names[:3]]
    example_str = ", ".join(f"{enum_name}.{e}" for e in examples)

    code = f"""\
{enum_name} = plugin_registry.create_enum(
    PluginCategory.{category_member}, "{enum_name}"
)
\"\"\"
Dynamic enum for {category_display} implementations.

Members are auto-generated from registered {category_display} plugins.
Example: {example_str}
\"\"\"
"""
    return code.split("\n")


# =============================================================================
# enums.pyi Generation
# =============================================================================


def generate_enums_pyi(categories: dict, registry: dict) -> str:
    """Generate the content of enums.pyi stub file.

    Args:
        categories: Category definitions from categories.yaml
        registry: Plugin registry from registry.yaml
    """
    category_names = get_category_names(categories)
    enum_names = get_enum_names(categories, registry)

    header = """\
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# AUTO-GENERATED FILE - DO NOT EDIT
# ============================================================================
# This file is automatically generated by tools/generate_plugin_enums.py
# Any manual changes will be overwritten on the next generation.
#
# To regenerate: python tools/generate_plugin_enums.py
# ============================================================================
\"\"\"
Type stubs for dynamically generated plugin enums.

These stubs provide IDE autocomplete and type checking for enum members
that are created dynamically at runtime from the plugin registry.
\"\"\"

from aiperf.common.enums import ExtensibleStrEnum

class PluginCategory(ExtensibleStrEnum):
    \"\"\"
    Dynamic enum for plugin categories.

    Each category represents a type of plugin that can be registered
    and used within the AIPerf framework.
    \"\"\"
"""
    lines = header.split("\n")

    for category in category_names:
        member_name = category.replace("-", "_").upper()
        lines.append(f'    {member_name} = "{category}"')

        # Add description as docstring if available
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

    # Generate enums for each category
    for category in category_names:
        category_info = categories.get(category, {})
        if not isinstance(category_info, dict):
            continue

        enum_name = category_info.get("enum")
        if not enum_name:
            continue

        types = registry.get(category, {})
        if not types:
            continue

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

        # Add enum members with descriptions
        for type_name in sorted(types.keys()):
            member_name = type_to_member_name(type_name)
            type_spec = types[type_name]
            description = get_description(type_spec)

            lines.append(f'    {member_name} = "{type_name}"')
            if description:
                desc_lines = description.strip().split("\n")
                if len(desc_lines) == 1:
                    lines.append(f'    """{desc_lines[0]}"""')
                else:
                    lines.append(f'    """{desc_lines[0]}')
                    for desc_line in desc_lines[1:]:
                        lines.append(f"    {desc_line.rstrip()}")
                    lines.append('    """')
            lines.append("")

        lines.append("")

    # Add __all__ export
    lines.append("__all__ = [")
    for name in enum_names:
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
        description="Generate enums.py and enums.pyi from categories.yaml and registry.yaml"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check if files are up-to-date"
    )
    parser.add_argument("--py-only", action="store_true", help="Only generate enums.py")
    parser.add_argument(
        "--pyi-only", action="store_true", help="Only generate enums.pyi"
    )
    args = parser.parse_args()

    # Determine which files to generate
    generate_py = not args.pyi_only
    generate_pyi = not args.py_only

    # Load data
    try:
        categories = load_categories()
        registry = load_registry()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Generate content
    py_content = generate_enums_py(categories, registry) if generate_py else None
    pyi_content = generate_enums_pyi(categories, registry) if generate_pyi else None

    if args.check:
        # Check mode: verify files are up-to-date
        errors = []

        if generate_py:
            if not ENUMS_PY.exists():
                errors.append(f"{ENUMS_PY} does not exist")
            else:
                current = ENUMS_PY.read_text(encoding="utf-8")
                if normalize_content(current) != normalize_content(py_content):
                    errors.append("enums.py is out of date")

        if generate_pyi:
            if not ENUMS_PYI.exists():
                errors.append(f"{ENUMS_PYI} does not exist")
            else:
                current = ENUMS_PYI.read_text(encoding="utf-8")
                if normalize_content(current) != normalize_content(pyi_content):
                    errors.append("enums.pyi is out of date")

        if errors:
            for error in errors:
                print(f"Error: {error}", file=sys.stderr)
            print("Run: python tools/generate_plugin_enums.py", file=sys.stderr)
            return 1

        print("Enum files are up-to-date")
        return 0

    # Write files only if they changed
    files_written = 0

    if generate_py:
        current = ENUMS_PY.read_text(encoding="utf-8") if ENUMS_PY.exists() else ""
        if normalize_content(current) != normalize_content(py_content):
            ENUMS_PY.write_text(py_content, encoding="utf-8")
            print(f"Generated: {ENUMS_PY}")
            files_written += 1

    if generate_pyi:
        current = ENUMS_PYI.read_text(encoding="utf-8") if ENUMS_PYI.exists() else ""
        if normalize_content(current) != normalize_content(pyi_content):
            ENUMS_PYI.write_text(pyi_content, encoding="utf-8")
            print(f"Generated: {ENUMS_PYI}")
            files_written += 1

    if files_written == 0:
        print("Enum files are already up-to-date")
    else:
        print("IDEs will now provide autocomplete for plugin enum members!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
