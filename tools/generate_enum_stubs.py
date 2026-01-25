#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Generate type stub files (.pyi) for dynamic plugin enums.

This script reads categories.yaml and registry.yaml and generates a .pyi stub
file that provides IDE autocomplete and type checking for dynamically created enums.

Usage:
    python tools/generate_enum_stubs.py

This will generate: src/aiperf/plugin/enums.pyi
"""

from pathlib import Path

from ruamel.yaml import YAML

yaml = YAML(typ="safe")


def load_categories() -> dict:
    """Load the plugin categories from categories.yaml."""
    categories_path = (
        Path(__file__).parent.parent / "src" / "aiperf" / "plugin" / "categories.yaml"
    )

    if not categories_path.exists():
        raise FileNotFoundError(f"Categories file not found: {categories_path}")

    with open(categories_path) as f:
        return yaml.load(f)


def load_registry() -> dict:
    """Load the plugin registry from registry.yaml."""
    registry_path = Path(__file__).parent.parent / "src" / "aiperf" / "registry.yaml"

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")

    with open(registry_path) as f:
        return yaml.load(f)


def type_to_member_name(type_name: str) -> str:
    """Convert type name to enum member name (UPPER_CASE)."""
    return type_name.replace("-", "_").upper()


def get_description(type_spec: str | dict) -> str | None:
    """Extract description from type spec.

    Args:
        type_spec: Either a string (simple format) or dict (full format)

    Returns:
        Description string if available, None otherwise
    """
    if isinstance(type_spec, dict):
        return type_spec.get("description")
    return None


def generate_stub_content(categories: dict, registry: dict) -> str:
    """Generate the content of the .pyi stub file.

    Args:
        categories: Category definitions from categories.yaml
        registry: Plugin registry from registry.yaml
    """
    lines = [
        "# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        '"""',
        "Type stubs for dynamically generated plugin enums.",
        "",
        "This file is AUTO-GENERATED from categories.yaml and registry.yaml.",
        "Run `python tools/generate_enum_stubs.py` to regenerate.",
        "",
        "These stubs provide IDE autocomplete and type checking for enum members",
        "that are created dynamically at runtime from the plugin registry.",
        '"""',
        "",
        "from aiperf.common.enums import ExtensibleStrEnum",
        "",
    ]

    # Get all categories from categories.yaml (excluding metadata keys)
    metadata_keys = {"schema_version"}
    category_names = sorted(k for k in categories if k not in metadata_keys)

    # Generate PluginCategory enum from categories
    lines.append("class PluginCategory(ExtensibleStrEnum):")
    lines.append('    """')
    lines.append("    Dynamic enum for plugin categories.")
    lines.append("")
    lines.append("    Each category represents a type of plugin that can be registered")
    lines.append("    and used within the AIPerf framework.")
    lines.append('    """')
    lines.append("")

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

    # Generate enums for each category that has an enum mapping in categories.yaml
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

        # Add class definition with description from categories.yaml
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

        # Add enum members with their string values and descriptions
        for type_name in sorted(types.keys()):
            member_name = type_to_member_name(type_name)
            type_spec = types[type_name]
            description = get_description(type_spec)

            lines.append(f'    {member_name} = "{type_name}"')
            if description:
                # Add description as a docstring comment after the member
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
    enum_names = ["PluginCategory"] + [
        categories[cat]["enum"]
        for cat in category_names
        if isinstance(categories.get(cat), dict)
        and categories[cat].get("enum")
        and registry.get(cat)
    ]

    lines.append("__all__ = [")
    for name in sorted(set(enum_names)):
        lines.append(f'    "{name}",')
    lines.append("]")

    return "\n".join(lines)


def main():
    """Generate the stub file."""
    # Load categories and registry
    categories = load_categories()
    registry = load_registry()

    # Generate stub content
    stub_content = generate_stub_content(categories, registry)

    # Write to file
    stub_path = Path(__file__).parent.parent / "src" / "aiperf" / "plugin" / "enums.pyi"
    stub_path.write_text(stub_content)

    print(f"Generated stub file: {stub_path}")
    print("   IDEs will now provide autocomplete for plugin enum members!")


if __name__ == "__main__":
    main()
