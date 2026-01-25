#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Generate JSON Schema files for categories.yaml and plugins.yaml.

This script generates:
1. categories.schema.json - Schema for plugin category definitions
2. plugins.schema.json - Schema for plugin type definitions (with typed metadata)

The plugins schema is dynamically generated from categories.yaml, providing:
- Explicit properties for each category (validates category names)
- Typed metadata schemas introspected from metadata_class when defined

Usage:
    python tools/generate_plugin_schemas.py [output_dir]

Arguments:
    output_dir  Directory to write schema files (default: src/aiperf/plugin)
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

# Paths
REPO_ROOT = Path(__file__).parent.parent
PLUGIN_DIR = REPO_ROOT / "src" / "aiperf" / "plugin"
SCHEMA_DIR = PLUGIN_DIR / "schema"
CATEGORIES_YAML = PLUGIN_DIR / "categories.yaml"

yaml_safe = YAML(typ="safe")


# =============================================================================
# Class Loading Utilities
# =============================================================================


def load_class_from_path(class_path: str) -> type:
    """Load a class from a module:ClassName path.

    Args:
        class_path: Fully qualified class path (e.g., "aiperf.common.models:MyModel")

    Returns:
        The loaded class

    Raises:
        ValueError: If class_path format is invalid
        ImportError: If module cannot be imported
        AttributeError: If class not found in module
    """
    if ":" not in class_path:
        raise ValueError(
            f"Invalid class path '{class_path}', expected 'module:ClassName'"
        )

    module_path, class_name = class_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_metadata_schema(class_path: str) -> dict[str, Any]:
    """Get JSON Schema for a metadata class.

    Supports Pydantic models (model_json_schema) and dataclasses.

    Args:
        class_path: Fully qualified class path (e.g., "aiperf.common.models:MyModel")

    Returns:
        JSON Schema dict for the class
    """
    cls = load_class_from_path(class_path)

    # Pydantic model
    if hasattr(cls, "model_json_schema"):
        return cls.model_json_schema()

    # Dataclass - convert to basic schema
    if dataclasses.is_dataclass(cls):
        properties = {}
        required = []
        for field in dataclasses.fields(cls):
            field_schema: dict[str, Any] = {
                "title": field.name.replace("_", " ").title()
            }

            # Basic type mapping
            origin = getattr(field.type, "__origin__", None)
            if field.type is str or origin is str:
                field_schema["type"] = "string"
            elif field.type is int:
                field_schema["type"] = "integer"
            elif field.type is bool:
                field_schema["type"] = "boolean"
            elif field.type is float:
                field_schema["type"] = "number"
            elif origin is list:
                field_schema["type"] = "array"
            else:
                field_schema["type"] = "string"  # Fallback

            properties[field.name] = field_schema

            if (
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            ):
                required.append(field.name)

        return {
            "type": "object",
            "title": cls.__name__,
            "properties": properties,
            "required": required,
        }

    raise TypeError(f"Class {class_path} is not a Pydantic model or dataclass")


# =============================================================================
# Categories Loading
# =============================================================================


def load_categories() -> dict[str, dict[str, Any]]:
    """Load categories from categories.yaml.

    Returns:
        Dict mapping category name to category spec.
    """
    if not CATEGORIES_YAML.exists():
        raise FileNotFoundError(f"Categories file not found: {CATEGORIES_YAML}")

    data = yaml_safe.load(CATEGORIES_YAML.read_text())

    # Filter out non-category keys
    return {
        k: v
        for k, v in data.items()
        if k not in ("schema_version",) and isinstance(v, dict)
    }


# =============================================================================
# Schema Generation
# =============================================================================


def generate_categories_schema() -> dict[str, Any]:
    """Generate JSON Schema for categories.yaml."""
    from aiperf.plugin.schema import CategoriesFile

    return CategoriesFile.model_json_schema()


def generate_plugins_schema() -> dict[str, Any]:
    """Generate JSON Schema for plugins.yaml based on categories.yaml.

    Reads categories.yaml to generate specific properties for each category,
    with metadata schemas introspected from the metadata_class if defined.
    """
    from aiperf.plugin.schema import PluginsFile, PluginTypeEntry

    base_schema = PluginsFile.model_json_schema()
    categories = load_categories()

    # Base plugin type entry schema (without metadata)
    plugin_entry_schema = PluginTypeEntry.model_json_schema()

    # Collect all $defs from metadata schemas to hoist to root level
    all_defs: dict[str, Any] = dict(base_schema.get("$defs", {}))

    # Build properties for each category
    category_properties: dict[str, Any] = {}

    for category_name, category_spec in categories.items():
        # Get description from category spec
        description = category_spec.get("description", f"Plugins for {category_name}")
        if isinstance(description, str):
            description = description.strip()

        # Build metadata schema if metadata_class is defined
        metadata_schema: dict[str, Any] | None = None
        if category_spec.get("metadata_class"):
            with contextlib.suppress(Exception):
                metadata_schema = get_metadata_schema(category_spec["metadata_class"])

        # Build the plugin entry schema for this category
        if metadata_schema:
            # Hoist any $defs from the metadata schema to root level
            if "$defs" in metadata_schema:
                all_defs.update(metadata_schema.pop("$defs"))

            # Custom entry schema with typed metadata
            entry_schema = {
                "additionalProperties": True,
                "description": "Full specification for a plugin type entry.",
                "properties": {
                    "class": {
                        "description": "Fully qualified class path (module:ClassName).",
                        "title": "Class",
                        "type": "string",
                    },
                    "description": {
                        "description": "Human-readable description of this plugin type.",
                        "title": "Description",
                        "type": "string",
                    },
                    "priority": {
                        "default": 0,
                        "description": "Priority for conflict resolution (higher wins).",
                        "title": "Priority",
                        "type": "integer",
                    },
                    "metadata": metadata_schema,
                },
                "required": ["class", "description"],
                "title": "PluginTypeEntry",
                "type": "object",
            }
        else:
            entry_schema = plugin_entry_schema

        # Category property: maps type names to entries
        category_properties[category_name] = {
            "type": "object",
            "description": description,
            "additionalProperties": entry_schema,
        }

    # Build final schema with explicit category properties
    schema = {
        "$defs": all_defs,
        "description": base_schema.get("description", ""),
        "properties": {
            "schema_version": base_schema["properties"]["schema_version"],
            "plugin": base_schema["properties"]["plugin"],
            **category_properties,
        },
        "required": base_schema.get("required", ["plugin"]),
        "title": base_schema.get("title", "PluginsFile"),
        "type": "object",
        "additionalProperties": False,  # Only allow known categories
    }

    return schema


# =============================================================================
# File Output
# =============================================================================


def normalize_content(content: str) -> str:
    """Normalize content for comparison."""
    return "\n".join(line.rstrip() for line in content.strip().split("\n"))


def write_schema_files(output_dir: str | Path = SCHEMA_DIR) -> int:
    """Write JSON Schema files to the specified directory.

    Args:
        output_dir: Directory to write schema files to.

    Returns:
        Number of files written (0 if all up-to-date).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    categories_schema = generate_categories_schema()
    plugins_schema = generate_plugins_schema()

    # Add JSON Schema declaration at the top for IDE recognition
    categories_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "categories.schema.json",
        **categories_schema,
    }
    plugins_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "plugins.schema.json",
        **plugins_schema,
    }

    files_written = 0

    # Write categories schema
    categories_path = output_path / "categories.schema.json"
    categories_content = json.dumps(categories_schema, indent=2) + "\n"
    current = categories_path.read_text() if categories_path.exists() else ""
    if normalize_content(current) != normalize_content(categories_content):
        categories_path.write_text(categories_content)
        print(f"Generated: {categories_path}")
        files_written += 1

    # Write plugins schema
    plugins_path = output_path / "plugins.schema.json"
    plugins_content = json.dumps(plugins_schema, indent=2) + "\n"
    current = plugins_path.read_text() if plugins_path.exists() else ""
    if normalize_content(current) != normalize_content(plugins_content):
        plugins_path.write_text(plugins_content)
        print(f"Generated: {plugins_path}")
        files_written += 1

    return files_written


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    output_dir = sys.argv[1] if len(sys.argv) > 1 else SCHEMA_DIR
    files_written = write_schema_files(output_dir)

    if files_written == 0:
        print("Schema files are already up-to-date")
    else:
        print(
            "IDEs will now provide validation and autocomplete for plugin YAML files!"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
