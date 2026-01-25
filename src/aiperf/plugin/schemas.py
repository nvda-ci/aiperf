# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for plugin YAML schema validation.

These models define the structure of categories.yaml and plugins.yaml files,
enabling JSON Schema generation for IDE support and validation.

Usage:
    # Generate JSON Schema
    from aiperf.plugin.schemas import CategoriesFile, PluginsFile

    categories_schema = CategoriesFile.model_json_schema()
    plugins_schema = PluginsFile.model_json_schema()
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from aiperf.common.models import AIPerfBaseModel

# =============================================================================
# Categories YAML Schema
# =============================================================================


class CategorySpec(AIPerfBaseModel):
    """Specification for a plugin category."""

    protocol: str = Field(
        description="Fully qualified class path (module:ClassName) for the Protocol/ABC."
    )
    metadata_class: str | None = Field(
        default=None,
        description="Class path for the metadata model. Schema is introspected dynamically.",
    )
    enum: str = Field(
        description="Name of the dynamic enum generated from registered plugins."
    )
    description: str = Field(
        description="Human-readable description of the category's purpose."
    )
    internal: bool = Field(
        default=False,
        description="Whether this category is internal infrastructure, not user-facing.",
    )


class CategoriesFile(AIPerfBaseModel):
    """Root model for categories.yaml file.

    Categories define plugin extension points with their protocols, enums,
    and optional metadata schemas.
    """

    schema_version: str = Field(
        default="1.0",
        description="Schema version for the categories file.",
    )

    # Categories are stored as additional fields beyond schema_version
    # Using model_extra to capture them
    model_config = {"extra": "allow"}

    @classmethod
    def model_json_schema(cls, **kwargs) -> dict[str, Any]:
        """Generate JSON Schema with additionalProperties for categories."""
        schema = super().model_json_schema(**kwargs)
        category_schema = CategorySpec.model_json_schema(**kwargs)

        # Merge $defs from CategorySpec into root schema so $refs resolve correctly
        if "$defs" in category_schema:
            schema.setdefault("$defs", {}).update(category_schema.pop("$defs"))

        # Allow additional properties that match CategorySpec
        schema["additionalProperties"] = category_schema
        return schema


# =============================================================================
# Plugins YAML Schema
# =============================================================================


class PluginPackageInfo(AIPerfBaseModel):
    """Metadata about the plugin package itself."""

    name: str = Field(description="Package name.")
    version: str = Field(description="Package version.")
    description: str = Field(description="Human-readable package description.")
    author: str = Field(description="Package author or organization.")
    builtin: bool = Field(
        default=False,
        description="Whether this is a built-in plugin package.",
    )


class PluginTypeEntry(AIPerfBaseModel):
    """Full specification for a plugin type entry."""

    class_: str = Field(
        alias="class",
        description="Fully qualified class path (module:ClassName).",
    )
    description: str = Field(
        description="Human-readable description of this plugin type."
    )
    priority: int = Field(
        default=0,
        description="Priority for conflict resolution (higher wins).",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Category-specific metadata values, schema defined in categories.yaml.",
    )

    model_config = {"populate_by_name": True}


class PluginsFile(AIPerfBaseModel):
    """Root model for plugins.yaml file.

    Plugins files register concrete implementations for plugin categories.
    Each category section maps type names to either:
    - A full PluginTypeEntry with class, description, priority, metadata
    - A simple string class path for shorthand notation
    """

    schema_version: str = Field(
        default="1.0",
        description="Schema version for the plugins file.",
    )
    plugin: PluginPackageInfo = Field(
        description="Metadata about the plugin package.",
    )

    # Plugin categories are stored as additional fields
    model_config = {"extra": "allow"}

    @classmethod
    def model_json_schema(cls, **kwargs) -> dict[str, Any]:
        """Generate JSON Schema with additionalProperties for plugin categories."""
        schema = super().model_json_schema(**kwargs)

        # Create a schema for category entries (dict of type name -> entry)
        category_entry_schema = {
            "type": "object",
            "additionalProperties": {
                "oneOf": [
                    # Short form: just a class path string
                    {"type": "string", "description": "Shorthand class path"},
                    # Full form: PluginTypeEntry
                    PluginTypeEntry.model_json_schema(**kwargs),
                ]
            },
        }

        schema["additionalProperties"] = category_entry_schema
        return schema


# =============================================================================
# Schema Generation Utilities
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
    import importlib

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
    import dataclasses

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


def generate_categories_schema() -> dict[str, Any]:
    """Generate JSON Schema for categories.yaml."""
    return CategoriesFile.model_json_schema()


def generate_plugins_schema() -> dict[str, Any]:
    """Generate JSON Schema for plugins.yaml."""
    return PluginsFile.model_json_schema()


def write_schema_files(output_dir: str = ".") -> None:
    """Write JSON Schema files to the specified directory.

    Args:
        output_dir: Directory to write schema files to.
    """
    import json
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    categories_schema = generate_categories_schema()
    plugins_schema = generate_plugins_schema()

    (output_path / "categories.schema.json").write_text(
        json.dumps(categories_schema, indent=2)
    )
    (output_path / "plugins.schema.json").write_text(
        json.dumps(plugins_schema, indent=2)
    )


if __name__ == "__main__":
    # Generate schemas when run directly
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    write_schema_files(output_dir)
    print(f"Schema files written to {output_dir}")
