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

To generate schema files:
    python tools/generate_plugin_schemas.py [output_dir]
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
