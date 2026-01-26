# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for plugin YAML schema validation.

These models define the structure of categories.yaml and plugins.yaml files,
enabling JSON Schema generation for IDE support and validation.

To generate schema files:
    python tools/generate_plugin_schemas.py [output_dir]

Default output directory: src/aiperf/plugin/schema/
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Categories YAML Schema
# =============================================================================


class CategorySpec(BaseModel):
    """Specification for a plugin category.

    Categories define extension points in AIPerf where plugins can provide
    custom implementations. Each category has a protocol that plugins must
    implement and an enum that lists all available plugin types.
    """

    protocol: str = Field(
        description=(
            "The interface that plugins in this category must implement. "
            "Use 'module.path:ClassName' format, e.g., 'aiperf.plugin.protocols:EndpointProtocol'."
        )
    )
    metadata_class: str | None = Field(
        default=None,
        description=(
            "Optional Pydantic model for category-specific metadata. "
            "When set, plugins can include typed metadata fields validated against this schema. "
            "Use 'module.path:ClassName' format."
        ),
    )
    enum: str = Field(
        description=(
            "Name of the enum that will be auto-generated from registered plugins. "
            "This enum is used in config files and APIs to select plugin types, "
            "e.g., 'EndpointType' generates EndpointType.CHAT, EndpointType.COMPLETIONS, etc."
        )
    )
    description: str = Field(
        description="Brief explanation of what this category is for and when to use it."
    )
    internal: bool = Field(
        default=False,
        description=(
            "Set to true for infrastructure categories not meant for end users. "
            "Internal categories are hidden from documentation and plugin listings."
        ),
    )


class CategoriesFile(BaseModel):
    """Root model for categories.yaml file.

    This file defines all plugin extension points in AIPerf. Each category
    specifies a protocol interface and generates an enum for type selection.

    Example:
        endpoint:
          protocol: aiperf.plugin.protocols:EndpointProtocol
          enum: EndpointType
          description: HTTP endpoint handlers for LLM APIs
    """

    # Categories are stored as additional fields beyond schema_version
    # Using model_extra to capture them
    model_config = ConfigDict(extra="allow")

    schema_version: str = Field(
        default="1.0",
        description="Version of the categories.yaml schema format. Used for backwards compatibility.",
    )

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


class PackageInfo(BaseModel):
    """Metadata about the plugin package.

    This section identifies your plugin package and is displayed in
    plugin listings and error messages.
    """

    name: str = Field(
        description=(
            "Unique identifier for your plugin package. "
            "Use your Python package name, e.g., 'my-aiperf-plugins'."
        )
    )
    version: str = Field(
        default="unknown",
        description="Semantic version of your plugin package, e.g., '1.0.0' or '2.1.3-beta'.",
    )
    description: str = Field(
        default="unknown",
        description="One-line summary of what your plugin package provides.",
    )
    author: str = Field(
        default="unknown",
        description="Author name, team, or organization, e.g., 'NVIDIA' or 'Jane Doe <jane@example.com>'.",
    )
    license: str = Field(
        default="unknown",
        description="License of the plugin package, e.g., 'Apache-2.0' or 'MIT'.",
    )
    homepage: str = Field(
        default="unknown",
        description="Homepage of the plugin package, e.g., 'https://example.com'.",
    )

    @property
    def is_builtin(self) -> bool:
        """Whether this is a built-in plugin package.

        A built-in plugin package is one that is included in the AIPerf core distribution.
        """
        return self.name == "aiperf"


class TypeSpec(BaseModel):
    """Specification for a plugin type.

    Each plugin type maps a name (like 'chat' or 'completions') to a Python
    class that implements the category's protocol.

    Example:
        chat:
          class: aiperf.endpoints.chat:ChatEndpoint
          description: OpenAI-compatible chat completions endpoint
    """

    model_config = ConfigDict(populate_by_name=True)

    class_: str = Field(
        alias="class",
        description=(
            "Python class that implements this plugin type. "
            "Use 'module.path:ClassName' format, e.g., 'aiperf.endpoints.chat:ChatEndpoint'."
        ),
    )
    description: str = Field(
        default="",
        description="Brief explanation of what this plugin type does and when to use it.",
    )
    priority: int = Field(
        default=0,
        description=(
            "Conflict resolution priority. When multiple packages register the same type name, "
            "the one with higher priority wins. Use 0 for normal plugins, higher values to "
            "override built-in implementations."
        ),
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Category-specific configuration for this plugin type. "
            "The allowed fields depend on the category's metadata_class in categories.yaml."
        ),
    )


class PluginsFile(BaseModel):
    """Root model for plugins.yaml file.

    This file registers plugin implementations for AIPerf. Each section after
    'plugin' corresponds to a category from categories.yaml and maps type names
    to their implementing classes.

    Example:
        schema_version: "1.0"
        plugin:
          name: my-plugins
          version: 1.0.0
          description: Custom endpoints for my use case
          author: My Team

        endpoint:
          my_custom:
            class: my_package.endpoints:MyCustomEndpoint
            description: Custom endpoint for my use case
    """

    # Plugin categories are stored as additional fields
    model_config = ConfigDict(extra="allow")

    schema_version: str = Field(
        default="1.0",
        description="Version of the plugins.yaml schema format. Use '1.0' for current format.",
    )
    plugin: PackageInfo = Field(
        description="Required section identifying your plugin package. See PluginPackageInfo for fields.",
    )

    @classmethod
    def model_json_schema(cls, **kwargs) -> dict[str, Any]:
        """Generate JSON Schema with additionalProperties for plugin categories."""
        schema = super().model_json_schema(**kwargs)

        # Create a schema for category entries (dict of type name -> PluginTypeEntry)
        category_entry_schema = {
            "type": "object",
            "additionalProperties": TypeSpec.model_json_schema(**kwargs),
        }

        schema["additionalProperties"] = category_entry_schema
        return schema
