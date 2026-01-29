# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for plugin YAML schema validation.

These models define the structure of categories.yaml and plugins.yaml files,
enabling JSON Schema generation for IDE support and validation.

To generate schema files:
    python tools/generate_plugin_artifacts.py --schemas
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
            "Use 'module.path:ClassName' format, e.g., 'aiperf.common.protocols:EndpointProtocol'."
        )
    )
    metadata_class: str | None = Field(
        default=None,
        description=(
            "Optional class path for category-specific metadata. "
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


class CategoriesManifest(BaseModel):
    """Root model for categories.yaml file.

    This file defines all plugin extension points in AIPerf. Each category
    specifies a protocol interface and generates an enum for type selection.

    Example:
        endpoint:
          protocol: aiperf.common.protocols:EndpointProtocol
          enum: EndpointType
          description: HTTP endpoint handlers for LLM APIs
    """

    # Categories are stored as additional fields beyond schema_version
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
# Plugin Metadata Classes
# =============================================================================
# These classes define the metadata schema for specific plugin categories.
# They are referenced by categories.yaml via metadata_class and used to
# validate the metadata field in plugins.yaml entries.
# =============================================================================


class EndpointMetadata(BaseModel):
    """Endpoint metadata for discovery and documentation.

    Defines capabilities and configuration for endpoint plugins.
    This metadata is specified in plugins.yaml under each endpoint entry.
    """

    metrics_title: str | None = Field(
        ..., description="Display title for metrics dashboard."
    )
    endpoint_path: str | None = Field(
        ..., description="API path (e.g., /v1/chat/completions)."
    )
    streaming_path: str | None = Field(
        default=None,
        description="Streaming API path if different from the endpoint path (e.g., /generate_stream).",
    )
    service_kind: str = Field(
        default="openai",
        description="The service kind of the endpoint (used for artifact naming).",
    )
    supports_streaming: bool = Field(
        ..., description="Whether endpoint supports streaming responses."
    )
    tokenizes_input: bool = Field(
        ..., description="Whether endpoint tokenizes text inputs."
    )
    produces_tokens: bool = Field(
        ..., description="Whether endpoint produces token-based output."
    )
    supports_audio: bool = Field(
        default=False, description="Whether endpoint accepts audio input."
    )
    supports_images: bool = Field(
        default=False, description="Whether endpoint accepts image input."
    )
    supports_videos: bool = Field(
        default=False, description="Whether endpoint accepts video input."
    )
    produces_audio: bool = Field(
        default=False, description="Whether endpoint produces audio-based outputs."
    )
    produces_images: bool = Field(
        default=False, description="Whether endpoint produces image-based outputs."
    )
    produces_videos: bool = Field(
        default=False, description="Whether endpoint produces video-based outputs."
    )


class TransportMetadata(BaseModel):
    """Transport metadata for discovery and documentation.

    Defines transport type and URL schemes for auto-detection.
    This metadata is specified in plugins.yaml under each transport entry.
    """

    transport_type: str = Field(
        description="Transport type identifier for this transport"
    )
    url_schemes: list[str] = Field(
        default_factory=list,
        description="URL schemes this transport handles (for auto-detection and validation).",
    )


class PlotMetadata(BaseModel):
    """Plot metadata for discovery and documentation.

    Defines display properties for plot type plugins.
    Description comes from the plugin entry, not metadata.
    """

    display_name: str = Field(description="Human-readable name for UI display.")
    category: str = Field(
        description="Plot category (per_request, aggregated, combined, comparison)."
    )


class ServiceMetadata(BaseModel):
    """Service metadata for discovery and configuration.

    Defines runtime configuration for service plugins.
    This metadata is specified in plugins.yaml under each service entry.
    """

    required: bool = Field(
        description="Whether the service is required for benchmark execution."
    )
    auto_start: bool = Field(
        description="Whether the service is automatically started by the system controller."
    )
    disable_gc: bool = Field(
        default=False,
        description="Whether to disable garbage collection in the service for timing-critical operations.",
    )


# =============================================================================
# Plugins YAML Schema
# =============================================================================


class PluginSpec(BaseModel):
    """Specification for a plugin entry.

    Each plugin entry maps a name (like 'chat' or 'completions') to a Python
    class that implements the category's protocol.

    Example:
        chat:
          class: aiperf.endpoints.openai_chat:ChatEndpoint
          description: OpenAI-compatible chat completions endpoint
    """

    model_config = ConfigDict(populate_by_name=True)

    class_: str = Field(
        alias="class",
        description=(
            "Python class that implements this plugin entry. "
            "Use 'module.path:ClassName' format, e.g., 'aiperf.endpoints.openai_chat:ChatEndpoint'."
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


class PluginsManifest(BaseModel):
    """Root model for plugins.yaml file.

    This file registers plugin implementations for AIPerf. Each top-level section
    corresponds to a category from categories.yaml and maps type names to their
    implementing classes.

    Note: Package metadata (name, version, author) comes from pyproject.toml
    via importlib.metadata, not from this file.

    Example:
        schema_version: "1.0"

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

    @classmethod
    def model_json_schema(cls, **kwargs) -> dict[str, Any]:
        """Generate JSON Schema with additionalProperties for plugin categories."""
        schema = super().model_json_schema(**kwargs)

        # Create a schema for category entries (dict of type name -> PluginSpec)
        category_entry_schema = {
            "type": "object",
            "additionalProperties": PluginSpec.model_json_schema(**kwargs),
        }

        schema["additionalProperties"] = category_entry_schema
        return schema
