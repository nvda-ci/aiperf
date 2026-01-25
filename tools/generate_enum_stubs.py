#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Generate type stub files (.pyi) for dynamic plugin enums.

This script reads registry.yaml and generates a .pyi stub file that provides
IDE autocomplete and type checking for dynamically created enums.

Usage:
    python tools/generate_enum_stubs.py

This will generate: src/aiperf/plugin/enums.pyi
"""

from pathlib import Path

from ruamel.yaml import YAML

yaml = YAML(typ="safe")

# Mapping from registry category names to enum class names
CATEGORY_TO_ENUM = {
    "arrival_pattern": "ArrivalPatternType",
    "communication": "CommunicationBackend",
    "communication_client": "CommClientType",
    "console_exporter": "ConsoleExporterType",
    "custom_dataset_loader": "CustomDatasetType",
    "data_exporter": "DataExporterType",
    "dataset_backing_store": "DatasetBackingStoreType",
    "dataset_client_store": "DatasetClientStoreType",
    "dataset_composer": "ComposerType",
    "dataset_sampler": "DatasetSamplingStrategy",
    "endpoint": "EndpointType",
    "plot": "PlotType",
    "ramp": "RampType",
    "record_processor": "RecordProcessorType",
    "results_processor": "ResultsProcessorType",
    "service": "ServiceType",
    "service_manager": "ServiceRunType",
    "timing_strategy": "TimingStrategyType",
    "transport": "TransportType",
    "ui": "UIType",
    "zmq_proxy": "ZMQProxyType",
}


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


def generate_stub_content(registry: dict) -> str:
    """Generate the content of the .pyi stub file."""
    lines = [
        "# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        '"""',
        "Type stubs for dynamically generated plugin enums.",
        "",
        "This file is AUTO-GENERATED from registry.yaml.",
        "Run `python tools/generate_enum_stubs.py` to regenerate.",
        "",
        "These stubs provide IDE autocomplete and type checking for enum members",
        "that are created dynamically at runtime from the plugin registry.",
        '"""',
        "",
        "from aiperf.common.enums import ExtensibleStrEnum",
        "",
    ]

    # Get all categories from registry (excluding metadata keys)
    metadata_keys = {"schema_version", "plugin"}
    categories = sorted(k for k in registry if k not in metadata_keys)

    # Generate PluginCategory enum from registry categories
    lines.append("class PluginCategory(ExtensibleStrEnum):")
    lines.append('    """Dynamic enum for plugin categories."""')
    lines.append("")

    for category in categories:
        member_name = category.replace("-", "_").upper()
        lines.append(f'    {member_name} = "{category}"')

    lines.append("")
    lines.append("")

    # Generate enums for each category that has a mapping
    for category in categories:
        enum_name = CATEGORY_TO_ENUM.get(category)
        if not enum_name:
            continue

        types = registry.get(category, {})
        if not types:
            continue

        # Add class definition
        lines.append(f"class {enum_name}(ExtensibleStrEnum):")
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
                lines.append(f'    """{description}"""')
            lines.append("")

        lines.append("")

    # Add __all__ export
    enum_names = ["PluginCategory"] + [
        CATEGORY_TO_ENUM[cat]
        for cat in categories
        if cat in CATEGORY_TO_ENUM and registry.get(cat)
    ]

    lines.append("__all__ = [")
    for name in sorted(set(enum_names)):
        lines.append(f'    "{name}",')
    lines.append("]")

    return "\n".join(lines)


def main():
    """Generate the stub file."""
    # Load registry
    registry = load_registry()

    # Generate stub content
    stub_content = generate_stub_content(registry)

    # Write to file
    stub_path = Path(__file__).parent.parent / "src" / "aiperf" / "plugin" / "enums.pyi"
    stub_path.write_text(stub_content)

    print(f"Generated stub file: {stub_path}")
    print("   IDEs will now provide autocomplete for plugin enum members!")


if __name__ == "__main__":
    main()
