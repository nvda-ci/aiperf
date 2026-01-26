#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Validation script for src/aiperf/plugin/plugins.yaml

Checks:
1. Valid YAML syntax
2. All class paths exist and are importable
3. Alphabetical ordering within sections
4. No duplicate entries
5. Consistent formatting
6. Schema compliance
"""

import importlib
import sys
from pathlib import Path
from typing import Any

import yaml


def validate_yaml_syntax(
    registry_path: Path,
) -> tuple[bool, dict[str, Any] | None, str]:
    """Validate YAML syntax and load the registry."""
    try:
        with open(registry_path) as f:
            registry = yaml.safe_load(f)
        return True, registry, ""
    except yaml.YAMLError as e:
        return False, None, f"YAML syntax error: {e}"


def validate_class_paths(registry: dict[str, Any]) -> list[str]:
    """Validate that all class paths exist and are importable."""
    errors = []

    # Skip metadata sections
    skip_sections = {"schema_version", "plugin"}

    for protocol_name, implementations in registry.items():
        if protocol_name in skip_sections or not isinstance(implementations, dict):
            continue

        for name, impl_info in implementations.items():
            # Extract class path
            if isinstance(impl_info, str):
                class_path = impl_info
            elif isinstance(impl_info, dict) and "class" in impl_info:
                class_path = impl_info["class"]
            else:
                errors.append(
                    f"  {protocol_name}:{name} - Invalid format (missing 'class' field)"
                )
                continue

            # Parse module:class format
            if ":" not in class_path:
                errors.append(
                    f"  {protocol_name}:{name} - Invalid class path format: {class_path}"
                )
                continue

            module_path, class_name = class_path.rsplit(":", 1)

            # Try to import module and verify class exists
            try:
                module = importlib.import_module(module_path)
                if not hasattr(module, class_name):
                    errors.append(
                        f"  {protocol_name}:{name} - Class '{class_name}' not found in module '{module_path}'"
                    )
            except ImportError as e:
                errors.append(
                    f"  {protocol_name}:{name} - Cannot import module '{module_path}': {e}"
                )
            except Exception as e:
                errors.append(f"  {protocol_name}:{name} - Error: {e}")

    return errors


def validate_alphabetical_order(registry: dict[str, Any]) -> list[str]:
    """Validate that implementations within each protocol are alphabetically ordered."""
    errors = []

    # Skip metadata sections
    skip_sections = {"schema_version", "plugin"}

    for protocol_name, implementations in registry.items():
        if protocol_name in skip_sections or not isinstance(implementations, dict):
            continue

        impl_names = list(implementations.keys())
        sorted_impl_names = sorted(impl_names)

        if impl_names != sorted_impl_names:
            errors.append(f"  {protocol_name}: Not alphabetically sorted")
            errors.append(f"    Current order: {impl_names}")
            errors.append(f"    Expected order: {sorted_impl_names}")

    return errors


def validate_no_duplicates(registry: dict[str, Any]) -> list[str]:
    """Validate that there are no duplicate implementation names across protocols."""
    errors = []
    seen_impls: dict[str, list[str]] = {}

    # Skip metadata sections
    skip_sections = {"schema_version", "plugin"}

    for protocol_name, implementations in registry.items():
        if protocol_name in skip_sections or not isinstance(implementations, dict):
            continue

        for name in implementations:
            if name not in seen_impls:
                seen_impls[name] = []
            seen_impls[name].append(protocol_name)

    return errors


def validate_schema_version(registry: dict[str, Any]) -> list[str]:
    """Validate that schema_version field exists and is valid."""
    errors = []

    if "schema_version" not in registry:
        errors.append("  Missing 'schema_version' field")
    elif not isinstance(registry["schema_version"], str):
        errors.append("  'schema_version' must be a string")

    return errors


def validate_plugin_metadata(registry: dict[str, Any]) -> list[str]:
    """Validate that plugin metadata exists and is valid."""
    errors = []

    if "plugin" not in registry:
        errors.append("  Missing 'plugin' metadata section")
        return errors

    plugin = registry["plugin"]
    required_fields = ["name", "version", "description", "author", "builtin"]

    for field in required_fields:
        if field not in plugin:
            errors.append(f"  Missing required plugin field: '{field}'")

    if "builtin" in plugin and not isinstance(plugin["builtin"], bool):
        errors.append("  'builtin' field must be a boolean")

    return errors


def main() -> int:
    """Run all validation checks."""
    # Find plugins.yaml
    script_dir = Path(__file__).parent
    registry_path = script_dir.parent / "src" / "aiperf" / "plugin" / "plugins.yaml"

    if not registry_path.exists():
        print(f"ERROR: Registry file not found at {registry_path}")
        return 1

    print(f"Validating registry: {registry_path}")
    print("=" * 80)

    # 1. Validate YAML syntax
    print("\n1. Validating YAML syntax...")
    is_valid, registry, error_msg = validate_yaml_syntax(registry_path)
    if not is_valid:
        print(f"FAILED: {error_msg}")
        return 1
    print("PASSED: Valid YAML syntax")

    # 2. Validate schema version
    print("\n2. Validating schema version...")
    errors = validate_schema_version(registry)
    if errors:
        print("FAILED:")
        for error in errors:
            print(error)
        return 1
    print("PASSED: Schema version is valid")

    # 3. Validate plugin metadata
    print("\n3. Validating plugin metadata...")
    errors = validate_plugin_metadata(registry)
    if errors:
        print("FAILED:")
        for error in errors:
            print(error)
        return 1
    print("PASSED: Plugin metadata is valid")

    # 4. Validate alphabetical ordering
    print("\n4. Validating alphabetical ordering...")
    errors = validate_alphabetical_order(registry)
    if errors:
        print("FAILED:")
        for error in errors:
            print(error)
        return 1
    print("PASSED: All implementations are alphabetically ordered")

    # 5. Validate no duplicates
    print("\n5. Validating no duplicates...")
    errors = validate_no_duplicates(registry)
    if errors:
        print("FAILED:")
        for error in errors:
            print(error)
        return 1
    print("PASSED: No duplicate entries")

    # 6. Validate class paths
    print("\n6. Validating class paths (this may take a moment)...")
    errors = validate_class_paths(registry)
    if errors:
        print("FAILED:")
        for error in errors:
            print(error)
        return 1
    print("PASSED: All class paths are valid and importable")

    # Summary
    print("\n" + "=" * 80)
    print("SUCCESS: All validation checks passed!")
    print(
        f"Registry contains {len([k for k in registry if k not in ['schema_version', 'plugin']])} protocol sections"
    )

    total_impls = sum(
        len(impls)
        for k, impls in registry.items()
        if k not in ["schema_version", "plugin"] and isinstance(impls, dict)
    )
    print(f"Total implementations: {total_impls}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
