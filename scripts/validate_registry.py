#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation script for AIPerf plugin registry."""

import sys

from aiperf.plugin import plugins
from aiperf.plugin.validate import validate_alphabetical_order, validate_registry


def main() -> int:
    """Run all validation checks."""
    print("Validating AIPerf plugin registry")
    print("=" * 80)

    all_passed = True

    # 1. Validate alphabetical ordering
    print("\n1. Validating alphabetical ordering...")
    order_errors = validate_alphabetical_order()
    if order_errors:
        print("FAILED:")
        for category, messages in order_errors.items():
            for msg in messages:
                print(f"  {category}: {msg}")
        all_passed = False
    else:
        print("PASSED: All entries are alphabetically ordered")

    # 2. Validate class paths
    print("\n2. Validating class paths (this may take a moment)...")
    class_errors = validate_registry(check_class=True)
    if class_errors:
        print("FAILED:")
        for category, errors in class_errors.items():
            for name, error in errors:
                print(f"  {category}:{name} - {error}")
        all_passed = False
    else:
        print("PASSED: All class paths are valid")

    # Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("SUCCESS: All validation checks passed!")
        categories = plugins.list_categories(include_internal=True)
        print(f"Registry contains {len(categories)} categories")
        total = sum(len(plugins.list_entries(cat)) for cat in categories)
        print(f"Total plugin entries: {total}")
        return 0
    else:
        print("FAILED: Validation errors found")
        return 1


if __name__ == "__main__":
    sys.exit(main())
