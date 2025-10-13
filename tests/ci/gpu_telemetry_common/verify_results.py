#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Validates GPU telemetry results from AIPerf benchmarking.

This script checks that GPU metrics were successfully collected and contain
reasonable values, ensuring the telemetry system is working correctly.
"""

import json
import sys
from pathlib import Path
from typing import Any


def find_aiperf_output(base_dir: Path) -> Path | None:
    """
    Find AIPerf output directory automatically.

    Searches for AIPerf output in common locations:
    - aiperf_output/ (legacy)
    - artifacts/*/ (current AIPerf output structure)

    Returns:
        Path to the directory containing the results, or None if not found.
    """
    # Try legacy location
    legacy_dir = base_dir / "aiperf_output"
    if legacy_dir.exists() and (legacy_dir / "summary.json").exists():
        return legacy_dir

    # Search for profile_export_aiperf.json in artifacts
    artifacts_dir = base_dir / "artifacts"
    if artifacts_dir.exists():
        for json_file in artifacts_dir.rglob("profile_export_aiperf.json"):
            return json_file.parent

    return None


def load_summary_json(output_dir: Path) -> dict[str, Any]:
    """
    Load and parse the AIPerf results JSON file.

    Tries both summary.json (legacy) and profile_export_aiperf.json (current).
    """
    # Try current filename first
    json_path = output_dir / "profile_export_aiperf.json"
    if not json_path.exists():
        # Fallback to legacy filename
        json_path = output_dir / "summary.json"

    if not json_path.exists():
        print(f"ERROR: Results file not found at {output_dir}")
        print("  Tried: profile_export_aiperf.json, summary.json")
        return {}

    try:
        with open(json_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse {json_path.name}: {e}")
        return {}


def validate_telemetry_data(data: dict[str, Any]) -> bool:
    """
    Validate that GPU telemetry data is present and contains expected metrics.

    Returns:
        True if validation passes, False otherwise.
    """
    if "telemetry_data" not in data:
        print("ERROR: No 'telemetry_data' section found in summary.json")
        return False

    telemetry = data["telemetry_data"]

    # Check summary section
    if "summary" not in telemetry:
        print("ERROR: No 'summary' section in telemetry_data")
        return False

    summary = telemetry["summary"]

    # Validate endpoints were successfully reached
    endpoints_successful = summary.get("endpoints_successful", [])
    if not endpoints_successful:
        print("ERROR: No successful DCGM endpoints found")
        print(f"  Endpoints tested: {summary.get('endpoints_tested', [])}")
        return False

    print(f"✓ Successfully connected to {len(endpoints_successful)} DCGM endpoint(s)")
    for endpoint in endpoints_successful:
        print(f"  • {endpoint}")

    # Check endpoints section
    if "endpoints" not in telemetry:
        print("ERROR: No 'endpoints' section in telemetry_data")
        return False

    endpoints = telemetry["endpoints"]
    if not endpoints:
        print("ERROR: 'endpoints' section is empty")
        return False

    # Validate GPU metrics for each endpoint
    all_metrics_valid = True
    required_metrics = ["gpu_power_usage", "gpu_utilization", "gpu_memory_used"]

    # Metrics that MUST be positive during active GPU work
    MUST_BE_POSITIVE = {
        "gpu_power_usage",
        "gpu_utilization",
        "gpu_memory_used",
        "energy_consumption",
        "sm_clock_frequency",
        "memory_clock_frequency",
        "gpu_temperature",
        "memory_temperature",
        "gpu_memory_free",
    }

    for endpoint_name, endpoint_data in endpoints.items():
        print(f"\nValidating endpoint: {endpoint_name}")

        if "gpus" not in endpoint_data:
            print(f"  ERROR: No 'gpus' section for endpoint {endpoint_name}")
            all_metrics_valid = False
            continue

        gpus = endpoint_data["gpus"]
        if not gpus:
            print(f"  ERROR: No GPUs found for endpoint {endpoint_name}")
            all_metrics_valid = False
            continue

        # Check each GPU
        for gpu_id, gpu_data in gpus.items():
            gpu_name = gpu_data.get("gpu_name", "Unknown")
            gpu_index = gpu_data.get("gpu_index", "?")
            print(f"  GPU {gpu_index}: {gpu_name}")

            if "metrics" not in gpu_data:
                print(f"    ERROR: No 'metrics' section for {gpu_id}")
                all_metrics_valid = False
                continue

            metrics = gpu_data["metrics"]

            # Validate required metrics
            for metric_name in required_metrics:
                if metric_name not in metrics:
                    print(f"    ERROR: Missing metric '{metric_name}'")
                    all_metrics_valid = False
                    continue

                metric_data = metrics[metric_name]
                avg_value = metric_data.get("avg", 0)
                count = metric_data.get("count", 0)
                unit = metric_data.get("unit", "")

                # Validate metric has data
                if count == 0:
                    print(
                        f"    ERROR: Metric '{metric_name}' has no data points (count=0)"
                    )
                    all_metrics_valid = False
                    continue

                # Fail on negative values (physically impossible)
                if avg_value < 0:
                    print(
                        f"    ERROR: Metric '{metric_name}' has negative avg={avg_value}"
                    )
                    all_metrics_valid = False
                    continue

                # Fail if critical metrics are zero (GPU should be working)
                if metric_name in MUST_BE_POSITIVE and avg_value == 0:
                    print(
                        f"    ERROR: Metric '{metric_name}' is 0 (expected > 0 during GPU work)"
                    )
                    all_metrics_valid = False
                    continue

                print(
                    f"    ✓ {metric_name}: avg={avg_value:.2f} {unit} (count={count})"
                )

    return all_metrics_valid


def main():
    """Main entry point for verification script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate GPU telemetry results from AIPerf benchmarking"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory to search for AIPerf output (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit path to AIPerf output directory (overrides auto-detection)",
    )
    args = parser.parse_args()

    # Use explicit output-dir if provided, otherwise auto-detect
    if args.output_dir:
        output_dir = args.output_dir
        if not output_dir.exists():
            print(f"ERROR: Specified output directory not found: {output_dir}")
            print(f"  Absolute path: {output_dir.absolute()}")
            sys.exit(1)
    else:
        output_dir = find_aiperf_output(args.base_dir)
        if output_dir is None:
            print(f"ERROR: Could not find AIPerf output in {args.base_dir}")
            print("  Searched for:")
            print(f"    - {args.base_dir / 'aiperf_output' / 'summary.json'}")
            print(
                f"    - {args.base_dir / 'artifacts' / '*' / 'profile_export_aiperf.json'}"
            )
            sys.exit(1)
        print(f"Found AIPerf output at: {output_dir}")

    print("=" * 70)
    print("GPU Telemetry Results Verification")
    print("=" * 70)

    # Load summary data
    summary_data = load_summary_json(output_dir)
    if not summary_data:
        sys.exit(1)

    # Validate telemetry data
    validation_passed = validate_telemetry_data(summary_data)

    print("\n" + "=" * 70)
    if validation_passed:
        print("✓ VALIDATION PASSED: GPU telemetry data is present and valid")
        print("=" * 70)
        sys.exit(0)
    else:
        print("✗ VALIDATION FAILED: Issues found in GPU telemetry data")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
