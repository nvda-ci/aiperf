#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyze histogram percentiles from server metrics JSON export.

Analyzes estimated percentiles computed using the polynomial histogram algorithm
with +Inf bucket estimation via back-calculation from total sum.

Usage:
    python scripts/analyze_histogram_percentiles.py <json_file>

Example:
    python scripts/analyze_histogram_percentiles.py artifacts/run4_c50/server_metrics_export.json
"""

import sys
from pathlib import Path

import orjson


def analyze_percentiles(json_path: Path) -> None:
    """Analyze histogram percentiles from JSON export file."""
    with open(json_path, "rb") as f:
        data = orjson.loads(f.read())

    print("=" * 100)
    print("HISTOGRAM PERCENTILE ANALYSIS")
    print("=" * 100)
    print("\nPercentiles estimated using polynomial histogram algorithm:")
    print("  - Learns per-bucket means from single-bucket scrape intervals")
    print("  - Uses exact sum constraint to improve observation placement")
    print("  - Back-calculates +Inf bucket observations for accurate tail percentiles")
    print()

    all_histograms = []

    for endpoint, ep_data in data["endpoints"].items():
        print(f"\n{'─' * 100}")
        print(f"ENDPOINT: {endpoint}")
        print(f"{'─' * 100}")

        endpoint_histograms = []

        for name, metric in ep_data["metrics"].items():
            if metric["type"] != "histogram":
                continue

            for series in metric["series"]:
                stats = series["stats"]
                labels = series.get("labels", {})
                label_str = (
                    ",".join(f"{k}={v}" for k, v in labels.items()) if labels else ""
                )

                # Check if percentile estimates exist
                if stats.get("p50_estimate") is None:
                    continue

                count_delta = stats.get("count_delta", 0)
                if count_delta == 0:
                    continue

                hist_info = {
                    "name": name,
                    "labels": label_str,
                    "count_delta": count_delta,
                    "avg": stats.get("avg", 0),
                    "stats": stats,
                }
                endpoint_histograms.append(hist_info)
                all_histograms.append(hist_info)

        # Print endpoint summary
        if endpoint_histograms:
            print(f"\nHistogram metrics: {len(endpoint_histograms)}")
            print()

            # Table header
            header = (
                f"{'Metric':<40} {'Count':>8} {'Avg':>10} "
                f"{'p50_est':>10} {'p90_est':>10} {'p95_est':>10} {'p99_est':>10}"
            )
            print(header)
            print("─" * len(header))

            for h in sorted(
                endpoint_histograms, key=lambda x: x["count_delta"], reverse=True
            ):
                name = h["name"]
                if h["labels"]:
                    name = (
                        f"{name}[{h['labels'][:15]}...]"
                        if len(h["labels"]) > 15
                        else f"{name}[{h['labels']}]"
                    )
                name = name[:38] + ".." if len(name) > 40 else name

                count = h["count_delta"]
                avg = h["avg"]
                stats = h["stats"]

                p50 = stats.get("p50_estimate")
                p90 = stats.get("p90_estimate")
                p95 = stats.get("p95_estimate")
                p99 = stats.get("p99_estimate")

                p50_str = f"{p50:.4f}" if p50 is not None else ""
                p90_str = f"{p90:.4f}" if p90 is not None else ""
                p95_str = f"{p95:.4f}" if p95 is not None else ""
                p99_str = f"{p99:.4f}" if p99 is not None else ""

                print(
                    f"{name:<40} {count:>8.0f} {avg:>10.4f} "
                    f"{p50_str:>10} {p90_str:>10} {p95_str:>10} {p99_str:>10}"
                )

    # Overall statistics
    print()
    print("=" * 100)
    print("OVERALL STATISTICS")
    print("=" * 100)

    print(f"\nTotal histogram series analyzed: {len(all_histograms)}")

    # Detailed view for top metrics by count
    print()
    print("=" * 100)
    print("DETAILED VIEW (Top 10 by observation count)")
    print("=" * 100)

    top_histograms = sorted(
        all_histograms, key=lambda x: x["count_delta"], reverse=True
    )[:10]

    for h in top_histograms:
        name = h["name"]
        labels = h["labels"]
        stats = h["stats"]

        print(f"\n{name}")
        if labels:
            print(f"  Labels: {labels}")
        print(f"  Observations: {h['count_delta']:.0f}")
        print(f"  Average: {h['avg']:.6f}")

        print()
        print(f"  {'Percentile':<12} {'Value':>12}")
        print(f"  {'─' * 26}")

        for pct in ["p50_estimate", "p90_estimate", "p95_estimate", "p99_estimate"]:
            val = stats.get(pct)
            val_str = f"{val:.6f}" if val is not None else "N/A"
            print(f"  {pct:<12} {val_str:>12}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_histogram_percentiles.py <json_file>")
        print(
            "Example: python analyze_histogram_percentiles.py artifacts/run4_c50/server_metrics_export.json"
        )
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    analyze_percentiles(json_path)


if __name__ == "__main__":
    main()
