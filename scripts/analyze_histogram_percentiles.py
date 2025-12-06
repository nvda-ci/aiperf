#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyze histogram percentiles from reprocessed server metrics JSON.

Compares three percentile estimation approaches:
1. Bucket interpolation (traditional linear interpolation between bucket boundaries)
2. Observed (extracted from single-observation scrape intervals)
3. Best-guess (includes +Inf bucket estimation via back-calculation)

Usage:
    python scripts/analyze_histogram_percentiles.py <json_file>

Example:
    python scripts/analyze_histogram_percentiles.py artifacts/run4_c50/server_metrics_reprocessed.json
"""

import sys
from pathlib import Path

import orjson


def analyze_percentiles(json_path: Path) -> None:
    """Analyze histogram percentiles from reprocessed JSON file."""
    with open(json_path, "rb") as f:
        data = orjson.loads(f.read())

    print("=" * 100)
    print("HISTOGRAM PERCENTILE ANALYSIS (3-Way Comparison)")
    print("=" * 100)
    print("\nApproaches:")
    print("  - Bucket:     Linear interpolation (returns ceiling for +Inf)")
    print("  - Observed:   Per-scrape extraction (skips +Inf observations)")
    print("  - Best-guess: Includes +Inf via back-calculation from total sum")
    print()

    all_histograms = []
    all_coverages = []
    all_differences_bucket_vs_observed = {"p50": [], "p90": [], "p95": [], "p99": []}
    all_differences_bucket_vs_best = {"p50": [], "p90": [], "p95": [], "p99": []}
    all_inf_bucket_fractions = []

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

                percentiles = stats.get("percentiles")
                if not percentiles:
                    continue

                bucket = percentiles.get("bucket", {})
                observed = percentiles.get("observed")
                best_guess = percentiles.get("best_guess")

                count_delta = stats.get("count_delta", 0)
                if count_delta == 0:
                    continue

                hist_info = {
                    "name": name,
                    "labels": label_str,
                    "count_delta": count_delta,
                    "bucket": bucket,
                    "observed": observed,
                    "best_guess": best_guess,
                }
                endpoint_histograms.append(hist_info)
                all_histograms.append(hist_info)

                if observed:
                    all_coverages.append(observed.get("coverage", 0))

                if best_guess:
                    inf_count = best_guess.get("inf_bucket_count", 0)
                    finite_count = best_guess.get("finite_observations_count", 0)
                    total = inf_count + finite_count
                    if total > 0:
                        all_inf_bucket_fractions.append(inf_count / total)

                # Calculate differences
                for pct in ["p50", "p90", "p95", "p99"]:
                    b_val = bucket.get(pct)

                    if observed:
                        o_val = observed.get(pct)
                        if b_val is not None and o_val is not None and b_val != 0:
                            diff_pct = (o_val - b_val) / b_val * 100
                            all_differences_bucket_vs_observed[pct].append(diff_pct)

                    if best_guess:
                        bg_val = best_guess.get(pct)
                        if b_val is not None and bg_val is not None and b_val != 0:
                            diff_pct = (bg_val - b_val) / b_val * 100
                            all_differences_bucket_vs_best[pct].append(diff_pct)

        # Print endpoint summary
        if endpoint_histograms:
            print(f"\nHistogram metrics: {len(endpoint_histograms)}")
            print()

            # Table header
            header = (
                f"{'Metric':<40} {'Count':>8} {'Coverage':>8} "
                f"{'InfFrac':>8} {'Conf':>6} {'p99 obs':>10} {'p99 best':>10}"
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
                observed = h["observed"]
                best_guess = h["best_guess"]

                coverage_str = ""
                inf_frac_str = ""
                conf_str = ""
                p99_obs_str = ""
                p99_best_str = ""

                if observed:
                    coverage = observed.get("coverage", 0)
                    coverage_str = f"{coverage:.1%}"
                    p99_obs = observed.get("p99")
                    if p99_obs:
                        p99_obs_str = f"{p99_obs:.4f}"

                if best_guess:
                    inf_count = best_guess.get("inf_bucket_count", 0)
                    finite_count = best_guess.get("finite_observations_count", 0)
                    total = inf_count + finite_count
                    if total > 0:
                        inf_frac_str = f"{inf_count / total:.1%}"
                    conf_str = best_guess.get("estimation_confidence", "")[:4]
                    p99_best = best_guess.get("p99")
                    if p99_best:
                        p99_best_str = f"{p99_best:.4f}"

                print(
                    f"{name:<40} {count:>8.0f} {coverage_str:>8} "
                    f"{inf_frac_str:>8} {conf_str:>6} {p99_obs_str:>10} {p99_best_str:>10}"
                )

    # Overall statistics
    print()
    print("=" * 100)
    print("OVERALL STATISTICS")
    print("=" * 100)

    print(f"\nTotal histogram series analyzed: {len(all_histograms)}")

    if all_coverages:
        print("\nObserved Coverage (exact observations / total):")
        print(f"  Min:    {min(all_coverages):.2%}")
        print(f"  Max:    {max(all_coverages):.2%}")
        print(f"  Mean:   {sum(all_coverages) / len(all_coverages):.2%}")
        print(f"  Median: {sorted(all_coverages)[len(all_coverages) // 2]:.2%}")

    if all_inf_bucket_fractions:
        print("\n+Inf Bucket Fraction (observations in +Inf / total):")
        print(f"  Min:    {min(all_inf_bucket_fractions):.2%}")
        print(f"  Max:    {max(all_inf_bucket_fractions):.2%}")
        print(
            f"  Mean:   {sum(all_inf_bucket_fractions) / len(all_inf_bucket_fractions):.2%}"
        )
        median_idx = len(all_inf_bucket_fractions) // 2
        print(f"  Median: {sorted(all_inf_bucket_fractions)[median_idx]:.2%}")

    print("\nPercentile differences vs Bucket interpolation:")
    print("\n  Observed vs Bucket (ignores +Inf):")
    for pct in ["p50", "p90", "p95", "p99"]:
        diffs = all_differences_bucket_vs_observed[pct]
        if diffs:
            print(
                f"    {pct}: mean={sum(diffs) / len(diffs):+.2f}%, MAE={sum(abs(d) for d in diffs) / len(diffs):.2f}%"
            )

    print("\n  Best-guess vs Bucket (includes +Inf):")
    for pct in ["p50", "p90", "p95", "p99"]:
        diffs = all_differences_bucket_vs_best[pct]
        if diffs:
            print(
                f"    {pct}: mean={sum(diffs) / len(diffs):+.2f}%, MAE={sum(abs(d) for d in diffs) / len(diffs):.2f}%"
            )

    # Detailed comparison for top metrics by count
    print()
    print("=" * 100)
    print("DETAILED COMPARISON (Top 10 by observation count)")
    print("=" * 100)

    top_histograms = sorted(
        all_histograms, key=lambda x: x["count_delta"], reverse=True
    )[:10]

    for h in top_histograms:
        name = h["name"]
        labels = h["labels"]
        bucket = h["bucket"]
        observed = h["observed"]
        best_guess = h["best_guess"]

        print(f"\n{name}")
        if labels:
            print(f"  Labels: {labels}")
        print(f"  Observations: {h['count_delta']:.0f}")

        if observed:
            exact = observed.get("exact_count", 0)
            placed = observed.get("bucket_placed_count", 0)
            coverage = observed.get("coverage", 0)
            print(
                f"  Observed: exact={exact}, bucket-placed={placed}, coverage={coverage:.2%}"
            )

        if best_guess:
            inf_count = best_guess.get("inf_bucket_count", 0)
            inf_mean = best_guess.get("inf_bucket_estimated_mean")
            conf = best_guess.get("estimation_confidence", "unknown")
            finite = best_guess.get("finite_observations_count", 0)
            learned = best_guess.get("buckets_with_learned_means", 0)
            inf_mean_str = f"{inf_mean:.4f}" if inf_mean is not None else "N/A"
            print(
                f"  Best-guess: +Inf count={inf_count}, +Inf mean={inf_mean_str}, confidence={conf}"
            )
            print(
                f"              finite={finite}, buckets with learned means={learned}"
            )

        print()
        print(
            f"  {'Percentile':<12} {'Bucket':>12} {'Observed':>12} {'Best-guess':>12} {'Obs diff':>12} {'Best diff':>12}"
        )
        print(f"  {'─' * 74}")

        for pct in ["p50", "p90", "p95", "p99", "p999"]:
            b_val = bucket.get(pct)
            o_val = observed.get(pct) if observed else None
            bg_val = best_guess.get(pct) if best_guess else None

            b_str = f"{b_val:.4f}" if b_val is not None else "N/A"
            o_str = f"{o_val:.4f}" if o_val is not None else "N/A"
            bg_str = f"{bg_val:.4f}" if bg_val is not None else "N/A"

            o_diff_str = ""
            if b_val is not None and o_val is not None and b_val != 0:
                diff = (o_val - b_val) / b_val * 100
                o_diff_str = f"{diff:+.2f}%"

            bg_diff_str = ""
            if b_val is not None and bg_val is not None and b_val != 0:
                diff = (bg_val - b_val) / b_val * 100
                bg_diff_str = f"{diff:+.2f}%"

            print(
                f"  {pct:<12} {b_str:>12} {o_str:>12} {bg_str:>12} {o_diff_str:>12} {bg_diff_str:>12}"
            )


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_histogram_percentiles.py <json_file>")
        print(
            "Example: python analyze_histogram_percentiles.py artifacts/run4_c50/server_metrics_reprocessed.json"
        )
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    analyze_percentiles(json_path)


if __name__ == "__main__":
    main()
