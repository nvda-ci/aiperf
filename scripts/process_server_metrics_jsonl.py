#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple script to read an exported server metrics JSONL file and compute summaries.

Reads the JSONL file and metadata file, processes through the post-processor engine,
and exports computed statistics to a new JSON file.

Usage:
    python scripts/process_server_metrics_jsonl.py <jsonl_file>

Example:
    python scripts/process_server_metrics_jsonl.py artifacts/run4_c50/server_metrics_export.jsonl
"""

import sys
from pathlib import Path

import orjson

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aiperf.common.models.export_models import (
    ServerMetricLabeledStats,
    ServerMetricsEndpointSummary,
    ServerMetricSummary,
)
from aiperf.common.models.server_metrics_models import (
    HistogramData,
    MetricFamily,
    MetricSample,
    MetricSchema,
    PrometheusMetricType,
    ServerMetricsHierarchy,
    ServerMetricsMetadataFile,
    ServerMetricsRecord,
    ServerMetricsSlimRecord,
    SummaryData,
    TimeRangeFilter,
)
from aiperf.exporters.display_units_utils import normalize_endpoint_display


def load_metadata(metadata_path: Path) -> ServerMetricsMetadataFile:
    """Load metadata file."""
    with open(metadata_path, "rb") as f:
        data = orjson.loads(f.read())
    return ServerMetricsMetadataFile.model_validate(data)


def slim_to_full_record(
    slim: ServerMetricsSlimRecord,
    metadata_file: ServerMetricsMetadataFile,
) -> ServerMetricsRecord:
    """Convert a slim record back to a full record using metadata."""
    endpoint_metadata = metadata_file.endpoints.get(slim.endpoint_url)
    schemas = endpoint_metadata.metric_schemas if endpoint_metadata else {}

    metrics: dict[str, MetricFamily] = {}

    for metric_name, slim_samples in slim.metrics.items():
        schema = schemas.get(metric_name, MetricSchema(type="gauge", description=""))
        metric_type = PrometheusMetricType(schema.type.lower())

        samples = []
        for slim_sample in slim_samples:
            # Convert slim sample back to full sample
            sample = MetricSample(
                labels=slim_sample.labels,
                value=slim_sample.value,
            )

            # Handle histogram
            if slim_sample.histogram is not None:
                sample.histogram = HistogramData(
                    buckets=slim_sample.histogram,
                    sum=slim_sample.sum,
                    count=slim_sample.count,
                )

            # Handle summary
            if slim_sample.summary is not None:
                sample.summary = SummaryData(
                    quantiles=slim_sample.summary,
                    sum=slim_sample.sum,
                    count=slim_sample.count,
                )

            samples.append(sample)

        metrics[metric_name] = MetricFamily(
            type=metric_type,
            description=schema.description or "",
            samples=samples,
        )

    return ServerMetricsRecord(
        endpoint_url=slim.endpoint_url,
        timestamp_ns=slim.timestamp_ns,
        endpoint_latency_ns=slim.endpoint_latency_ns,
        metrics=metrics,
    )


def parse_metric_key(full_key: str) -> tuple[str, dict[str, str] | None]:
    """Parse a metric key with label suffix into base name and labels dict."""
    if "|" not in full_key:
        return full_key, None

    base_name, label_str = full_key.split("|", 1)

    labels = {}
    for pair in label_str.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            labels[k] = v

    return base_name, labels if labels else None


def compute_summaries(
    hierarchy: ServerMetricsHierarchy,
    time_filter: TimeRangeFilter | None = None,
) -> dict[str, ServerMetricsEndpointSummary]:
    """Compute server metrics summaries from hierarchy."""
    summaries: dict[str, ServerMetricsEndpointSummary] = {}

    for endpoint_url, endpoint_data in hierarchy.endpoints.items():
        endpoint_display = normalize_endpoint_display(endpoint_url)

        # Build unified metrics dict: base_metric_name -> ServerMetricSummary
        metrics: dict[str, ServerMetricSummary] = {}

        # Each metric type knows how to export itself
        for (
            metric_key,
            metric_type,
            export_stats,
        ) in endpoint_data.time_series.iter_export_stats(time_filter):
            base_name, labels = parse_metric_key(metric_key)

            series_stats = ServerMetricLabeledStats(
                labels=labels if labels else None,
                stats=export_stats,
            )

            if base_name not in metrics:
                # Look up description from metadata
                schema = endpoint_data.metadata.metric_schemas.get(base_name)
                description = schema.description if schema else ""

                metrics[base_name] = ServerMetricSummary(
                    type=metric_type,
                    description=description,
                    series=[series_stats],
                )
            else:
                metrics[base_name].series.append(series_stats)

        # Compute collection stats
        ts = endpoint_data.time_series
        scrape_count = len(ts)
        duration_seconds = (
            (ts.last_timestamp_ns - ts.first_timestamp_ns) / 1e9
            if scrape_count > 1
            else 0.0
        )
        latencies = ts._scrape_latencies_ns
        avg_scrape_latency_ms = (
            sum(latencies) / len(latencies) / 1e6 if latencies else 0.0
        )

        # Get info metrics from metadata
        info_metrics = endpoint_data.metadata.info_metrics or None

        summaries[endpoint_display] = ServerMetricsEndpointSummary(
            endpoint_url=endpoint_url,
            duration_seconds=duration_seconds,
            scrape_count=scrape_count,
            avg_scrape_latency_ms=avg_scrape_latency_ms,
            info_metrics=info_metrics,
            metrics=metrics,
        )

    return summaries


def main():
    if len(sys.argv) < 2:
        print("Usage: python process_server_metrics_jsonl.py <jsonl_file>")
        print(
            "Example: python process_server_metrics_jsonl.py artifacts/run4_c50/server_metrics_export.jsonl"
        )
        sys.exit(1)

    jsonl_path = Path(sys.argv[1])
    if not jsonl_path.exists():
        print(f"Error: File not found: {jsonl_path}")
        sys.exit(1)

    # Derive metadata path from jsonl path
    metadata_path = jsonl_path.parent / "server_metrics_metadata.json"
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        sys.exit(1)

    # Output path (different name to avoid conflicts)
    output_path = jsonl_path.parent / "server_metrics_reprocessed.json"

    print(f"Reading JSONL: {jsonl_path}")
    print(f"Reading metadata: {metadata_path}")

    # Load metadata
    metadata_file = load_metadata(metadata_path)
    print(f"Loaded metadata for {len(metadata_file.endpoints)} endpoints")

    # Build hierarchy from JSONL records
    hierarchy = ServerMetricsHierarchy()
    record_count = 0
    error_count = 0

    with open(jsonl_path, "rb") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                data = orjson.loads(line)
                slim_record = ServerMetricsSlimRecord.model_validate(data)
                full_record = slim_to_full_record(slim_record, metadata_file)
                hierarchy.add_record(full_record)
                record_count += 1
            except orjson.JSONDecodeError as e:
                print(f"  Warning: Invalid JSON at line {line_num}: {e}")
                error_count += 1
                continue
            except Exception as e:
                print(f"  Warning: Failed to process line {line_num}: {e}")
                error_count += 1
                continue

            if record_count % 100 == 0:
                print(f"  Processed {record_count} records...")

    print(f"Processed {record_count} total records")
    if error_count > 0:
        print(f"  Skipped {error_count} records due to errors")
    print(f"Endpoints found: {list(hierarchy.endpoints.keys())}")

    # Compute summaries (no time filter - use all data)
    print("Computing summaries...")
    summaries = compute_summaries(hierarchy)

    # Build output structure
    output_data = {
        "endpoints": {
            endpoint: summary.model_dump(mode="json", exclude_none=True)
            for endpoint, summary in summaries.items()
        }
    }

    # Write output
    print(f"Writing output: {output_path}")
    with open(output_path, "wb") as f:
        f.write(orjson.dumps(output_data, option=orjson.OPT_INDENT_2))

    print("Done!")

    # Print summary stats
    for endpoint, summary in summaries.items():
        print(f"\n{endpoint}:")
        print(f"  Duration: {summary.duration_seconds:.2f}s")
        print(f"  Scrapes: {summary.scrape_count}")
        print(f"  Avg latency: {summary.avg_scrape_latency_ms:.2f}ms")
        print(f"  Metrics: {len(summary.metrics)}")

        # Show a few histogram metrics with percentiles
        histogram_count = 0
        for name, metric in summary.metrics.items():
            if metric.type == "histogram" and histogram_count < 3:
                for series in metric.series[:1]:
                    stats = series.stats
                    if hasattr(stats, "percentiles") and stats.percentiles:
                        bucket = stats.percentiles.bucket
                        observed = stats.percentiles.observed
                        print(f"    {name}:")
                        if bucket:
                            print(
                                f"      bucket: p50={bucket.p50:.4f}, p99={bucket.p99:.4f}"
                                if bucket.p50
                                else "      bucket: None"
                            )
                        if observed:
                            p50_str = (
                                f"{observed.p50:.4f}"
                                if observed.p50 is not None
                                else "None"
                            )
                            print(
                                f"      observed: p50={p50_str}, coverage={observed.coverage:.2%}"
                            )
                        histogram_count += 1


if __name__ == "__main__":
    main()
