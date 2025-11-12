# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.base_models import AIPerfBaseModel


class HistogramData(AIPerfBaseModel):
    """Structured histogram data with buckets, sum, and count."""

    buckets: dict[str, float] = Field(
        description="Bucket upper bounds to counts {le: value}"
    )
    sum: float | None = Field(default=None, description="Sum of all observed values")
    count: float | None = Field(
        default=None, description="Total number of observations"
    )


class SummaryData(AIPerfBaseModel):
    """Structured summary data with quantiles, sum, and count."""

    quantiles: dict[str, float] = Field(
        description="Quantile to value {quantile: value}"
    )
    sum: float | None = Field(default=None, description="Sum of all observed values")
    count: float | None = Field(
        default=None, description="Total number of observations"
    )


# Note: SlimHistogramData and SlimSummaryData removed in favor of array-based format
# Histogram/summary data now stored as arrays directly in SlimMetricSample with
# sum/count as optional fields at the sample level


class MetricSample(AIPerfBaseModel):
    """Single metric sample with labels and value."""

    labels: dict[str, str] | None = Field(
        default=None,
        description="Metric labels (excluding histogram/summary special labels). None if no labels.",
    )
    value: float | None = Field(
        default=None, description="Simple metric value (counter/gauge)"
    )
    histogram: HistogramData | None = Field(
        default=None, description="Histogram data if metric is histogram type"
    )
    summary: SummaryData | None = Field(
        default=None, description="Summary data if metric is summary type"
    )


class SlimMetricSample(AIPerfBaseModel):
    """Slim metric sample with minimal data using array-based format.

    Optimized for ZMQ messages and JSONL export. Uses array-based format for
    histogram/summary data for maximum compactness:
    - Type and help text are in schema
    - Histogram bucket labels (le values) are in schema, only counts sent here as array
    - Summary quantile labels are in schema, only values sent here as array
    - sum/count are optional fields at sample level (used for histogram/summary)

    Format examples:
    - Counter/Gauge: {"value": 42.0} or {"labels": {...}, "value": 42.0}
    - Histogram: {"histogram": [10, 25, 50, ...], "sum": 100.0, "count": 50}
    - Summary: {"summary": [0.5, 0.95, 0.99, ...], "sum": 100.0, "count": 50}
    """

    labels: dict[str, str] | None = Field(
        default=None,
        description="Metric labels (excluding histogram/summary special labels). None if no labels.",
    )
    value: float | None = Field(
        default=None, description="Simple metric value (counter/gauge)"
    )
    histogram: list[float] | None = Field(
        default=None,
        description="Histogram bucket counts in order matching schema bucket_labels",
    )
    summary: list[float] | None = Field(
        default=None,
        description="Summary quantile values in order matching schema quantile_labels",
    )
    sum: float | None = Field(
        default=None,
        description="Sum of all observed values (for histogram/summary)",
    )
    count: float | None = Field(
        default=None,
        description="Total number of observations (for histogram/summary)",
    )


class MetricFamily(AIPerfBaseModel):
    """Group of related metrics with same name and type."""

    type: PrometheusMetricType = Field(description="Metric type as enum")
    help: str = Field(description="Metric description from HELP text")
    samples: list[MetricSample] = Field(
        description="Metric samples grouped by base labels"
    )


class MetricSchema(AIPerfBaseModel):
    """Schema information for a metric (type, help text, and bucket/quantile labels).

    Sent once per metric in ServerMetricsMetadata to avoid repeating in every record.
    For histograms and summaries, includes static bucket/quantile labels so only
    counts/values need to be sent in each record.
    """

    type: PrometheusMetricType = Field(description="Metric type as enum")
    help: str = Field(description="Metric description from HELP text")
    bucket_labels: list[str] | None = Field(
        default=None,
        description="Histogram bucket upper bounds (le values) in order. Only for histogram metrics.",
    )
    quantile_labels: list[str] | None = Field(
        default=None,
        description="Summary quantile labels in order. Only for summary metrics.",
    )


class ServerMetricsSlimRecord(AIPerfBaseModel):
    """Slim server metrics record containing only time-varying data.

    This record excludes static metadata (endpoint_url, metric types, help text)
    to reduce ZMQ message size and JSONL file size. The metadata and schemas are sent
    once separately via ServerMetricsMetadataMessage.

    Format is optimized with flat structure and slim samples:
    - Metrics map directly to sample lists (no 'samples' key nesting)
    - Histogram/summary samples only include counts/values (bucket/quantile labels in schema)
    """

    timestamp_ns: int = Field(
        description="Nanosecond wall-clock timestamp when metrics were collected (time_ns)"
    )
    metrics: dict[str, list[SlimMetricSample]] = Field(
        description="Metrics grouped by family name, mapping directly to slim sample list"
    )


class ServerMetricsRecord(AIPerfBaseModel):
    """Single server metrics data point from Prometheus endpoint.

    This record contains all metrics scraped from one Prometheus endpoint at one point in time.
    Used for hierarchical storage: endpoint_url -> time series data.
    """

    timestamp_ns: int = Field(
        description="Nanosecond wall-clock timestamp when metrics were collected (time_ns)"
    )
    endpoint_url: str = Field(
        description="Source Prometheus metrics endpoint URL (e.g., 'http://localhost:8081/metrics')"
    )
    metrics: dict[str, MetricFamily] = Field(
        description="Metrics grouped by family name"
    )

    def to_slim(self) -> ServerMetricsSlimRecord:
        """Convert to slim record using array-based format for histograms/summaries.

        Creates flat structure where metrics map directly to slim sample lists.
        For histograms and summaries, uses array format with bucket counts/quantile values
        and sum/count at the sample level.

        Returns:
            ServerMetricsSlimRecord with only timestamp and slim samples (flat structure)
        """
        slim_metrics = {}
        for name, family in self.metrics.items():
            slim_samples = []
            for sample in family.samples:
                # Convert histogram data to array format (only counts)
                histogram_counts = None
                hist_sum = None
                hist_count = None
                if sample.histogram:
                    # Extract counts in sorted order by bucket label
                    sorted_buckets = sorted(
                        sample.histogram.buckets.items(), key=lambda x: float(x[0])
                    )
                    histogram_counts = [count for _, count in sorted_buckets]
                    hist_sum = sample.histogram.sum
                    hist_count = sample.histogram.count

                # Convert summary data to array format (only values)
                summary_values = None
                summ_sum = None
                summ_count = None
                if sample.summary:
                    # Extract values in sorted order by quantile label
                    sorted_quantiles = sorted(
                        sample.summary.quantiles.items(), key=lambda x: float(x[0])
                    )
                    summary_values = [value for _, value in sorted_quantiles]
                    summ_sum = sample.summary.sum
                    summ_count = sample.summary.count

                slim_samples.append(
                    SlimMetricSample(
                        labels=sample.labels,
                        value=sample.value,
                        histogram=histogram_counts,
                        summary=summary_values,
                        sum=hist_sum if hist_sum is not None else summ_sum,
                        count=hist_count if hist_count is not None else summ_count,
                    )
                )
            slim_metrics[name] = slim_samples

        return ServerMetricsSlimRecord(
            timestamp_ns=self.timestamp_ns,
            metrics=slim_metrics,
        )


class ServerMetricsMetadata(AIPerfBaseModel):
    """Metadata for a server metrics endpoint that doesn't change over time.

    Includes metric schemas (type and help text) to avoid sending them in every record.
    """

    endpoint_url: str = Field(description="Prometheus metrics endpoint URL")
    endpoint_display: str = Field(description="Human-readable endpoint display name")
    metric_schemas: dict[str, MetricSchema] = Field(
        default_factory=dict,
        description="Metric schemas (type and help) sent once to avoid repetition",
    )
