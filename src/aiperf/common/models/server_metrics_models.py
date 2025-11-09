# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from pydantic import Field

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.metric_utils import compute_histogram_delta
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.error_models import ErrorDetails, ErrorDetailsCount

if TYPE_CHECKING:
    from aiperf.common.models.record_models import MetricResult


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


class MetricSample(AIPerfBaseModel):
    """Single metric sample with labels and value."""

    labels: dict[str, str] = Field(
        description="Metric labels (excluding histogram/summary special labels)"
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


class MetricFamily(AIPerfBaseModel):
    """Group of related metrics with same name and type."""

    type: PrometheusMetricType = Field(description="Metric type as enum")
    help: str = Field(description="Metric description from HELP text")
    samples: list[MetricSample] = Field(
        description="Metric samples grouped by base labels"
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


class ServerMetricsMetadata(AIPerfBaseModel):
    """Metadata for a server metrics endpoint that doesn't change over time."""

    endpoint_url: str = Field(description="Prometheus metrics endpoint URL")
    endpoint_display: str = Field(description="Human-readable endpoint display name")


class ServerMetricsSnapshotTimeSeries(AIPerfBaseModel):
    """Time series of complete Prometheus snapshots for one endpoint.

    Uses the shared compute_metric_statistics() utility for consistent statistics
    computation across GPU telemetry and server metrics. Supports time-window
    filtering and histogram-based percentile estimation.
    """

    snapshots: list[tuple[int, dict[str, MetricFamily]]] = Field(
        default_factory=list, description="List of (timestamp_ns, metrics) tuples"
    )
    max_snapshots: int = Field(
        default=300, description="Maximum snapshots to retain (prevents OOM)"
    )

    def append_snapshot(
        self, timestamp_ns: int, metrics: dict[str, MetricFamily]
    ) -> None:
        """Add new snapshot.

        Args:
            timestamp_ns: Timestamp when measurements were taken
            metrics: Complete Prometheus metrics snapshot
        """
        self.snapshots.append((timestamp_ns, metrics))

        # Enforce retention limit to prevent unbounded memory growth
        if len(self.snapshots) > self.max_snapshots:
            # Keep most recent snapshots
            self.snapshots = self.snapshots[-self.max_snapshots :]

    def filter_by_time_window(
        self, min_timestamp_ns: int | None, max_timestamp_ns: int | None
    ) -> "ServerMetricsSnapshotTimeSeries":
        """Create a new time series filtered to a specific time window.

        Args:
            min_timestamp_ns: Start of time window (inclusive), or None for no lower bound
            max_timestamp_ns: End of time window (inclusive), or None for no upper bound

        Returns:
            New ServerMetricsSnapshotTimeSeries with only snapshots in the time window
        """
        filtered_snapshots = []
        for ts, metrics in self.snapshots:
            if min_timestamp_ns is not None and ts < min_timestamp_ns:
                continue
            if max_timestamp_ns is not None and ts > max_timestamp_ns:
                continue
            filtered_snapshots.append((ts, metrics))

        return ServerMetricsSnapshotTimeSeries(snapshots=filtered_snapshots)

    @staticmethod
    def _labels_match(
        sample_labels: dict[str, str], filter_labels: dict[str, str]
    ) -> bool:
        """Helper to match labels: empty filter dict means 'match any labels'.

        Args:
            sample_labels: Labels from a metric sample
            filter_labels: Labels to filter by (empty dict matches any labels)

        Returns:
            True if labels match, False otherwise
        """
        if not filter_labels:  # Empty filter matches any labels
            return True
        return sample_labels == filter_labels

    def get_metric_values(
        self, metric_name: str, labels: dict[str, str]
    ) -> list[tuple[float, int]]:
        """Extract time series data for a specific metric with specific labels.

        Args:
            metric_name: Name of the metric family to extract
            labels: Label combination to match (empty dict matches any labels)

        Returns:
            List of (value, timestamp_ns) tuples for the specified metric and labels
        """

        values = []
        for timestamp_ns, metrics in self.snapshots:
            if metric_name not in metrics:
                continue

            metric_family = metrics[metric_name]
            for sample in metric_family.samples:
                if (
                    self._labels_match(sample.labels, labels)
                    and sample.value is not None
                ):
                    values.append((sample.value, timestamp_ns))
                    break

        return values

    def get_histogram_delta(
        self, metric_name: str, labels: dict[str, str]
    ) -> tuple[dict[str, float], float, float] | None:
        """Get histogram delta between first and last snapshot in time series.

        For cumulative histogram metrics, computes the delta by subtracting
        the first snapshot from the last snapshot. This gives the histogram
        data for just the time window.

        Args:
            metric_name: Name of the histogram metric family
            labels: Label combination to match

        Returns:
            Tuple of (buckets_delta, sum_delta, count_delta) or None if not found
        """
        if not self.snapshots:
            return None

        first_histogram = None
        last_histogram = None

        # Find first histogram
        for _, metrics in self.snapshots:
            if metric_name not in metrics:
                continue
            metric_family = metrics[metric_name]
            if metric_family.type != PrometheusMetricType.HISTOGRAM:
                continue
            for sample in metric_family.samples:
                if self._labels_match(sample.labels, labels) and sample.histogram:
                    first_histogram = sample.histogram
                    break
            if first_histogram:
                break

        # Find last histogram
        for _, metrics in reversed(self.snapshots):
            if metric_name not in metrics:
                continue
            metric_family = metrics[metric_name]
            for sample in metric_family.samples:
                if self._labels_match(sample.labels, labels) and sample.histogram:
                    last_histogram = sample.histogram
                    break
            if last_histogram:
                break

        if not first_histogram or not last_histogram:
            return None

        try:
            buckets_delta = compute_histogram_delta(
                first_histogram.buckets, last_histogram.buckets
            )
            sum_delta = last_histogram.sum - first_histogram.sum
            count_delta = last_histogram.count - first_histogram.count

            return buckets_delta, sum_delta, count_delta
        except ValueError:
            # Bucket boundaries don't match between snapshots (Prometheus server config changed)
            # Return None to indicate invalid data - caller will raise NoMetricValue
            return None

    def get_counter_delta(
        self, metric_name: str, labels: dict[str, str]
    ) -> float | None:
        """Get counter delta between first and last snapshot in time series.

        For cumulative counter metrics, computes the delta by subtracting
        the first snapshot from the last snapshot.

        Args:
            metric_name: Name of the counter metric family
            labels: Label combination to match (empty dict matches any labels)

        Returns:
            Delta value or None if not found
        """
        if not self.snapshots:
            return None

        first_value = None
        last_value = None

        # Find first counter value
        for _, metrics in self.snapshots:
            if metric_name not in metrics:
                continue
            metric_family = metrics[metric_name]
            if metric_family.type != PrometheusMetricType.COUNTER:
                continue
            for sample in metric_family.samples:
                if (
                    self._labels_match(sample.labels, labels)
                    and sample.value is not None
                ):
                    first_value = sample.value
                    break
            if first_value is not None:
                break

        # Find last counter value
        for _, metrics in reversed(self.snapshots):
            if metric_name not in metrics:
                continue
            metric_family = metrics[metric_name]
            for sample in metric_family.samples:
                if (
                    self._labels_match(sample.labels, labels)
                    and sample.value is not None
                ):
                    last_value = sample.value
                    break
            if last_value is not None:
                break

        if first_value is None or last_value is None:
            return None

        return last_value - first_value

    def get_summary_delta(
        self, metric_name: str, labels: dict[str, str]
    ) -> tuple[dict[str, float], float, float] | None:
        """Get summary delta between first and last snapshot in time series.

        For cumulative summary metrics, computes the delta by subtracting
        the first snapshot from the last snapshot. Quantiles are taken from
        the last snapshot.

        Args:
            metric_name: Name of the summary metric family
            labels: Label combination to match (empty dict matches any labels)

        Returns:
            Tuple of (quantiles, sum_delta, count_delta) or None if not found
        """
        if not self.snapshots:
            return None

        first_summary = None
        last_summary = None

        # Find first summary
        for _, metrics in self.snapshots:
            if metric_name not in metrics:
                continue
            metric_family = metrics[metric_name]
            if metric_family.type != PrometheusMetricType.SUMMARY:
                continue
            for sample in metric_family.samples:
                if self._labels_match(sample.labels, labels) and sample.summary:
                    first_summary = sample.summary
                    break
            if first_summary:
                break

        # Find last summary
        for _, metrics in reversed(self.snapshots):
            if metric_name not in metrics:
                continue
            metric_family = metrics[metric_name]
            for sample in metric_family.samples:
                if self._labels_match(sample.labels, labels) and sample.summary:
                    last_summary = sample.summary
                    break
            if last_summary:
                break

        if not first_summary or not last_summary:
            return None

        # Compute deltas - use quantiles from last snapshot (most recent)
        sum_delta = last_summary.sum - first_summary.sum
        count_delta = last_summary.count - first_summary.count

        return last_summary.quantiles, sum_delta, count_delta

    def to_metric_result(
        self, metric_name: str, labels: dict[str, str], tag: str, header: str, unit: str
    ) -> "MetricResult":
        """Convert metric time series to MetricResult with statistical summary.

        This method intelligently handles different metric types:
        - Histograms: Uses bucket data to estimate percentiles (via delta)
        - Counters: Uses delta between first and last value
        - Gauges: Uses all values in time window
        - Summaries: Uses pre-computed quantiles from last snapshot

        Args:
            metric_name: Name of the metric family to analyze
            labels: Label combination to match
            tag: Unique identifier for this metric (used by dashboard, exports, API)
            header: Human-readable name for display
            unit: Unit of measurement (e.g., "req/s", "ms", "%")

        Returns:
            MetricResult with min/max/avg/percentiles computed appropriately

        Raises:
            NoMetricValue: If no data points are available for the specified metric
        """
        from aiperf.common.exceptions import NoMetricValue
        from aiperf.common.metric_utils import (
            compute_metric_statistics,
            compute_metric_statistics_from_histogram,
        )
        from aiperf.common.models.record_models import MetricResult

        if not self.snapshots:
            raise NoMetricValue(
                f"No snapshots available for metric '{metric_name}' "
                f"(time series has {len(self.snapshots)} snapshots)"
            )

        # Determine metric type from first snapshot
        metric_type = None
        for _, metrics in self.snapshots:
            if metric_name in metrics:
                metric_type = metrics[metric_name].type
                break

        if not metric_type:
            # List available metrics for debugging
            available_metrics = set()
            for _, metrics in self.snapshots:
                available_metrics.update(metrics.keys())
            raise NoMetricValue(
                f"Metric '{metric_name}' not found in any of {len(self.snapshots)} snapshot(s). "
                f"Available metrics: {sorted(list(available_metrics))[:10]}..."
            )

        # Handle histogram metrics: use bucket data to estimate percentiles
        if metric_type == PrometheusMetricType.HISTOGRAM:
            histogram_delta = self.get_histogram_delta(metric_name, labels)
            if histogram_delta:
                buckets, sum_delta, count_delta = histogram_delta
                if count_delta > 0:
                    metric_result = compute_metric_statistics_from_histogram(
                        buckets=buckets,
                        sum_value=sum_delta,
                        count=count_delta,
                        tag=tag,
                        header=header,
                        unit=unit,
                        metric_name=metric_name,
                    )
                    # Store raw histogram delta for JSON export
                    metric_result.raw_histogram_delta = histogram_delta
                    return metric_result
            raise NoMetricValue(f"No valid histogram data for '{metric_name}'")

        # Handle counter metrics: use delta
        if metric_type == PrometheusMetricType.COUNTER:
            counter_delta = self.get_counter_delta(metric_name, labels)
            if counter_delta is not None:
                # Return a simplified MetricResult for counters
                from aiperf.common.models import MetricResult

                metric_result = MetricResult(
                    tag=tag,
                    header=header,
                    unit=unit,
                    min=counter_delta,
                    max=counter_delta,
                    avg=counter_delta,
                    std=0.0,
                    count=1,
                    current=counter_delta,
                    p1=counter_delta,
                    p5=counter_delta,
                    p10=counter_delta,
                    p25=counter_delta,
                    p50=counter_delta,
                    p75=counter_delta,
                    p90=counter_delta,
                    p95=counter_delta,
                    p99=counter_delta,
                )
                # Store raw counter delta for JSON export
                metric_result.raw_counter_delta = counter_delta
                return metric_result
            raise NoMetricValue(f"No valid counter data for '{metric_name}'")

        # Handle summary metrics: use pre-computed quantiles from last snapshot
        if metric_type == PrometheusMetricType.SUMMARY:
            summary_delta = self.get_summary_delta(metric_name, labels)
            if summary_delta:
                quantiles, sum_delta, count_delta = summary_delta
                # Create MetricResult from summary quantiles
                from aiperf.common.models import MetricResult

                # Map quantiles to percentiles (0.5 -> p50, 0.95 -> p95, etc.)
                metric_result = MetricResult(
                    tag=tag,
                    header=header,
                    unit=unit,
                    avg=sum_delta / count_delta if count_delta > 0 else 0.0,
                    count=int(count_delta),
                    min=quantiles.get("0.0", quantiles.get("0.01", None)),
                    max=quantiles.get("1.0", quantiles.get("0.99", None)),
                    p50=quantiles.get("0.5", None),
                    p90=quantiles.get("0.9", None),
                    p95=quantiles.get("0.95", None),
                    p99=quantiles.get("0.99", None),
                )
                # Store raw summary data for JSON export
                metric_result.raw_summary_delta = summary_delta
                return metric_result
            raise NoMetricValue(f"No valid summary data for '{metric_name}'")

        # Handle gauge metrics (and default): use time series
        data_points = self.get_metric_values(metric_name, labels)
        return compute_metric_statistics(
            data_points=data_points,
            tag=tag,
            header=header,
            unit=unit,
            metric_name=f"{metric_name} with labels {labels}",
        )


class ServerMetricsData(AIPerfBaseModel):
    """Complete server metrics data for one endpoint: metadata + time series."""

    metadata: ServerMetricsMetadata = Field(description="Static endpoint information")
    time_series: ServerMetricsSnapshotTimeSeries = Field(
        default_factory=ServerMetricsSnapshotTimeSeries,
        description="Time series of complete snapshots",
    )

    def add_record(self, record: ServerMetricsRecord) -> None:
        """Add server metrics record as a complete snapshot.

        Args:
            record: New server metrics data point from Prometheus collector
        """
        self.time_series.append_snapshot(record.timestamp_ns, record.metrics)

    def get_metric_result(
        self,
        metric_name: str,
        labels: dict[str, str],
        tag: str,
        header: str,
        unit: str,
        min_timestamp_ns: int | None = None,
        max_timestamp_ns: int | None = None,
    ) -> "MetricResult":
        """Get MetricResult for a specific metric with specific labels.

        Optionally filters to a time window before computing statistics.
        This ensures server metrics align with the actual inference benchmark
        time window (min_request_timestamp_ns to max_response_timestamp_ns).

        Args:
            metric_name: Name of the metric family to analyze
            labels: Label combination to match
            tag: Unique identifier for this metric
            header: Human-readable name for display
            unit: Unit of measurement
            min_timestamp_ns: Optional start of time window (inclusive)
            max_timestamp_ns: Optional end of time window (inclusive)

        Returns:
            MetricResult with statistical summary for the specified metric
        """
        time_series = self.time_series

        # Filter to time window if specified
        if min_timestamp_ns is not None or max_timestamp_ns is not None:
            time_series = time_series.filter_by_time_window(
                min_timestamp_ns, max_timestamp_ns
            )

        return time_series.to_metric_result(metric_name, labels, tag, header, unit)


class ServerMetricsHierarchy(AIPerfBaseModel):
    """Hierarchical storage: endpoint_url -> complete server metrics data.

    Structure:
    {
        "http://localhost:8081/metrics": ServerMetricsData(metadata + time series),
        "http://node2:9090/metrics": ServerMetricsData(metadata + time series)
    }
    """

    endpoints: dict[str, ServerMetricsData] = Field(
        default_factory=dict,
        description="Dict: endpoint_url -> server metrics data",
    )

    def add_record(self, record: ServerMetricsRecord) -> None:
        """Add server metrics record to hierarchical storage.

        Args:
            record: New server metrics data from Prometheus endpoint

        Note: Automatically creates hierarchy levels as needed
        """
        if record.endpoint_url not in self.endpoints:
            metadata = ServerMetricsMetadata(
                endpoint_url=record.endpoint_url,
                endpoint_display=self._normalize_endpoint_display(record.endpoint_url),
            )
            self.endpoints[record.endpoint_url] = ServerMetricsData(metadata=metadata)

        self.endpoints[record.endpoint_url].add_record(record)

    @staticmethod
    def _normalize_endpoint_display(url: str) -> str:
        """Normalize endpoint URL for display.

        Args:
            url: Full endpoint URL

        Returns:
            Cleaned URL for display (removes http://, /metrics suffix)
        """
        display = url.replace("http://", "").replace("https://", "")
        if display.endswith("/metrics"):
            display = display[: -len("/metrics")]
        return display


class ServerMetricsResults(AIPerfBaseModel):
    """Results from server metrics collection during a profile run.

    This class contains all server metrics data collected during
    a benchmarking session, separate from inference performance results.
    """

    metrics_data: ServerMetricsHierarchy = Field(
        description="Hierarchical metrics data organized by endpoint"
    )
    start_ns: int = Field(description="Start time of metrics collection in nanoseconds")
    end_ns: int = Field(description="End time of metrics collection in nanoseconds")
    endpoints_configured: list[str] = Field(
        default_factory=list,
        description="List of Prometheus endpoint URLs configured",
    )
    endpoints_successful: list[str] = Field(
        default_factory=list,
        description="List of Prometheus endpoint URLs that successfully provided data",
    )
    error_summary: list[ErrorDetailsCount] = Field(
        default_factory=list,
        description="A list of the unique error details and their counts",
    )
    aggregated_metrics: list["MetricResult"] = Field(
        default_factory=list,
        description="Aggregated statistics for tracked metrics across the entire run",
    )


class ProcessServerMetricsResult(AIPerfBaseModel):
    """Result of server metrics processing - mirrors ProcessTelemetryResult pattern."""

    results: ServerMetricsResults = Field(
        description="The processed server metrics results"
    )
    errors: list[ErrorDetails] = Field(
        default_factory=list,
        description="Any errors that occurred while processing server metrics data",
    )
