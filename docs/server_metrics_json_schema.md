<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Server Metrics JSON Export Schema

This document describes the structure and semantics of every field in the server metrics JSON export format.

## Overview

The server metrics JSON export provides aggregated statistics from Prometheus metrics collected during a benchmark run. Metrics are organized in a hybrid format:
- **Metrics keyed by name** for O(1) lookup
- **Flat stats within each series** for easy access

```
data["metrics"]["metric_name"]["series"][0]["p99"]
```

---

## Top-Level Structure

```json
{
  "summary": { ... },
  "metrics": { ... }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `summary` | object | Collection metadata and endpoint information |
| `metrics` | object | Metrics keyed by name, each containing type info and series data |

---

## Summary Section

```json
"summary": {
  "endpoints_configured": ["localhost:10000", "localhost:10001"],
  "endpoints_successful": ["localhost:10000", "localhost:10001"],
  "start_time": "2025-12-06T08:52:09.880074",
  "end_time": "2025-12-06T08:52:30.688461",
  "endpoint_info": { ... }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `endpoints_configured` | array[string] | Normalized endpoint identifiers that were configured for scraping |
| `endpoints_successful` | array[string] | Normalized endpoint identifiers that returned data |
| `start_time` | datetime | When metrics collection started (ISO 8601) |
| `end_time` | datetime | When metrics collection ended (ISO 8601) |
| `endpoint_info` | object | Per-endpoint collection metadata |

### Endpoint Info

```json
"endpoint_info": {
  "localhost:10000": {
    "endpoint_url": "http://localhost:10000/metrics",
    "duration_seconds": 21.78,
    "scrape_count": 68,
    "avg_scrape_latency_ms": 325.94,
    "avg_scrape_period_ms": 325.12
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `endpoint_url` | string | Full URL used to scrape metrics |
| `duration_seconds` | float | Total time this endpoint was scraped |
| `scrape_count` | int | Number of successful scrapes from this endpoint |
| `avg_scrape_latency_ms` | float | Average time to complete each scrape (network + parsing) |
| `avg_scrape_period_ms` | float | Average time between consecutive scrapes |

---

## Metrics Section

Each metric entry has this structure:

```json
"metrics": {
  "metric_name": {
    "type": "gauge|counter|histogram|summary",
    "description": "Metric description from HELP text",
    "unit": "seconds|tokens|requests|...",
    "series": [ ... ]
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Prometheus metric type: `gauge`, `counter`, `histogram`, or `summary` |
| `description` | string | Human-readable description from Prometheus HELP text |
| `unit` | string or null | Unit inferred from metric name suffix (e.g., `_seconds` → `"seconds"`) |
| `series` | array | Statistics for each unique label combination |

---

## Series Fields (Common)

Every series entry contains these common fields:

```json
{
  "endpoint": "localhost:10000",
  "endpoint_url": "http://localhost:10000/metrics",
  "labels": {"model": "qwen/qwen3-0.6b"}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `endpoint` | string | Normalized endpoint identifier (host:port) |
| `endpoint_url` | string | Full endpoint URL |
| `labels` | object or null | Prometheus labels for this time series. `null` if metric has no labels. |

---

## Gauge Metrics

Gauges represent point-in-time values that can go up or down (e.g., current queue depth, memory usage).

### Gauge with Variation (multiple observations)

```json
{
  "endpoint": "localhost:10000",
  "endpoint_url": "http://localhost:10000/metrics",
  "labels": {"model": "qwen/qwen3-0.6b"},
  "avg": 35.31,
  "min": 0.0,
  "max": 48.0,
  "std": 16.45,
  "p50": 42.5,
  "p90": 48.0,
  "p95": 48.0,
  "p99": 48.0,
  "estimated_percentiles": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `avg` | float | Mean of all observed values during collection |
| `min` | float | Minimum observed value |
| `max` | float | Maximum observed value |
| `std` | float | Standard deviation of observed values |
| `p50` | float | 50th percentile (median) |
| `p90` | float | 90th percentile |
| `p95` | float | 95th percentile |
| `p99` | float | 99th percentile |
| `estimated_percentiles` | bool | `false` for gauges (percentiles computed from exact sample data) |

**Example interpretation** (`dynamo_frontend_inflight_requests`):
- "On average, 35 requests were in-flight"
- "In-flight requests ranged from 0 to 48"
- "99% of the time, in-flight requests were at or below 48"

### Gauge with No Variation (constant value)

When a gauge never changes during collection, only `avg` and `observation_count` are included:

```json
{
  "endpoint": "localhost:10000",
  "endpoint_url": "http://localhost:10000/metrics",
  "labels": {"model": "Qwen/Qwen3-0.6B"},
  "observation_count": 1,
  "avg": 40960.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `observation_count` | int | Number of times the value was observed (1 = constant) |
| `avg` | float | The constant value |

**Example interpretation** (`dynamo_frontend_model_context_length`):
- "Context length is 40960 tokens (constant configuration value)"

---

## Counter Metrics

Counters are monotonically increasing values (e.g., total requests processed, total tokens generated).

### Counter with Activity (delta > 0)

```json
{
  "endpoint": "localhost:10000",
  "endpoint_url": "http://localhost:10000/metrics",
  "labels": {"model": "qwen/qwen3-0.6b"},
  "delta": 5275.0,
  "rate_per_second": 242.15,
  "rate_avg": 263.02,
  "rate_min": 22.38,
  "rate_max": 2948.65,
  "rate_std": 804.92
}
```

| Field | Type | Description |
|-------|------|-------------|
| `delta` | float | **Total increase** in counter value during collection period |
| `rate_per_second` | float | **Overall rate**: `delta / duration_seconds` |
| `rate_avg` | float | **Time-weighted average rate** between change points |
| `rate_min` | float | **Minimum instantaneous rate** observed between consecutive scrapes |
| `rate_max` | float | **Maximum instantaneous rate** observed between consecutive scrapes |
| `rate_std` | float | **Standard deviation** of point-to-point rates |

**Example interpretation** (`dynamo_frontend_output_tokens`):
- `delta: 5275` → "5275 tokens were generated during the benchmark"
- `rate_per_second: 242.15` → "Overall throughput was 242 tokens/second"
- `rate_avg: 263.02` → "Average instantaneous rate was 263 tokens/second"
- `rate_min: 22.38` → "Slowest period saw 22 tokens/second"
- `rate_max: 2948.65` → "Fastest burst reached 2949 tokens/second"

### Counter with No Activity (delta = 0)

When a counter doesn't change, only `delta` is included:

```json
{
  "endpoint": "localhost:10000",
  "endpoint_url": "http://localhost:10000/metrics",
  "labels": {"status": "error"},
  "delta": 0.0
}
```

**Example interpretation** (`dynamo_frontend_requests` with `status: error`):
- "No error requests occurred during the benchmark"

---

## Histogram Metrics

Histograms track distributions of values (e.g., request latencies, token counts). Prometheus histograms maintain cumulative bucket counts and a running sum.

```json
{
  "endpoint": "localhost:10000",
  "endpoint_url": "http://localhost:10000/metrics",
  "labels": {"model": "qwen/qwen3-0.6b"},
  "observation_count": 50,
  "avg": 14.67,
  "p50": 15.76,
  "p90": 20.43,
  "p95": 21.01,
  "p99": 21.48,
  "estimated_percentiles": true,
  "delta": 733.55,
  "rate_per_second": 31.66,
  "observations_per_second": 2.16,
  "buckets": {
    "0": 0,
    "1.9": 4,
    "3.4": 5,
    "12": 8,
    "22": 50,
    "+Inf": 50
  }
}
```

### Observation Fields

| Field | Type | Description |
|-------|------|-------------|
| `observation_count` | int | Number of values observed (histogram updates) during collection |
| `avg` | float | Mean observed value: `sum_delta / observation_count` |
| `observations_per_second` | float | Observation rate: `observation_count / duration_seconds` |

### Percentile Fields

| Field | Type | Description |
|-------|------|-------------|
| `p50` | float | Estimated 50th percentile (median) |
| `p90` | float | Estimated 90th percentile |
| `p95` | float | Estimated 95th percentile |
| `p99` | float | Estimated 99th percentile |
| `estimated_percentiles` | bool | `true` for histograms (percentiles interpolated from buckets) |

### Sum Fields

| Field | Type | Description |
|-------|------|-------------|
| `delta` | float | **Sum delta**: Total of all observed values during collection |
| `rate_per_second` | float | **Sum rate**: `delta / duration_seconds` |

### Bucket Data

| Field | Type | Description |
|-------|------|-------------|
| `buckets` | object | Map of bucket upper bounds to cumulative counts |

Bucket keys are the upper bound (as strings), values are cumulative counts of observations ≤ that bound.

---

### Histogram Field Semantics by Use Case

The meaning of histogram fields depends on what the histogram measures:

#### Request-Level Histograms (e.g., `request_duration_seconds`)

| Field | Semantic Meaning | Example |
|-------|------------------|---------|
| `observation_count` | Number of requests | 50 requests |
| `observations_per_second` | Request throughput | 2.16 requests/second |
| `avg` | Mean request duration | 14.67 seconds |
| `delta` | Total time spent on requests | 733.55 seconds |
| `rate_per_second` | **Concurrency metric**: seconds of request time per second of real time | 31.66 (≈32 concurrent requests) |
| `p99` | 99th percentile latency | 21.48 seconds |

#### Token-Level Histograms (e.g., `input_sequence_tokens`)

| Field | Semantic Meaning | Example |
|-------|------------------|---------|
| `observation_count` | Number of requests | 50 requests |
| `observations_per_second` | Request throughput | 2.29 requests/second |
| `avg` | Mean tokens per request | 986 tokens |
| `delta` | Total tokens processed | 49,311 tokens |
| `rate_per_second` | **Token throughput** | 2,264 tokens/second |
| `p99` | 99th percentile tokens | 2,193 tokens |

#### Per-Token Histograms (e.g., `inter_token_latency_seconds`)

| Field | Semantic Meaning | Example |
|-------|------------------|---------|
| `observation_count` | Number of tokens generated | 4,906 tokens |
| `observations_per_second` | Token generation rate | 225 tokens/second |
| `avg` | Mean inter-token latency | 3.7 ms |
| `delta` | Total ITL time | 18.19 seconds |
| `rate_per_second` | **ITL utilization**: seconds of ITL per second | 0.83 |
| `p99` | 99th percentile ITL | 9.26 ms |

---

## Summary Metrics

Prometheus summaries are similar to histograms but compute quantiles server-side. They include pre-computed quantile values.

```json
{
  "endpoint": "localhost:10000",
  "endpoint_url": "http://localhost:10000/metrics",
  "labels": {"model": "qwen/qwen3-0.6b"},
  "observation_count": 1000,
  "avg": 0.15,
  "p50": 0.12,
  "p90": 0.25,
  "p95": 0.35,
  "p99": 0.50,
  "estimated_percentiles": false,
  "delta": 150.0,
  "rate_per_second": 6.89,
  "observations_per_second": 45.9,
  "quantiles": {
    "0.5": 0.12,
    "0.9": 0.25,
    "0.95": 0.35,
    "0.99": 0.50
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | object | Server-computed quantile values from Prometheus (keys are quantile strings like `"0.5"`, `"0.99"`) |
| `estimated_percentiles` | bool | `false` for summaries (quantiles are server-computed, not estimated) |

All other fields have the same semantics as histograms.

---

## Info Metrics

Info metrics (ending in `_info`) are special gauges with value 1.0 used to export static labels (version info, configuration, etc.).

```json
"vllm:cache_config_info": {
  "type": "gauge",
  "description": "Cache configuration",
  "unit": "info",
  "series": [
    {
      "endpoint": "localhost:10001",
      "endpoint_url": "http://localhost:10001/metrics",
      "labels": {
        "block_size": "16",
        "cache_dtype": "auto",
        "num_gpu_blocks": "71670"
      }
    }
  ]
}
```

Info metrics contain **only labels** - no statistical fields. The important data is entirely in the `labels` object.

---

## Field Presence Rules

Fields are omitted when not applicable to reduce JSON size:

| Condition | Fields Omitted |
|-----------|----------------|
| Info metric (`_info` suffix) | All stats (only `endpoint`, `endpoint_url`, `labels` present) |
| Gauge with no variation (constant) | `min`, `max`, `std`, `p50`, `p90`, `p95`, `p99` |
| Counter with no activity (`delta = 0`) | `rate_per_second`, `rate_avg`, `rate_min`, `rate_max`, `rate_std` |
| Histogram/Summary with no observations | All stats except `observation_count` |
| Metric has no labels | `labels` field is `null` or omitted |
| Unit cannot be inferred from name | `unit` field is `null` or omitted |

---

## Unit Inference

Units are inferred from metric name suffixes:

| Suffix | Unit |
|--------|------|
| `_seconds`, `_seconds_total` | `seconds` |
| `_ms`, `_ms_total`, `_milliseconds` | `milliseconds` |
| `_bytes`, `_bytes_total` | `bytes` |
| `_total`, `_count` | `count` |
| `_tokens`, `_tokens_total` | `tokens` |
| `_requests`, `_requests_total` | `requests` |
| `_ratio` | `ratio` |
| `_percent`, `_perc` | `percent` |
| `_info` | `info` |

See `display_units_utils.py` for the complete list.

---

## Example Queries

### Find all metrics with p99 > 1 second
```python
for name, metric in data["metrics"].items():
    for series in metric["series"]:
        if series.get("p99", 0) > 1.0 and metric.get("unit") == "seconds":
            print(f"{name}: p99={series['p99']:.2f}s")
```

### Calculate total tokens generated across all endpoints
```python
total = sum(
    series.get("delta", 0)
    for series in data["metrics"]["dynamo_frontend_output_tokens"]["series"]
)
```

### Find highest throughput endpoint
```python
max_throughput = max(
    (series["rate_per_second"], series["endpoint"])
    for series in data["metrics"]["dynamo_frontend_output_tokens"]["series"]
)
```
