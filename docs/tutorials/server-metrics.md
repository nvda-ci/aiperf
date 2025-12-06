<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# Server Metrics Collection

AIPerf automatically collects metrics from Prometheus-compatible endpoints exposed by LLM inference servers (vLLM, SGLang, TRT-LLM, Dynamo, etc.).

## Quick Start

Server metrics are **collected by default** - just run AIPerf normally:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --url localhost:8000 \
    --concurrency 4 \
    --request-count 100
```

AIPerf automatically:
1. Queries the `/metrics` endpoint on your inference server
2. Collects metrics at configurable intervals
3. Exports time-series data to `server_metrics_export.jsonl`
4. Exports aggregated statistics to `server_metrics.json`
5. Saves metric schemas to `server_metrics_metadata.json`

### Adding Custom Endpoints

```bash
# Single endpoint
aiperf profile --model MODEL ... --server-metrics http://localhost:8081

# Multiple endpoints (distributed deployment)
aiperf profile --model MODEL ... --server-metrics \
    http://node1:8081 \
    http://node2:8081
```

### Disabling Server Metrics

```bash
export AIPERF_SERVER_METRICS_ENABLED=false
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `AIPERF_SERVER_METRICS_COLLECTION_INTERVAL` | 0.1s | Collection frequency |
| `AIPERF_SERVER_METRICS_COLLECTION_FLUSH_PERIOD` | 2.0s | Wait time for final metrics after benchmark |
| `AIPERF_SERVER_METRICS_DEFAULT_BACKEND_PORTS` | empty | Additional ports to check during auto-discovery |
| `AIPERF_SERVER_METRICS_REACHABILITY_TIMEOUT` | 5s | Timeout for endpoint reachability tests |

## Output Files

### 1. Time-Series: `server_metrics_export.jsonl`

Line-delimited JSON with metrics snapshots over time:

```json
{
  "endpoint_url": "http://localhost:8000/metrics",
  "timestamp_ns": 1763591215220757503,
  "endpoint_latency_ns": 719764167,
  "metrics": {
    "vllm:num_requests_running": [{"value": 12.0}],
    "vllm:gpu_cache_usage_perc": [{"value": 0.72}],
    "vllm:request_success_total": [{"value": 1500.0}],
    "vllm:time_to_first_token_seconds": [{
      "histogram": {"0.01": 145.0, "0.1": 1498.0, "+Inf": 1500.0},
      "sum": 32.456,
      "count": 1500.0
    }]
  }
}
```

**Fields:**
- `endpoint_url`: Source Prometheus endpoint
- `timestamp_ns`: Collection timestamp (nanoseconds)
- `endpoint_latency_ns`: Time to fetch metrics
- `metrics`: All metrics from this endpoint
  - Counter/Gauge: `{"value": N}` or `{"labels": {...}, "value": N}`
  - Histogram: `{"histogram": {"le": count}, "sum": N, "count": N}`
  - Summary: `{"summary": {"quantile": value}, "sum": N, "count": N}`

**Deduplication:** Consecutive identical metrics are deduplicated per endpoint to reduce file size while preserving accurate timestamps for when metrics changed.

### 2. Aggregated Statistics: `server_metrics.json`

Aggregated statistics computed from time-series data. Metrics from all endpoints are merged together, with each series item tagged with its source endpoint.

Info metrics (ending in `_info`) are included as gauges with `value: 1.0` since they represent static configuration/version information.

```json
{
  "summary": {
    "endpoints_configured": ["localhost:8000"],
    "endpoints_successful": ["localhost:8000"],
    "start_time": "2025-12-04T10:15:30.123456",
    "end_time": "2025-12-04T10:20:35.789012",
    "endpoint_info": {
      "localhost:8000": {
        "endpoint_url": "http://localhost:8000/metrics",
        "duration_seconds": 305.5,
        "scrape_count": 61,
        "avg_scrape_latency_ms": 12.5,
        "avg_scrape_period_ms": 5008.2
      }
    }
  },
  "metrics": {
    "vllm_version_info": {
      "description": "vLLM version information.",
      "type": "gauge",
      "series": [{
        "endpoint": "localhost:8000",
        "labels": {"version": "0.6.0", "build": "abc123"},
        "value": 1.0
      }]
    },
    "vllm:gpu_cache_usage_perc": {
      "description": "GPU KV-cache usage.",
      "type": "gauge",
      "series": [{
        "endpoint": "localhost:8000",
        "stats": {"avg": 0.65, "min": 0.45, "max": 0.85, "std": 0.12, "p50": 0.64, "p90": 0.79, "p95": 0.82, "p99": 0.84}
      }]
    },
    "vllm:request_success_total": {
      "description": "Count of successfully processed requests.",
      "type": "counter",
      "series": [{
        "endpoint": "localhost:8000",
        "stats": {"delta": 1000.0, "rate_overall": 200.0, "rate_avg": 195.0, "rate_min": 150.0, "rate_max": 280.0, "rate_std": 35.2}
      }]
    },
    "vllm:time_to_first_token_seconds": {
      "description": "Histogram of time to first token.",
      "type": "histogram",
      "series": [{
        "endpoint": "localhost:8000",
        "stats": {
          "count_delta": 1000,
          "count_rate": 200.0,
          "sum_delta": 125.5,
          "sum_rate": 25.1,
          "avg": 0.1255,
          "p50_estimate": 0.0823,
          "p90_estimate": 0.2156,
          "p95_estimate": 0.3421,
          "p99_estimate": 0.8765,
          "buckets": {"0.01": 50, "0.1": 450, "1.0": 980, "+Inf": 1000}
        }
      }]
    }
  }
}
```

### 3. CSV Export: `server_metrics.csv`

Tabular export for spreadsheet and pandas analysis. Sections are separated by blank lines, each with its own column headers based on metric type. Sections appear in order: gauge, counter, histogram, summary (only if metrics of that type exist).

```csv
Endpoint,Type,Metric,Labels,avg,min,max,std,p50,p90,p95,p99
localhost:8000,gauge,vllm:gpu_cache_usage_perc,,0.6500,0.4500,0.8500,0.1200,0.6400,0.7900,0.8200,0.8400
localhost:8000,gauge,vllm:num_requests_running,,4.2500,1.0000,8.0000,2.1000,4.0000,7.0000,7.5000,7.9000

Endpoint,Type,Metric,Labels,delta,rate_overall,rate_avg,rate_min,rate_max,rate_std
localhost:8000,counter,vllm:prompt_tokens_total,,125000,2500.0000,2450.0000,1800.0000,3200.0000,420.0000
localhost:8000,counter,vllm:request_success_total,,1000.0000,200.0000,195.0000,150.0000,280.0000,35.2000

Endpoint,Type,Metric,Labels,count_delta,sum_delta,avg,rate,0.01,0.1,1.0,+Inf
localhost:8000,histogram,vllm:e2e_request_latency_seconds,,1000.0000,856.2000,0.8562,200.0000,10.0000,150.0000,920.0000,1000.0000
localhost:8000,histogram,vllm:time_to_first_token_seconds,,1000.0000,125.5000,0.1255,200.0000,50.0000,450.0000,980.0000,1000.0000

Endpoint,Type,Metric,Labels,count_delta,sum_delta,avg,rate,0.5,0.9,0.95,0.99
localhost:8000,summary,vllm:request_latency_seconds,,1000.0000,245.8000,0.2458,200.0000,0.2200,0.3800,0.4500,0.6200
```

**Column structure by metric type:**
- **Gauge:** `Endpoint,Type,Metric,Labels,avg,min,max,std,p50,p90,p95,p99`
- **Counter:** `Endpoint,Type,Metric,Labels,delta,rate_overall,rate_avg,rate_min,rate_max,rate_std`
- **Histogram:** `Endpoint,Type,Metric,Labels,count_delta,sum_delta,avg,rate,<bucket_boundaries>...`
- **Summary:** `Endpoint,Type,Metric,Labels,count_delta,sum_delta,avg,rate,<quantile_keys>...`

**Formatting details:**
- Metrics are sorted alphabetically by metric name, then endpoint, then labels
- Numbers are formatted to 4 decimal places; missing values are empty
- Labels are formatted as `key1=value1,key2=value2` (sorted by key)
- Histograms with different bucket boundaries get separate sub-sections
- Summaries with different quantile keys get separate sub-sections

**Loading with pandas:**
```python
from io import StringIO
import pandas as pd

with open("server_metrics.csv") as f:
    content = f.read()

# Split on blank lines and parse each section
sections = [pd.read_csv(StringIO(s)) for s in content.strip().split('\n\n') if s.strip()]

# Access by metric type
by_type = {df["Type"].iloc[0]: df for df in sections}
gauges = by_type.get("gauge")
counters = by_type.get("counter")
histograms = by_type.get("histogram")
summaries = by_type.get("summary")
```

### 4. Metadata: `server_metrics_metadata.json`

Metric schemas and info metrics:

```json
{
  "endpoints": {
    "http://localhost:8000/metrics": {
      "endpoint_url": "http://localhost:8000/metrics",
      "info_metrics": {
        "python_info": {
          "description": "Python platform information",
          "labels": [{"implementation": "CPython", "version": "3.12.10"}]
        }
      },
      "metric_schemas": {
        "vllm:num_requests_running": {"type": "gauge", "description": "Number of requests running on GPU."},
        "vllm:request_success_total": {"type": "counter", "description": "Count of successfully processed requests."},
        "vllm:time_to_first_token_seconds": {"type": "histogram", "description": "Time to first token in seconds."}
      }
    }
  }
}
```

## Statistics by Metric Type

### Gauge (point-in-time values)

For gauges that vary during collection:
```json
{"stats": {"avg": 0.65, "min": 0.45, "max": 0.85, "std": 0.12, "p50": 0.64, "p90": 0.79, "p95": 0.82, "p99": 0.84}}
```

For constant gauges (std == 0), a simplified format is used:
```json
{"value": 42.0}
```

This allows checking `if "stats" in series` to determine if the metric varied during collection.

### Counter (cumulative totals)

```json
{
  "delta": 1000.0,
  "rate_overall": 200.0,
  "rate_avg": 195.0,
  "rate_min": 150.0,
  "rate_max": 280.0,
  "rate_std": 35.2
}
```

- `delta`: Total change over the period
- `rate_overall`: Overall throughput (delta/duration) - always available
- `rate_avg`: Time-weighted average rate between change points (longer intervals contribute more)
- `rate_min/max/std`: Statistics computed between *change points* only (when counter value actually changed)

Note: Change-point detection avoids misleading rates when sampling faster than server update frequency. Returns `null` if no changes occurred.

### Histogram (distributions)

```json
{
  "count_delta": 1000,
  "sum_delta": 125.5,
  "avg": 0.1255,
  "count_rate": 200.0,
  "sum_rate": 25.1,
  "p50_estimate": 0.0823,
  "p90_estimate": 0.2156,
  "p95_estimate": 0.3421,
  "p99_estimate": 0.8765,
  "buckets": {"0.01": 50, "0.1": 450, "1.0": 980, "+Inf": 1000}
}
```

- `count_rate`: Observations per second (count_delta/duration)
- `sum_rate`: Sum per second (sum_delta/duration). Useful for token/byte histograms.
- `p50_estimate`, `p90_estimate`, `p95_estimate`, `p99_estimate`: Percentile estimates using polynomial histogram algorithm (see below)
- `buckets`: Cumulative bucket counts (le="less than or equal") for downstream analysis

#### Histogram Percentile Estimation

AIPerf uses a polynomial histogram algorithm to estimate percentiles from Prometheus histogram buckets. This approach provides more accurate estimates than traditional bucket interpolation, especially for tail percentiles (p99) when observations fall in the +Inf bucket.

**Algorithm:**
1. **Learn per-bucket means**: When all observations in a scrape interval fall into a single bucket, we learn the exact mean for that bucket
2. **Sum-constrained observation generation**: Place observations using learned means (or midpoint fallback), then adjust positions to match the exact histogram sum
3. **+Inf bucket back-calculation**: Estimate +Inf bucket observations using `inf_sum = total_sum - estimated_finite_sum`

**Note:** All histogram percentile estimates are approximations. For the most accurate data, use the `avg` field (sum/count) which is exact. Tail percentiles (p99) should be treated with appropriate skepticism, especially when many observations fall in the +Inf bucket.

For histograms with no observations during collection, a simplified format is used:
```json
{"observation_count": 0}
```

This allows checking `if "stats" in series` to determine if the histogram had any observations.

### Summary (server-computed quantiles)

```json
{
  "count_delta": 1000,
  "sum_delta": 245.8,
  "avg": 0.2458,
  "count_rate": 200.0,
  "sum_rate": 49.16,
  "quantiles": {"0.5": 0.220, "0.9": 0.380, "0.95": 0.450, "0.99": 0.620}
}
```

- `count_rate`: Observations per second (count_delta/duration)
- `sum_rate`: Sum per second (sum_delta/duration)
- `quantiles`: Cumulative values computed by the server over all observations since server start (not period-specific)

For summaries with no observations during collection, a simplified format is used:
```json
{"observation_count": 0}
```

This allows checking `if "stats" in series` to determine if the summary had any observations.

## Labeled Metrics

When metrics have labels, each unique label combination is aggregated separately. With multiple endpoints, series from all endpoints are merged together:

```json
{
  "dynamo_frontend_requests": {
    "type": "counter",
    "series": [
      {"endpoint": "node1:8000", "labels": {"model": "llama-3", "status": "success"}, "stats": {"delta": 800.0, "rate_avg": 160.0}},
      {"endpoint": "node1:8000", "labels": {"model": "llama-3", "status": "error"}, "stats": {"delta": 5.0, "rate_avg": 1.0}},
      {"endpoint": "node2:8000", "labels": {"model": "llama-3", "status": "success"}, "stats": {"delta": 750.0, "rate_avg": 150.0}}
    ]
  }
}
```

## Common Metrics

### vLLM
- **Queue:** `vllm:num_requests_running`, `vllm:num_requests_waiting`
- **Cache:** `vllm:gpu_cache_usage_perc`, `vllm:gpu_prefix_cache_hit_rate`
- **Throughput:** `vllm:avg_prompt_throughput_toks_per_s`, `vllm:avg_generation_throughput_toks_per_s`
- **Latency:** `vllm:time_to_first_token_seconds`, `vllm:e2e_request_latency_seconds`
- **Counters:** `vllm:request_success_total`, `vllm:prompt_tokens_total`

### Dynamo
- **Frontend:** `dynamo_frontend_requests`, `dynamo_frontend_queued_requests`, `dynamo_frontend_time_to_first_token_seconds`
- **Component:** `dynamo_component_requests_total`, `dynamo_component_request_duration_seconds`
