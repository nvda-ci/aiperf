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

Aggregated statistics computed from time-series data:

```json
{
  "summary": {
    "endpoints_configured": ["http://localhost:8000/metrics"],
    "endpoints_successful": ["http://localhost:8000/metrics"],
    "start_time": "2025-12-04T10:15:30.123456",
    "end_time": "2025-12-04T10:20:35.789012"
  },
  "endpoints": {
    "http://localhost:8000/metrics": {
      "endpoint_url": "http://localhost:8000/metrics",
      "duration_seconds": 305.5,
      "scrape_count": 61,
      "avg_scrape_latency_ms": 12.5,
      "metrics": {
        "vllm:gpu_cache_usage_perc": {
          "description": "GPU KV-cache usage.",
          "type": "gauge",
          "series": [{
            "labels": null,
            "stats": {"avg": 0.65, "min": 0.45, "max": 0.85, "std": 0.12, "p50": 0.64, "p90": 0.79, "p95": 0.82, "p99": 0.84}
          }]
        },
        "vllm:request_success_total": {
          "description": "Count of successfully processed requests.",
          "type": "counter",
          "series": [{
            "labels": null,
            "stats": {"delta": 1000.0, "rate_overall": 200.0, "rate_avg": 195.0, "rate_min": 150.0, "rate_max": 280.0, "rate_std": 35.2}
          }]
        },
        "vllm:time_to_first_token_seconds": {
          "description": "Histogram of time to first token.",
          "type": "histogram",
          "series": [{
            "labels": null,
            "stats": {
              "count_delta": 1000.0,
              "sum_delta": 125.5,
              "avg": 0.1255,
              "rate": 200.0,
              "buckets": {"0.01": 50.0, "0.1": 450.0, "1.0": 980.0, "+Inf": 1000.0}
            }
          }]
        }
      }
    }
  }
}
```

### 3. Metadata: `server_metrics_metadata.json`

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

```json
{"avg": 0.65, "min": 0.45, "max": 0.85, "std": 0.12, "p50": 0.64, "p90": 0.79, "p95": 0.82, "p99": 0.84}
```

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
  "count_delta": 1000.0,
  "sum_delta": 125.5,
  "avg": 0.1255,
  "rate": 200.0,
  "buckets": {"0.01": 50.0, "0.1": 450.0, "1.0": 980.0, "+Inf": 1000.0}
}
```

- `buckets`: Cumulative bucket counts (le="less than or equal"). Use for percentile computation or downstream analysis.

### Summary (server-computed quantiles)

```json
{
  "count_delta": 1000.0,
  "sum_delta": 245.8,
  "avg": 0.2458,
  "rate": 200.0,
  "quantiles": {"0.5": 0.220, "0.9": 0.380, "0.95": 0.450, "0.99": 0.620}
}
```

- `quantiles`: Cumulative values computed by the server over all observations since server start (not period-specific)

## Labeled Metrics

When metrics have labels, each unique label combination is aggregated separately:

```json
{
  "dynamo_frontend_requests": {
    "type": "counter",
    "series": [
      {"labels": {"model": "llama-3", "status": "success"}, "stats": {"delta": 800.0, "rate_avg": 160.0}},
      {"labels": {"model": "llama-3", "status": "error"}, "stats": {"delta": 5.0, "rate_avg": 1.0}}
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
