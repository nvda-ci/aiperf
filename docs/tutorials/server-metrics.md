<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# Server Metrics Collection with AIPerf

This guide shows you how to use AIPerf's automatic server metrics collection feature. Server metrics provide insights into LLM inference server performance, including request counts, latencies, cache utilization, and custom application metrics.

## Overview

AIPerf **automatically collects metrics by default** from Prometheus-compatible endpoints exposed by LLM inference servers like vLLM, SGLang, TRT-LLM, and others. These metrics complement AIPerf's client-side measurements with server-side observability data.

**What You'll Learn:**
- How automatic server metrics collection works (enabled by default)
- Configure additional custom Prometheus endpoints
- Understand the output files and data format
- Use server metrics for performance analysis

## Quick Start

### Basic Usage

Server metrics are **automatically collected** for the inference endpoint port - just run AIPerf normally:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --url localhost:8000 \
    --concurrency 4 \
    --request-count 100
```

**What happens automatically:**
1. AIPerf queries the Prometheus `/metrics` endpoint on your inference server (checks `--url` port)
2. Collects metrics every `AIPERF_SERVER_METRICS_COLLECTION_INTERVAL` (configurable)
3. Exports time-series data to `server_metrics_export.jsonl` file
4. Saves metadata about collected metrics to `server_metrics_metadata.json` file

> [!TIP]
> No flag needed! Server metrics are collected by default. Use `--server-metrics <urls>` to add additional endpoints, or set `AIPERF_SERVER_METRICS_ENABLED=false` to disable.

### Automatic Endpoint

By default, AIPerf automatically discovers and queries the Prometheus `/metrics` endpoint on your inference server (checks `--url` port) and any additional ports specified via `AIPERF_SERVER_METRICS_DEFAULT_BACKEND_PORTS` (comma-separated string or JSON array).

> [!TIP]
> **Default Port Handling:** When your inference URL has no explicit port (e.g., `https://api.example.com/v1/chat`), AIPerf uses the default port for the scheme (443 for HTTPS, 80 for HTTP) before checking additional ports from `AIPERF_SERVER_METRICS_DEFAULT_BACKEND_PORTS`.

### Custom Endpoint URLs

Specify additional custom Prometheus endpoints explicitly:

```bash
# Single custom endpoint
aiperf profile --model MODEL ... --server-metrics http://localhost:8081

# Multiple endpoints (multi-node or multiple services)
aiperf profile --model MODEL ... --server-metrics \
    http://node1:8081 \
    http://node2:8081 \
    http://monitoring:9090
```

> [!NOTE]
> URLs can be specified with or without the `http://` prefix and `/metrics` suffix. AIPerf normalizes them automatically:
> - `localhost:8000` → `http://localhost:8000/metrics`
> - `http://server:9090` → `http://server:9090/metrics`
> - `localhost:8081/metrics` → `http://localhost:8081/metrics`

### Disabling Server Metrics

To disable automatic server metrics collection:

```bash
export AIPERF_SERVER_METRICS_ENABLED=false
```

This completely disables server metrics collection for the run.

## Understanding Server Metrics

### What Metrics Are Collected?

AIPerf collects **all metrics** exposed by Prometheus-compatible endpoints, with automatic filtering:

- **Collected:** All counter, gauge, histogram, and summary metrics
- **Automatically Filtered:** Metrics ending with `_created` (internal Prometheus timestamps)

Common metrics from LLM inference servers include:

#### vLLM Metrics Examples
- **Queue Metrics:** `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:num_requests_swapped`
- **Cache Utilization:** `vllm:gpu_cache_usage_perc`, `vllm:cpu_cache_usage_perc`
- **Prefix Cache:** `vllm:gpu_prefix_cache_hit_rate`, `vllm:cpu_prefix_cache_hit_rate`
- **Throughput:** `vllm:avg_prompt_throughput_toks_per_s`, `vllm:avg_generation_throughput_toks_per_s`
- **Latency Histograms:** `vllm:time_to_first_token_seconds`, `vllm:time_per_output_token_seconds`, `vllm:e2e_request_latency_seconds`
- **Token Distributions:** `vllm:request_prompt_tokens`, `vllm:request_generation_tokens`
- **Counters:** `vllm:request_success_total`, `vllm:prompt_tokens_total`, `vllm:generation_tokens_total`, `vllm:num_preemptions_total`

#### Dynamo Frontend Metrics Examples
- **Request Metrics:** `dynamo_frontend_requests`
- **Latency Distributions:** `dynamo_frontend_time_to_first_token_seconds`, `dynamo_frontend_request_duration_seconds`, `dynamo_frontend_inter_token_latency_seconds`
- **Queue Metrics:** `dynamo_frontend_queued_requests`, `dynamo_frontend_inflight_requests`
- **Token Metrics:** `dynamo_frontend_input_sequence_tokens`, `dynamo_frontend_output_sequence_tokens`
- **Model Config:** `dynamo_frontend_model_context_length`, `dynamo_frontend_model_total_kv_blocks`

#### Dynamo Component Metrics Examples
- **Request Metrics:** `dynamo_component_requests_total`, `dynamo_component_errors_total`
- **Data Transfer:** `dynamo_component_request_bytes_total`, `dynamo_component_response_bytes_total`
- **Task Metrics:** `dynamo_component_tasks_issued_total`, `dynamo_component_tasks_success_total`, `dynamo_component_tasks_failed_total`
- **Performance:** `dynamo_component_request_duration_seconds`, `dynamo_component_inflight_requests`
- **System:** `dynamo_component_uptime_seconds`
- **NATS Metrics:** `dynamo_component_nats_client_in_messages`, `dynamo_component_nats_service_requests_total`

#### Prometheus Metric Types Supported
- **Counter:** Cumulative values (e.g., total requests, total tokens)
- **Gauge:** Point-in-time values (e.g., cache utilization %)
- **Histogram:** Distribution with buckets (e.g., latency percentiles)
- **Summary:** Pre-computed quantiles (e.g., p50, p90, p99)

### Output Files

AIPerf generates two files per benchmark run:

#### 1. Time-Series Data: `server_metrics_export.jsonl`

Line-delimited JSON with metrics snapshots collected over time (from real Dynamo run):

```json
{"endpoint_url":"http://localhost:8081/metrics","timestamp_ns":1763591215213919629,"endpoint_latency_ns":712690779,"metrics":{"dynamo_component_requests":[{"labels":{"dynamo_component":"prefill","dynamo_endpoint":"generate","model":"openai/gpt-oss-20b"},"value":360.0}],"dynamo_component_nats_client_in_messages":[{"value":59284.0}],"dynamo_component_request_duration_seconds":[{"labels":{"dynamo_component":"prefill","dynamo_endpoint":"generate","model":"openai/gpt-oss-20b"},"histogram":{"0.005":0.0,"0.01":0.0,"0.025":123.0,"0.05":327.0,"0.1":348.0,"0.25":360.0,"+Inf":360.0},"sum":12.232215459,"count":360.0}]}}
{"endpoint_url":"http://localhost:8000/metrics","timestamp_ns":1763591215220757503,"endpoint_latency_ns":719764167,"metrics":{"dynamo_frontend_requests":[{"labels":{"endpoint":"chat_completions","model":"openai/gpt-oss-20b","status":"success"},"value":1000.0}],"dynamo_frontend_queued_requests":[{"labels":{"model":"openai/gpt-oss-20b"},"value":0.0}],"dynamo_frontend_time_to_first_token_seconds":[{"labels":{"model":"openai/gpt-oss-20b"},"histogram":{"0":0.0,"0.0047":0.0,"0.1":835.0,"0.47":892.0,"1":899.0,"10":1000.0,"+Inf":1000.0},"sum":765.15823571,"count":1000.0}]}}
```

**Each line contains:**
- `endpoint_url`: Source Prometheus endpoint
- `timestamp_ns`: Collection timestamp (nanoseconds since epoch)
- `endpoint_latency_ns`: Time to fetch metrics from endpoint (nanoseconds)
- `metrics`: Dictionary of metric families with samples
  - **Counter/Gauge:** `{"value": 42.0}` or `{"labels": {...}, "value": 42.0}`
  - **Histogram:** `{"histogram": {"le": count, ...}, "sum": X, "count": N}` (le = bucket upper bounds)
  - **Summary:** `{"summary": {"quantile": value, ...}, "sum": X, "count": N}` (quantile = percentile labels)

**Space Optimization with Deduplication:**
The file is automatically **deduplicated** per endpoint to reduce file size while preserving accurate timeline information:

1. **First occurrence** of metrics → always written (marks start of period)
2. **Consecutive identical metrics** → skipped and counted
3. **Change detected** → last duplicate written (marks end of period), then new record written (marks start of new period)

**Example:** Input `A,A,A,B,B,C,D,D,D,D` → Output `A,A,B,B,C,D,D`

This ensures you have actual timestamp observations for when metrics changed, enabling accurate duration calculations and time-series analysis. Deduplication uses equality comparison on the metrics dictionary for each endpoint separately.

#### 2. Metadata: `server_metrics_metadata.json`

Pretty-printed JSON with metric schemas, info metrics, and documentation:

```json
{
  "endpoints": {
    "http://localhost:8000/metrics": {
      "endpoint_url": "http://localhost:8000/metrics",
      "info_metrics": {
        "python_info": {
          "description": "Python platform information",
          "labels": [
            {
              "implementation": "CPython",
              "major": "3",
              "minor": "12",
              "patchlevel": "10",
              "version": "3.12.10"
            }
          ]
        }
      },
      "metric_schemas": {
        "dynamo_frontend_inflight_requests": {
          "type": "gauge",
          "description": "Number of inflight requests"
        },
        "dynamo_frontend_queued_requests": {
          "type": "gauge",
          "description": "Number of queued requests"
        },
        "dynamo_frontend_time_to_first_token_seconds": {
          "type": "histogram",
          "description": "Time to first token in seconds"
        },
        "dynamo_frontend_request_duration_seconds": {
          "type": "histogram",
          "description": "Request duration in seconds"
        },
        "dynamo_frontend_requests": {
          "type": "counter",
          "description": "Total number of requests processed"
        }
      }
    }
  }
}
```

**Contains:**
- Metric names and types (counter, gauge, histogram, summary)
- Description text explaining what each metric measures
- Endpoint URLs and display names

**Info Metrics:**
- Info metrics (ending in _info) contain static system information that doesn't change over time.
- We store only the labels (not values) since the labels contain the actual information and values are typically just 1.0.
- The description text explains what the info metric measures.

> [!TIP]
> Use the metadata file to understand what metrics are available and how to interpret the JSONL data.

> [!NOTE]
> **Output Directory Structure:** Files are created in `artifacts/{run-name}/` where `{run-name}` is automatically generated from your model, endpoint type, schedule, and concurrency (e.g., `Qwen_Qwen3-0.6B-openai-chat-concurrency4`).
>
> **Custom Filenames:** When using `--profile-export-prefix custom_name`, files become:
> - `artifacts/{run-name}/custom_name_server_metrics.jsonl`
> - `artifacts/{run-name}/custom_name_server_metrics_metadata.json`

## Configuration Options

### Environment Variables

Customize collection behavior with environment variables:

**Available Settings:**

| Environment Variable | Default | Range | Description |
|---------------------|---------|-------|-------------|
| `AIPERF_SERVER_METRICS_COLLECTION_INTERVAL` | 0.1s | 0.01s - 300s | Metrics collection frequency |
| `AIPERF_SERVER_METRICS_COLLECTION_FLUSH_PERIOD` | 2.0s | 0.0s - 30s | Wait time for final metrics after benchmark |
| `AIPERF_SERVER_METRICS_DEFAULT_BACKEND_PORTS` | empty list | comma-separated | Additional ports to check during auto-discovery (beyond inference endpoint port) |
| `AIPERF_SERVER_METRICS_REACHABILITY_TIMEOUT` | 5s | 1s - 300s | Timeout for endpoint reachability tests |
| `AIPERF_SERVER_METRICS_SHUTDOWN_DELAY` | 5.0s | 1.0s - 300s | Delay before shutting down to allow final command response transmission |

## Multi-Node Server Metrics

For distributed LLM deployments (tensor parallelism, pipeline parallelism), collect metrics from all nodes:

```bash
# Example: 4-node distributed Dynamo deployment
aiperf profile \
    --model meta-llama/Llama-3.1-70B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --url http://node-0:8000 \
    --concurrency 16 \
    --request-count 500 \
    --server-metrics \
        node-0:8000 \
        node-1:8081 \
        node-2:8081 \
        node-3:8081
```

**Output Structure:**
Each endpoint's metrics are stored separately in the JSONL file with its `endpoint_url` field, allowing you to:
- Analyze per-node performance
- Detect load imbalances
- Monitor distributed system health

## Understanding the Data Format

### JSONL Record Structure

Each line in `server_metrics_export.jsonl` is a JSON object containing ALL metrics from one endpoint at one point in time.

**Example from Dynamo frontend:**

```json
{
  "endpoint_url": "http://localhost:8000/metrics",
  "timestamp_ns": 1763591215220757503,
  "endpoint_latency_ns": 719764167,
  "metrics": {
    "dynamo_frontend_requests": [
      {
        "labels": {
          "endpoint": "chat_completions",
          "model": "openai/gpt-oss-20b",
          "request_type": "unary",
          "status": "success"
        },
        "value": 1000.0
      }
    ],
    "dynamo_frontend_queued_requests": [
      {
        "labels": {"model": "openai/gpt-oss-20b"},
        "value": 0.0
      }
    ],
    "dynamo_frontend_time_to_first_token_seconds": [
      {
        "labels": {"model": "openai/gpt-oss-20b"},
        "histogram": {
          "0": 0.0,
          "0.0022": 0.0,
          "0.0047": 0.0,
          "0.01": 0.0,
          "0.022": 0.0,
          "0.047": 0.0,
          "0.1": 835.0,
          "0.22": 888.0,
          "0.47": 892.0,
          "1": 899.0,
          "2.2": 900.0,
          "4.7": 900.0,
          "10": 1000.0,
          "22": 1000.0,
          "48": 1000.0,
          "100": 1000.0,
          "220": 1000.0,
          "480": 1000.0,
          "+Inf": 1000.0
        },
        "sum": 765.1582357100003,
        "count": 1000.0
      }
    ],
    "dynamo_frontend_request_duration_seconds": [
      {
        "labels": {"model": "openai/gpt-oss-20b"},
        "histogram": {
          "0": 0.0,
          "1.9": 0.0,
          "3.4": 10.0,
          "6.3": 212.0,
          "12": 554.0,
          "22": 969.0,
          "40": 1000.0,
          "75": 1000.0,
          "140": 1000.0,
          "260": 1000.0,
          "+Inf": 1000.0
        },
        "sum": 11336.94603903602,
        "count": 1000.0
      }
    ]
  }
}
```

**Top-Level Fields:**
- `endpoint_url`: Source Prometheus endpoint URL
- `timestamp_ns`: Unix timestamp in nanoseconds when metrics were collected
- `endpoint_latency_ns`: Time taken to fetch metrics from endpoint (nanoseconds)
- `metrics`: Dictionary containing ALL metrics from this endpoint at this timestamp

**Sample Structure by Type:**
- **Counter/Gauge:** `{"labels": {...}, "value": N}`
- **Histogram:** `{"labels": {...}, "histogram": {"le": count, ...}, "sum": N, "count": N}`
- **Summary:** `{"labels": {...}, "summary": {"quantile": value, ...}, "sum": N, "count": N}`

### Multi-Endpoint Data (Interleaved)

When collecting from multiple endpoints, records are **interleaved by write time** (when deduplication completes), not strictly by collection time.

**Example from Dynamo with component (8081) and frontend (8000) endpoints:**

```jsonl
{"endpoint_url":"http://localhost:8081/metrics","timestamp_ns":1763591215213919629,"endpoint_latency_ns":712690779,"metrics":{...}}
{"endpoint_url":"http://localhost:8000/metrics","timestamp_ns":1763591215220757503,"endpoint_latency_ns":719764167,"metrics":{...}}
{"endpoint_url":"http://localhost:8081/metrics","timestamp_ns":1763591215313945146,"endpoint_latency_ns":100025517,"metrics":{...}}
{"endpoint_url":"http://localhost:8000/metrics","timestamp_ns":1763591215320776712,"endpoint_latency_ns":100831209,"metrics":{...}}
{"endpoint_url":"http://localhost:8081/metrics","timestamp_ns":1763591215414013463,"endpoint_latency_ns":100068317,"metrics":{...}}
{"endpoint_url":"http://localhost:8000/metrics","timestamp_ns":1763591215421601721,"endpoint_latency_ns":100825009,"metrics":{...}}
```

**Key Points:**
- Records are **NOT** strictly alternating between endpoints
- **Deduplication** causes multiple consecutive records from same endpoint (first occurrence + last duplicate before change)
- Use `endpoint_url` field to filter/group by endpoint during analysis
- Each endpoint is collected and deduplicated independently
- Timestamps reflect actual collection times, not write order

### Example: vLLM Metrics Data

**Example from vLLM inference server:**

```json
{
  "endpoint_url": "http://localhost:8000/metrics",
  "timestamp_ns": 1763591240123456789,
  "endpoint_latency_ns": 42134567,
  "metrics": {
    "vllm:num_requests_running": [
      {"value": 12.0}
    ],
    "vllm:num_requests_waiting": [
      {"value": 3.0}
    ],
    "vllm:gpu_cache_usage_perc": [
      {"value": 0.72}
    ],
    "vllm:gpu_prefix_cache_hit_rate": [
      {"value": 0.85}
    ],
    "vllm:avg_prompt_throughput_toks_per_s": [
      {"value": 1523.4}
    ],
    "vllm:avg_generation_throughput_toks_per_s": [
      {"value": 892.1}
    ],
    "vllm:request_success_total": [
      {"value": 1500.0}
    ],
    "vllm:prompt_tokens_total": [
      {"value": 75000.0}
    ],
    "vllm:generation_tokens_total": [
      {"value": 45000.0}
    ],
    "vllm:time_to_first_token_seconds": [
      {
        "histogram": {
          "0.001": 0.0,
          "0.005": 12.0,
          "0.01": 145.0,
          "0.02": 789.0,
          "0.04": 1234.0,
          "0.06": 1456.0,
          "0.08": 1489.0,
          "0.1": 1498.0,
          "0.25": 1500.0,
          "0.5": 1500.0,
          "1.0": 1500.0,
          "+Inf": 1500.0
        },
        "sum": 32.456,
        "count": 1500.0
      }
    ],
    "vllm:e2e_request_latency_seconds": [
      {
        "histogram": {
          "0.01": 0.0,
          "0.025": 5.0,
          "0.05": 23.0,
          "0.075": 78.0,
          "0.1": 234.0,
          "0.15": 567.0,
          "0.2": 890.0,
          "0.3": 1234.0,
          "0.4": 1456.0,
          "0.5": 1489.0,
          "1.0": 1500.0,
          "2.5": 1500.0,
          "+Inf": 1500.0
        },
        "sum": 245.678,
        "count": 1500.0
      }
    ]
  }
}
```

**Key vLLM Metrics Shown:**
- **Gauges:** Current queue sizes, cache utilization percentages, throughput rates
- **Counters:** Cumulative success counts and token totals
- **Histograms:** Latency distributions with bucket counts showing percentile breakdowns

### Example: Dynamo Component Metrics Data

**Example from Dynamo component (work handler) endpoint:**

```json
{
  "endpoint_url": "http://localhost:8081/metrics",
  "timestamp_ns": 1763591240234567890,
  "endpoint_latency_ns": 38765432,
  "metrics": {
    "dynamo_component_requests_total": [
      {"value": 2340.0}
    ],
    "dynamo_component_errors_total": [
      {"value": 12.0}
    ],
    "dynamo_component_request_bytes_total": [
      {"value": 1170000.0}
    ],
    "dynamo_component_response_bytes_total": [
      {"value": 4656000.0}
    ],
    "dynamo_component_tasks_issued_total": [
      {"value": 2340.0}
    ],
    "dynamo_component_tasks_success_total": [
      {"value": 2328.0}
    ],
    "dynamo_component_tasks_failed_total": [
      {"value": 12.0}
    ],
    "dynamo_component_inflight_requests": [
      {"value": 8.0}
    ],
    "dynamo_component_uptime_seconds": [
      {"value": 3456.0}
    ],
    "dynamo_component_request_duration_seconds": [
      {
        "histogram": {
          "0.005": 0.0,
          "0.01": 23.0,
          "0.025": 456.0,
          "0.05": 1234.0,
          "0.1": 2012.0,
          "0.25": 2298.0,
          "0.5": 2328.0,
          "1.0": 2340.0,
          "+Inf": 2340.0
        },
        "sum": 287.654,
        "count": 2340.0
      }
    ]
  }
}
```

**Key Dynamo Component Metrics Shown:**
- **Counters:** Request counts, error counts, data transfer bytes, task completion status
- **Gauges:** Concurrent request count, uptime tracking
- **Histograms:** Request duration distributions

### Metadata File Structure

The `server_metrics_metadata.json` file describes all collected metrics:

```json
{
  "endpoints": {
    "http://localhost:8000/metrics": {
      "endpoint_url": "http://localhost:8000/metrics",
      "info_metrics": {
        "python_info": {
          "description": "Python platform information",
          "labels": [
            {
              "implementation": "CPython",
              "major": "3",
              "minor": "12",
              "patchlevel": "10",
              "version": "3.12.10"
            }
          ]
        }
      },
      "metric_schemas": {
        "vllm:num_requests_running": {
          "type": "gauge",
          "description": "Number of requests currently running on GPU."
        },
        "vllm:num_requests_waiting": {
          "type": "gauge",
          "description": "Number of requests waiting to be processed."
        },
        "vllm:gpu_cache_usage_perc": {
          "type": "gauge",
          "description": "GPU KV-cache usage. 1 means 100 percent usage."
        },
        "vllm:gpu_prefix_cache_hit_rate": {
          "type": "gauge",
          "description": "GPU prefix cache block hit rate."
        },
        "vllm:request_success_total": {
          "type": "counter",
          "description": "Count of successfully processed requests."
        },
        "vllm:time_to_first_token_seconds": {
          "type": "histogram",
          "description": "Histogram of time to first token in seconds."
        },
        "vllm:e2e_request_latency_seconds": {
          "type": "histogram",
          "description": "Histogram of end to end request latency in seconds."
        }
      }
    },
    "http://localhost:8081/metrics": {
      "endpoint_url": "http://localhost:8081/metrics",
      "metric_schemas": {
        "dynamo_component_requests_total": {
          "type": "counter",
          "description": "Total component requests processed."
        },
        "dynamo_component_errors_total": {
          "type": "counter",
          "description": "Total processing errors."
        },
        "dynamo_component_inflight_requests": {
          "type": "gauge",
          "description": "Concurrent requests being processed."
        },
        "dynamo_component_request_duration_seconds": {
          "type": "histogram",
          "description": "Component request duration in seconds."
        }
      }
    }
  }
}
```

**Use the metadata file to:**
- Discover what metrics are available from each endpoint
- Understand metric types (counter, gauge, histogram, summary)
- Read metric descriptions and understand what they measure
- View info metrics with system/platform information

## Aggregated Statistics Export

AIPerf automatically computes comprehensive statistics from the collected time-series data and exports them to a separate JSON file for easy analysis.

### Output File: `server_metrics.json`

This file contains aggregated statistics for all collected metrics, organized by endpoint and metric type. The aggregations follow Prometheus and LLM serving best practices (vLLM, TensorRT-LLM).

### Aggregation Statistics by Metric Type

Each Prometheus metric type has type-specific aggregations optimized for its semantics:

#### Gauge Metrics - Point-in-Time Values

**Example metrics**: `vllm:kv_cache_usage_perc`, `vllm:num_requests_waiting`, `dynamo_frontend_queued_requests`

**Statistics computed**:
```json
{
  "sample_count": 50,
  "duration_seconds": 5.0,
  "avg": 0.65,
  "min": 0.45,
  "max": 0.85,
  "std": 0.12,
  "p50": 0.64,
  "p90": 0.79,
  "p95": 0.82,
  "p99": 0.84
}
```

**Use cases**:
- Monitor KV cache utilization trends with `avg` and percentiles
- Track queue depth with `p90`, `p95`, `p99` for capacity planning

#### Counter Metrics - Cumulative Totals

**Example metrics**: `vllm:request_success_total`, `vllm:prompt_tokens_total`, `dynamo_component_requests_total`

**Statistics computed**:
```json
{
  "sample_count": 50,
  "duration_seconds": 5.0,
  "delta": 1000.0,
  "rate_avg": 200.0,
  "rate_min": 150.0,
  "rate_max": 280.0,
  "rate_std": 35.2,
  "burst_count": 3
}
```

**Key fields**:
- `delta`: Total change over the aggregation period
- `rate_avg`: Average rate per second (delta/duration) - primary metric
- `rate_std`: Rate variability (higher = more variable traffic)
- `burst_count`: Number of intervals where rate > (avg + 2×std). Indicates traffic spikes.

**Use cases**:
- Identify traffic patterns: `burst_count > 0` indicates bursty traffic
- Monitor throughput stability with `rate_std`
- Capacity planning with `rate_max` (peak throughput)

**Best Practice**:
> Following Prometheus best practices, always use `rate_*` fields (not raw `delta`) for analysis. Rates account for counter resets and provide normalized per-second metrics.

#### Histogram Metrics - Latency Distributions

**Example metrics**: `vllm:time_to_first_token_seconds`, `vllm:e2e_request_latency_seconds`, `triton:request_duration`

**Statistics computed**:
```json
{
  "sample_count": 50,
  "duration_seconds": 5.0,
  "observation_count": 1000,
  "sum": 125.5,
  "avg": 0.1255,
  "observation_rate": 200.0,
  "estimated_percentiles": {
    "p50": 0.110,
    "p90": 0.180,
    "p95": 0.220,
    "p99": 0.350
  },
  "buckets": {
    "0.01": 50.0, "0.1": 450.0, "1.0": 980.0, "+Inf": 1000.0
  }
}
```

**Key fields**:
- `observation_count`: Total observations (e.g., total requests measured)
- `avg`: Average value per observation (e.g., average latency per request)
- `observation_rate`: Average observation rate (observations per second)
- `estimated_percentiles`: p50/p90/p95/p99 for SLO monitoring using standard Prometheus histogram_quantile algorithm
- `buckets`: Raw bucket data for custom analysis

**Use cases**:
- **SLO monitoring**: Use `estimated_percentiles.p95` or `p99` for latency SLO compliance (e.g., "p99 TTFT < 500ms")
- **Performance analysis**: Compare `avg` across runs to track latency trends
- **Custom analysis**: Use `buckets` for downstream tooling

**Important Notes**:
> [!NOTE]
> **Percentile Accuracy**: `estimated_percentiles` use linear interpolation between bucket boundaries and may have ±10-30% error depending on bucket granularity. These match the standard Prometheus `histogram_quantile()` function used for SLO monitoring.

**Histogram vs Summary**:
- **Histograms**: Aggregatable across instances using `histogram_quantile()` - use for distributed systems
- **Summaries**: Exact quantiles but NOT aggregatable - use for single-instance monitoring

#### Summary Metrics - Exact Quantiles

**Example metrics**: `vllm:request_latency_seconds` (if exposed as summary type)

**Statistics computed**:
```json
{
  "sample_count": 50,
  "duration_seconds": 5.0,
  "observation_count": 1000,
  "sum": 245.8,
  "avg": 0.2458,
  "observation_rate": 200.0,
  "quantiles": {
    "p50": 0.220,
    "p90": 0.380,
    "p95": 0.450,
    "p99": 0.620
  },
  "quantile_spread": {
    "p99_p50_ratio": 2.82,
    "p95_p50_ratio": 2.05,
    "p90_p50_ratio": 1.73
  }
}
```

**Key fields**:
- `quantiles`: Exact quantiles from server (p50, p90, p95, p99) - not estimates
- `quantile_spread`: Tail latency ratios indicating distribution shape
  - `p99_p50_ratio`: Overall tail behavior
  - `p95_p50_ratio`: Upper percentile spread
  - `p90_p50_ratio`: Common SLO threshold ratio

**Use cases**:
- **SLO compliance**: Use exact `quantiles` values for strict latency SLO monitoring
- **Long tail detection**: `p99_p50_ratio > 5.0` indicates problematic tail latency requiring investigation

**Interpreting Quantile Spreads**:
```
p99_p50_ratio = 2.0  → ✅ Good - tight distribution
p99_p50_ratio = 5.0  → ⚠️  Warning - significant tail
p99_p50_ratio = 10.0 → 🚨 Critical - investigate causes
```

### JSON Export Structure

The `server_metrics.json` file contains:

```json
{
  "summary": {
    "endpoints_configured": ["http://localhost:8000/metrics", "http://localhost:8081/metrics"],
    "endpoints_successful": ["http://localhost:8000/metrics", "http://localhost:8081/metrics"],
    "start_time": "2025-12-04T10:15:30.123456",
    "end_time": "2025-12-04T10:20:35.789012"
  },
  "endpoints": {
    "http://localhost:8000/metrics": {
      "endpoint_url": "http://localhost:8000/metrics",
      "info_metrics": {
        "python_info": {
          "description": "Python platform information",
          "labels": [{"implementation": "CPython", "version": "3.12.10"}]
        }
      },
      "metrics": {
        "vllm:kv_cache_usage_perc": {
          "description": "GPU KV-cache usage. 1 means 100 percent usage.",
          "type": "gauge",
          "series": [
            {
              "labels": null,
              "stats": {
                "sample_count": 50,
                "avg": 0.65,
                "min": 0.45,
                "max": 0.85,
                "p50": 0.64,
                "p90": 0.79,
                "p95": 0.82,
                "p99": 0.84
              }
            }
          ]
        },
        "vllm:request_success_total": {
          "description": "Count of successfully processed requests.",
          "type": "counter",
          "series": [
            {
              "labels": null,
              "stats": {
                "delta": 1000.0,
                "rate_avg": 200.0,
                "burst_count": 3,
                ...
              }
            }
          ]
        },
        "vllm:time_to_first_token_seconds": {
          "description": "Histogram of time to first token in seconds.",
          "type": "histogram",
          "series": [
            {
              "labels": null,
              "stats": {
                "observation_count": 1000,
                "avg": 0.1255,
                "observation_rate": 200.0,
                "estimated_percentiles": {
                  "p50": 0.110,
                  "p90": 0.180,
                  "p95": 0.220,
                  "p99": 0.350
                },
                "buckets": {...}
              }
            }
          ]
        }
      }
    }
  }
}
```

### Advanced Analysis Examples

#### Detecting Traffic Spikes

```json
{
  "vllm:request_success_total": {
    "rate_avg": 200.0,
    "rate_std": 45.0,
    "burst_count": 5
  }
}
```

**Interpretation**: 5 traffic spikes detected where instantaneous rate exceeded (avg + 2×std). Indicates bursty traffic that may cause performance degradation or resource contention.

#### Analyzing Tail Latency

```json
{
  "vllm:e2e_request_latency_seconds": {
    "quantiles": {
      "p50": 0.220,
      "p99": 1.540
    },
    "quantile_spread": {
      "p99_p50_ratio": 7.0
    }
  }
}
```

**Analysis**: `p99_p50_ratio = 7.0` indicates a long tail distribution. The slowest 1% of requests take 7x longer than the median - investigate causes (cache misses, resource contention, etc.)

### Labeled Metrics (Time Series)

When metrics have labels (e.g., per-model, per-endpoint), each unique label combination is aggregated separately:

```json
{
  "dynamo_frontend_requests": {
    "description": "Total number of requests processed",
    "type": "counter",
    "series": [
      {
        "labels": {"endpoint": "chat", "model": "llama-3", "status": "success"},
        "stats": {
          "delta": 800.0,
          "rate_avg": 160.0,
          ...
        }
      },
      {
        "labels": {"endpoint": "chat", "model": "llama-3", "status": "error"},
        "stats": {
          "delta": 5.0,
          "rate_avg": 1.0,
          ...
        }
      }
    ]
  }
}
```

Each label combination gets independent statistics, enabling per-model, per-endpoint analysis.

## Next Steps

- **GPU Telemetry:** Combine server metrics with [GPU telemetry](gpu-telemetry.md) for comprehensive observability<br>

## Summary

Server metrics collection in AIPerf provides:

✅ **Enabled by default** - automatic discovery of Prometheus endpoints (checks inference endpoint port)<br>
✅ **Comprehensive collection** of all exposed metrics (counters, gauges, histograms, summaries)<br>
✅ **Industry-standard aggregations** - p50/p90/p95/p99 percentiles following Prometheus best practices<br>
✅ **Histogram percentile estimation** - SLO monitoring with `histogram_quantile()` algorithm<br>
✅ **Burst detection** - automatic identification of traffic spikes for capacity planning<br>
✅ **Tail latency analysis** - quantile spread ratios for detecting long-tail distributions<br>
✅ **Efficient storage** with automatic deduplication (per endpoint)<br>
✅ **Multi-node support** for distributed deployments<br>
✅ **Easy analysis** with JSONL time-series and JSON aggregated statistics<br>
