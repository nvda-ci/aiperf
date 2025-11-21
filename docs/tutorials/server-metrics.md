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

Server metrics are **automatically collected** - just run AIPerf normally:

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
1. AIPerf discovers and queries Prometheus `/metrics` endpoint on your inference server (checks `--url` port)
2. Collects metrics every 100ms (configurable)
3. Exports time-series data to JSONL file
4. Saves metadata about collected metrics

> [!TIP]
> No flag needed! Server metrics are collected by default. Use `--server-metrics <urls>` to add additional endpoints, or set `AIPERF_SERVER_METRICS_ENABLED=false` to disable.

### Automatic Endpoint Discovery

By default, AIPerf automatically:

1. **Extracts the hostname and port** from your inference endpoint URL (`--url`)
2. **Tests the same port** as your inference endpoint on that hostname
3. **Optionally tests additional ports** if configured via `AIPERF_SERVER_METRICS_DEFAULT_BACKEND_PORTS`
4. **Collects from all reachable endpoints that respond to the `/metrics` endpoint**

**Example:**
```bash
# Inference URL: http://gpu-server:8000/v1/chat/completions
# AIPerf automatically tries:
#   - http://gpu-server:8000/metrics  (same port as inference endpoint)
#   - Plus any ports specified in AIPERF_SERVER_METRICS_DEFAULT_BACKEND_PORTS (if set)
```

> [!NOTE]
> By default, `AIPERF_SERVER_METRICS_DEFAULT_BACKEND_PORTS` is empty. AIPerf only checks the inference endpoint's port unless you configure additional ports to check.

> [!TIP]
> **Default Port Handling:** When your inference URL has no explicit port (e.g., `https://api.example.com/v1/chat`), AIPerf uses the default port for the scheme (443 for HTTPS, 80 for HTTP) before checking additional ports from `DEFAULT_BACKEND_PORTS`.

### Custom Endpoint URLs

Specify custom Prometheus endpoints explicitly:

```bash
# Single custom endpoint
aiperf profile --model MODEL ... --server-metrics http://localhost:9400/metrics

# Multiple endpoints (multi-node or multiple services)
aiperf profile --model MODEL ... --server-metrics \
    http://node1:8081/metrics \
    http://node2:8081/metrics \
    http://monitoring:9090/metrics
```

> [!NOTE]
> URLs can be specified with or without the `http://` prefix and `/metrics` suffix. AIPerf normalizes them automatically:
> - `localhost:8000` → `http://localhost:8000/metrics`
> - `http://server:9090` → `http://server:9090/metrics`

### Disabling Server Metrics

To disable automatic server metrics collection:

```bash
AIPERF_SERVER_METRICS_ENABLED=false aiperf profile --model MODEL ...
```

This completely disables server metrics collection for the run.

## Understanding Server Metrics

### What Metrics Are Collected?

AIPerf collects **all metrics** exposed by Prometheus-compatible endpoints, with automatic filtering:

- **Collected:** All counter, gauge, histogram, and summary metrics
- **Automatically Filtered:** Metrics ending with `_created` (internal Prometheus timestamps)

Common metrics from LLM inference servers include:

#### Dynamo Metrics Examples
- **Request Metrics:** `dynamo_frontend_requests`, `dynamo_component_requests`
- **Latency Distributions:** `dynamo_frontend_time_to_first_token_seconds`, `dynamo_frontend_request_duration_seconds`, `dynamo_frontend_inter_token_latency_seconds`
- **Queue Metrics:** `dynamo_frontend_queued_requests`, `dynamo_frontend_inflight_requests`, `dynamo_component_inflight_requests`
- **Token Metrics:** `dynamo_frontend_input_sequence_tokens`, `dynamo_frontend_output_sequence_tokens`
- **Model Config:** `dynamo_frontend_model_context_length`, `dynamo_frontend_model_total_kv_blocks`
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

### Metadata File Structure

The `server_metrics_metadata.json` file describes all collected metrics:

```json
{
  "endpoints": {
    "endpoint_url": {
      "endpoint_url": "http://localhost:8000/metrics",
      "metric_schemas": {
        "metric_name": {
          "type": "counter|gauge|histogram|summary",
          "help": "Metric description from Prometheus"
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

## Next Steps

- **GPU Telemetry:** Combine server metrics with [GPU telemetry](gpu-telemetry.md) for comprehensive observability<br>

## Summary

Server metrics collection in AIPerf provides:

✅ **Enabled by default** - automatic discovery of Prometheus endpoints (checks inference endpoint port)<br>
✅ **Comprehensive collection** of all exposed metrics (counters, gauges, histograms, summaries)<br>
✅ **Efficient storage** with automatic deduplication (per endpoint)<br>
✅ **Multi-node support** for distributed deployments<br>
✅ **Easy analysis** with JSONL format and metadata schemas<br>
