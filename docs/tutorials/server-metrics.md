<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# Server Metrics with AIPerf

This guide shows you how to collect server-side metrics (requests, throughput, latency, KV cache, CPU, memory, etc.) from your inference server during AIPerf benchmarking. Server metrics provide insights into server performance and resource usage while running inference workloads.

## Overview

AIPerf automatically collects server metrics from your inference server's Prometheus `/metrics` endpoint. This works with any inference server that exposes Prometheus metrics, including:
- **Dynamo** (AI-Dynamo inference server with comprehensive metrics)
- **vLLM** (with Prometheus metrics enabled)
- **SGLang** (with Prometheus metrics enabled)
- **TRT-LLM** (with Prometheus metrics enabled)
- **Any custom inference server** that exposes Prometheus-compatible metrics

## Understanding Server Metrics in AIPerf

AIPerf provides server metrics collection with automatic endpoint discovery and optional console display.

### How Server Metrics Collection Works

| Usage | Command | What Gets Collected | Console Display | JSONL Export |
|-------|---------|---------------------|-----------------|--------------|
| **No flag** | `aiperf profile --url localhost:8000 ...` | Inference endpoint + `/metrics` + default endpoints (:8081, :7777, :2379) | ❌ No | ✅ Yes |
| **Flag only** | `aiperf profile --url localhost:8000 ... --server-metrics` | Inference endpoint + `/metrics` + default endpoints | ✅ Yes | ✅ Yes |
| **Custom URLs** | `aiperf profile --url localhost:8000 ... --server-metrics http://worker:9000/metrics` | Inference endpoint + defaults + custom URLs | ✅ Yes | ✅ Yes |

> [!IMPORTANT]
> Server metrics are **ALWAYS collected automatically** from multiple default endpoints, regardless of whether the `--server-metrics` flag is used:
> - **Inference endpoint**: Auto-derived from `--url` (e.g., `http://localhost:8000/metrics`)
> - **Additional defaults**: `localhost:8081/metrics`, `localhost:7777/metrics`, `localhost:2379/metrics`
>
> The flag primarily controls whether metrics are displayed on the console and allows you to specify additional custom Prometheus endpoints (e.g., separate frontend/backend servers).

> [!NOTE]
> When specifying custom Prometheus endpoint URLs, the `http://` prefix is optional. URLs like `localhost:8081` will automatically be treated as `http://localhost:8081`. Both formats work identically.

### Automatic Endpoint Discovery

AIPerf automatically collects server metrics from multiple default endpoints:

```bash
# When you run: aiperf profile --url localhost:8000 ...
# AIPerf automatically tries ALL of these endpoints:
#   - http://localhost:8000/metrics  (auto-derived from --url)
#   - http://localhost:8081/metrics  (default backend/worker)
#   - http://localhost:7777/metrics  (default backend/worker)
#   - http://localhost:2379/metrics  (default service, e.g., etcd)

# When you run: aiperf profile --url http://server:9090 ...
# AIPerf automatically tries:
#   - http://server:9090/metrics     (auto-derived from --url)
#   - http://localhost:8081/metrics  (default)
#   - http://localhost:7777/metrics  (default)
#   - http://localhost:2379/metrics  (default)
```

This means server metrics from distributed systems "just work" without any additional configuration! Only reachable endpoints will be collected and exported.

---

# Using Server Metrics with Dynamo

Dynamo provides comprehensive inference server metrics out of the box, including:
- **Request metrics**: Total requests, in-flight requests, success/error rates
- **Throughput metrics**: Requests per second, tokens per second
- **Latency metrics**: Time to first token (TTFT), inter-token latency (ITL), end-to-end latency
- **KV Cache statistics**: Active blocks, cache usage, memory pressure
- **Component metrics**: Per-component request bytes, response bytes, processing time
- **Model configuration**: Workers, total KV blocks, block size

## Setup Dynamo Server

```bash
# Set environment variables
export AIPERF_REPO_TAG="main"
export DYNAMO_PREBUILT_IMAGE_TAG="nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.1"
export MODEL="Qwen/Qwen3-0.6B"

# Download the Dynamo container
docker pull ${DYNAMO_PREBUILT_IMAGE_TAG}
export DYNAMO_REPO_TAG=$(docker run --rm --entrypoint "" ${DYNAMO_PREBUILT_IMAGE_TAG} cat /workspace/version.txt | cut -d'+' -f2)

# Start up required services
curl -O https://raw.githubusercontent.com/ai-dynamo/dynamo/${DYNAMO_REPO_TAG}/deploy/docker-compose.yml
docker compose -f docker-compose.yml down || true
docker compose -f docker-compose.yml up -d

# Launch Dynamo in the background
docker run \
  --rm \
  --gpus all \
  --network host \
  ${DYNAMO_PREBUILT_IMAGE_TAG} \
    /bin/bash -c "python3 -m dynamo.frontend & python3 -m dynamo.vllm --model ${MODEL} --enforce-eager --no-enable-prefix-caching" > server.log 2>&1 &
```

```bash
# Set up AIPerf
docker run \
  -it \
  --rm \
  --gpus all \
  --network host \
  -e AIPERF_REPO_TAG=${AIPERF_REPO_TAG} \
  -e MODEL=${MODEL} \
  ubuntu:24.04

apt update && apt install -y curl git

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

uv venv --python 3.10

source .venv/bin/activate

git clone -b ${AIPERF_REPO_TAG} --depth 1 https://github.com/ai-dynamo/aiperf.git

uv pip install ./aiperf
```

## Verify Dynamo is Running

```bash
# Wait for Dynamo API to be ready (up to 15 minutes)
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"max_completion_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "Dynamo not ready after 15min"; exit 1; }
```

```bash
# Wait for Prometheus metrics endpoint to be ready (up to 2 minutes after Dynamo is ready)
echo "Dynamo ready, waiting for metrics endpoint to be available..."
timeout 120 bash -c 'while true; do STATUS=$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/metrics); if [ "$STATUS" = "200" ]; then if curl -s localhost:8000/metrics | grep -q "dynamo_"; then break; fi; fi; echo "Waiting for Prometheus metrics..."; sleep 5; done' || { echo "Prometheus metrics not found after 2min"; exit 1; }
echo "Server metrics are now available"
```

## Run AIPerf Benchmark (Without Console Display)

Server metrics are collected automatically:

```bash
# Server metrics collected from http://localhost:8000/metrics automatically
# Console display: NO, JSONL export: YES
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 200 \
    --output-tokens-stddev 0 \
    --extra-inputs min_tokens:200 \
    --extra-inputs ignore_eos:true \
    --concurrency 4 \
    --request-count 64 \
    --warmup-request-count 1 \
    --num-dataset-entries 8 \
    --random-seed 100
```

Server metrics are automatically exported to `server_metrics_export.jsonl` in the artifacts directory, even without the `--server-metrics` flag!

## Run AIPerf Benchmark (With Console Display)

Add `--server-metrics` to see metrics in the console:

```bash
# Server metrics collected AND displayed in console
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 200 \
    --output-tokens-stddev 0 \
    --extra-inputs min_tokens:200 \
    --extra-inputs ignore_eos:true \
    --concurrency 4 \
    --request-count 64 \
    --warmup-request-count 1 \
    --num-dataset-entries 8 \
    --random-seed 100 \
    --server-metrics
```

---

# Using Server Metrics with Other Inference Servers

This path works with **vLLM, SGLang, TRT-LLM, or any inference server that exposes Prometheus metrics**. We'll use vLLM as an example.

## Setup vLLM Server with Prometheus Metrics

vLLM automatically exposes Prometheus metrics on the `/metrics` endpoint when you start the server.

```bash
# Set environment variables
export MODEL="Qwen/Qwen3-0.6B"
export AIPERF_REPO_TAG="main"

# Start vLLM with Prometheus metrics enabled (enabled by default)
docker pull vllm/vllm-openai:latest

docker run -d --name vllm-server \
  --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000
```

```bash
# Set up AIPerf
docker run \
  -it \
  --rm \
  --gpus all \
  --network host \
  -e AIPERF_REPO_TAG=${AIPERF_REPO_TAG} \
  -e MODEL=${MODEL} \
  ubuntu:24.04

apt update && apt install -y curl git

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

uv venv --python 3.10

source .venv/bin/activate

git clone -b ${AIPERF_REPO_TAG} --depth 1 https://github.com/ai-dynamo/aiperf.git

uv pip install ./aiperf
```

> [!NOTE]
> Replace the vLLM command above with your preferred backend (SGLang, TRT-LLM, etc.). As long as the server exposes Prometheus metrics on `/metrics`, AIPerf will automatically collect them.

## Verify vLLM is Running

```bash
# Wait for vLLM inference server to be ready (up to 15 minutes)
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }

# Wait for Prometheus metrics endpoint to be available (up to 2 minutes after vLLM is ready)
echo "vLLM ready, waiting for metrics endpoint to be available..."
timeout 120 bash -c 'while true; do OUTPUT=$(curl -s localhost:8000/metrics); if echo "$OUTPUT" | grep -q "vllm_"; then break; fi; echo "Waiting for Prometheus metrics..."; sleep 5; done' || { echo "Prometheus metrics not found after 2min"; exit 1; }
echo "Server metrics are now available"
```

## Run AIPerf Benchmark

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 200 \
    --output-tokens-stddev 0 \
    --extra-inputs min_tokens:200 \
    --extra-inputs ignore_eos:true \
    --concurrency 4 \
    --request-count 64 \
    --warmup-request-count 1 \
    --num-dataset-entries 8 \
    --random-seed 100 \
    --server-metrics
```

---

## Multi-Endpoint Server Metrics Example

For distributed setups with separate frontend and backend servers, you can collect metrics from all components simultaneously:

```bash
# Example: Collecting metrics from Dynamo frontend + backend workers
# Note: The inference endpoint http://localhost:8000/metrics is always attempted
#       automatically in addition to these custom URLs
# URLs can be specified with or without the http:// prefix
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 200 \
    --output-tokens-stddev 0 \
    --extra-inputs min_tokens:200 \
    --extra-inputs ignore_eos:true \
    --concurrency 4 \
    --request-count 64 \
    --warmup-request-count 1 \
    --num-dataset-entries 8 \
    --random-seed 100 \
    --server-metrics localhost:8081 localhost:7777 localhost:2379
```

This will collect server metrics from:
- `http://localhost:8000/metrics` (inference endpoint, automatically derived)
- `http://localhost:8081/metrics` (worker/backend service 1)
- `http://localhost:7777/metrics` (worker/backend service 2)
- `http://localhost:2379/metrics` (additional service, e.g., etcd)

All metrics are displayed on the console and saved to the `server_metrics_export.jsonl` file.

---

## How Dynamic Metric Discovery Works

AIPerf uses **dynamic field discovery** to automatically collect and display metrics from any Prometheus-compatible endpoint without requiring manual configuration. This means:

✅ **No configuration needed** - Works with any metrics your server exposes
✅ **Automatic display names** - Converts `kvstats_gpu_cache_usage_percent` → "KVStats GPU Cache Usage Percent"
✅ **Smart unit detection** - Automatically infers units from field names (seconds, bytes, percent, count)
✅ **Acronym-aware formatting** - Recognizes common terms like HTTP, GPU, KV, TTFT, and displays them correctly

### How It Works

1. **Metric Collection**: AIPerf fetches all metrics from your server's Prometheus `/metrics` endpoint
2. **Field Discovery**: Automatically discovers all available metric fields from the server response
3. **Display Formatting**: Transforms Prometheus metric names into human-readable labels using intelligent parsing:
   - Splits on underscores: `vllm_num_requests_running` → `vllm`, `num`, `requests`, `running`
   - Recognizes acronyms: `vllm`, `http`, `kv`, `gpu`, `ttft`, `api`, etc. → uppercase
   - Capitalizes other words: `num`, `requests`, `running` → "Num", "Requests", "Running"
   - Final result: **"vLLM Num Requests Running"**
4. **Unit Inference**: Automatically determines units from naming patterns:
   - `*_seconds` or `*_seconds_*` → seconds
   - `*_bytes` or `*_bytes_*` → bytes
   - `*_percent` or `*_percentage` → percent
   - Everything else → count

This approach means **AIPerf works with any Prometheus metrics out of the box** - whether from Dynamo, vLLM, SGLang, TRT-LLM, or custom inference servers. No code changes needed when new metrics are added!

---

## Example Metrics by Server Type

While AIPerf automatically collects **all** available metrics, here are examples of common metrics you'll see from different inference servers:

### Dynamo Metrics
Dynamo exposes comprehensive metrics including:
- **Request tracking**: Total requests, in-flight requests, success/error rates
- **Component metrics**: Per-component request/response bytes, processing time
- **KV Cache statistics**: Active blocks, cache usage percentage, allocation/copy time
- **Frontend metrics**: TTFT, inter-token latency, end-to-end latency
- **Model config**: Total KV blocks, block size, worker count

### vLLM Metrics
vLLM exposes metrics like:
- **Request metrics**: `vllm_num_requests_running`, `vllm_num_requests_waiting`
- **Cache metrics**: `vllm_cache_usage_perc`, `vllm_gpu_cache_usage_perc`
- **Token metrics**: `vllm_avg_prompt_throughput_toks_per_s`, `vllm_avg_generation_throughput_toks_per_s`
- **Time metrics**: `vllm_time_to_first_token_seconds`, `vllm_time_per_output_token_seconds`

### Custom Server Metrics
Any custom inference server that exposes Prometheus metrics will work automatically. For example:
```
my_server_request_count_total
my_server_latency_seconds
my_server_gpu_memory_bytes
```
All will be automatically collected and displayed as:
- **My Server Request Count Total (count)**
- **My Server Latency Seconds (seconds)**
- **My Server GPU Memory Bytes (bytes)**

> [!TIP]
> The list above is just examples! AIPerf automatically discovers and displays **all metrics** your server exposes, regardless of naming conventions. Add new metrics to your server and they'll appear automatically in AIPerf's output.

---

## Example Console Display

```
                          NVIDIA AIPerf | Server Metrics Summary
                                1/1 Server endpoints reachable
                                      • localhost:8000 ✔

                        localhost:8000 | dynamo-frontend | my-server-hostname
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┓
┃                               Metric ┃       avg ┃       min ┃       max ┃       p99 ┃       p90 ┃       p50 ┃     std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━┩
│                     Requests Total () │     64.00 │     64.00 │     64.00 │     64.00 │     64.00 │     64.00 │    0.00 │
│                 Requests In-Flight () │      2.35 │      0.00 │      4.00 │      4.00 │      4.00 │      3.00 │    1.42 │
│      Component In-Flight Requests (%) │      2.18 │      0.00 │      4.00 │      4.00 │      4.00 │      2.00 │    1.38 │
│       Component Request Bytes (bytes) │  4,256.80 │  4,096.00 │  4,608.00 │  4,608.00 │  4,608.00 │  4,352.00 │  142.35 │
│      Component Response Bytes (bytes) │ 12,845.20 │ 12,288.00 │ 13,824.00 │ 13,824.00 │ 13,824.00 │ 13,056.00 │  427.05 │
│         KVStats Active Blocks (count) │     18.45 │      0.00 │     32.00 │     32.00 │     32.00 │     20.00 │    9.67 │
│          KVStats GPU Cache Usage (%)  │     15.00 │      0.00 │     25.00 │     25.00 │     25.00 │     16.00 │    8.00 │
│    Frontend In-Flight Requests (rate) │      2.35 │      0.00 │      4.00 │      4.00 │      4.00 │      3.00 │    1.42 │
│ Frontend Time to First Token (seconds)│      0.08 │      0.05 │      0.12 │      0.12 │      0.11 │      0.08 │    0.02 │
│        Model Total KV Blocks (blocks) │    128.00 │    128.00 │    128.00 │    128.00 │    128.00 │    128.00 │    0.00 │
│                  Model Workers (count)│      1.00 │      1.00 │      1.00 │      1.00 │      1.00 │      1.00 │    0.00 │
└──────────────────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴─────────┘
```

---

## JSONL Export Format

Server metrics are always exported to `server_metrics_export.jsonl` in the artifacts directory. Each line is a JSON object representing one server at one point in time:

```json
{
  "timestamp_ns": 1736832498450123456,
  "server_url": "http://localhost:8000/metrics",
  "server_id": "dynamo-frontend-localhost-8000",
  "server_type": "dynamo-frontend",
  "hostname": "my-server-hostname",
  "instance": "localhost:8000",
  "metrics_data": {
    "requests_total": 64.0,
    "requests_in_flight": 3.0,
    "component_inflight_requests": 3.0,
    "component_request_bytes_total": 4352.0,
    "component_response_bytes_total": 13056.0,
    "kvstats_active_blocks": 24.0,
    "kvstats_gpu_cache_usage_percent": 18.75,
    "frontend_inflight_requests": 3.0,
    "frontend_time_to_first_token_seconds": 0.085,
    "model_total_kv_blocks": 128.0,
    "model_workers": 1.0
  }
}
```

This time-series format is perfect for:
- **Time-series analysis** with tools like Pandas, DuckDB, or Polars
- **Correlation analysis** between server metrics and inference performance
- **Custom visualization** with Matplotlib, Plotly, or Grafana
- **Performance debugging** by examining metrics over time

---

## Integration with GPU Telemetry

Server metrics work seamlessly with GPU telemetry. You can collect both simultaneously:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 200 \
    --output-tokens-stddev 0 \
    --extra-inputs min_tokens:200 \
    --extra-inputs ignore_eos:true \
    --concurrency 4 \
    --request-count 64 \
    --warmup-request-count 1 \
    --num-dataset-entries 8 \
    --random-seed 100 \
    --gpu-telemetry \
    --server-metrics
```

This gives you complete visibility:
- **Client-side metrics**: Request latency, throughput, errors (always collected)
- **Server-side metrics**: KV cache, request queues, component performance (from Prometheus)
- **GPU metrics**: Power, utilization, memory, temperature (from DCGM)

All three metric types are time-aligned and exported to separate JSONL files for correlation analysis!

---

## Best Practices

1. **Always enable for production profiling**: Server metrics are lightweight and provide invaluable insights
2. **Use with GPU telemetry**: Combined metrics show the complete picture
3. **Analyze JSONL exports**: Time-series data reveals patterns not visible in aggregated statistics
4. **Monitor KV cache**: High cache pressure often indicates performance bottlenecks
5. **Track TTFT separately**: Time to first token is a critical user experience metric
6. **Collect from all components**: In distributed setups, collect from frontend AND backend servers

---

## Troubleshooting

### Server metrics not collected?

1. **Check if `/metrics` endpoint exists**:
   ```bash
   curl http://localhost:8000/metrics
   ```

2. **Verify Prometheus metrics are exposed**:
   ```bash
   curl http://localhost:8000/metrics | grep -E "(dynamo_|vllm_|requests_)"
   ```

3. **Check AIPerf logs**: Look for server metrics connection messages in the output

### Console display not showing?

Make sure you're using the `--server-metrics` flag:
```bash
aiperf profile ... --server-metrics
```

### Want to disable collection?

Server metrics collection is automatic and lightweight. If you really want to disable it, you would need to modify the inference server to not expose the `/metrics` endpoint (not recommended).

---

## Summary

- ✅ Server metrics are **ALWAYS collected automatically** from your inference endpoint
- ✅ Use `--server-metrics` flag to enable console display
- ✅ JSONL export is always generated if data is collected
- ✅ Works with Dynamo, vLLM, SGLang, TRT-LLM, and any Prometheus-compatible server
- ✅ Provides comprehensive insights: requests, throughput, latency, KV cache, and more
- ✅ Integrates seamlessly with GPU telemetry for complete observability
