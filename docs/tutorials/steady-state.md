<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Steady-State Measurement

Steady-state measurement provides highly accurate throughput and latency metrics by measuring only during a stable measurement window while the server is under continuous load.

## Overview

Traditional benchmarking includes warmup overhead and tail latencies from requests that start near the end of the benchmark. Steady-state mode solves this with a **3-loop methodology**:

1. **Warmup Loop (a)**: Requests that bring the server to operating temperature
2. **Measurement Loop (b)**: The precise window where metrics are captured
3. **Tail Loop (c)**: Requests that keep the server loaded while measurement requests complete

This approach ensures:

- **Accurate throughput**: Only tokens emitted during the measurement window are counted
- **Stable latencies**: Measurements occur while the server is at steady-state load
- **No tail bias**: Late-finishing warmup or early-starting tail requests don't pollute metrics
- **Precise timing**: The measurement window is exactly defined from the first measurement request to the last measurement response

## How It Works

When `--steady-state` is enabled, AIPerf loops through your dataset continuously:

```
Dataset: [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8]

Timeline:
├─ Warmup (a1-a8)      → Requests submitted, excluded from metrics
├─ Measurement (b1-b8) → measurement_start_ns = b1 sent
│                      → measurement_end_ns = max(b1...b8 completion times)
├─ Tail (c1-c8)        → Continue submitting to maintain load
└─ Grace period        → Wait, then cancel remaining requests
```

**Key behaviors:**

- `measurement_start_ns`: When the first measurement request (b1) is sent
- `measurement_end_ns`: When the *last measurement request to complete* finishes (not when b8 is sent)
- Tail requests continue during the grace period to keep the server loaded
- Only TTFT/ITL events occurring within `[measurement_start_ns, measurement_end_ns]` are counted

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--steady-state` | `false` | Enable steady-state measurement mode |
| `--steady-state-grace-period` | `2.0` | Seconds to continue submitting after measurement loop completes |
| `--steady-state-count-tail` | `false` | Include cancelled tail requests in metrics |
| `--request-count` | dataset size | Number of requests in each loop (measurement window size) |

## Basic Usage

### Setting Up the Server

```bash
# Start vLLM server
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000 &
```

```bash
# Wait for server to be ready
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
```

### Simple Steady-State Benchmark

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --concurrency 64 \
    --request-rate 100 \
    --steady-state \
    --synthetic-input-tokens-mean 500 \
    --output-tokens-mean 200
```

This runs:
1. 10 warmup requests (default `--request-count`)
2. 10 measurement requests (metrics captured here)
3. Tail requests until grace period expires
4. Cancellation of remaining in-flight requests

### High-Throughput Steady-State

For accurate throughput measurement under heavy load:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --concurrency 128 \
    --request-rate 500 \
    --request-count 100 \
    --steady-state \
    --steady-state-grace-period 5.0 \
    --synthetic-input-tokens-mean 256 \
    --output-tokens-mean 128 \
    --random-seed 42
```

This configuration:
- Uses 100 requests per loop for statistical significance
- Higher concurrency and request rate saturate the server
- 5-second grace period ensures measurement requests complete under load
- Fixed seed for reproducibility

## Understanding the Output

With steady-state mode, the reported metrics reflect only the measurement window:

- **`output_token_throughput`**: Tokens emitted during `[measurement_start, measurement_end]` divided by window duration
- **`request_throughput`**: Measurement requests completed per second within the window
- **`time_to_first_token`**: TTFT for requests where the first token arrived within the window
- **`inter_token_latency`**: ITL for individual tokens emitted within the window
- **`benchmark_duration`**: Exactly `measurement_end_ns - measurement_start_ns`

### Event-Level Filtering

Steady-state mode performs fine-grained filtering at the token level:

- A request that **starts in warmup** but **receives tokens during measurement** will have those tokens counted
- A request that **starts during measurement** but **finishes during tail** will have tokens counted until `measurement_end_ns`
- Only the portion of each request that overlaps with the measurement window contributes to metrics

This provides the most accurate steady-state throughput measurement possible.

## Use Cases

### Capacity Planning

Determine maximum sustainable throughput:

```bash
# Test at increasing request rates
for rate in 100 200 300 400 500; do
    echo "Testing at $rate req/s"
    aiperf profile \
        --model Qwen/Qwen3-0.6B \
        --endpoint-type chat \
        --endpoint /v1/chat/completions \
        --streaming \
        --url localhost:8000 \
        --concurrency 256 \
        --request-rate $rate \
        --request-count 50 \
        --steady-state \
        --synthetic-input-tokens-mean 500 \
        --output-tokens-mean 200 \
        --artifact-dir ./results/rate_$rate
done
```

### SLA Validation

Verify latency requirements under sustained load:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --concurrency 64 \
    --request-rate 50 \
    --request-count 200 \
    --steady-state \
    --steady-state-grace-period 10.0 \
    --synthetic-input-tokens-mean 1000 \
    --output-tokens-mean 500
```

The larger request count (200) provides better statistical confidence for p99 latency measurements.

### Comparing Configurations

Steady-state mode makes A/B comparisons more reliable:

```bash
# Test configuration A
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --url localhost:8000 \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --concurrency 64 \
    --request-rate 100 \
    --request-count 100 \
    --steady-state \
    --random-seed 12345 \
    --artifact-dir ./results/config_a

# Test configuration B (different server settings)
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --url localhost:8001 \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --concurrency 64 \
    --request-rate 100 \
    --request-count 100 \
    --steady-state \
    --random-seed 12345 \
    --artifact-dir ./results/config_b
```

Using the same seed and steady-state measurement ensures fair comparison.

## Best Practices

> [!TIP]
> **Choose request count based on your goal:**
> - **Quick validation**: 10-20 requests per loop
> - **Throughput measurement**: 50-100 requests for good average
> - **Latency percentiles (p99)**: 200+ requests for statistical significance

> [!TIP]
> **Set grace period based on expected response time:**
> - For short responses (< 1s): 2-3 second grace period
> - For long responses (> 5s): 10+ second grace period
> - Rule of thumb: `grace_period >= 2 * expected_max_response_time`

> [!IMPORTANT]
> **Ensure concurrency can sustain load:**
> If `concurrency < request_count`, requests will queue up during the measurement loop. Set `concurrency >= request_count` for true steady-state behavior.

> [!NOTE]
> **Warmup is automatic:**
> The first loop through the dataset serves as warmup. You don't need to set `--warmup-request-count` separately (though you can for additional warmup).

## Quick Reference

**Enable steady-state:**
```bash
--steady-state
```

**Control measurement window size:**
```bash
--request-count 100  # 100 requests in measurement window
```

**Extend grace period for long responses:**
```bash
--steady-state-grace-period 10.0  # 10 seconds
```

**Include tail requests in metrics (not recommended for accuracy):**
```bash
--steady-state-count-tail
```

**Full example:**
```bash
aiperf profile \
    --model <model> \
    --url <server> \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --concurrency 64 \
    --request-rate 100 \
    --request-count 100 \
    --steady-state \
    --steady-state-grace-period 5.0 \
    --synthetic-input-tokens-mean 500 \
    --output-tokens-mean 200
```
