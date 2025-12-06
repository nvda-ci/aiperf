<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Server Metrics Reference

This document provides semantic meanings for all metrics collected from NVIDIA Dynamo and vLLM inference servers.

## Table of Contents

- [Dynamo Frontend Metrics](#dynamo-frontend-metrics)
- [Dynamo Component Metrics](#dynamo-component-metrics)
- [vLLM Metrics](#vllm-metrics)

---

## Dynamo Frontend Metrics

The Dynamo frontend is the HTTP entry point that receives client requests and routes them to backend workers.

### Request Flow Metrics

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `dynamo_frontend_requests` | counter | requests | Total LLM requests processed. Use `delta` for requests during benchmark, `rate_per_second` for throughput. Labels: `endpoint`, `model`, `request_type`, `status`. |
| `dynamo_frontend_inflight_requests` | gauge | requests | Requests currently being processed. High values indicate saturation. Labels: `model`. |
| `dynamo_frontend_queued_requests` | gauge | requests | Requests waiting in HTTP queue before processing. Non-zero values indicate backpressure. Labels: `model`. |
| `dynamo_frontend_disconnected_clients` | gauge | clients | Number of client connections that disconnected (possibly due to timeouts). |

### Latency Metrics

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `dynamo_frontend_request_duration_seconds` | histogram | seconds | **End-to-end request latency** from HTTP receive to response complete. Key metric for SLA compliance. Use `p99` for tail latency. Labels: `model`. |
| `dynamo_frontend_time_to_first_token_seconds` | histogram | seconds | **Time to first token (TTFT)** - latency until first token is generated. Critical for perceived responsiveness. Labels: `model`. |
| `dynamo_frontend_inter_token_latency_seconds` | histogram | seconds | **Inter-token latency (ITL)** - time between consecutive tokens. Lower is better for streaming UX. Labels: `model`. |

### Token Metrics

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `dynamo_frontend_output_tokens` | counter | tokens | Total output tokens generated. `rate_per_second` = output token throughput. Labels: `model`. |
| `dynamo_frontend_input_sequence_tokens` | histogram | tokens | **Input sequence length distribution**. `avg` = mean prompt length, `p99` = longest prompts. Labels: `model`. |
| `dynamo_frontend_output_sequence_tokens` | histogram | tokens | **Output sequence length distribution**. `avg` = mean response length. Labels: `model`. |

### Model Configuration (Static)

These are constant values that don't change during the benchmark. Only `avg` is meaningful.

| Metric | Type | Description |
|--------|------|-------------|
| `dynamo_frontend_model_context_length` | gauge | Maximum context window size in tokens (e.g., 40960). |
| `dynamo_frontend_model_kv_cache_block_size` | gauge | KV cache block size in tokens (e.g., 16). |
| `dynamo_frontend_model_max_num_batched_tokens` | gauge | Maximum tokens that can be batched together. |
| `dynamo_frontend_model_max_num_seqs` | gauge | Maximum concurrent sequences per worker. |
| `dynamo_frontend_model_total_kv_blocks` | gauge | Total KV cache blocks available per worker. |
| `dynamo_frontend_model_migration_limit` | gauge | Maximum request migrations allowed (0 = disabled). |

---

## Dynamo Component Metrics

Dynamo components are backend workers that execute inference. These metrics come from the worker process level.

### Request Processing

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `dynamo_component_requests` | counter | requests | Requests processed by this worker. Compare across workers to check load balancing. |
| `dynamo_component_inflight_requests` | gauge | requests | Requests currently executing on this worker. |
| `dynamo_component_request_duration_seconds` | histogram | seconds | Worker-level request processing time. Compare to frontend duration to measure routing overhead. |
| `dynamo_component_errors` | counter | errors | Errors in work handler. Non-zero indicates problems. |

### Data Transfer

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `dynamo_component_request_bytes` | counter | bytes | Total bytes received in requests. `rate_per_second` = inbound bandwidth. |
| `dynamo_component_response_bytes` | counter | bytes | Total bytes sent in responses. `rate_per_second` = outbound bandwidth. |

### KV Cache Statistics

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `dynamo_component_kvstats_active_blocks` | gauge | blocks | KV cache blocks currently in use. |
| `dynamo_component_kvstats_total_blocks` | gauge | blocks | Total KV cache blocks available. |
| `dynamo_component_kvstats_gpu_cache_usage_percent` | gauge | percent | **GPU cache utilization** (0.0-1.0). High values (>0.9) may cause preemptions. |
| `dynamo_component_kvstats_gpu_prefix_cache_hit_rate` | gauge | ratio | Prefix cache hit rate. Higher = better reuse of cached prefixes. |

### NATS Messaging (Internal)

These track the internal NATS messaging system used for component communication.

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `dynamo_component_nats_client_connection_state` | gauge | - | Connection state: 0=disconnected, 1=connected, 2=reconnecting. |
| `dynamo_component_nats_client_current_connections` | gauge | connections | Active NATS connections. |
| `dynamo_component_nats_client_in_messages` | gauge | messages | Messages received via NATS. |
| `dynamo_component_nats_client_out_messages` | gauge | messages | Messages sent via NATS. |
| `dynamo_component_nats_client_in_total_bytes` | gauge | bytes | Bytes received via NATS. |
| `dynamo_component_nats_client_out_overhead_bytes` | gauge | bytes | Bytes sent via NATS. |
| `dynamo_component_nats_service_active_services` | gauge | services | Active NATS services in component. |
| `dynamo_component_nats_service_active_endpoints` | gauge | endpoints | Active NATS endpoints. |
| `dynamo_component_nats_service_requests_total` | gauge | requests | Total NATS service requests. |
| `dynamo_component_nats_service_errors_total` | gauge | errors | NATS service errors. |
| `dynamo_component_nats_service_processing_ms_total` | gauge | milliseconds | Total NATS processing time. |
| `dynamo_component_nats_service_processing_ms_avg` | gauge | milliseconds | Average NATS processing time. |

### System

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `dynamo_component_uptime_seconds` | gauge | seconds | Worker uptime. Useful for detecting restarts. |

---

## vLLM Metrics

vLLM is the inference engine. These metrics provide deep visibility into model execution.

### Cache & Memory

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `vllm:kv_cache_usage_perc` | gauge | percent | **KV cache utilization** (0.0-1.0). Key capacity indicator. Values near 1.0 cause performance degradation. |
| `vllm:prefix_cache_hits` | counter | hits | Tokens served from prefix cache. Higher = better prompt reuse. |
| `vllm:prefix_cache_queries` | counter | queries | Tokens queried against prefix cache. `hits/queries` = hit rate. |
| `vllm:num_preemptions` | counter | preemptions | Requests preempted due to memory pressure. Non-zero indicates capacity issues. |

### Queue State

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `vllm:num_requests_running` | gauge | - | Requests currently in model execution batch. Indicates batch size. |
| `vllm:num_requests_waiting` | gauge | - | Requests queued waiting for execution. High values indicate saturation. |

### Token Throughput

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `vllm:prompt_tokens` | counter | tokens | Prefill tokens processed. `rate_per_second` = prefill throughput. |
| `vllm:generation_tokens` | counter | tokens | Generation tokens produced. `rate_per_second` = decode throughput. |
| `vllm:iteration_tokens_total` | histogram | tokens | Tokens processed per engine step. Indicates batch efficiency. |
| `vllm:request_success` | counter | - | Successfully completed requests. |

### Request-Level Latency Breakdown

These histograms show where time is spent for each request:

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `vllm:e2e_request_latency_seconds` | histogram | seconds | **Total request latency** inside vLLM (queue + inference). |
| `vllm:request_queue_time_seconds` | histogram | seconds | Time spent in **WAITING** phase (queued before execution). |
| `vllm:request_prefill_time_seconds` | histogram | seconds | Time spent in **PREFILL** phase (processing input tokens). |
| `vllm:request_decode_time_seconds` | histogram | seconds | Time spent in **DECODE** phase (generating output tokens). |
| `vllm:request_inference_time_seconds` | histogram | seconds | Time spent in **RUNNING** phase (prefill + decode). |

### Token-Level Latency

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `vllm:time_to_first_token_seconds` | histogram | seconds | **TTFT** - time from request start to first output token. |
| `vllm:inter_token_latency_seconds` | histogram | seconds | **ITL** - time between consecutive output tokens. |
| `vllm:time_per_output_token_seconds` | histogram | seconds | *(Deprecated)* Use `vllm:inter_token_latency_seconds` instead. |
| `vllm:request_time_per_output_token_seconds` | histogram | seconds | Average time per token for each request (total_time / num_tokens). |

### Request Parameters

These histograms show the distribution of request parameters:

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `vllm:request_prompt_tokens` | histogram | tokens | Input token count per request. Same as `dynamo_frontend_input_sequence_tokens`. |
| `vllm:request_generation_tokens` | histogram | tokens | Output token count per request. Same as `dynamo_frontend_output_sequence_tokens`. |
| `vllm:request_max_num_generation_tokens` | histogram | tokens | Maximum tokens requested per request (`max_tokens` parameter). |
| `vllm:request_params_max_tokens` | histogram | tokens | Distribution of `max_tokens` API parameter. |
| `vllm:request_params_n` | histogram | - | Distribution of `n` parameter (number of completions per request). |

### Configuration Info

| Metric | Type | Description |
|--------|------|-------------|
| `vllm:cache_config_info` | gauge (info) | Static cache configuration. Labels contain: `block_size`, `cache_dtype`, `num_gpu_blocks`, etc. |

---

## Key Metrics for Common Questions

### "What is my throughput?"

| Question | Metric | Field |
|----------|--------|-------|
| Requests per second | `dynamo_frontend_requests` | `rate_per_second` |
| Output tokens per second | `dynamo_frontend_output_tokens` | `rate_per_second` |
| Input tokens per second | `vllm:prompt_tokens` | `rate_per_second` |

### "What is my latency?"

| Question | Metric | Field |
|----------|--------|-------|
| End-to-end p99 latency | `dynamo_frontend_request_duration_seconds` | `p99` |
| Time to first token (p99) | `dynamo_frontend_time_to_first_token_seconds` | `p99` |
| Inter-token latency (p99) | `dynamo_frontend_inter_token_latency_seconds` | `p99` |
| Average request latency | `dynamo_frontend_request_duration_seconds` | `avg` |

### "Am I hitting capacity limits?"

| Question | Metric | Field | Threshold |
|----------|--------|-------|-----------|
| KV cache full? | `vllm:kv_cache_usage_perc` | `avg`, `max` | >0.9 is concerning |
| Requests being preempted? | `vllm:num_preemptions` | `delta` | >0 indicates memory pressure |
| Queue building up? | `vllm:num_requests_waiting` | `avg`, `max` | Growing over time is bad |
| Requests queuing? | `dynamo_frontend_queued_requests` | `avg`, `max` | Non-zero indicates backpressure |

### "What does my workload look like?"

| Question | Metric | Field |
|----------|--------|-------|
| Average prompt length | `dynamo_frontend_input_sequence_tokens` | `avg` |
| Longest prompts (p99) | `dynamo_frontend_input_sequence_tokens` | `p99` |
| Average response length | `dynamo_frontend_output_sequence_tokens` | `avg` |
| Longest responses (p99) | `dynamo_frontend_output_sequence_tokens` | `p99` |

### "Where is time being spent?"

Use the vLLM request latency breakdown:

```
Total latency = Queue time + Prefill time + Decode time

vllm:e2e_request_latency_seconds ≈
    vllm:request_queue_time_seconds +
    vllm:request_prefill_time_seconds +
    vllm:request_decode_time_seconds
```

| Phase | Metric | What it means |
|-------|--------|---------------|
| Queue | `vllm:request_queue_time_seconds` | Waiting for GPU resources |
| Prefill | `vllm:request_prefill_time_seconds` | Processing input tokens |
| Decode | `vllm:request_decode_time_seconds` | Generating output tokens |

---

## Metric Labels

Common labels used across metrics:

| Label | Description | Example Values |
|-------|-------------|----------------|
| `model` | Model identifier | `qwen/qwen3-0.6b`, `Qwen/Qwen3-0.6B` |
| `endpoint` | API endpoint | `chat_completions`, `completions` |
| `request_type` | Request type | `stream`, `unary` |
| `status` | Request outcome | `success`, `error` |

---

## Notes

1. **Dynamo vs vLLM metrics**: Dynamo metrics measure at the HTTP/routing layer, vLLM metrics measure inside the inference engine. Use Dynamo for user-facing SLAs, vLLM for debugging performance.

2. **Counter vs Gauge interpretation**:
   - Counters: Use `delta` for total change, `rate_per_second` for rate
   - Gauges: Use `avg` for typical value, `max` for peak, `p99` for tail behavior

3. **Histogram percentiles**: Histogram percentiles (`p50`, `p90`, `p95`, `p99`) are *estimated* from bucket boundaries. Exact values depend on bucket configuration.

4. **Multiple endpoints**: When scraping multiple vLLM instances, each series includes an `endpoint` field to identify the source.
