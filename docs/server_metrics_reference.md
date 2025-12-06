<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Server Metrics Reference

This document provides semantic meanings for all metrics collected from NVIDIA Dynamo, vLLM, SGLang, and TensorRT-LLM inference servers.

## Table of Contents

- [Dynamo Frontend Metrics](#dynamo-frontend-metrics)
- [Dynamo Component Metrics](#dynamo-component-metrics)
- [vLLM Metrics](#vllm-metrics)
- [SGLang Metrics](#sglang-metrics)
- [TensorRT-LLM Metrics](#tensorrt-llm-metrics)

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

## SGLang Metrics

SGLang is a fast inference engine with RadixAttention for efficient prefix caching. These metrics provide visibility into SGLang's scheduling and execution.

### Throughput & Performance

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `sglang:gen_throughput` | gauge | tokens/s | **Generation throughput** in tokens per second. Real-time throughput indicator. |
| `sglang:cache_hit_rate` | gauge | ratio | Prefix cache hit rate (0.0-1.0). Higher = better prompt reuse via RadixAttention. |
| `sglang:token_usage` | gauge | ratio | Token usage ratio (0.0-1.0). Indicates memory utilization. |
| `sglang:utilization` | gauge | ratio | Overall utilization. -1.0 indicates idle, 0.0+ indicates active. |

### Queue State

| Metric | Type | Description |
|--------|------|-------------|
| `sglang:num_running_reqs` | gauge | Requests currently executing in the batch. |
| `sglang:num_queue_reqs` | gauge | Requests in the waiting queue. High values indicate saturation. |
| `sglang:num_used_tokens` | gauge | Total tokens currently in use across all requests. |
| `sglang:num_running_reqs_offline_batch` | gauge | Low-priority offline batch requests running. |
| `sglang:num_paused_reqs` | gauge | Requests paused by async weight sync. |
| `sglang:num_retracted_reqs` | gauge | Requests that were retracted/preempted. |
| `sglang:num_grammar_queue_reqs` | gauge | Requests waiting for grammar processing. |

### Disaggregated Inference Queues

For disaggregated prefill/decode deployments:

| Metric | Type | Description |
|--------|------|-------------|
| `sglang:num_prefill_prealloc_queue_reqs` | gauge | Requests in prefill preallocation queue. |
| `sglang:num_prefill_inflight_queue_reqs` | gauge | Requests in prefill inflight queue. |
| `sglang:num_decode_prealloc_queue_reqs` | gauge | Requests in decode preallocation queue. |
| `sglang:num_decode_transfer_queue_reqs` | gauge | Requests in decode transfer queue. |

### Request Latency Breakdown

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `sglang:queue_time_seconds` | histogram | seconds | Time spent in **WAITING** queue before execution starts. |
| `sglang:per_stage_req_latency_seconds` | histogram | seconds | Per-stage latency breakdown. Label `stage` identifies the phase (see below). |

**Stage labels for `sglang:per_stage_req_latency_seconds`:**

| Stage | Description |
|-------|-------------|
| `prefill_waiting` | Time waiting before prefill begins |
| `prefill_bootstrap` | Time to bootstrap prefill (scheduling overhead) |
| `prefill_prepare` | Time preparing prefill batch |
| `prefill_forward` | Time executing prefill forward pass |
| `prefill_transfer_kv_cache` | Time transferring KV cache (disaggregated mode) |
| `decode_waiting` | Time waiting before decode begins |
| `decode_bootstrap` | Time to bootstrap decode |
| `decode_prepare` | Time preparing decode batch |
| `decode_transferred` | Total time in transferred/decode phase |

### KV Cache Transfer (Disaggregated)

For disaggregated prefill/decode deployments:

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `sglang:kv_transfer_latency_ms` | gauge | milliseconds | KV cache transfer latency. |
| `sglang:kv_transfer_speed_gb_s` | gauge | GB/s | KV cache transfer throughput. |
| `sglang:kv_transfer_alloc_ms` | gauge | milliseconds | Time waiting for KV cache allocation. |
| `sglang:kv_transfer_bootstrap_ms` | gauge | milliseconds | KV transfer bootstrap time. |
| `sglang:pending_prealloc_token_usage` | gauge | ratio | Token usage for pending preallocated tokens. |

### Speculative Decoding

| Metric | Type | Description |
|--------|------|-------------|
| `sglang:spec_accept_rate` | gauge | Speculative decoding acceptance rate (accepted/draft tokens). Higher = better speculation. |
| `sglang:spec_accept_length` | gauge | Average acceptance length of speculative decoding. |

### System Configuration

| Metric | Type | Description |
|--------|------|-------------|
| `sglang:is_cuda_graph` | gauge | Whether batch is using CUDA graph (1=yes, 0=no). |
| `sglang:engine_startup_time` | gauge | Engine startup time in seconds. |
| `sglang:engine_load_weights_time` | gauge | Time to load model weights in seconds. |
| `sglang:mamba_usage` | gauge | Token usage for Mamba layers (hybrid models). |
| `sglang:swa_token_usage` | gauge | Token usage for sliding window attention layers. |

---

## TensorRT-LLM Metrics

TensorRT-LLM (trtllm) is NVIDIA's high-performance inference engine. These metrics track request processing within the TensorRT-LLM backend.

### Request Latency

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `trtllm:e2e_request_latency_seconds` | histogram | seconds | **End-to-end request latency** from submission to completion. Use `p99` for tail latency. |
| `trtllm:request_queue_time_seconds` | histogram | seconds | Time spent in **WAITING** phase (queued before execution). |
| `trtllm:time_to_first_token_seconds` | histogram | seconds | **TTFT** - time from request start to first output token. |
| `trtllm:time_per_output_token_seconds` | histogram | seconds | Time per output token (inter-token latency). |

### Request Completion

| Metric | Type | Description |
|--------|------|-------------|
| `trtllm:request_success` | counter | Successfully completed requests. Label `finished_reason` indicates completion reason (e.g., `length`, `stop`). |

**Common labels:**

| Label | Description | Example Values |
|-------|-------------|----------------|
| `engine_type` | Engine type | `trtllm` |
| `model_name` | Model identifier | `Qwen/Qwen3-0.6B` |
| `finished_reason` | Why request completed | `length`, `stop`, `error` |

---

## Key Metrics for Common Questions

### "What is my throughput?"

| Question | Metric | Field |
|----------|--------|-------|
| Requests per second | `dynamo_frontend_requests` | `rate_per_second` |
| Output tokens per second | `dynamo_frontend_output_tokens` | `rate_per_second` |
| Input tokens per second | `vllm:prompt_tokens` | `rate_per_second` |
| Generation throughput (SGLang) | `sglang:gen_throughput` | `avg` |

### "What is my latency?"

| Question | Metric | Field |
|----------|--------|-------|
| End-to-end p99 latency | `dynamo_frontend_request_duration_seconds` | `p99` |
| Time to first token (p99) | `dynamo_frontend_time_to_first_token_seconds` | `p99` |
| Inter-token latency (p99) | `dynamo_frontend_inter_token_latency_seconds` | `p99` |
| Average request latency | `dynamo_frontend_request_duration_seconds` | `avg` |
| TTFT (TensorRT-LLM) | `trtllm:time_to_first_token_seconds` | `p99` |
| E2E latency (TensorRT-LLM) | `trtllm:e2e_request_latency_seconds` | `p99` |
| Queue time (SGLang) | `sglang:queue_time_seconds` | `p99` |

### "Am I hitting capacity limits?"

| Question | Metric | Field | Threshold |
|----------|--------|-------|-----------|
| KV cache full? | `vllm:kv_cache_usage_perc` | `avg`, `max` | >0.9 is concerning |
| Requests being preempted? | `vllm:num_preemptions` | `delta` | >0 indicates memory pressure |
| Queue building up? | `vllm:num_requests_waiting` | `avg`, `max` | Growing over time is bad |
| Requests queuing? | `dynamo_frontend_queued_requests` | `avg`, `max` | Non-zero indicates backpressure |
| Token usage (SGLang) | `sglang:token_usage` | `avg`, `max` | >0.9 is concerning |
| Queue depth (SGLang) | `sglang:num_queue_reqs` | `avg`, `max` | Growing over time is bad |
| Queue time (TensorRT-LLM) | `trtllm:request_queue_time_seconds` | `avg` | High values indicate saturation |

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

**SGLang latency breakdown** (via `sglang:per_stage_req_latency_seconds` with `stage` label):

| Phase | Stage Label | What it means |
|-------|-------------|---------------|
| Queue | `prefill_waiting` | Waiting before prefill |
| Prefill | `prefill_forward` | Prefill forward pass execution |
| KV Transfer | `prefill_transfer_kv_cache` | KV cache transfer (disaggregated) |
| Decode | `decode_transferred` | Decode phase execution |

**TensorRT-LLM latency breakdown:**

| Phase | Metric | What it means |
|-------|--------|---------------|
| Queue | `trtllm:request_queue_time_seconds` | Waiting for GPU resources |
| TTFT | `trtllm:time_to_first_token_seconds` | Time to first output token |
| Total | `trtllm:e2e_request_latency_seconds` | Complete request duration |

---

## Metric Labels

Common labels used across metrics:

| Label | Description | Example Values |
|-------|-------------|----------------|
| `model` | Model identifier | `qwen/qwen3-0.6b`, `Qwen/Qwen3-0.6B` |
| `endpoint` | API endpoint | `chat_completions`, `completions` |
| `request_type` | Request type | `stream`, `unary` |
| `status` | Request outcome | `success`, `error` |

**SGLang-specific labels:**

| Label | Description | Example Values |
|-------|-------------|----------------|
| `engine_type` | Engine type | `unified` |
| `model_name` | Model identifier | `Qwen/Qwen3-0.6B` |
| `tp_rank` | Tensor parallel rank | `0`, `1`, ... |
| `pp_rank` | Pipeline parallel rank | `0`, `1`, ... |
| `stage` | Processing stage | `prefill_forward`, `decode_transferred` |
| `pid` | Process ID | `670` |

**TensorRT-LLM-specific labels:**

| Label | Description | Example Values |
|-------|-------------|----------------|
| `engine_type` | Engine type | `trtllm` |
| `model_name` | Model identifier | `Qwen/Qwen3-0.6B` |
| `finished_reason` | Completion reason | `length`, `stop`, `error` |

---

## Notes

1. **Dynamo vs backend metrics**: Dynamo metrics measure at the HTTP/routing layer, while vLLM/SGLang/TensorRT-LLM metrics measure inside the inference engine. Use Dynamo for user-facing SLAs, backend metrics for debugging performance.

2. **Counter vs Gauge interpretation**:
   - Counters: Use `delta` for total change, `rate_per_second` for rate
   - Gauges: Use `avg` for typical value, `max` for peak, `p99` for tail behavior

3. **Histogram percentiles**: Histogram percentiles (`p50`, `p90`, `p95`, `p99`) are *estimated* from bucket boundaries. Exact values depend on bucket configuration.

4. **Multiple endpoints**: When scraping multiple instances, each series includes an `endpoint` field to identify the source.

5. **Backend-specific metrics**:
   - **vLLM**: Comprehensive metrics for cache, queue state, and request phases
   - **SGLang**: RadixAttention-based caching, disaggregated inference support, speculative decoding stats
   - **TensorRT-LLM**: Focused on latency breakdown (queue, TTFT, e2e) and request completion
