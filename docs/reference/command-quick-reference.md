<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Command Quick Reference

**Quick visual guide showing exactly what you'll see when running AIPerf commands.**

---

## Basic Commands

### Check Version

```bash
$ aiperf --version
```
```
0.1.0
```

### Get Help

```bash
$ aiperf --help
```
```
Usage: aiperf COMMAND

NVIDIA AIPerf

╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ profile    Run the Profile subcommand.                                       │
│ --help -h  Display this message and exit.                                    │
│ --version  Display application version.                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## Profile Command

### Simple Benchmark (Concurrency Mode)

```bash
$ aiperf profile \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --url http://localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --concurrency 8 \
    --request-count 100 \
    --synthetic-input-tokens-mean 512 \
    --output-tokens-mean 256
```

**Expected Output (Initialization):**
```
23:07:28.809795 INFO     Starting AIPerf System
23:07:31.580751 INFO     Registered Dataset Manager (id: 'dataset_manager_43448f70')
23:07:31.586297 INFO     Registered Worker Manager (id: 'worker_manager_2662dd2f')
23:07:31.595391 INFO     Registered Timing Manager (id: 'timing_manager_106cf7fd')
23:07:31.615191 INFO     Registered Records Manager (id: 'records_manager_c61d5cdd')
23:07:31.707697 INFO     Registered Worker (id: 'worker_5cabb891')
23:07:31.710912 INFO     Registered Record Processor (id: 'record_processor_50b14de3')
23:07:31.987169 INFO     AIPerf System is CONFIGURING
23:07:32.594891 INFO     AIPerf System is CONFIGURED
23:07:32.597896 INFO     AIPerf System is PROFILING
```

**Expected Output (Progress with Simple UI):**
```
Profiling: 100/100 |████████████████████████| 100% [03:24<00:00]
Processing Records: 100/100 |███████████████| 100% [00:01<00:00]
```

**Expected Output (Completion):**
```
23:11:15.234567 INFO     Benchmark completed successfully
23:11:15.345678 INFO     Results saved to: artifacts/llama-3.1-8b-chat-concurrency8/
```

**Generated Artifacts:**
```
artifacts/llama-3.1-8b-chat-concurrency8/
├── profile_export_aiperf.json    # Aggregated metrics
├── profile_export.jsonl           # Per-request details
└── logs/                          # Service logs
```

---

### Request Rate Benchmark

```bash
$ aiperf profile \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --url http://localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --request-rate 10.0 \
    --benchmark-duration 300 \
    --synthetic-input-tokens-mean 512 \
    --output-tokens-mean 256
```

**Expected Output (Configuration):**
```
23:15:42.123456 INFO     Starting AIPerf System
...
23:15:45.456789 INFO     Using Request_Rate strategy
23:15:45.567890 INFO     Credit issuing strategy RequestRateStrategy initialized
                         with 1 phase(s): [CreditPhaseConfig(type=CreditPhase.PROFILING,
                         total_expected_requests=None, expected_duration_sec=300)]
23:15:46.678901 INFO     AIPerf System is PROFILING
```

**Expected Output (Progress):**
```
Profiling: ∞ |█████████████           | [03:00 / 05:00]
Processing Records: 1500/1500 |████████| 100%
```

---

### Benchmark with GPU Telemetry

```bash
$ aiperf profile \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --url http://localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --concurrency 4 \
    --request-count 50 \
    --synthetic-input-tokens-mean 512 \
    --output-tokens-mean 256 \
    --gpu-telemetry localhost:9400
```

**Expected Output (GPU Summary at End):**
```
                          NVIDIA AIPerf | GPU Telemetry Summary
                               1/1 DCGM endpoints reachable
                                    • localhost:9400 ✔

                      localhost:9400 | GPU 0 | NVIDIA H100 80GB HBM3
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┓
┃                       Metric ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p50 ┃   std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━┩
│          GPU Power Usage (W) │   348.69 │   120.57 │   386.02 │   386.02 │   386.02 │   378.34 │ 85.97 │
│      Energy Consumption (MJ) │     0.24 │     0.23 │     0.25 │     0.25 │     0.25 │     0.23 │  0.01 │
│          GPU Utilization (%) │    45.82 │     0.00 │    66.00 │    66.00 │    66.00 │    66.00 │ 24.52 │
│  Memory Copy Utilization (%) │    21.10 │     0.00 │    29.00 │    29.00 │    29.00 │    29.00 │ 10.11 │
│         GPU Memory Used (GB) │    92.70 │    92.70 │    92.70 │    92.70 │    92.70 │    92.70 │  0.00 │
│         GPU Memory Free (GB) │     9.39 │     9.39 │     9.39 │     9.39 │     9.39 │     9.39 │  0.00 │
│     SM Clock Frequency (MHz) │ 1,980.00 │ 1,980.00 │ 1,980.00 │ 1,980.00 │ 1,980.00 │ 1,980.00 │  0.00 │
│ Memory Clock Frequency (MHz) │ 2,619.00 │ 2,619.00 │ 2,619.00 │ 2,619.00 │ 2,619.00 │ 2,619.00 │  0.00 │
│      Memory Temperature (°C) │    45.99 │    41.00 │    48.00 │    48.00 │    48.00 │    46.00 │  2.08 │
│         GPU Temperature (°C) │    38.87 │    33.00 │    41.00 │    41.00 │    41.00 │    39.00 │  2.38 │
│           XID Errors (count) │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │  0.00 │
└──────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴───────┘
```

**Generated Artifacts:**
```
artifacts/llama-3.1-8b-chat-concurrency4/
├── profile_export_aiperf.json
├── profile_export.jsonl
├── gpu_telemetry_export.json
├── gpu_telemetry_export.jsonl
└── logs/
```

---

## Common Errors and Solutions

### Error 1: Invalid Endpoint Type

**Command:**
```bash
$ aiperf profile --model test-model --endpoint-type openai-chat-completions
```

**Error:**
```
╭─ Error ──────────────────────────────────────────────────────────────────────╮
│ 1 validation error for UserConfig                                            │
│ endpoint.type                                                                │
│   Input should be 'chat', 'completions', 'embeddings' or 'responses'         │
│ [type=enum, input_value='openai-chat-completions', input_type=str]           │
│     For further information visit https://errors.pydantic.dev/2.12/v/enum    │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**Solution:** Use `--endpoint-type chat` (not `openai-chat-completions`)

---

### Error 2: Model Not Found

**Command:**
```bash
$ aiperf profile --model fake-model --url http://localhost:8000
```

**Error:**
```
23:07:32.157372 ERROR    Failed to handle command profile_configure:
                         InitializationError: fake-model is not a local folder
                         and is not a valid model identifier listed on
                         'https://huggingface.co/models'

                         RepositoryNotFoundError: 404 Client Error.
                         Repository Not Found for url:
                         https://huggingface.co/fake-model/resolve/main/tokenizer_config.json.
```

**Solution:** Use a valid HuggingFace model name or provide `--tokenizer <path>` with a local tokenizer

---

### Error 3: Server Not Reachable

**Command:**
```bash
$ aiperf profile --model meta-llama/Llama-3.1-8B-Instruct --url http://localhost:9999
```

**Error:**
```
23:10:45.123456 ERROR    Worker worker_abc123: Connection error to http://localhost:9999
                         ConnectionError: Failed to establish connection
```

**Solution:** Ensure the inference server is running and accessible at the specified URL

---

### Error 4: All Requests Failed (Timeout/Connection)

**Command:**
```bash
$ aiperf profile \
    --model test-model \
    --url http://localhost:8000 \
    --endpoint-type chat \
    --request-count 5 \
    --concurrency 1
```

**Output (When Server Not Running):**
```
23:15:02.804469 NOTICE   Phase completed: type=CreditPhase.PROFILING
                         sent=5 completed=5
23:15:02.804968 INFO     Processed 0 valid requests and 5 errors (5 total).
23:15:02.805739 INFO     Processing records results...
23:15:03.324454 INFO     Exporting all records
23:15:03.324454 INFO     Exporting console data


      NVIDIA AIPerf | Error Summary
┏━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
┃ Code ┃         Type ┃ Message ┃ Count ┃
┡━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
│  N/A │ TimeoutError │         │     5 │
└──────┴──────────────┴─────────┴───────┘


            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┓
┃ Metric ┃ avg ┃ min ┃ max ┃ p99 ┃ p90 ┃ p75 ┃ std ┃
┡━━━━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━┩
└────────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

CLI Command: aiperf profile --model "test-model" --url "http://localhost:8000"
--endpoint-type "chat" --synthetic-input-tokens-mean 100 --output-tokens-mean 50
--request-count 5 --concurrency 1 --ui-type None

CSV Export: artifacts/test-model-openai-chat-concurrency1/profile_export_aiperf.csv
JSON Export: artifacts/test-model-openai-chat-concurrency1/profile_export_aiperf.json
```

**What Happened:**
- Benchmark initialized successfully
- All 5 requests were sent but timed out (no server responding)
- Error summary table shows 5 TimeoutError
- Metrics table is empty (no successful requests to compute metrics from)
- Export files are still created (they contain error details)

**Solution:** Start the inference server before running the benchmark:
```bash
# Example: Start vLLM server
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# Then run AIPerf
aiperf profile --model meta-llama/Llama-3.1-8B-Instruct --url http://localhost:8000
```

---

## UI Types

### Dashboard UI (Default)

```bash
$ aiperf profile --model MODEL --url URL --ui-type dashboard
```

**Features:**
- Real-time metrics dashboard with multiple tabs
- Live GPU telemetry visualization
- Request throughput graphs
- Token latency histograms
- Interactive TUI interface

**Preview:**
```
┌─ AIPerf Dashboard ────────────────────────────────────────────────────────────┐
│ [Metrics] [GPU Telemetry] [Logs]                                              │
│                                                                                │
│ Request Latency (ms)                     TTFT (ms)                            │
│ ┌──────────────────────┐                ┌──────────────────────┐             │
│ │  Avg:  3,652.14      │                │  Avg:    30.81       │             │
│ │  P50:  3,644.08      │                │  P50:    30.67       │             │
│ │  P99:  3,909.33      │                │  P99:    34.34       │             │
│ └──────────────────────┘                └──────────────────────┘             │
│                                                                                │
│ Progress: 64/64 requests completed                                            │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

### Simple UI

```bash
$ aiperf profile --model MODEL --url URL --ui-type simple
```

**Output:**
```
Profiling: 64/64 |████████████████████████| 100% [03:24<00:00]
Processing Records: 64/64 |█████████████████| 100% [00:01<00:00]
```

---

### No UI (Logs Only)

```bash
$ aiperf profile --model MODEL --url URL --ui-type none
```

**Output:**
```
23:07:28.809795 INFO     Starting AIPerf System
23:07:31.580751 INFO     Registered Dataset Manager (id: 'dataset_manager_43448f70')
...
23:07:32.597896 INFO     AIPerf System is PROFILING
23:11:15.234567 INFO     Benchmark completed successfully
```

---

## Output Artifacts Summary

### Default Artifacts

Every successful run creates:

```
artifacts/<model>-<endpoint>-<config>/
├── profile_export_aiperf.json     # Aggregated statistics (always created)
├── profile_export.jsonl            # Per-request records (always created)
└── logs/                           # Service logs (always created)
    ├── dataset_manager_*.log
    ├── timing_manager_*.log
    ├── worker_manager_*.log
    ├── worker_*.log
    ├── record_processor_*.log
    ├── records_manager_*.log
    └── system_controller.log
```

### With GPU Telemetry (`--gpu-telemetry <endpoint>`)

```
artifacts/<model>-<endpoint>-<config>/
├── profile_export_aiperf.json
├── profile_export.jsonl
├── gpu_telemetry_export.json       # Added with GPU telemetry
├── gpu_telemetry_export.jsonl      # Added with GPU telemetry
└── logs/
```

### With Server Metrics (`--server-metrics <endpoint>`)

```
artifacts/<model>-<endpoint>-<config>/
├── profile_export_aiperf.json
├── profile_export.jsonl
├── server_metrics_export.json      # Added with server metrics
├── server_metrics_export.jsonl     # Added with server metrics
└── logs/
```

### With CSV Export (`--export-level records`)

```
artifacts/<model>-<endpoint>-<config>/
├── profile_export_aiperf.json
├── profile_export_aiperf.csv       # Added with records export
├── profile_export.jsonl
└── logs/
```

### With Raw Data (`--export-level raw`)

```
artifacts/<model>-<endpoint>-<config>/
├── profile_export_aiperf.json
├── profile_export.jsonl
├── _raw.jsonl                      # Added with raw export
└── logs/
```

---

## Quick Metric Reference

Key metrics you'll find in `profile_export_aiperf.json`:

| Metric | Unit | Description |
|--------|------|-------------|
| `request_throughput` | requests/sec | Requests completed per second |
| `request_latency` | ms | End-to-end request latency |
| `time_to_first_token` (TTFT) | ms | Time until first token arrives |
| `inter_token_latency` (ITL) | ms | Average time between tokens |
| `output_token_throughput` | tokens/sec | Output tokens generated per second |
| `prefill_throughput` | tokens/sec | Input tokens processed per second |
| `output_token_throughput_per_user` | tokens/sec/user | Per-request token generation speed |

All metrics include: `avg`, `min`, `max`, `p50`, `p90`, `p99`, `std`

---

## Next Steps

- **Detailed Schemas:** See [command-outputs-and-artifacts.md](./command-outputs-and-artifacts.md)
- **All Metrics:** See [../metrics_reference.md](../metrics_reference.md)
- **CLI Options:** See [../cli_options.md](../cli_options.md)
- **Tutorials:** See [../tutorials/](../tutorials/)

---

**For complete examples with real JSON/JSONL data, see [command-outputs-and-artifacts.md](./command-outputs-and-artifacts.md)**
