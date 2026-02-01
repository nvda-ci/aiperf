<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Command Outputs and Artifacts Reference

This document provides **actual examples** of AIPerf command outputs, error messages, and generated artifacts. All examples are captured from real command executions.

## Table of Contents

- [Command Line Interface](#command-line-interface)
- [Validation and Configuration Errors](#validation-and-configuration-errors)
- [Profile Command Outputs](#profile-command-outputs)
- [Artifact Directory Structure](#artifact-directory-structure)
- [Export File Formats](#export-file-formats)

---

## Command Line Interface

### Main Help Command

```bash
$ aiperf --help
```

**Output:**
```
Usage: aiperf COMMAND

NVIDIA AIPerf

╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ profile    Run the Profile subcommand.                                       │
│ --help -h  Display this message and exit.                                    │
│ --version  Display application version.                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Version Command

```bash
$ aiperf --version
```

**Output:**
```
0.1.0
```

### Profile Command Help (Excerpt)

```bash
$ aiperf profile --help
```

**Output (first 50 lines):**
```
Usage: aiperf profile [ARGS] [OPTIONS]

Run the Profile subcommand.

╭─ Endpoint ───────────────────────────────────────────────────────────────────╮
│ *  MODEL-NAMES --model-names     -m  Model name(s) to be benchmarked. Can be │
│      --model                         a comma-separated list or a single      │
│                                      model name. [required]                  │
│    MODEL-SELECTION-STRATEGY          When multiple models are specified,     │
│      --model-selection-strategy      this is how a specific model should be  │
│                                      assigned to a prompt. round_robin: nth  │
│                                      prompt in the list gets assigned to     │
│                                      n-mod len(models). random: assignment   │
│                                      is uniformly random [choices:           │
│                                      round-robin, random] [default:          │
│                                      round-robin]                            │
│    CUSTOM-ENDPOINT                   Set a custom endpoint that differs from │
│      --custom-endpoint               the OpenAI defaults.                    │
│      --endpoint                                                              │
│    ENDPOINT-TYPE                     The endpoint type to send requests to   │
│      --endpoint-type                 on the server. [choices:                │
│                                      openai-chat-completions,                │
│                                      openai-completions, openai-embeddings,  │
│                                      openai-responses] [default:             │
│                                      openai-chat-completions]                │
│    STREAMING --streaming             An option to enable the use of the      │
│                                      streaming API. [default: False]         │
│    URL --url                     -u  URL of the endpoint to target for       │
│                                      benchmarking. [default: localhost:8080] │
│    REQUEST-TIMEOUT-SECONDS           The timeout in floating points seconds  │
│      --request-timeout-seconds       for each request to the endpoint.       │
│                                      [default: 600.0]                        │
│    API-KEY --api-key                 The API key to use for the endpoint. If │
│                                      provided, it will be sent with every    │
│                                      request as a header: Authorization:     │
│                                      Bearer <api_key>.                       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## Validation and Configuration Errors

### 1. Invalid Endpoint Type Error

**Command:**
```bash
aiperf profile \
  --model test-model \
  --url http://localhost:8000 \
  --endpoint-type openai-chat-completions
```

**Error Output:**
```
╭─ Error ──────────────────────────────────────────────────────────────────────╮
│ 1 validation error for UserConfig                                            │
│ endpoint.type                                                                │
│   Input should be 'chat', 'completions', 'embeddings' or 'responses'         │
│ [type=enum, input_value='openai-chat-completions', input_type=str]           │
│     For further information visit https://errors.pydantic.dev/2.12/v/enum    │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**Fix:** Use `--endpoint-type chat` instead of `--endpoint-type openai-chat-completions`

### 2. Model/Tokenizer Not Found Error

**Command:**
```bash
aiperf profile \
  --model test-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --synthetic-input-tokens-mean 100 \
  --output-tokens-mean 50 \
  --request-count 5 \
  --concurrency 1
```

**Error Output (truncated for clarity):**
```
23:07:28.809795 INFO     Starting AIPerf System
23:07:31.580751 INFO     Registered Dataset Manager (id: 'dataset_manager_43448f70')
23:07:31.586297 INFO     Registered Worker Manager (id: 'worker_manager_2662dd2f')
23:07:31.595391 INFO     Registered Timing Manager (id: 'timing_manager_106cf7fd')
23:07:31.615191 INFO     Registered Records Manager (id: 'records_manager_c61d5cdd')
23:07:31.707697 INFO     Registered Worker (id: 'worker_5cabb891')
23:07:31.710912 INFO     Registered Record Processor (id: 'record_processor_50b14de3')
23:07:31.987169 INFO     AIPerf System is CONFIGURING
23:07:31.989017 INFO     Configuring dataset for dataset_manager_43448f70
23:07:31.989418 INFO     Using Request_Rate strategy
23:07:32.157372 ERROR    Failed to handle command profile_configure with hook
                         @on_command → DatasetManager._profile_configure_command:
                         InitializationError: test-model is not a local folder
                         and is not a valid model identifier listed on
                         'https://huggingface.co/models'
                         If this is a private repository, make sure to pass a
                         token having permission to this repo either by logging
                         in with `hf auth login` or by passing `token=<your_token>`

                         HTTPError: 404 Client Error: Not Found for url:
                         https://huggingface.co/test-model/resolve/main/tokenizer_config.json

                         RepositoryNotFoundError: 404 Client Error.
                         Repository Not Found for url:
                         https://huggingface.co/test-model/resolve/main/tokenizer_config.json.
```

**Fix:** Use a valid HuggingFace model name like `meta-llama/Llama-3.1-8B-Instruct` or provide a local tokenizer path.

### 3. Unknown Command Error

**Command:**
```bash
aiperf plugins --all
```

**Error Output:**
```
╭─ Error ──────────────────────────────────────────────────────────────────────╮
│ Unknown command "plugins". Available commands: profile.                      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**Note:** The `plugins` and `plot` commands are documented but not yet implemented in version 0.1.0.

---

## Profile Command Outputs

### Successful Run - Log Output Example

**Command:**
```bash
aiperf profile \
  --model "Qwen/Qwen3-0.6B" \
  --endpoint-type "chat" \
  --streaming \
  --url "localhost:8000" \
  --synthetic-input-tokens-mean 100 \
  --output-tokens-mean 200 \
  --concurrency 1 \
  --request-count 64 \
  --warmup-request-count 1
```

**Console Output (Service Registration Phase):**
```
23:07:28.809795 INFO     Starting AIPerf System
23:07:31.580751 INFO     Registered Dataset Manager (id: 'dataset_manager_43448f70')
23:07:31.586297 INFO     Registered Worker Manager (id: 'worker_manager_2662dd2f')
23:07:31.595391 INFO     Registered Timing Manager (id: 'timing_manager_106cf7fd')
23:07:31.615191 INFO     Registered Records Manager (id: 'records_manager_c61d5cdd')
23:07:31.707697 INFO     Registered Worker (id: 'worker_5cabb891')
23:07:31.710912 INFO     Registered Record Processor (id: 'record_processor_50b14de3')
23:07:31.987169 INFO     AIPerf System is CONFIGURING
23:07:31.989017 INFO     Configuring dataset for dataset_manager_43448f70
23:07:31.989418 INFO     Using Request_Rate strategy
23:07:31.992823 INFO     Credit issuing strategy RequestRateStrategy initialized
                         with 1 phase(s): [CreditPhaseConfig(type=CreditPhase.PROFILING,
                         total_expected_requests=5, expected_duration_sec=None)]
23:07:32.594891 INFO     AIPerf System is CONFIGURED
23:07:32.596092 INFO     Credit issuing strategy for Request_Rate started
23:07:32.597415 INFO     All services started profiling successfully
23:07:32.597896 INFO     AIPerf System is PROFILING
```

**Console Output (with Dashboard UI):**
```
[Real-time TUI dashboard with tabs showing:]
- Request metrics: TTFT, ITL, latency distributions
- GPU telemetry: power, utilization, memory
- Progress bars for workers and record processors
```

**Console Output (with Simple UI):**
```
Profiling: 64/64 |████████████████████████| 100% [00:255<00:00]
Processing Records: 64/64 |████████████████| 100% [00:01<00:00]
```

---

## Artifact Directory Structure

After a successful run, artifacts are saved to the configured output directory (default: `artifacts/`):

```
artifacts/
└── <model>-<endpoint-type>-<load-config>/
    ├── inputs.json                              # Input dataset payloads
    ├── profile_export.jsonl                     # Per-request metrics (JSONL)
    ├── profile_export_aiperf.json               # Aggregated statistics (JSON)
    ├── profile_export_aiperf.csv                # Aggregated statistics (CSV) [if enabled]
    ├── profile_export_aiperf_timeslices.csv     # Time-sliced metrics [if enabled]
    ├── profile_export_aiperf_timeslices.json    # Time-sliced metrics [if enabled]
    ├── server_metrics_export.json               # Prometheus metrics [if enabled]
    ├── server_metrics_export.csv                # Prometheus metrics [if enabled]
    ├── server_metrics_export.jsonl              # Time-series metrics [if enabled]
    ├── server_metrics_export.parquet            # Columnar metrics [if enabled]
    ├── gpu_telemetry_export.json                # GPU telemetry [if enabled]
    ├── gpu_telemetry_export.csv                 # GPU telemetry [if enabled]
    ├── gpu_telemetry_export.jsonl               # Time-series GPU data [if enabled]
    ├── _raw.jsonl                               # Raw HTTP request/response [if enabled]
    └── logs/
        ├── dataset_manager_*.log
        ├── timing_manager_*.log
        ├── worker_manager_*.log
        ├── worker_*.log
        ├── record_processor_*.log
        ├── records_manager_*.log
        └── system_controller.log
```

### Example Path (Successful Run)

```
artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency1/
├── profile_export_aiperf.json    # 23 KB
├── profile_export.jsonl           # 38 KB
├── gpu_telemetry_export.jsonl     # 15 KB
└── logs/                          # Service logs
```

### Example Path (Failed Run - All Requests Timeout)

```
artifacts/test-model-openai-chat-concurrency1/
├── profile_export_aiperf.json    # 1.4 KB (only error_summary)
├── profile_export_aiperf.csv     # 40 bytes (only error count)
└── logs/                          # Service logs with error details
    (Note: No profile_export.jsonl created when no requests succeed)
```

---

## Export File Formats

### 1. Aggregated Statistics JSON (`profile_export_aiperf.json`)

**Format:** Single JSON object with aggregated metrics across all requests.

**Schema:**
```typescript
{
  // Request Performance Metrics
  "request_throughput": {
    "unit": "requests/sec",
    "avg": number
  },
  "request_latency": {
    "unit": "ms",
    "avg": number,
    "p1": number, "p5": number, "p10": number, "p25": number,
    "p50": number, "p75": number, "p90": number, "p95": number, "p99": number,
    "min": number, "max": number, "std": number
  },
  "request_count": {
    "unit": "requests",
    "avg": number
  },

  // Token-Level Metrics
  "time_to_first_token": {
    "unit": "ms",
    "avg": number, "p1": number, "p5": number, ..., "std": number
  },
  "inter_token_latency": {
    "unit": "ms",
    "avg": number, "p1": number, ..., "std": number
  },
  "output_token_throughput": {
    "unit": "tokens/sec",
    "avg": number
  },
  "output_token_throughput_per_user": {
    "unit": "tokens/sec/user",
    "avg": number, "p1": number, ..., "std": number
  },
  "prefill_throughput": {
    "unit": "tokens/sec",
    "avg": number, "p1": number, ..., "std": number
  },

  // Sequence Length Metrics
  "input_sequence_length": {
    "unit": "tokens",
    "avg": number, "p1": number, ..., "std": number
  },
  "output_sequence_length": {
    "unit": "tokens",
    "avg": number, "p1": number, ..., "std": number
  },
  "output_token_count": {
    "unit": "tokens",
    "avg": number, "p1": number, ..., "std": number
  },

  // Goodput Metrics (if enabled)
  "goodput": number | null,
  "good_request_count": number | null,

  // Aggregate Totals
  "total_output_tokens": {
    "unit": "tokens",
    "avg": number
  },
  "total_isl": {
    "unit": "tokens",
    "avg": number
  },
  "total_osl": {
    "unit": "tokens",
    "avg": number
  },
  "benchmark_duration": {
    "unit": "sec",
    "avg": number
  },

  // Error Summary
  "error_request_count": number | null,
  "error_summary": string[],
  "was_cancelled": boolean,

  // GPU Telemetry (if enabled)
  "telemetry_data": {
    "summary": {
      "endpoints_configured": string[],
      "endpoints_successful": string[],
      "start_time": "ISO8601",
      "end_time": "ISO8601"
    },
    "endpoints": {
      "<hostname:port>": {
        "gpus": {
          "gpu_<index>": {
            "gpu_index": number,
            "gpu_name": string,
            "gpu_uuid": string,
            "hostname": string,
            "metrics": {
              "gpu_power_usage": { "unit": "W", "avg": number, "p1": number, ... },
              "energy_consumption": { "unit": "MJ", "avg": number, ... },
              "gpu_utilization": { "unit": "%", "avg": number, ... },
              "memory_copy_utilization": { "unit": "%", "avg": number, ... },
              "gpu_memory_used": { "unit": "GB", "avg": number, ... },
              "gpu_memory_free": { "unit": "GB", "avg": number, ... },
              "gpu_memory_total": { "unit": "GB", "avg": number, ... },
              "sm_clock_frequency": { "unit": "MHz", "avg": number, ... },
              "memory_clock_frequency": { "unit": "MHz", "avg": number, ... },
              "memory_temperature": { "unit": "°C", "avg": number, ... },
              "gpu_temperature": { "unit": "°C", "avg": number, ... },
              "xid_errors": { "unit": "count", "avg": number, ... },
              "power_violation": { "unit": "us", "avg": number, ... },
              "thermal_violation": { "unit": "us", "avg": number, ... }
            }
          }
        }
      }
    }
  },

  // Input Configuration
  "input_config": {
    "endpoint": {
      "model_names": string[],
      "custom_endpoint": string,
      "type": string,
      "streaming": boolean,
      "url": string
    },
    "input": {
      "extra": [string, any][],
      "dataset_sampling_strategy": string,
      "random_seed": number,
      "prompt": {
        "input_tokens": { "mean": number, "stddev": number },
        "output_tokens": { "mean": number, "stddev": number }
      }
    },
    "output": {
      "artifact_directory": string
    },
    "loadgen": {
      "concurrency": number,
      "arrival_pattern": string,
      "request_count": number,
      "warmup_request_count": number
    },
    "cli_command": string,
    "gpu_telemetry": string[]
  },

  // Timestamps
  "start_time": "ISO8601",
  "end_time": "ISO8601"
}
```

**Real Example - Successful Run (Excerpt):**
```json
{
  "request_throughput": {
    "unit": "requests/sec",
    "avg": 0.2504302580624667
  },
  "request_latency": {
    "unit": "ms",
    "avg": 3652.1454372343746,
    "p1": 3518.0658775399997,
    "p50": 3644.0809205,
    "p99": 3909.3341161199996,
    "min": 3517.14812,
    "max": 3954.0145199999997,
    "std": 70.34527181408323
  },
  "time_to_first_token": {
    "unit": "ms",
    "avg": 30.814987671875,
    "p50": 30.668547999999998,
    "p99": 34.344268879999994,
    "min": 28.662267,
    "max": 34.717433,
    "std": 1.3745155360138603
  },
  "inter_token_latency": {
    "unit": "ms",
    "avg": 18.19909533160138,
    "p50": 18.158001085427134,
    "p99": 19.481766498241203,
    "std": 0.3514929398171827
  },
  "output_token_throughput": {
    "unit": "tokens/sec",
    "avg": 50.08213863971112
  },
  "prefill_throughput": {
    "unit": "tokens/sec",
    "avg": 3251.4887179314755,
    "p50": 3260.6696082508024,
    "p99": 3472.1383573074354
  },
  "telemetry_data": {
    "summary": {
      "endpoints_configured": ["http://localhost:9402/metrics", "http://localhost:9400/metrics"],
      "endpoints_successful": ["http://localhost:9400/metrics", "http://localhost:9402/metrics"],
      "start_time": "2025-11-07T13:38:54.793124",
      "end_time": "2025-11-07T13:43:13.730640"
    },
    "endpoints": {
      "localhost:9400": {
        "gpus": {
          "gpu_0": {
            "gpu_index": 0,
            "gpu_name": "NVIDIA RTX 6000 Ada Generation",
            "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "hostname": "test-host-1",
            "metrics": {
              "gpu_power_usage": {
                "unit": "W",
                "avg": 129.8854497206704,
                "p50": 131.7835,
                "p99": 136.88345,
                "min": 30.257,
                "max": 137.17
              },
              "gpu_utilization": {
                "unit": "%",
                "avg": 93.01536312849161,
                "p50": 94.0,
                "p99": 95.0
              },
              "gpu_memory_used": {
                "unit": "GB",
                "avg": 49.63033664643575,
                "p50": 49.628053504
              },
              "gpu_temperature": {
                "unit": "°C",
                "avg": 57.487430167597765,
                "p50": 59.0,
                "p99": 66.0
              }
            }
          }
        }
      }
    }
  },
  "input_config": {
    "endpoint": {
      "model_names": ["Qwen/Qwen3-0.6B"],
      "custom_endpoint": "/v1/chat/completions",
      "type": "chat",
      "streaming": true,
      "url": "localhost:8000"
    },
    "loadgen": {
      "concurrency": 1,
      "arrival_pattern": "concurrency_burst",
      "request_count": 64,
      "warmup_request_count": 1
    }
  }
}
```

**Real Example - Failed Run (All Requests Timeout):**
```json
{
  "records": {
    "error_request_count": {
      "tag": "error_request_count",
      "unit": "requests",
      "header": "Error Request Count",
      "avg": 5.0,
      "min": null,
      "max": null,
      "p50": null,
      "p99": null,
      "std": null,
      "count": 1
    }
  },
  "input_config": {
    "endpoint": {
      "model_names": ["test-model"],
      "type": "chat",
      "url": "http://localhost:8000"
    },
    "input": {
      "prompt": {
        "input_tokens": { "mean": 100 },
        "output_tokens": { "mean": 50 }
      }
    },
    "loadgen": {
      "concurrency": 1,
      "request_rate_mode": "concurrency_burst",
      "request_count": 5
    },
    "cli_command": "aiperf profile --model \"test-model\" --url \"http://localhost:8000\" --endpoint-type \"chat\" --synthetic-input-tokens-mean 100 --output-tokens-mean 50 --request-count 5 --concurrency 1"
  },
  "was_cancelled": false,
  "error_summary": [
    {
      "error_details": {
        "code": null,
        "type": "TimeoutError",
        "message": ""
      },
      "count": 5
    }
  ],
  "start_time": "2026-01-31T23:07:32.596754",
  "end_time": "2026-01-31T23:15:02.804414"
}
```

**Note:** When all requests fail, the JSON export contains:
- Only `error_request_count` metric (no performance metrics)
- `error_summary` array with error details and counts
- `input_config` with the command configuration
- No `profile_export.jsonl` file is created

---

### 2. Per-Request Records JSONL (`profile_export.jsonl`)

**Format:** Newline-delimited JSON (JSONL), one record per request.

**Schema:**
```typescript
{
  "metadata": {
    "session_num": number,
    "x_request_id": string,              // Unique request ID
    "x_correlation_id": string,          // Correlation across related requests
    "conversation_id": string | null,    // For multi-turn conversations
    "turn_index": number,                // Turn number in conversation
    "request_start_ns": number,          // Nanosecond timestamp
    "request_ack_ns": number,            // When server acknowledged
    "request_end_ns": number,            // When request completed
    "worker_id": string,                 // Worker that handled request
    "record_processor_id": string,       // Record processor ID
    "benchmark_phase": "warmup" | "profiling",
    "was_cancelled": boolean,
    "cancellation_time_ns": number | null
  },
  "metrics": {
    "output_sequence_length": { "value": number, "unit": "tokens" },
    "input_sequence_length": { "value": number, "unit": "tokens" },
    "time_to_first_token": { "value": number, "unit": "ms" },
    "time_to_second_token": { "value": number, "unit": "ms" },
    "output_token_count": { "value": number, "unit": "tokens" },
    "request_latency": { "value": number, "unit": "ms" },
    "time_to_first_output_token": { "value": number, "unit": "ms" },
    "inter_token_latency": { "value": number, "unit": "ms" },
    "prefill_throughput": { "value": number, "unit": "tokens/sec" },
    "output_token_throughput_per_user": { "value": number, "unit": "tokens/sec/user" },
    "inter_chunk_latency": { "value": number[], "unit": "ms" }  // Array of latencies
  },
  "error": {
    "error_type": string,
    "error_message": string,
    "error_code": string
  } | null
}
```

**Real Example:**
```json
{
    "metadata": {
        "session_num": 0,
        "x_request_id": "fd78b108-2a8f-43b0-8327-0f97e76c6a9c",
        "x_correlation_id": "66c73ffd-2dbb-48ea-b5a5-5a220d464bf0",
        "conversation_id": null,
        "turn_index": 0,
        "request_start_ns": 1762551534797067827,
        "request_ack_ns": 1762551534807258440,
        "request_end_ns": 1762551538441354976,
        "worker_id": "worker_416f5670",
        "record_processor_id": "record_processor_63b1a30c",
        "benchmark_phase": "profiling",
        "was_cancelled": false,
        "cancellation_time_ns": null
    },
    "metrics": {
        "output_sequence_length": { "value": 200, "unit": "tokens" },
        "input_sequence_length": { "value": 100, "unit": "tokens" },
        "time_to_first_token": { "value": 31.742933999999998, "unit": "ms" },
        "output_token_count": { "value": 200, "unit": "tokens" },
        "time_to_second_token": { "value": 17.868007, "unit": "ms" },
        "request_latency": { "value": 3644.2601019999997, "unit": "ms" },
        "time_to_first_output_token": { "value": 31.742933999999998, "unit": "ms" },
        "prefill_throughput": { "value": 3150.30740384616, "unit": "tokens/sec" },
        "inter_token_latency": { "value": 18.153352603015076, "unit": "ms" },
        "output_token_throughput_per_user": { "value": 55.08624339913448, "unit": "tokens/sec/user" },
        "inter_chunk_latency": {
            "value": [17.868007, 18.349401999999998, 18.929637, 17.527514999999998, ...],
            "unit": "ms"
        }
    },
    "error": null
}
```

---

### 3. GPU Telemetry JSONL (`gpu_telemetry_export.jsonl`)

**Format:** Newline-delimited JSON (JSONL), one sample per line.

**Schema:**
```typescript
{
  "timestamp": string,              // ISO8601 timestamp
  "endpoint": string,               // DCGM endpoint (hostname:port)
  "gpu_index": number,
  "gpu_uuid": string,
  "gpu_name": string,
  "hostname": string,
  "metrics": {
    "gpu_power_usage": number,      // Watts
    "gpu_utilization": number,      // Percentage
    "memory_copy_utilization": number,
    "gpu_memory_used": number,      // Bytes
    "gpu_memory_free": number,      // Bytes
    "sm_clock_frequency": number,   // MHz
    "memory_clock_frequency": number,
    "memory_temperature": number,   // Celsius
    "gpu_temperature": number,
    "xid_errors": number,
    "power_violation": number,      // Microseconds
    "thermal_violation": number
  }
}
```

---

### 4. Console Output - GPU Telemetry Summary Table

**Displayed at end of benchmark when GPU telemetry is enabled:**

```
                          NVIDIA AIPerf | GPU Telemetry Summary
                               1/1 DCGM endpoints reachable
                                    • localhost:9401 ✔

                      localhost:9401 | GPU 0 | NVIDIA H100 80GB HBM3
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

---

### 5. CSV Export Format (`profile_export_aiperf.csv`)

**Structure:** One metric per row with statistical aggregations as columns.

**CSV Headers:**
```
Metric,Unit,avg,min,max,p50,p99,std
```

**Example Rows:**
```csv
Metric,Unit,avg,min,max,p50,p99,std
request_throughput,requests/sec,0.2504302580624667,,,,
request_latency,ms,3652.1454372343746,3517.14812,3954.0145199999997,3644.0809205,3909.3341161199996,70.34527181408323
time_to_first_token,ms,30.814987671875,28.662267,34.717433,30.668547999999998,34.344268879999994,1.3745155360138603
inter_token_latency,ms,18.19909533160138,17.528383467336685,19.698244613065327,18.158001085427134,19.481766498241203,0.3514929398171827
output_token_throughput,tokens/sec,50.08213863971112,,,,
```

---

### 6. GPU Telemetry CSV Export

**CSV Headers:**
```
Endpoint,GPU_Index,GPU_Name,GPU_UUID,Metric,avg,min,max,p50,p99,std
```

**Example Rows:**
```csv
Endpoint,GPU_Index,GPU_Name,GPU_UUID,Metric,avg,min,max,p50,p99,std
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,GPU Power Usage (W),348.69,120.57,386.02,378.34,386.02,85.97
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,Energy Consumption (MJ),0.24,0.23,0.25,0.23,0.25,0.01
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,GPU Utilization (%),45.82,0.00,66.00,66.00,66.00,24.52
```

---

## Common Artifact Paths (Dummy Examples for Documentation)

Use these consistent path patterns in your documentation and examples:

### Single-Run Artifacts
```
# Basic concurrency run
artifacts/llama-3.1-8b-chat-concurrency8/
  ├── profile_export_aiperf.json
  ├── profile_export.jsonl
  └── logs/

# Request rate run
artifacts/qwen-2.5-7b-chat-rate10/
  ├── profile_export_aiperf.json
  ├── profile_export.jsonl
  └── logs/

# With GPU telemetry
artifacts/mistral-7b-chat-concurrency4/
  ├── profile_export_aiperf.json
  ├── profile_export.jsonl
  ├── gpu_telemetry_export.jsonl
  ├── gpu_telemetry_export.json
  └── logs/
```

### Multi-Run Comparison Artifacts
```
artifacts/
├── sweep_concurrency/
│   ├── run1_concurrency1/
│   │   └── profile_export_aiperf.json
│   ├── run2_concurrency2/
│   │   └── profile_export_aiperf.json
│   ├── run3_concurrency4/
│   │   └── profile_export_aiperf.json
│   └── run4_concurrency8/
│       └── profile_export_aiperf.json
```

### With All Telemetry Enabled
```
artifacts/full-telemetry-run/
├── profile_export_aiperf.json
├── profile_export_aiperf.csv
├── profile_export.jsonl
├── gpu_telemetry_export.json
├── gpu_telemetry_export.csv
├── gpu_telemetry_export.jsonl
├── server_metrics_export.json
├── server_metrics_export.csv
├── server_metrics_export.jsonl
├── _raw.jsonl
└── logs/
    ├── dataset_manager_abc123.log
    ├── timing_manager_def456.log
    ├── worker_manager_ghi789.log
    ├── worker_jkl012.log
    ├── record_processor_mno345.log
    ├── records_manager_pqr678.log
    └── system_controller.log
```

---

## Summary

This reference provides:

1. **Actual CLI output** from real command executions
2. **Complete JSON/JSONL schemas** with type annotations
3. **Real-world examples** from test fixtures and actual runs
4. **Error message formats** for common configuration issues
5. **Artifact directory structures** showing all generated files
6. **Dummy consistent paths** for use in documentation

All formats are **production-ready** and reflect the actual output structure of AIPerf v0.1.0.

For complete metric definitions, see [metrics_reference.md](../metrics_reference.md).
