<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Working with Profile Export Files

This guide demonstrates how to programmatically work with AIPerf benchmark output files using the native Pydantic data models.

## Overview

AIPerf generates multiple output formats after each benchmark run, each optimized for different analysis workflows:

- [**`inputs.json`**](#input-dataset-json) - Complete input dataset with formatted payloads for each request
- [**`profile_export.jsonl`**](#per-request-records-jsonl) - Per-request metric records in JSON Lines format with one record per line
- [**`profile_export_aiperf.json`**](#aggregated-statistics-json) - Aggregated statistics and user configuration as a single JSON object
- [**`profile_export_aiperf.csv`**](#aggregated-statistics-csv) - Aggregated statistics in CSV format

## Data Models

AIPerf uses Pydantic models for type-safe parsing and validation of all benchmark output files. These models ensure data integrity and provide IDE autocompletion support.

### Core Models

```python
from aiperf.common.models import (
    MetricRecordInfo,
    MetricRecordMetadata,
    MetricValue,
    ErrorDetails,
    InputsFile,
    SessionPayloads,
)
```

| Model | Description | Source |
|-------|-------------|--------|
| `MetricRecordInfo` | Complete per-request record including metadata, metrics, and error information | [record_models.py](../../aiperf/common/models/record_models.py) |
| `MetricRecordMetadata` | Request metadata: timestamps, IDs, worker identifiers, and phase information | [record_models.py](../../aiperf/common/models/record_models.py) |
| `MetricValue` | Individual metric value with associated unit of measurement | [record_models.py](../../aiperf/common/models/record_models.py) |
| `ErrorDetails` | Error information including HTTP code, error type, and descriptive message | [error_models.py](../../aiperf/common/models/error_models.py) |
| `InputsFile` | Container for all input dataset sessions with formatted payloads for each turn | [dataset_models.py](../../aiperf/common/models/dataset_models.py) |
| `SessionPayloads` | Single conversation session with session ID and list of formatted request payloads | [dataset_models.py](../../aiperf/common/models/dataset_models.py) |

## Output File Formats

### Input Dataset (JSON)

**File:** `artifacts/my-run/inputs.json`

A structured representation of all input datasets converted to the payload format used by the endpoint.

**Structure:**
```json
{
  "data": [
    {
      "session_id": "a5cdb1fe-19a3-4ed0-9e54-ed5ed6dc5578",
      "payloads": [
        { ... }  // formatted payload based on the endpoint type.
      ]
    }
  ]
}
```

**Key fields:**
- `session_id`: Unique identifier for the conversation. This can be used to correlate inputs with results.
- `payloads`: Array of formatted request payloads (one per turn in multi-turn conversations)

### Per-Request Records (JSONL)

**File:** `artifacts/my-run/profile_export.jsonl`

The JSONL output contains one record per line, for each request sent during the benchmark. Each record includes request metadata, computed metrics, and error information if the request failed.

#### Successful Request Record

```json
{
  "metadata": {
    "session_num": 45,
    "x_request_id": "7609a2e7-aa53-4ab1-98f4-f35ecafefd25",
    "x_correlation_id": "32ee4f33-cfca-4cfc-988f-79b45408b909",
    "conversation_id": "77aa5b0e-b305-423f-88d5-c00da1892599",
    "turn_index": 0,
    "request_start_ns": 1759813207532900363,
    "request_ack_ns": 1759813207650730976,
    "request_end_ns": 1759813207838764604,
    "worker_id": "worker_359d423a",
    "record_processor_id": "record_processor_1fa47cd7",
    "benchmark_phase": "profiling",
    "was_cancelled": false,
    "cancellation_time_ns": null
  },
  "metrics": {
    "input_sequence_length": {"value": 550, "unit": "tokens"},
    "time_to_first_token": {"value": 255.88656799999998, "unit": "ms"},
    "request_latency": {"value": 297.52522799999997, "unit": "ms"},
    "output_token_count": {"value": 9, "unit": "tokens"},
    "time_to_first_token": {"value": 4.8984369999999995, "unit": "ms"},
    "inter_chunk_latency": {"value": [4.898437, 5.316006, 4.801489, 5.674918, 4.811467, 5.097998, 5.504797, 5.533548], "unit": "ms"},
    "output_sequence_length": {"value": 9, "unit": "tokens"},
    "inter_token_latency": {"value": 5.2048325, "unit": "ms"},
    "output_token_throughput_per_user": {"value": 192.1291415237666, "unit": "tokens/sec/user"}
  },
  "error": null
}
```

**Metadata Fields:**
- `session_num`: Sequential request number across the entire benchmark (0-indexed).
  - For single-turn conversations, this will be the request index across all requests in the benchmark.
  - For multi-turn conversations, this will be the index of the user session across all sessions in the benchmark.
- `x_request_id`: Unique identifier for this specific request. This is sent to the endpoint as the X-Request-ID header.
- `x_correlation_id`: Unique identifier for the user session. This is the same for all requests in the same user session for multi-turn conversations. This is sent to the endpoint as the X-Correlation-ID header.
- `conversation_id`: ID of the input dataset conversation. This can be used to correlate inputs with results.
- `turn_index`: Position within a multi-turn conversation (0-indexed), or 0 for single-turn conversations.
- `request_start_ns`: Epoch time in nanoseconds when request was initiated by AIPerf.
- `request_ack_ns`: Epoch time in nanoseconds when server acknowledged the request. This is only applicable to streaming requests.
- `request_end_ns`: Epoch time in nanoseconds when the last response was received from the endpoint.
- `worker_id`: ID of the AIPerf worker that executed the request against the endpoint.
- `record_processor_id`: ID of the AIPerf record processor that processed the results from the server.
- `benchmark_phase`: Phase of the benchmark. Currently only `profiling` is supported.
- `was_cancelled`: Whether the request was cancelled during execution (such as when `--request-cancellation-rate` is enabled).
- `cancellation_time_ns`: Epoch time in nanoseconds when the request was cancelled (if applicable).

**Metrics:**
See the [Complete Metrics Reference](../metrics_reference.md) page for a list of all metrics and their descriptions. Will always be null for failed requests.

#### Failed Request Record

```json
{
  "metadata": {
    "session_num": 80,
    "x_request_id": "c35e4b1b-6775-4750-b875-94cd68e5ec15",
    "x_correlation_id": "77ecf78d-b848-4efc-9579-cd695c6e89c4",
    "conversation_id": "9526b41d-5dbc-41a5-a353-99ae06a53bc5",
    "turn_index": 0,
    "request_start_ns": 1759879161119147826,
    "request_ack_ns": null,
    "request_end_ns": 1759879161119772754,
    "worker_id": "worker_6006099d",
    "record_processor_id": "record_processor_fdeeec8f",
    "benchmark_phase": "profiling",
    "was_cancelled": true,
    "cancellation_time_ns": 1759879161119772754
  },
  "metrics": {
    "error_isl": {"value": 550, "unit": "tokens"}
  },
  "error": {
    "code": 499,
    "type": "RequestCancellationError",
    "message": "Request was cancelled after 0.000 seconds"
  }
}
```

**Error Fields:**
- `code`: HTTP status code or custom error code
- `type`: Classification of the error (e.g., timeout, cancellation, server error). Typically the python exception class name.
- `message`: Human-readable error description


### Aggregated Statistics (JSON)

**File:** `artifacts/my-run/profile_export_aiperf.json`

A single JSON object containing statistical summaries (min, max, mean, percentiles) for all metrics across the entire benchmark run, error summary, and the user configuration used for the benchmark.

#### Complete File Structure

The following example shows the complete structure of a `profile_export_aiperf.json` file with all major sections:

```json
{
  // ==================== Benchmark Metrics ====================
  // Metrics with full statistical distributions
  "request_throughput": {
    "unit": "requests/sec",
    "avg": 13.16084556379392
  },
  "request_latency": {
    "unit": "ms",
    "avg": 6550.356151749999,
    "p1": 5523.63769363,
    "p5": 5641.17552705,
    "p10": 5649.248782399999,
    "p25": 5678.539702,
    "p50": 5799.897624499999,
    "p75": 7314.955492249999,
    "p90": 8594.209632299999,
    "p95": 8980.84214965,
    "p99": 9381.628943169999,
    "min": 4996.247445999999,
    "max": 9383.367922,
    "std": 1171.13614778278
  },
  "time_to_first_token": {
    "unit": "ms",
    "avg": 1588.797775365,
    "p50": 787.5439584999999,
    "p95": 4051.42644575,
    "min": 382.242961,
    "max": 4121.087335
  },
  "output_sequence_length": {
    "unit": "tokens",
    "avg": 96.85,
    "p50": 97.0,
    "min": 91.0,
    "max": 97.0
  },
  // ... other metrics with statistical distributions

  // ==================== Benchmark Summary ====================
  "benchmark_duration": {
    "unit": "sec",
    "avg": 15.196591969000002
  },
  "request_count": {
    "unit": "requests",
    "avg": 200.0
  },
  "total_isl": {
    "unit": "tokens",
    "avg": 110000.0
  },
  "total_osl": {
    "unit": "tokens",
    "avg": 19370.0
  },

  // ==================== Benchmark Status ====================
  "was_cancelled": true,
  "start_time": "2025-10-17T20:22:08.939332",
  "end_time": "2025-10-17T20:22:24.967633",

  // ==================== Error Tracking ====================
  "error_summary": [
    {
      "error_details": {
        "code": 499,
        "type": "RequestCancellationError",
        "message": "Request was cancelled after 0.000 seconds"
      },
      "count": 81
    },
    {
      "error_details": {
        "code": 500,
        "type": "InternalServerError",
        "message": "Server encountered an internal error"
      },
      "count": 3
    }
  ],
  "goodput": null,
  "good_request_count": null,
  "error_request_count": null,
  "error_isl": null,
  "total_error_isl": null,

  // ==================== GPU Telemetry Data ====================
  "telemetry_data": {
    "summary": {
      "endpoints_configured": ["http://localhost:9401/metrics"],
      "endpoints_successful": ["http://localhost:9401/metrics"],
      "start_time": "2025-10-17T20:22:08.939332",
      "end_time": "2025-10-17T20:22:24.970213"
    },
    "endpoints": {
      "localhost:9401": {
        "gpus": {
          "gpu_0": {
            "gpu_index": 0,
            "gpu_name": "NVIDIA RTX 6000 Ada Generation",
            "gpu_uuid": "GPU-f35d4dc1-e000-3faa-e3cf-787ede5e8761",
            "hostname": "d3bc1424aff3",
            "metrics": {
              "gpu_power_usage": {
                "unit": "W",
                "avg": 43.37,
                "p50": 26.68,
                "p95": 190.24,
                "min": 26.68,
                "max": 299.28,
                "std": 65.35
              },
              "gpu_utilization": {
                "unit": "%",
                "avg": 33.22,
                "p50": 29.0,
                "p95": 70.40,
                "min": 29.0,
                "max": 98.0,
                "std": 16.54
              },
              "gpu_memory_used": {
                "unit": "GB",
                "avg": 47.03,
                "min": 47.03,
                "max": 47.03
              },
              "gpu_temperature": {
                "unit": "°C",
                "avg": 41.22,
                "p50": 40.0,
                "max": 60.0
              }
              // ... other GPU metrics
            }
          }
        }
      }
    }
  },

  // ==================== Input Configuration ====================
  "input_config": {
    "endpoint": {
      "model_names": ["openai/gpt-oss-20b"],
      "type": "chat",
      "streaming": true,
      "url": "localhost:9000"
    },
    "input": {
      "random_seed": 0,
      "prompt": {
        "output_tokens": {
          "mean": 100
        }
      }
    },
    "loadgen": {
      "concurrency": 100,
      "request_rate_mode": "concurrency_burst",
      "request_count": 1000000000
    },
    "cli_command": "aiperf profile -m \"openai/gpt-oss-20b\" --url \"localhost:9000\" --streaming --concurrency 100 --endpoint-type \"chat\" --num-requests 1000000000 --random-seed 0 --osl 100 --record-processors 4 --gpu-telemetry \"http://localhost:9400/metrics\" \"http://localhost:9401/metrics\"",
    "gpu_telemetry": [
      "http://localhost:9400/metrics",
      "http://localhost:9401/metrics"
    ]
  }
}
```

> [!NOTE]
> The `input_config` section is a representation of the internal configuration, and it is not guaranteed to be stable across AIPerf versions.

#### Key Field Categories

**Metric Fields (for metrics with statistical distributions):**
- `unit`: Unit of measurement for the metric
- `avg`: Mean value across all requests
- `p1`, `p5`, `p10`, `p25`, `p50`, `p75`, `p90`, `p95`, `p99`: Percentile values
- `min`: Minimum value observed
- `max`: Maximum value observed
- `std`: Standard deviation

**Benchmark Status Fields:**
- `was_cancelled`: Boolean indicating if the benchmark was manually stopped before completion
- `start_time`: ISO 8601 timestamp when benchmark started
- `end_time`: ISO 8601 timestamp when benchmark ended

**Error Tracking Fields:**
- `error_summary`: Array of error summaries grouped by error type
  - `error_details`: Detailed information about the error
    - `code`: HTTP status code or custom error code
    - `type`: Error classification (typically the Python exception class name)
    - `message`: Human-readable error description
  - `count`: Number of times this specific error occurred during the benchmark
- `goodput`: Percentage of successful requests (null if not tracked)
- `good_request_count`: Total number of successful requests (null if not tracked)
- `error_request_count`: Total number of failed requests (null if no errors)
- `error_isl`: Input sequence length statistics for failed requests (null if no errors)
- `total_error_isl`: Total input sequence length across all failed requests (null if no errors)

**GPU Telemetry Metrics (when enabled via `--gpu-telemetry`):**
- `gpu_power_usage`: Power consumption in watts
- `energy_consumption`: Total energy used in megajoules
- `gpu_utilization`: GPU compute utilization percentage
- `memory_copy_utilization`: Memory copy engine utilization percentage
- `gpu_memory_used`: GPU memory used in gigabytes
- `gpu_memory_free`: GPU memory free in gigabytes
- `sm_clock_frequency`: Streaming multiprocessor clock frequency in MHz
- `memory_clock_frequency`: Memory clock frequency in MHz
- `gpu_temperature`: GPU temperature in Celsius
- `memory_temperature`: Memory temperature in Celsius
- `xid_errors`: Count of GPU hardware errors

### Aggregated Statistics (CSV)

**File:** `artifacts/my-run/profile_export_aiperf.csv`

Contains the same aggregated statistics as the JSON format, but in a spreadsheet-friendly structure optimized for analysis in Excel, Google Sheets, or data analysis tools like pandas.

#### Structure

The CSV file is organized into three distinct sections separated by blank lines:

1. **Statistical Distribution Metrics** - Metrics with full statistical distributions
2. **Summary Metrics** - Single-value aggregate metrics
3. **GPU Telemetry Metrics** - Hardware telemetry data (if enabled)

#### Section 1: Statistical Distribution Metrics

The first section contains metrics that have statistical distributions across all requests:

**Example Snippet:**
```csv
Metric,avg,min,max,p1,p5,p10,p25,p50,p75,p90,p95,p99,std
Request Latency (ms),6550.36,4996.25,9383.37,5523.64,5641.18,5649.25,5678.54,5799.90,7314.96,8594.21,8980.84,9381.63,1171.14
Time to First Token (ms),1588.80,382.24,4121.09,403.48,547.97,641.77,659.03,787.54,2439.00,3695.10,4051.43,4119.33,1189.99
Inter Token Latency (ms),51.76,46.81,56.30,50.41,50.50,50.53,50.86,51.98,52.17,53.18,54.32,55.87,1.23
Output Sequence Length (tokens),96.85,91.00,97.00,91.00,97.00,97.00,97.00,97.00,97.00,97.00,97.00,97.00,0.88
```

**Columns:**
- `Metric`: Human-readable metric name with unit in parentheses
- `avg`: Mean value across all requests
- `min`: Minimum value observed
- `max`: Maximum value observed
- `p1`, `p5`, `p10`, `p25`, `p50`, `p75`, `p90`, `p95`, `p99`: Percentile values
- `std`: Standard deviation

#### Section 2: Summary Metrics

The second section contains aggregate metrics that have a single value for the entire benchmark:

**Example Snippet:**
```csv
Metric,Value
Benchmark Duration (sec),15.20
Request Throughput (requests/sec),13.16
Request Count,200.00
Output Token Throughput (tokens/sec),1274.63
```

**Columns:**
- `Metric`: Human-readable metric name with unit in parentheses
- `Value`: The aggregate value for the entire benchmark run

#### Section 3: GPU Telemetry Metrics

When GPU telemetry is enabled via `--gpu-telemetry`, the third section contains hardware metrics with one row per GPU per metric:

**Example Snippet:**
```csv
Endpoint,GPU_Index,GPU_Name,GPU_UUID,Metric,avg,min,max,p1,p5,p10,p25,p50,p75,p90,p95,p99,std
localhost:9401,0,NVIDIA RTX 6000 Ada Generation,GPU-f35d4dc1-e000-3faa-e3cf-787ede5e8761,GPU Power Usage (W),43.37,26.68,299.28,26.68,26.68,26.68,26.68,26.68,26.68,26.68,190.24,299.28,65.35
localhost:9401,0,NVIDIA RTX 6000 Ada Generation,GPU-f35d4dc1-e000-3faa-e3cf-787ede5e8761,GPU Utilization (%),33.22,29.00,98.00,29.00,29.00,29.00,29.00,29.00,29.00,29.00,70.40,98.00,16.54
localhost:9401,0,NVIDIA RTX 6000 Ada Generation,GPU-f35d4dc1-e000-3faa-e3cf-787ede5e8761,GPU Memory Used (GB),47.03,47.03,47.03,47.03,47.03,47.03,47.03,47.03,47.03,47.03,47.03,47.03,0.00
```

**Columns:**
- `Endpoint`: Telemetry endpoint hostname and port
- `GPU_Index`: Zero-based GPU index on the system
- `GPU_Name`: GPU model name (e.g., "NVIDIA RTX 6000 Ada Generation")
- `GPU_UUID`: Unique GPU identifier
- `Metric`: Human-readable telemetry metric name with unit
- `avg`, `min`, `max`, `p1`-`p99`, `std`: Statistical values (same as Section 1)


## Working with Output Data

AIPerf output files can be parsed using the native Pydantic models for type-safe data handling and analysis.

### Synchronous Loading
```python
from aiperf.common.models import MetricRecordInfo

def load_records(file_path: Path) -> list[MetricRecordInfo]:
    """Load artifacts/my-run/profile_export.jsonl file into structured Pydantic models in sync mode."""
    records = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = MetricRecordInfo.model_validate_json(line)
                records.append(record)
    return records
```

### Asynchronous Loading

For large benchmark runs with thousands of requests, use async file I/O for better performance:

```python
import aiofiles
from aiperf.common.models import MetricRecordInfo

async def process_streaming_records_async(file_path: Path) -> None:
    """Load artifacts/my-run/profile_export.jsonl file into structured Pydantic models in async mode and process the streaming records."""
    async with aiofiles.open(file_path, encoding="utf-8") as f:
        async for line in f:
            if line.strip():
                record = MetricRecordInfo.model_validate_json(line)
                # ... Process the streaming records here ...
```

### Working with Input Datasets

Load and analyze the `inputs.json` file to understand what data was sent during the benchmark:

```python
from pathlib import Path
from aiperf.common.models import InputsFile

def load_inputs_file(file_path: Path) -> InputsFile:
    """Load inputs.json file into structured Pydantic model."""
    with open(file_path, encoding="utf-8") as f:
        return InputsFile.model_validate_json(f.read())

inputs = load_inputs_file(Path("artifacts/my-run/inputs.json"))
```

### Correlating Inputs with Results

Combine `artifacts/my-run/inputs.json` with `artifacts/my-run/profile_export.jsonl` for deeper analysis:

```python
from pathlib import Path
from aiperf.common.models import InputsFile, MetricRecordInfo

def correlate_inputs_and_results(inputs_path: Path, results_path: Path):
    """Correlate input prompts with performance metrics."""
    # Load inputs
    with open(inputs_path, encoding="utf-8") as f:
        inputs = InputsFile.model_validate_json(f.read())

    # Create session lookup
    session_inputs = {session.session_id: session for session in inputs.data}

    # Process results and correlate
    with open(results_path, encoding="utf-8") as f:
        for line in f:
          if not line.strip():
            continue

          record = MetricRecordInfo.model_validate_json(line)

          # Find corresponding input
          conv_id = record.metadata.conversation_id
          if conv_id not in session_inputs:
              raise ValueError(f"Conversation ID {conv_id} not found in inputs")

          session = session_inputs[conv_id]
          turn_idx = record.metadata.turn_index

          if turn_idx >= len(session.payloads):
              raise ValueError(f"Turn index {turn_idx} is out of range for session {conv_id}")

          # Assign the raw request payload to the record, and print it out
          # You can do this because AIPerf models allow extra fields to be added to the model.
          payload = session.payloads[turn_idx]
          record.payload = payload
          print(record.model_dump_json(indent=2))

correlate_inputs_and_results(
    Path("artifacts/my-run/inputs.json"),
    Path("artifacts/my-run/profile_export.jsonl")
)
```
