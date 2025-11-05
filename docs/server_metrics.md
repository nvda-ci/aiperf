<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Server Metrics Collection & Export

## Overview

AIPerf automatically collects server-side metrics from inference servers during benchmarking runs. This provides comprehensive visibility into server behavior, resource utilization, and performance characteristics alongside client-side latency measurements.

**Key Features:**
- **Automatic Collection**: Always attempts collection from inference endpoint + default ports
- **Multiple Endpoints**: Supports collecting from multiple server instances simultaneously
- **Dynamic Discovery**: Automatically discovers and exports all available metrics without hardcoded lists
- **Multiple Export Formats**: JSONL (raw records), CSV (statistical summaries), JSON (hierarchical aggregates)
- **Rich Console Display**: Optional formatted tables showing all server metrics in real-time
- **Prometheus Compatible**: Parses standard Prometheus `/metrics` endpoints

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         User Runs Benchmark                             │
│  aiperf profile --url http://server:8000 --server-metrics additional    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      ServerMetricsManager                               │
│  • Derives endpoint from --url + /metrics                               │
│  • Tests reachability of all endpoints                                  │
│  • Creates ServerMetricsDataCollector per endpoint                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              ServerMetricsDataCollector (per endpoint)                  │
│  • Polls /metrics endpoint every 0.5s                                   │
│  • Parses Prometheus format metrics                                     │
│  • Creates ServerMetricRecord with timestamp + metrics                  │
│  • Sends to RecordsManager via message bus                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         RecordsManager                                  │
│  • Aggregates records from all endpoints                                │
│  • Buffers and writes to server_metrics_export.jsonl                    │
│  • Passes to ServerMetricsResultsProcessor                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  ServerMetricsResultsProcessor                          │
│  • Computes statistics per metric (avg, min, max, p99, p90, p50, std)   │
│  • Organizes by endpoint → server → metrics hierarchy                   │
│  • Creates ServerMetricsResults object                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                             Exporters                                   │
│  • CSV Exporter:      metrics.csv (statistical summary)                 │
│  • JSON Exporter:     aiperf.json (hierarchical aggregate)              │
│  • Console Exporter:  Rich tables (if --server-metrics flag used)       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Base Classes

Server metrics leverage generic base classes for code reuse:

- **`BaseMetricsDataCollector`**: Async polling, reachability testing, metric parsing
- **`BaseMetricsManager`**: Multi-collector lifecycle management, endpoint configuration
- **`BaseMetricsConsoleExporter`**: Dynamic field discovery, table formatting

**Key Design Principle**: Dynamic field discovery means **no hardcoded metric lists**. Any metric exposed by the server will be automatically collected and exported.

## Configuration

### Automatic Collection (Default)

```bash
aiperf profile \
    --url http://inference-server:8000 \
    --model nvidia/llama-3.1-nemotron-70b-instruct \
    --endpoint-type chat \
    --request-count 100
```

**Default Behavior:**
- Automatically tries: `http://inference-server:8000/metrics`
- Also tries: `localhost:8081/metrics`, `localhost:7777/metrics`, `localhost:2379/metrics`
- Exports JSONL and includes in aggregate JSON/CSV if any endpoint reachable
- No console display unless `--server-metrics` flag provided

### Explicit Additional Endpoints

```bash
aiperf profile \
    --url http://inference-server:8000 \
    --server-metrics \
        http://server2:8001/metrics \
        http://server3:8002/metrics \
    --model nvidia/llama-3.1-nemotron-70b-instruct \
    --endpoint-type chat \
    --request-count 100
```

**With `--server-metrics` Flag:**
- Collects from inference endpoint + additional specified endpoints
- **Enables console display** of server metrics tables
- All endpoints exported to JSONL, CSV, and JSON

### Environment Variables

Default endpoints can be configured via environment:

```bash
export AIPERF_SERVER_METRICS_DEFAULT_ENDPOINTS="localhost:8081,localhost:7777"
export AIPERF_SERVER_METRICS_COLLECTION_INTERVAL="1.0"  # seconds
```

## Metrics Categories

Server metrics are organized into categories based on their source and purpose:

### System Resource Metrics
| Metric | Description | Unit |
|--------|-------------|------|
| `process_cpu_usage_percent` | CPU utilization of server process | % |
| `process_cpu_seconds` | Cumulative CPU time | seconds |
| `memory_usage_bytes` | Current memory usage | bytes |
| `memory_total_bytes` | Total available memory | bytes |
| `process_resident_memory_bytes` | RSS memory | bytes |
| `process_virtual_memory_bytes` | Virtual memory | bytes |
| `process_open_fds` | Open file descriptors | count |

### HTTP Server Metrics
| Metric | Description | Unit |
|--------|-------------|------|
| `http_requests_in_flight` | Current concurrent requests | count |
| `http_requests` | Total requests served | count |
| `http_responses_2xx` | Successful responses | count |
| `http_responses_4xx` | Client error responses | count |
| `http_responses_5xx` | Server error responses | count |
| `http_request_duration_seconds` | Request duration histogram | seconds |
| `http_response_size_bytes` | Response size | bytes |

### Frontend Metrics (vLLM/Dynamo)
| Metric | Description | Unit |
|--------|-------------|------|
| `dynamo_frontend_requests` | Total frontend requests | count |
| `dynamo_frontend_inflight_requests` | Current inflight requests | count |
| `dynamo_frontend_queued_requests` | Queued requests | count |
| `dynamo_frontend_time_to_first_token_seconds` | TTFT latency | seconds |
| `dynamo_frontend_inter_token_latency_seconds` | ITL latency | seconds |
| `dynamo_frontend_input_sequence_tokens` | Input tokens processed | count |
| `dynamo_frontend_output_sequence_tokens` | Output tokens generated | count |

### Component Metrics (vLLM/Dynamo Backend)
| Metric | Description | Unit |
|--------|-------------|------|
| `dynamo_component_requests` | Total component requests | count |
| `dynamo_component_inflight_requests` | Current inflight | count |
| `dynamo_component_request_duration_seconds` | Request duration | seconds |
| `dynamo_component_system_uptime_seconds` | System uptime | seconds |
| `dynamo_component_request_bytes` | Request size | bytes |
| `dynamo_component_response_bytes` | Response size | bytes |

### KV Cache Metrics (vLLM)
| Metric | Description | Unit |
|--------|-------------|------|
| `dynamo_component_kvstats_total_blocks` | Total KV cache blocks | count |
| `dynamo_component_kvstats_active_blocks` | Active KV cache blocks | count |
| `dynamo_component_kvstats_gpu_cache_usage_percent` | GPU cache utilization | % |
| `dynamo_component_kvstats_gpu_prefix_cache_hit_rate` | Prefix cache hit rate | % |

### Model Configuration Metrics (Static)
| Metric | Description | Unit |
|--------|-------------|------|
| `dynamo_frontend_model_total_kv_blocks` | Total KV blocks configured | count |
| `dynamo_frontend_model_max_num_seqs` | Max concurrent sequences | count |
| `dynamo_frontend_model_workers` | Number of worker processes | count |
| `dynamo_frontend_model_context_length` | Model context window | tokens |
| `dynamo_frontend_model_kv_cache_block_size` | KV cache block size | count |
| `dynamo_frontend_model_max_num_batched_tokens` | Max batched tokens | count |
| `dynamo_frontend_model_migration_limit` | Migration limit | count |

### Network Metrics
| Metric | Description | Unit |
|--------|-------------|------|
| `network_receive_bytes` | Bytes received | bytes |
| `network_transmit_bytes` | Bytes transmitted | bytes |

## Export Formats

### 1. JSONL Export (`server_metrics_export.jsonl`)

**Raw timestamped records** - one line per collection interval per endpoint.

```jsonl
{"timestamp_ns":1730000000000000000,"server_url":"http://127.0.0.1:39827/server1/metrics","collector_id":"collector_http___127_0_0_1_39827_server1_metrics","server_id":"server-0","instance":"server-0:8080","server_type":"ai-server","metrics_data":{"cpu_system_seconds":15.39,"cpu_user_seconds":35.90,"dynamo_component_inflight_requests":10.0,"dynamo_component_kvstats_active_blocks":767.0,"dynamo_component_kvstats_gpu_cache_usage_percent":0.38,"dynamo_component_kvstats_gpu_prefix_cache_hit_rate":0.56,"dynamo_component_kvstats_total_blocks":2000.0,"dynamo_component_request_bytes":387480.45,"dynamo_component_request_duration_seconds":0.12,"dynamo_component_requests":235.0,"dynamo_component_response_bytes":581222.67,"dynamo_component_system_uptime_seconds":69955.53,"dynamo_frontend_inflight_requests":14.0,"dynamo_frontend_input_sequence_tokens":36225.0,"dynamo_frontend_inter_token_latency_seconds":0.01,"dynamo_frontend_model_context_length":8192.0,"dynamo_frontend_model_kv_cache_block_size":16.0,"dynamo_frontend_model_max_num_batched_tokens":8192.0,"dynamo_frontend_model_max_num_seqs":64.0,"dynamo_frontend_model_migration_limit":100.0,"dynamo_frontend_model_total_kv_blocks":2000.0,"dynamo_frontend_model_workers":4.0,"dynamo_frontend_output_sequence_tokens":19685.0,"dynamo_frontend_queued_requests":7.0,"dynamo_frontend_request_duration_seconds":0.19,"dynamo_frontend_requests":157.0,"dynamo_frontend_time_to_first_token_seconds":0.07,"http_request_duration_seconds":0.31,"http_requests":157.0,"http_requests_in_flight":17.0,"http_response_size_bytes":3748.46,"http_responses_2xx":148.0,"http_responses_4xx":5.0,"http_responses_5xx":0.0,"memory_total_bytes":275000000000.0,"memory_usage_bytes":157000000000.0,"network_receive_bytes":322901.33,"network_transmit_bytes":645802.67,"process_cpu_seconds":51.29,"process_cpu_usage_percent":0.53,"process_open_fds":331.0,"process_resident_memory_bytes":128000000000.0,"process_virtual_memory_bytes":187000000000.0},"error":null}
{"timestamp_ns":1730000000500000000,"server_url":"http://127.0.0.1:39827/server2/metrics","collector_id":"collector_http___127_0_0_1_39827_server2_metrics","server_id":"server-1","instance":"server-1:8080","server_type":"ai-server","metrics_data":{"cpu_system_seconds":14.79,"cpu_user_seconds":34.50,"dynamo_component_inflight_requests":9.0,"dynamo_component_kvstats_active_blocks":708.0,"dynamo_component_kvstats_gpu_cache_usage_percent":0.35,"dynamo_component_kvstats_gpu_prefix_cache_hit_rate":0.52,"dynamo_component_kvstats_total_blocks":2000.0,"dynamo_component_request_bytes":356823.12,"dynamo_component_request_duration_seconds":0.11,"dynamo_component_requests":217.0,"dynamo_component_response_bytes":535234.68,"dynamo_component_system_uptime_seconds":69955.53,"dynamo_frontend_inflight_requests":13.0,"dynamo_frontend_input_sequence_tokens":33407.0,"dynamo_frontend_inter_token_latency_seconds":0.01,"dynamo_frontend_model_context_length":8192.0,"dynamo_frontend_model_kv_cache_block_size":16.0,"dynamo_frontend_model_max_num_batched_tokens":8192.0,"dynamo_frontend_model_max_num_seqs":64.0,"dynamo_frontend_model_migration_limit":100.0,"dynamo_frontend_model_total_kv_blocks":2000.0,"dynamo_frontend_model_workers":4.0,"dynamo_frontend_output_sequence_tokens":18148.0,"dynamo_frontend_queued_requests":6.0,"dynamo_frontend_request_duration_seconds":0.18,"dynamo_frontend_requests":145.0,"dynamo_frontend_time_to_first_token_seconds":0.07,"http_request_duration_seconds":0.29,"http_requests":145.0,"http_requests_in_flight":16.0,"http_response_size_bytes":3689.34,"http_responses_2xx":137.0,"http_responses_4xx":5.0,"http_responses_5xx":0.0,"memory_total_bytes":275000000000.0,"memory_usage_bytes":153000000000.0,"network_receive_bytes":297456.89,"network_transmit_bytes":594913.78,"process_cpu_seconds":49.29,"process_cpu_usage_percent":0.51,"process_open_fds":318.0,"process_resident_memory_bytes":125000000000.0,"process_virtual_memory_bytes":184000000000.0},"error":null}
```

**Use Cases:**
- Time-series analysis and visualization
- Correlation with request latencies
- Debugging performance issues
- Custom post-processing and analysis

**Structure:**
```typescript
{
  timestamp_ns: number;           // Nanosecond timestamp
  server_url: string;             // Metrics endpoint URL
  collector_id: string;           // Internal collector ID
  server_id: string;              // Unique server identifier
  instance: string;               // Server instance (e.g., "server-0:8080")
  server_type: string;            // Server type (e.g., "ai-server")
  metrics_data: {                 // All discovered metrics
    [metric_name: string]: number;
  };
  error: ErrorDetails | null;     // Collection error if any
}
```

### 2. CSV Export (`metrics.csv`)

**Statistical summaries** - aggregated statistics per metric per server.

```csv
Metric,avg,min,max,p99,p90,p50,std

# Server Metrics
## Server: 127.0.0.1:39827/server1
server_endpoint,127.0.0.1:39827/server1/metrics
server_id,server-0
server_type,ai-server
hostname,unknown
instance,server-0:8080

### CPU System Seconds (sec)
cpu_system_seconds,15.39,10.26,20.52,20.41,19.49,15.39,4.19

### CPU User Seconds (sec)
cpu_user_seconds,35.90,23.94,47.87,47.63,45.48,35.90,9.77

### Dynamo Component Inflight Requests (count)
dynamo_component_inflight_requests,10.00,9.00,11.00,10.98,10.80,10.00,0.82

### Dynamo Component Kvstats Active Blocks (count)
dynamo_component_kvstats_active_blocks,767.33,690.00,816.00,815.60,812.00,796.00,55.29

### Dynamo Component Kvstats GPU Cache Usage Percent (%)
dynamo_component_kvstats_gpu_cache_usage_percent,0.38,0.34,0.41,0.41,0.41,0.40,0.03

### HTTP Requests In Flight (count)
http_requests_in_flight,17.67,17.00,18.00,18.00,18.00,18.00,0.47

### Memory Usage Bytes (GB)
memory_usage_bytes,1.57e+11,1.54e+11,1.58e+11,1.58e+11,1.58e+11,1.57e+11,1.63e+09

### Process CPU Usage Percent (%)
process_cpu_usage_percent,0.53,0.53,0.53,0.53,0.53,0.53,0.00

## Server: 127.0.0.1:39827/server2
server_endpoint,127.0.0.1:39827/server2/metrics
server_id,server-1
server_type,ai-server
hostname,unknown
instance,server-1:8080

### CPU System Seconds (sec)
cpu_system_seconds,14.79,9.86,19.72,19.62,18.73,14.79,4.02

[... additional servers ...]
```

**Use Cases:**
- Quick performance overview
- Excel/spreadsheet analysis
- Performance reporting
- Automated performance regression detection

**Features:**
- Organized by server endpoint
- Hierarchical sections with metadata
- Human-readable metric names
- Automatic unit inference
- Full statistical distribution (avg, min, max, p99, p90, p50, std)

### 3. JSON Export (`aiperf.json`)

**Hierarchical aggregate** - nested structure for programmatic access.

```json
{
  "metadata": {
    "benchmark_start_time": "2025-01-07T09:05:31.234Z",
    "benchmark_end_time": "2025-01-07T09:05:45.678Z",
    "duration_seconds": 14.444,
    "aiperf_version": "1.0.0",
    "model": "nvidia/llama-3.1-nemotron-70b-instruct"
  },
  "llm_metrics": {
    "request_latency": {
      "avg": 8.12,
      "min": 4.27,
      "max": 44.27,
      "p99": 34.73,
      "p90": 12.38,
      "p50": 5.95,
      "std": 6.02,
      "unit": "ms"
    },
    "time_to_first_token": {
      "avg": 3.83,
      "min": 2.05,
      "max": 12.84,
      "p99": 11.26,
      "p90": 6.57,
      "p50": 2.68,
      "std": 2.17,
      "unit": "ms"
    }
  },
  "server_metrics": {
    "endpoints": {
      "http://127.0.0.1:39827/server1/metrics": {
        "servers": {
          "server-0": {
            "metadata": {
              "server_type": "ai-server",
              "hostname": "unknown",
              "instance": "server-0:8080"
            },
            "metrics": {
              "cpu_system_seconds": {
                "avg": 15.39,
                "min": 10.26,
                "max": 20.52,
                "p99": 20.41,
                "p90": 19.49,
                "p50": 15.39,
                "std": 4.19,
                "unit": "sec"
              },
              "process_cpu_usage_percent": {
                "avg": 0.53,
                "min": 0.53,
                "max": 0.53,
                "p99": 0.53,
                "p90": 0.53,
                "p50": 0.53,
                "std": 0.00,
                "unit": "%"
              },
              "memory_usage_bytes": {
                "avg": 1.57e11,
                "min": 1.54e11,
                "max": 1.58e11,
                "p99": 1.58e11,
                "p90": 1.58e11,
                "p50": 1.57e11,
                "std": 1.63e9,
                "unit": "GB"
              },
              "dynamo_component_kvstats_active_blocks": {
                "avg": 767.33,
                "min": 690.0,
                "max": 816.0,
                "p99": 815.6,
                "p90": 812.0,
                "p50": 796.0,
                "std": 55.29,
                "unit": "count"
              },
              "http_requests_in_flight": {
                "avg": 17.67,
                "min": 17.0,
                "max": 18.0,
                "p99": 18.0,
                "p90": 18.0,
                "p50": 18.0,
                "std": 0.47,
                "unit": "count"
              }
            }
          }
        }
      },
      "http://127.0.0.1:39827/server2/metrics": {
        "servers": {
          "server-1": {
            "metadata": {
              "server_type": "ai-server",
              "hostname": "unknown",
              "instance": "server-1:8080"
            },
            "metrics": {
              "cpu_system_seconds": {
                "avg": 14.79,
                "min": 9.86,
                "max": 19.72,
                "unit": "sec"
              }
            }
          }
        }
      }
    }
  }
}
```

**Use Cases:**
- CI/CD integration
- Automated performance analysis
- Machine learning on performance data
- Dashboard/visualization tools

**Structure:**
```typescript
{
  metadata: {
    benchmark_start_time: string;
    benchmark_end_time: string;
    duration_seconds: number;
    aiperf_version: string;
    model: string;
  };
  llm_metrics: {
    [metric_name: string]: StatisticalSummary;
  };
  server_metrics: {
    endpoints: {
      [endpoint_url: string]: {
        servers: {
          [server_id: string]: {
            metadata: {
              server_type: string;
              hostname: string;
              instance: string;
            };
            metrics: {
              [metric_name: string]: StatisticalSummary;
            };
          };
        };
      };
    };
  };
}

type StatisticalSummary = {
  avg: number;
  min: number;
  max: number;
  p99: number;
  p90: number;
  p50: number;
  std: number;
  unit: string;
};
```

### 4. Console Output (Rich Tables)

**Real-time display** - enabled with `--server-metrics` flag.

```
                          NVIDIA AIPerf | Server Metrics Summary
                              3/3 Server endpoints reachable
                                • 127.0.0.1:39827/server1 ✔
                                • 127.0.0.1:39827/server2 ✔
                                     • localhost:8081 ✔

                   127.0.0.1:39827/server1 | ai-server | server-0:8080
┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃    Metric ┃       avg ┃        min ┃       max ┃        p99 ┃       p90 ┃        p50 ┃       std ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━┩
│       CPU │     15.39 │      10.26 │     20.52 │      20.41 │     19.49 │      15.39 │      4.19 │
│    System │           │            │           │            │           │            │           │
│   Seconds │           │            │           │            │           │            │           │
│     (sec) │           │            │           │            │           │            │           │
│  CPU User │     35.90 │      23.94 │     47.87 │      47.63 │     45.48 │      35.90 │      9.77 │
│   Seconds │           │            │           │            │           │            │           │
│     (sec) │           │            │           │            │           │            │           │
│    Dynamo │     10.00 │       9.00 │     11.00 │      10.98 │     10.80 │      10.00 │      0.82 │
│ Component │           │            │           │            │           │            │           │
│  Inflight │           │            │           │            │           │            │           │
│  Requests │           │            │           │            │           │            │           │
│   (count) │           │            │           │            │           │            │           │
│    Dynamo │    767.33 │     690.00 │    816.00 │     815.60 │    812.00 │     796.00 │     55.29 │
│ Component │           │            │           │            │           │            │           │
│   Kvstats │           │            │           │            │           │            │           │
│    Active │           │            │           │            │           │            │           │
│    Blocks │           │            │           │            │           │            │           │
│   (count) │           │            │           │            │           │            │           │
│    Dynamo │      0.38 │       0.34 │      0.41 │       0.41 │      0.41 │       0.40 │      0.03 │
│ Component │           │            │           │            │           │            │           │
│   Kvstats │           │            │           │            │           │            │           │
│ GPU Cache │           │            │           │            │           │            │           │
│     Usage │           │            │           │            │           │            │           │
│   Percent │           │            │           │            │           │            │           │
│       (%) │           │            │           │            │           │            │           │
│      HTTP │     17.67 │      17.00 │     18.00 │      18.00 │     18.00 │      18.00 │      0.47 │
│  Requests │           │            │           │            │           │            │           │
│ In Flight │           │            │           │            │           │            │           │
│   (count) │           │            │           │            │           │            │           │
│    Memory │  1.57e+11 │   1.54e+11 │  1.58e+11 │   1.58e+11 │  1.58e+11 │   1.57e+11 │  1.63e+09 │
│     Usage │           │            │           │            │           │            │           │
│     Bytes │           │            │           │            │           │            │           │
│      (GB) │           │            │           │            │           │            │           │
│   Process │      0.53 │       0.53 │      0.53 │       0.53 │      0.53 │       0.53 │      0.00 │
│ CPU Usage │           │            │           │            │           │            │           │
│   Percent │           │            │           │            │           │            │           │
│       (%) │           │            │           │            │           │            │           │
└───────────┴───────────┴────────────┴───────────┴────────────┴───────────┴────────────┴───────────┘

                   127.0.0.1:39827/server2 | ai-server | server-1:8080
[... similar table for server2 ...]
```

**Features:**
- Formatted metric names (acronyms capitalized: CPU, GPU, HTTP, KV, etc.)
- Automatic unit inference and display
- Full statistical distribution
- Server metadata in title
- Rich colors and formatting

## Use Cases

### 1. Performance Debugging

**Scenario**: Benchmark shows increasing latency over time.

**Investigation**:
1. Check JSONL export for time-series correlation
2. Look for:
   - Rising `memory_usage_bytes` (memory leak?)
   - Increasing `http_requests_in_flight` (saturation?)
   - High `dynamo_component_kvstats_gpu_cache_usage_percent` (cache pressure?)
   - Growing `dynamo_frontend_queued_requests` (backlog building up?)

```python
import pandas as pd
import json

# Load server metrics time series
with open('server_metrics_export.jsonl') as f:
    records = [json.loads(line) for line in f]

df = pd.DataFrame([
    {
        'timestamp': r['timestamp_ns'],
        'memory_usage_gb': r['metrics_data']['memory_usage_bytes'] / 1e9,
        'requests_in_flight': r['metrics_data']['http_requests_in_flight'],
        'cache_usage_pct': r['metrics_data']['dynamo_component_kvstats_gpu_cache_usage_percent']
    }
    for r in records
])

# Plot correlation with request latencies
df.plot(x='timestamp', y=['memory_usage_gb', 'requests_in_flight', 'cache_usage_pct'])
```

### 2. Capacity Planning

**Scenario**: Need to determine if current server can handle increased load.

**Analysis**:
```python
# Load aggregate statistics
with open('aiperf.json') as f:
    results = json.load(f)

server_metrics = results['server_metrics']['endpoints']['http://server:8000/metrics']['servers']['server-0']['metrics']

# Check resource headroom
cpu_usage = server_metrics['process_cpu_usage_percent']['avg']  # 0.53 (53%)
memory_usage = server_metrics['memory_usage_bytes']['avg']       # 157GB
memory_total = server_metrics['memory_total_bytes']['avg']       # 275GB
cache_usage = server_metrics['dynamo_component_kvstats_gpu_cache_usage_percent']['avg']  # 0.38 (38%)

print(f"CPU Headroom: {100 - cpu_usage*100:.1f}%")
print(f"Memory Headroom: {(memory_total - memory_usage) / 1e9:.1f} GB")
print(f"Cache Headroom: {100 - cache_usage*100:.1f}%")

# Estimate capacity for increased load
current_rps = results['llm_metrics']['request_throughput']['avg']
estimated_max_rps = current_rps / min(cpu_usage, cache_usage/100)
print(f"Estimated max RPS: {estimated_max_rps:.1f}")
```

### 3. A/B Testing Server Configurations

**Scenario**: Compare two different server configurations.

```bash
# Run benchmark with config A
aiperf profile \
    --url http://server-a:8000 \
    --model llama-3.1 \
    --request-count 1000 \
    --profile-export-prefix config_a

# Run benchmark with config B
aiperf profile \
    --url http://server-b:8000 \
    --model llama-3.1 \
    --request-count 1000 \
    --profile-export-prefix config_b
```

**Comparison**:
```python
import json

with open('config_a_aiperf.json') as f:
    config_a = json.load(f)
with open('config_b_aiperf.json') as f:
    config_b = json.load(f)

# Compare key metrics
metrics_to_compare = [
    'dynamo_component_kvstats_gpu_cache_usage_percent',
    'dynamo_frontend_time_to_first_token_seconds',
    'process_cpu_usage_percent',
    'memory_usage_bytes'
]

for metric in metrics_to_compare:
    a_val = config_a['server_metrics']['endpoints']['...']['servers']['...']['metrics'][metric]['avg']
    b_val = config_b['server_metrics']['endpoints']['...']['servers']['...']['metrics'][metric]['avg']
    improvement = ((a_val - b_val) / a_val) * 100
    print(f"{metric}: Config A={a_val:.2f}, Config B={b_val:.2f}, Δ={improvement:+.1f}%")
```

### 4. Multi-Server Load Distribution Analysis

**Scenario**: Running multiple server instances, want to verify balanced load.

```python
import json

with open('aiperf.json') as f:
    results = json.load(f)

servers = results['server_metrics']['endpoints']

for endpoint_url, endpoint_data in servers.items():
    for server_id, server_data in endpoint_data['servers'].items():
        rps = server_data['metrics']['http_requests']['avg']
        cpu = server_data['metrics']['process_cpu_usage_percent']['avg']
        cache = server_data['metrics']['dynamo_component_kvstats_gpu_cache_usage_percent']['avg']

        print(f"{server_id}: RPS={rps:.1f}, CPU={cpu*100:.1f}%, Cache={cache*100:.1f}%")

# Expected output:
# server-0: RPS=157.7, CPU=53.0%, Cache=38.0%
# server-1: RPS=145.0, CPU=51.0%, Cache=35.0%
# server-2: RPS=132.3, CPU=47.5%, Cache=32.0%
```

## Advanced Configuration

### Custom Metrics Collection Interval

```python
# In user code or config file
import os
os.environ['AIPERF_SERVER_METRICS_COLLECTION_INTERVAL'] = '2.0'  # 2 seconds
```

### Disable Automatic Collection

```python
# Set empty list for default endpoints
os.environ['AIPERF_SERVER_METRICS_DEFAULT_ENDPOINTS'] = ''
```

### Custom Metrics Endpoint Path

```bash
aiperf profile \
    --url http://server:8000 \
    --server-metrics http://server:9090/custom-metrics \
    --model nvidia/llama-3.1-nemotron-70b-instruct
```

## Implementation Details

### Dynamic Field Discovery

Server metrics use **no hardcoded metric lists**. All metrics are discovered dynamically:

```python
# Pseudo-code of dynamic discovery in BaseMetricsConsoleExporter
def _create_metrics_table(resource_data):
    # Get first snapshot to discover available metrics
    first_snapshot = resource_data.time_series.snapshots[0]

    # Discover all scalar metrics from actual data
    metric_names = sorted(first_snapshot.metrics.keys())

    for metric_name in metric_names:
        display_name = format_metric_display_name(metric_name)  # e.g., "http_requests_in_flight" → "HTTP Requests In Flight"
        unit_enum = infer_metric_unit(metric_name)              # e.g., "bytes" → MetricUnit.BYTES

        # Add to table with statistics
        table.add_row(display_name, avg, min, max, p99, p90, p50, std, unit)

    # Discover histogram metrics (if any)
    if hasattr(first_snapshot, 'histograms'):
        for histogram_name in sorted(first_snapshot.histograms.keys()):
            # Add histogram rows...
```

**Benefits:**
- New metrics automatically exported
- No code changes needed for custom server implementations
- Prometheus-compatible with any `/metrics` endpoint

### Metric Name Formatting

```python
# src/aiperf/common/metrics/metrics_display_utils.py

METRIC_NAME_ACRONYMS = {
    "api", "cpu", "d2d", "d2h", "dcgm", "e2e", "fb", "fds",
    "gpu", "h2d", "http", "id", "kv", "kvbm", "sm", "ttft",
    "url", "uuid", "vllm", "xid",
}

def format_metric_display_name(field_name: str) -> str:
    """
    Convert: "dynamo_component_kvstats_gpu_cache_usage_percent"
    To:      "Dynamo Component Kvstats GPU Cache Usage Percent"
    """
    return " | ".join(
        " ".join(
            word.upper() if word.lower() in METRIC_NAME_ACRONYMS else word.capitalize()
            for word in key.split("_")
        )
        for key in field_name.split(":")
    )
```

### Unit Inference

```python
def infer_metric_unit(field_name: str) -> MetricUnitT:
    """Infer unit from metric name patterns."""
    name_lower = field_name.lower()

    # GPU-specific patterns
    if "gpu" in name_lower and "cache" in name_lower and "usage" in name_lower:
        return MetricUnit.PERCENTAGE

    # General patterns
    if name_lower.endswith("_bytes"):
        return MetricUnit.BYTES
    if name_lower.endswith("_seconds"):
        return MetricUnit.SECONDS
    if name_lower.endswith("_percent") or "usage" in name_lower:
        return MetricUnit.PERCENTAGE
    if "count" in name_lower or name_lower.endswith("_total"):
        return MetricUnit.COUNT

    return MetricUnit.COUNT  # Default
```

## Troubleshooting

### No Server Metrics Collected

**Symptoms:**
- Console shows "0/3 Server endpoints reachable"
- No `server_metrics_export.jsonl` file created

**Possible Causes:**

1. **Server doesn't expose `/metrics` endpoint**
   ```bash
   # Test manually
   curl http://your-server:8000/metrics
   ```

   **Solution**: Ensure server has Prometheus metrics endpoint enabled.

2. **Firewall/network blocking**
   ```bash
   # Test connectivity
   telnet your-server 8000
   ```

   **Solution**: Check network access, firewall rules, Docker networking.

3. **Incorrect URL**
   ```bash
   # AIPerf automatically appends /metrics
   # If your endpoint is http://server:8000/custom-metrics, specify:
   aiperf profile --server-metrics http://server:8000/custom-metrics
   ```

### Metrics Not in CSV/JSON

**Symptoms:**
- JSONL contains metrics but CSV/JSON missing them

**Cause**: Metrics flagged as `EXPERIMENTAL` or `INTERNAL` are filtered from exports.

**Solution**: Check metric registry flags in codebase.

### High Memory Usage

**Symptoms:**
- AIPerf process using excessive memory during long benchmarks

**Cause**: Large number of metric records buffered in memory.

**Solution**:
- Increase `AIPERF_SERVER_METRICS_COLLECTION_INTERVAL` for longer benchmarks
- Process JSONL file in streaming fashion rather than loading all at once

### Duplicate Metrics

**Symptoms:**
- Same metric appears multiple times with slightly different names

**Cause**: Server exposes metrics with labels that get flattened to different names.

**Example**:
```
http_request_duration_seconds{quantile="0.5"} 0.1
http_request_duration_seconds{quantile="0.9"} 0.5
http_request_duration_seconds{quantile="0.99"} 1.2
```

These become: `http_request_duration_seconds_quantile_0_5`, `http_request_duration_seconds_quantile_0_9`, etc.

**Solution**: This is expected behavior. Each label combination is treated as a separate metric.

## Performance Impact

Server metrics collection has minimal impact on benchmark accuracy:

- **Network Overhead**: ~1 HTTP GET per 0.5s per endpoint (typically <1KB response)
- **CPU Overhead**: ~0.1-0.2% additional CPU for parsing and processing
- **Memory Overhead**: ~10-50 MB per 10,000 records
- **Latency Impact**: Collection happens asynchronously, no blocking of request path

**Recommendation**: Leave server metrics collection enabled by default for full observability.

## Best Practices

1. **Always Use `--server-metrics` Flag for Important Benchmarks**
   - Enables console display for immediate feedback
   - Critical for debugging performance issues

2. **Monitor KV Cache Metrics for vLLM/Dynamo Servers**
   - `dynamo_component_kvstats_gpu_cache_usage_percent` - cache pressure
   - `dynamo_component_kvstats_active_blocks` - active memory blocks
   - High cache usage (>90%) can cause performance degradation

3. **Correlate Server and Client Metrics**
   - Plot `time_to_first_token` vs `dynamo_frontend_time_to_first_token_seconds`
   - Compare `http_requests_in_flight` with request concurrency
   - Check if server saturation correlates with latency spikes

4. **Use JSONL for Time-Series Analysis**
   - CSV/JSON are aggregated summaries
   - JSONL preserves full temporal information
   - Essential for understanding performance over time

5. **Export to Monitoring Systems**
   ```python
   # Example: Export to Prometheus Pushgateway
   import json
   import requests

   with open('server_metrics_export.jsonl') as f:
       for line in f:
           record = json.loads(line)
           metrics_data = record['metrics_data']

           # Push to Prometheus
           for metric_name, value in metrics_data.items():
               requests.post(
                   'http://pushgateway:9091/metrics/job/aiperf',
                   data=f'{metric_name} {value}\n'
               )
   ```

6. **Archive Benchmark Results with Server Metrics**
   - Include `server_metrics_export.jsonl` in artifact archives
   - Critical for historical performance analysis
   - Enables future investigation of incidents

## Architecture Decision Records (ADRs)

### Why Dynamic Field Discovery?

**Decision**: Metrics are discovered dynamically from server responses rather than using hardcoded lists.

**Rationale**:
- Different server implementations expose different metrics
- New metrics added by servers are automatically supported
- Custom/proprietary servers work without code changes
- Reduces maintenance burden (no hardcoded mappings to update)

**Trade-off**: Cannot validate metric existence upfront, but this is acceptable since metrics are best-effort observations.

### Why Three Export Formats?

**Decision**: Export to JSONL (raw), CSV (summary), and JSON (hierarchical).

**Rationale**:
- **JSONL**: Time-series analysis, streaming processing, debugging
- **CSV**: Human-readable, Excel-compatible, quick inspection
- **JSON**: Programmatic access, CI/CD integration, dashboards

Each format serves distinct use cases. Redundancy is intentional for usability.

### Why Automatic Collection?

**Decision**: Always attempt server metrics collection from inference endpoint, even without flag.

**Rationale**:
- Low overhead (~0.1% CPU, minimal network)
- Critical for post-hoc debugging (can't recreate production issues)
- Matches behavior of GPU telemetry (also automatic)
- Console display opt-in via flag prevents information overload

**Trade-off**: Slightly more network traffic, but negligible impact in practice.

## Future Enhancements

- **Metric Correlation Analysis**: Automatic correlation between server metrics and request latencies
- **Anomaly Detection**: Statistical outlier detection in server metric time series
- **Visualization Dashboard**: Real-time charting of server metrics during benchmark
- **Custom Metric Collectors**: Plugin system for non-Prometheus servers
- **Differential Analysis**: Built-in A/B comparison of server configurations

## References

- [Prometheus Metric Types](https://prometheus.io/docs/concepts/metric_types/)
- [vLLM Metrics Documentation](https://docs.vllm.ai/en/latest/serving/metrics.html)
- [OpenMetrics Specification](https://openmetrics.io/)
- [AIPerf Base Classes](../src/aiperf/common/metrics/)
- [Integration Tests](../tests/integration/test_server_metrics_collection.py)
