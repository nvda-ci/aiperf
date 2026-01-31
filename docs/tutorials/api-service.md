<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# API Service

The `APIService` is a unified HTTP + WebSocket server that exposes benchmark metrics and real-time streaming on a single port.

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--api-port` | None (disabled) | AIPerf API port (enables HTTP + WebSocket endpoints) |
| `--api-host` | `127.0.0.1` | AIPerf API host (requires `--api-port`) |

**Example:**
```bash
aiperf --api-port 9090
aiperf --api-port 9090 --api-host 0.0.0.0  # expose externally
```

## HTTP Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | API documentation page |
| `GET /dashboard` | Live metrics dashboard UI |
| `GET /metrics` | Prometheus-format metrics (for scrapers) |
| `GET /api/metrics` | JSON metrics |
| `GET /api/status` | Benchmark state/phase |
| `GET /api/progress` | Request completion progress |
| `GET /api/workers` | Worker health status |
| `GET /api/config` | Benchmark configuration |
| `GET /health` | Health check |

## WebSocket Endpoint

| Endpoint | Description |
|----------|-------------|
| `WS /ws` | Real-time ZMQ message stream with dynamic subscriptions |

## Key Features

- **Dual export**: Prometheus `/metrics` for scraping + JSON `/api/metrics` for programmatic access
- **Real-time streaming**: WebSocket clients can subscribe to specific ZMQ message types (e.g., `realtime_metrics`, `worker_status_summary`)
- **Dynamic subscriptions**: Clients send `{"type": "subscribe", "message_types": [...]}` to receive filtered messages
- **Dashboard UI**: Built-in `/dashboard` page for live visualization

## WebSocket Protocol

### Subscribe to message types

```json
{"type": "subscribe", "message_types": ["realtime_metrics", "worker_status_summary"]}
```

Response:
```json
{"type": "subscribed", "message_types": ["realtime_metrics", "worker_status_summary"]}
```

Use `"*"` to subscribe to all message types.

### Unsubscribe

```json
{"type": "unsubscribe", "message_types": ["realtime_metrics"]}
```

### Ping/Pong

```json
{"type": "ping"}
```

Response:
```json
{"type": "pong"}
```

## Supported Message Types

The following message types have built-in handlers and are forwarded to WebSocket clients:

- `realtime_metrics` - Live benchmark metrics (filtered and converted to display units)
- `realtime_telemetry_metrics` - GPU telemetry data
- `credit_phase_start` - Phase start events
- `credit_phase_progress` - Phase progress updates
- `credit_phase_complete` - Phase completion events
- `worker_status_summary` - Worker health status
- `processing_stats` - Record processing statistics
- `all_records_received` - Benchmark completion signal

Other valid `MessageType` values can be subscribed to dynamically.
