<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Feature Request: Move Data Exporters to Records Manager

## Summary

Move data exporters from SystemController to RecordsManager to provide direct access to raw accumulator data, enabling more powerful export capabilities including enhanced Parquet export functionality.

## Motivation

### Current Architecture

Data exporters currently operate in a two-tier system:

1. **In RecordsManager** (privileged access):
   - Parquet exporter runs directly with access to raw `ServerMetricsAccumulator`
   - Has access to full time-series data, individual timestamps, bucket distributions

2. **In SystemController** (aggregated data only):
   - JSON and CSV exporters receive `ServerMetricsResults` via ZMQ
   - Only have access to pre-computed statistics (p50, p99, avg, etc.)
   - Cannot access raw time-series or per-timestamp data

### The Problem

The Parquet exporter was placed in RecordsManager out of necessity - the accumulator object is not serializable through ZMQ. This created an architectural inconsistency:

| Exporter | Location | Data Access |
|----------|----------|-------------|
| Parquet | RecordsManager | Raw accumulator (full time-series) |
| JSON | SystemController | Aggregated statistics only |
| CSV | SystemController | Aggregated statistics only |

This split causes several issues:

1. **Inconsistent capabilities**: Parquet can export raw data; JSON/CSV cannot
2. **Limited flexibility**: New export formats requiring raw data must be specially handled
3. **Code duplication**: Export logic scattered across two services
4. **Testing complexity**: Different data paths for different exporters

### Why Raw Data Access Matters

The Parquet exporter demonstrates the value of raw data access. It currently has access to:

- Raw timestamps (nanosecond precision)
- Individual metric values at each scrape interval
- Per-bucket histogram distributions
- Delta calculations from reference points
- Complete label sets for all metrics

Other exporters could benefit from similar access for:
- Time-series JSON export for visualization tools
- Raw CSV for spreadsheet analysis
- Custom aggregation windows (not just full benchmark duration)
- Detailed histogram bucket data in any format

## Proposed Solution

### Move All Data Exporters to RecordsManager

Consolidate export functionality in RecordsManager where raw data is available:

```
Before:
┌─────────────────────┐     ZMQ (aggregated)    ┌──────────────────────┐
│   RecordsManager    │ ─────────────────────── │   SystemController   │
│                     │                         │                      │
│ - Accumulator       │                         │ - ExporterManager    │
│ - Parquet Export    │                         │ - JSON/CSV Exporters │
└─────────────────────┘                         └──────────────────────┘

After:
┌──────────────────────────────────────────────┐
│              RecordsManager                  │
│                                              │
│ - ServerMetricsAccumulator                   │
│ - GPUTelemetryAccumulator                    │
│ - ExporterManager                            │
│   - Parquet Exporter (raw data)              │
│   - JSON Exporter (raw or aggregated)        │
│   - CSV Exporter (raw or aggregated)         │
│   - Future exporters (raw data)              │
└──────────────────────────────────────────────┘
```

### Implementation Steps

1. **Move ExporterManager to RecordsManager**
   - Relocate `exporter_manager.py` logic to RecordsManager service
   - Update factory instantiation to pass accumulator objects directly

2. **Update ExporterConfig**
   - Add optional accumulator references for raw data access.

   Current implementation:
   ```python
   @dataclass(slots=True)
   class ExporterConfig:
       results: ProfileResults | None
       user_config: UserConfig
       service_config: ServiceConfig | None
       telemetry_results: TelemetryExportData | None
       server_metrics_results: ServerMetricsResults | None = None
   ```

   Proposed addition:
   ```python
   @dataclass(slots=True)
   class ExporterConfig:
       results: ProfileResults | None
       user_config: UserConfig
       service_config: ServiceConfig | None
       telemetry_results: TelemetryExportData | None
       server_metrics_results: ServerMetricsResults | None = None
       # New fields for raw data access
       server_metrics_accumulator: ServerMetricsAccumulatorProtocol | None = None
       gpu_telemetry_accumulator: GPUTelemetryAccumulatorProtocol | None = None
   ```

3. **Unify Parquet Export with Other Exporters**
   - Move `ServerMetricsParquetExporter` to use `DataExporterFactory`
   - Use existing `DataExporterType.SERVER_METRICS_PARQUET` registration
   - Remove special-case handling in accumulator's `_export_parquet_if_enabled()`

4. **Enhance JSON/CSV Exporters (optional)**
   - Add option for raw time-series export mode
   - Support custom aggregation windows
   - Include histogram bucket details

5. **Update SystemController**
   - Remove export handling from SystemController
   - SystemController receives export completion notification via ZMQ

### File Changes

| File | Change |
|------|--------|
| `records/records_manager.py` | Add ExporterManager instantiation and execution |
| `exporters/exporter_config.py` | Add accumulator fields |
| `exporters/exporter_manager.py` | Update to accept accumulators |
| `server_metrics/parquet_exporter.py` | Refactor to use DataExporterFactory pattern |
| `server_metrics/accumulator.py` | Remove `_export_parquet_if_enabled()` |
| `controller/system_controller.py` | Remove export handling |
| `common/enums/data_exporter_enums.py` | No changes needed (SERVER_METRICS_PARQUET already exists) |

## Benefits

1. **Unified export architecture**: All exporters in one location with consistent data access
2. **Raw data for all formats**: JSON/CSV can export time-series when needed
3. **Simplified code**: Remove cross-process data serialization for exports
4. **Better extensibility**: New exporters automatically get raw data access
5. **Improved Parquet exporter**: Follows standard factory pattern, easier to maintain

## Considerations

### Memory Usage

Raw accumulators hold full time-series in memory. This is already the case for Parquet export. Moving other exporters doesn't increase memory usage since they would share the same accumulator.

### Export Timing

Exports should occur after all records are processed but before accumulators are cleared. Current RecordsManager lifecycle supports this in `_process_server_metrics_results()`.

### Backward Compatibility

- Aggregated results still computed for console output and summary
- JSON/CSV exporters default to current aggregated output
- Raw export mode is opt-in via configuration

## Alternatives Considered

### 1. Serialize Accumulator to ZMQ

Rejected because:
- Accumulator contains numpy arrays and complex nested structures
- Serialization would be expensive and error-prone
- Would duplicate data in memory during transfer

### 2. Stream Raw Data via ZMQ

Rejected because:
- High volume of messages for large benchmarks
- Adds complexity without clear benefit
- Still requires reassembly in SystemController

### 3. Keep Split Architecture

Current state - suboptimal due to inconsistent capabilities and code organization.

## Related Files

- `src/aiperf/records/records_manager.py` - RecordsManager service
- `src/aiperf/exporters/` - Current exporter implementations
- `src/aiperf/server_metrics/parquet_exporter.py` - Parquet exporter with raw access
- `src/aiperf/server_metrics/accumulator.py` - Server metrics accumulator
- `src/aiperf/controller/system_controller.py` - SystemController (current export location)
- `src/aiperf/common/factories.py` - DataExporterFactory
- `src/aiperf/common/protocols.py` - Protocol definitions for accumulators

## Open Questions

1. Should raw export mode be enabled by default or require explicit opt-in?
2. What naming convention for raw vs aggregated exports? (e.g., `metrics.json` vs `metrics_raw.json`)
3. Should GPU telemetry exporters follow the same pattern?

## Priority

Medium - This is a code quality and extensibility improvement that enables future export enhancements.
ga
