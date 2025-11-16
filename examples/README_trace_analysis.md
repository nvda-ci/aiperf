<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Trace Data Analysis Tool

Analyze detailed HTTP trace data from AIPerf benchmark results.

## Usage

```bash
python examples/analyze_trace_data.py results/profile_export.jsonl
```

## What It Analyzes

The script provides comprehensive statistics on:

### Connection Performance
- **Connection reuse rate** - How often connections are reused vs created
- **Connection establishment time** - Time to create new connections (TCP handshake, TLS)
- **Queue wait time** - Time spent waiting in connection pool

### DNS Performance
- **DNS cache hit rate** - Percentage of DNS lookups served from cache
- **DNS resolution time** - Time to resolve hostnames (when not cached)

### Request Timing
- **Request send time** - Time to transmit request to server
- **TTFB (Time to First Byte)** - Server processing + network latency
- **TTLB (Time to Last Byte)** - Total response time

### Streaming Performance (for SSE/streaming responses)
- **Chunks per request** - Number of chunks received
- **Inter-chunk latency (ICL)** - Time between consecutive chunks
- **Chunk size distribution** - Size of individual response chunks

### Bandwidth
- **Request/response sizes** - Data transferred
- **Transfer efficiency** - Network transfer time vs server processing

### Token Analytics (NEW!)
- **Input/output token counts** - Token statistics with percentiles
- **Token throughput** - Tokens generated per second (min/max/p50/p95/p99)
- **Time per token** - Average generation time per output token
- **Bytes per token** - Network efficiency (payload size per token)
- **Chunks per token** - Streaming granularity (SSE chunks per token)

### Token-to-Response Correlation (NEW!)
- **Token generation performance** - Overall tokens/sec and time/token
- **Streaming characteristics** - How tokens map to chunks
- **Network efficiency per token** - Data transfer cost per token
- **Tokens per chunk** - Content density in streaming responses

## Example Output

```
================================================================================
  OVERVIEW
================================================================================
  Total records:                        1,000
  Records with trace data:              1,000 (100.0%)
  Streaming requests:                     850
  Non-streaming requests:                 150

================================================================================
  CONNECTION STATISTICS
================================================================================
  Connections created:                     25
  Connections reused:                     975
  Connection reuse rate:                 97.5%

Connection Establishment Time:
  Metric                         Value
  ------------------------------ ---------------
  Min                              1.23ms
  Max                             45.67ms
  Mean                             8.42ms
  Median (p50)                     7.11ms
  p90                             15.23ms
  p95                             22.45ms
  p99                             38.90ms

================================================================================
  DNS STATISTICS
================================================================================
  DNS resolutions performed:              10
  DNS cache hits:                         990
  DNS cache hit rate:                   99.0%

================================================================================
  RESPONSE TIMING - TIME TO FIRST BYTE (TTFB)
================================================================================
TTFB (includes network + server processing):
  Metric                         Value
  ------------------------------ ---------------
  Min                             12.34ms
  Max                            234.56ms
  Mean                            45.67ms
  Median (p50)                    42.11ms
  p90                             78.90ms
  p95                             98.45ms
  p99                            156.78ms

================================================================================
  STREAMING STATISTICS
================================================================================
Chunks per Request:
  Metric                         Value
  ------------------------------ ---------------
  Min                                  5
  Max                                150
  Mean                                42
  Median                              38

Inter-Chunk Latency (ICL):
  Metric                         Value
  ------------------------------ ---------------
  Min                              5.23ms
  Max                            125.67ms
  Mean                            28.42ms
  Median (p50)                    25.11ms
  p90                             45.23ms
  p95                             62.45ms
  p99                             98.90ms

================================================================================
  TOKEN ANALYTICS
================================================================================

Input Tokens:
  Metric                         Value
  ------------------------------ ---------------
  Min                                    64
  Max                                   512
  Mean                                  240
  Median                              192.00
  Total                                 960

Output Tokens:
  Metric                         Value
  ------------------------------ ---------------
  Min                                   128
  Max                                  1024
  Mean                                  480
  Median                              384.00
  Total                                1920

Token Throughput (tokens/second):
  Metric                         Value
  ------------------------------ ---------------
  Min                              1280.00
  Max                              4551.11
  Mean                             2666.67
  Median (p50)                     3413.33
  p90                              4551.11
  p95                              4551.11
  p99                              4551.11

Time per Output Token:
  Metric                         Value
  ------------------------------ ---------------
  Min                              219.73μs
  Max                              781.25μs
  Mean                             499.27μs
  Median (p50)                     703.12μs
  p90                              781.25μs
  p95                              781.25μs
  p99                              781.25μs

Bytes per Token (network efficiency):
  Metric                         Value
  ------------------------------ ---------------
  Min                                 5.0B
  Max                                32.0B
  Mean                               14.75B
  Median                             11.0B

================================================================================
  TOKEN-TO-RESPONSE CORRELATION
================================================================================

  Token Generation Performance:
    Avg tokens/second:                 2666.67
    Avg time/token:                   499.27μs

  Streaming Characteristics:
    Avg output tokens:                  480.00
    Avg inter-chunk latency:           21.67ms
    Avg chunks per token:                0.009
    Avg tokens per chunk:               117.03

  Network Efficiency:
    Avg bytes per token:                14.75B
    Total response bytes:              17.00KB
    Total output tokens:                 1,920
    Overall bytes/token:                 9.07B

================================================================================
  NETWORK EFFICIENCY INSIGHTS
================================================================================

  Average breakdown:
    Server processing + TTFB:           45.67ms
    Network transfer time:              32.15ms
    Transfer/Processing ratio:            0.70x
```

## Key Insights to Look For

### Network & Connection Health
1. **High connection reuse** (>90%) = Good connection pooling
2. **High DNS cache hit rate** (>95%) = Efficient DNS caching
3. **Low TTFB** = Fast server or good network
4. **Consistent ICL** (low p99/p50 ratio) = Stable streaming
5. **Transfer/Processing ratio**:
   - `< 0.5x` = Server-bound (slow generation)
   - `> 2.0x` = Network-bound (slow connection)

### Token Performance (NEW!)
6. **Token throughput** (tokens/second):
   - `> 100 tok/s` = Fast generation
   - `< 20 tok/s` = Slow generation (potential bottleneck)
   - Check p95/p99 for tail latency issues
7. **Time per token**:
   - `< 10ms` = Excellent
   - `10-50ms` = Good
   - `> 100ms` = Needs investigation
8. **Bytes per token**:
   - Lower is better (more efficient encoding)
   - Typical: 3-8 bytes/token for efficient APIs
   - `> 20 bytes/token` = Inefficient payload or includes metadata
9. **Tokens per chunk**:
   - Higher = More tokens per network round-trip (better efficiency)
   - `< 1.0` = Very fine-grained streaming (may have overhead)
   - `> 10` = Coarse-grained batching (better throughput, higher latency)

## Integration with AIPerf

This tool works with any AIPerf benchmark that has:
- `export_level: records` or `export_level: raw` in the config
- HTTP transport with aiohttp trace enabled (default)

The trace data is automatically captured and included in `profile_export.jsonl`.
