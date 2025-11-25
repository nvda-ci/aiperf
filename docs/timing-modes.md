# AIPerf Timing Modes Guide

<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Overview

AIPerf provides flexible timing control for benchmarking LLM endpoints through two main timing modes: **REQUEST_RATE** and **FIXED_SCHEDULE**. These modes determine how and when requests are sent to the endpoint under test.

**Key Concepts:**
- **Timing Mode**: Top-level strategy for issuing credits (permissions to send requests)
- **Credit**: A permission token to make a single request (turn) to an LLM endpoint
- **Turn**: A single interaction in a conversation (user sends message, receives response)
- **Conversation**: A sequence of turns between a user and an LLM (single or multi-turn)

**Important**: The timing mode is **automatically selected** based on your configuration options—you don't specify it directly. The selection follows this priority:

1. **FIXED_SCHEDULE** if:
   - `--fixed-schedule` flag is set, OR
   - `mooncake_trace` custom dataset type is used with timestamps in the file

2. **REQUEST_RATE** if:
   - `--request-rate` is specified

3. **Default**: REQUEST_RATE with CONCURRENCY_BURST mode (automatically sets `--concurrency 1` if not provided)

---

## Timing Modes

### REQUEST_RATE Mode

**Description**: Issues 1 credit per turn with smart routing and sticky credit support for subsequent turns. This mode is ideal for synthetic load generation with precise control over request patterns.

#### Request Rate Sub-Modes

REQUEST_RATE mode has three sub-modes controlled by `--request-rate-mode`:

##### CONSTANT (Fixed Inter-Arrival Times)

**Behavior**: Generates requests at a fixed rate with consistent intervals between requests.

**Requirements**:
- `--request-rate` must be set and `> 0`

**Calculation**: `interval = 1 / request_rate` seconds

**Use Case**: Consistent, predictable load for baseline performance testing

**Example**:
```bash
aiperf \
  --endpoint-type openai \
  --model gpt-4 \
  --request-rate 10 \
  --request-rate-mode constant \
  --request-count 100
```
Sends exactly 100ms between each request (10 req/s).

##### POISSON (Exponentially Distributed Inter-Arrival Times) ⭐ DEFAULT

**Behavior**: Generates requests using a Poisson process with exponentially distributed inter-arrival times.

**Requirements**:
- `--request-rate` must be set and `> 0`

**Calculation**: `interval ~ Exponential(request_rate)`

**Use Case**: Models realistic random traffic patterns where requests arrive randomly but at a consistent average rate

**Example**:
```bash
aiperf \
  --endpoint-type openai \
  --model gpt-4 \
  --request-rate 10 \
  --request-count 100
```
Averages 10 req/s but with natural randomness in timing.

##### CONCURRENCY_BURST (Send as Fast as Possible)

**Behavior**: Sends requests as fast as possible, limited only by the max concurrency setting. No delay between requests.

**Requirements**:
- `--concurrency` must be set and `>= 1`
- `--request-rate` must NOT be specified

**Use Case**: Maximum load testing, stress testing up to a concurrency limit

**Example**:
```bash
aiperf \
  --endpoint-type openai \
  --model gpt-4 \
  --concurrency 50 \
  --request-count 100
```
Bursts up to 50 concurrent requests as fast as the endpoint can handle them.

#### Multi-Turn Conversation Behavior

The REQUEST_RATE strategy handles multi-turn conversations differently based on whether concurrency is configured. This is a **critical implementation detail** that affects conversation timing.

##### WITH Concurrency (Conversation-Centric Mode)

**Behavior**:
- Semaphore slot acquired when conversation starts
- Slot held for **entire conversation** (not released between turns)
- Subsequent turns **respect `delay_ms`** from dataset
- Max N conversations in flight simultaneously

**When to use**:
- Model realistic user behavior with think time
- Preserve conversation timing patterns from dataset
- Ensure conversations don't exceed concurrency limit

**Example**:
```bash
aiperf \
  --endpoint-type openai \
  --model gpt-4 \
  --request-rate 10 \
  --request-rate-mode poisson \
  --concurrency 20 \
  --input-file multi_turn_dataset.jsonl \
  --custom-dataset-type multi_turn
```

**Timeline**:
```
Dataset: Conversation with 2 turns, delay_ms=100 between them
Request rate: 10 req/s (100ms average intervals)

t=0ms:   Send turn0 (acquire semaphore slot)
t=20ms:  Turn0 returns, schedule turn1 with 100ms delay
t=120ms: Send turn1 (delay respected, semaphore held for 120ms total)
Result: Delay respected, conversation treated as atomic unit
```

##### WITHOUT Concurrency (Rate-Centric Mode)

**Behavior**:
- No semaphore, request rate strictly enforced
- Subsequent turns **queued** and sent at next rate interval
- **`delay_ms` IGNORED** to maintain precise request rate
- ⚠️ Warning logged when `delay_ms > 0` is ignored

**When to use**:
- Stress testing at exact rate
- Don't care about conversation timing patterns
- Prioritize rate precision over realism

**Example**:
```bash
aiperf \
  --endpoint-type openai \
  --model gpt-4 \
  --request-rate 10 \
  --request-rate-mode constant \
  --input-file multi_turn_dataset.jsonl \
  --custom-dataset-type multi_turn
```

**Timeline**:
```
Dataset: Conversation with 2 turns, delay_ms=100 between them
Request rate: 10 req/s (100ms fixed intervals)

t=0ms:   Send turn0
t=20ms:  Turn0 returns, queue turn1
t=100ms: Main loop sends turn1 (at rate interval, delay ignored)
Result: Rate maintained at exactly 10 req/s, delays not respected
```

⚠️ **Warning Example**: `Conv conv_123 turn 1: delay_ms=100 ignored (rate-centric mode)`

#### Advanced: Combining Concurrency and Request Rate

**💡 Hidden Feature**: You CAN specify both `--concurrency` and `--request-rate` together!

**Behavior**:
- Semaphore limits max concurrent requests
- Rate generator controls timing of new requests
- Both constraints applied simultaneously

**Use Case**: "Send at 100 req/s but never exceed 50 concurrent requests"

**Example**:
```bash
aiperf \
  --endpoint-type openai \
  --model gpt-4 \
  --request-rate 100 \
  --request-rate-mode poisson \
  --concurrency 50 \
  --request-count 1000
```

This configuration:
- Attempts to send ~100 req/s (Poisson distribution)
- But never exceeds 50 concurrent requests
- Useful for rate-limited endpoints with concurrency limits

**Code Reference**: `src/aiperf/timing/request_rate_strategy.py:168-171`

---

### FIXED_SCHEDULE Mode

**Description**: Replays multi-turn conversations with precise timing from trace data. Ideal for trace replay scenarios where you want to reproduce real-world workload patterns.

#### Architecture

FIXED_SCHEDULE uses a **dual-queue architecture**:

1. **Absolute Schedule**: First turns with absolute timestamps
   - Sent sequentially by main loop at precise times
   - Example: `{0: [conv1_turn0], 100: [conv2_turn0, conv3_turn0]}`

2. **Pending Queue**: Subsequent turns per conversation
   - Sent dynamically when previous turn completes
   - Example: `{"conv1": deque([turn1, turn2]), "conv2": deque([turn1])}`

```
┌─────────────────────┐     ┌──────────────────────────────┐
│ Absolute Schedule   │     │ Pending Queue                │
│ =================== │     │ =========================    │
│ {                   │     │ {                            │
│   0ms:  [C1_T0]     │     │   "conv1": [C1_T1, C1_T2]    │
│   50ms: [C2_T0]     │     │   "conv2": [C2_T1]           │
│   100ms:[C3_T0]     │     │   "conv3": [C3_T1]           │
│ }                   │     │ }                            │
└─────────────────────┘     └──────────────────────────────┘
```

#### Timing Modes

Fixed schedule supports two timing modes for turns:

##### Absolute Timestamps (`timestamp_ms`)

**Description**: Turn scheduled at specific time relative to schedule start

**Use Case**: Replaying exact trace timing

**Example**: Turn at `timestamp_ms=1500` means "send 1500ms after schedule starts"

**Requirements**:
- **All first turns (turn_index=0) MUST have `timestamp_ms`**
- Subsequent turns CAN use `timestamp_ms`

##### Inter-Turn Delays (`delay_ms`)

**Description**: Turn scheduled X milliseconds after previous turn completes

**Use Case**: Modeling think time / user interaction patterns

**Example**: `delay_ms=100` means "send 100ms after previous turn returns"

**Requirements**:
- Subsequent turns CAN use `delay_ms`
- **Each subsequent turn must have EITHER `timestamp_ms` OR `delay_ms`**

##### Mixed Mode (Common Pattern)

**Description**: First turn uses absolute timestamp (controls workload distribution), subsequent turns use delays or timestamps (models user behavior)

**Example Dataset**:
```jsonl
{"conversation_id": "conv1", "turns": [
  {"turn_index": 0, "timestamp_ms": 0, "prompt": "Hello"},
  {"turn_index": 1, "delay_ms": 1000, "prompt": "Follow-up"}
]}
{"conversation_id": "conv2", "turns": [
  {"turn_index": 0, "timestamp_ms": 500, "prompt": "Hi"},
  {"turn_index": 1, "timestamp_ms": 2000, "prompt": "Question"}
]}
```

#### Offset Options

Control which portion of the trace to replay:

##### Auto-Offset (`--fixed-schedule-auto-offset`)

**Behavior**: Automatically offsets timestamps so first timestamp is considered 0

**Use Case**: Normalize traces that don't start at 0

**Example**:
```bash
aiperf \
  --input-file trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --fixed-schedule-auto-offset
```

If trace has timestamps `[1000, 1500, 2000]`, they become `[0, 500, 1000]`

##### Start Offset (`--fixed-schedule-start-offset`)

**Behavior**: Start replay at specific offset (milliseconds)

**Use Case**: Skip warm-up period in trace, start at interesting point

**Example**:
```bash
aiperf \
  --input-file trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --fixed-schedule-start-offset 60000  # Start at 1 minute
```

⚠️ **Constraint**: Cannot be used with `--fixed-schedule-auto-offset`

##### End Offset (`--fixed-schedule-end-offset`)

**Behavior**: End replay at specific offset (milliseconds)

**Use Case**: Only run a subset of the trace

**Example**:
```bash
aiperf \
  --input-file trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --fixed-schedule-start-offset 60000 \
  --fixed-schedule-end-offset 120000  # Run 1 minute of trace (from 1min to 2min)
```

⚠️ **Constraint**: `start_offset` must be `<= end_offset`

---

## Configuration Options Reference

### Timing Duration Options

| Option | Type | Default | Validation | Description |
|--------|------|---------|------------|-------------|
| `--benchmark-duration` | `float \| None` | `None` | `>= 1` | Duration in seconds for benchmarking |
| `--benchmark-grace-period` | `float` | `30.0` | `>= 0` | Grace period in seconds to wait for responses after benchmark duration ends. Responses received within this period are included in metrics. |

### Request Control Options

| Option | Type | Default | Validation | Description |
|--------|------|---------|------------|-------------|
| `--concurrency` | `int \| None` | `None` | `>= 1` | Max concurrent requests |
| `--request-rate` | `float \| None` | `None` | `> 0` | Requests per second |
| `--request-rate-mode` | `RequestRateMode` | `POISSON` | enum | Request rate mode: `constant`, `poisson`, `concurrency_burst` |
| `--request-count` | `int` | `10` | `>= 1` | Number of requests for measurement |
| `--warmup-request-count` | `int` | `0` | `>= 0` | Number of warmup requests before benchmarking |
| `--conversation-num` | `int \| None` | `None` | `>= 1` | Total number of unique conversations to generate |

### Fixed Schedule Options

| Option | Type | Default | Validation | Description |
|--------|------|---------|------------|-------------|
| `--fixed-schedule` | `bool` | `False` | - | Enable fixed schedule mode (normally auto-inferred) |
| `--fixed-schedule-auto-offset` | `bool` | `False` | - | Auto-offset timestamps so first is 0 |
| `--fixed-schedule-start-offset` | `int \| None` | `None` | `>= 0` | Start offset in milliseconds |
| `--fixed-schedule-end-offset` | `int \| None` | `None` | `>= 0` | End offset in milliseconds |

### Cancellation Options

| Option | Type | Default | Validation | Description |
|--------|------|---------|------------|-------------|
| `--request-cancellation-rate` | `float` | `0.0` | `0.0 - 100.0` | Percentage of requests to cancel |
| `--request-cancellation-delay` | `float` | `0.0` | `>= 0.0` | Delay in seconds before cancelling requests |

---

## Compatibility Matrix

### Option Combinations

| Option 1 | Option 2 | Compatible? | Notes |
|----------|----------|-------------|-------|
| `--request-rate` | `--concurrency` | ✅ **YES** | Both constraints applied. Rate generator controls timing, semaphore limits concurrency |
| `--request-rate` | `--request-rate-mode concurrency_burst` | ❌ **NO** | CONCURRENCY_BURST requires no rate specified |
| `--benchmark-duration` | `--request-count` | ❌ **NO** | Duration-based and count-based are mutually exclusive |
| `--benchmark-grace-period` | `--benchmark-duration` | ✅ **Required** | Grace period requires duration-based benchmarking |
| `--benchmark-grace-period` | `--request-count` | ❌ **NO** | Grace period only works with duration-based |
| `--request-count` | `--conversation-num` | ❌ **NO** | Use only `--conversation-num` for multi-turn |
| `--fixed-schedule` | `--input-file` | ✅ **Required** | Fixed schedule requires input file |
| `--fixed-schedule-start-offset` | `--fixed-schedule-auto-offset` | ❌ **NO** | Start offset and auto-offset are mutually exclusive |
| `--fixed-schedule-start-offset` | `--fixed-schedule-end-offset` | ✅ **YES** | Start must be <= end |
| `--concurrency` | `--conversation-num` | ✅ **YES** | Concurrency cannot exceed conversation_num |
| `--concurrency` | `--request-count` | ✅ **YES** | Concurrency cannot exceed request_count (if explicitly set) |

### Timing Mode Selection Priority

| Conditions | Selected Mode | Notes |
|------------|---------------|-------|
| `--fixed-schedule` set | FIXED_SCHEDULE | Explicit override |
| Mooncake trace with timestamps | FIXED_SCHEDULE | Auto-detected |
| `--request-rate` set | REQUEST_RATE | With specified sub-mode |
| None of above | REQUEST_RATE (CONCURRENCY_BURST) | Default, auto-sets `--concurrency 1` |

---

## Explicit Validation Rules

These are enforced at configuration time with clear error messages:

### 1. CONCURRENCY_BURST Mode Constraints
**Rule**: CONCURRENCY_BURST mode cannot be used when `--request-rate` is specified

**Error**: `"Request rate mode cannot be <RequestRateMode.CONCURRENCY_BURST: 'concurrency_burst'> when a request rate is specified."`

**Code**: `user_config.py:64-66`

### 2. Benchmark Mode Mutual Exclusivity
**Rule**: `--benchmark-duration` and `--request-count` cannot be used together

**Error**: `"Count-based and duration-based benchmarking cannot be used together. Use either --request-count or --benchmark-duration."`

**Code**: `user_config.py:81-88`

### 3. Grace Period Requires Duration
**Rule**: `--benchmark-grace-period` can only be used with `--benchmark-duration`

**Error**: `"--benchmark-grace-period can only be used with duration-based benchmarking (--benchmark-duration)."`

**Code**: `user_config.py:90-97`

### 4. Fixed Schedule Requires Input File
**Rule**: `--fixed-schedule` requires `--input-file` to be provided

**Error**: `"Fixed schedule requires a file to be provided"`

**Code**: `input_config.py:54-58`

### 5. Fixed Schedule Offset Mutual Exclusivity
**Rule**: `--fixed-schedule-start-offset` and `--fixed-schedule-auto-offset` cannot be used together

**Error**: `"The --fixed-schedule-start-offset and --fixed-schedule-auto-offset options cannot be used together"`

**Code**: `input_config.py:61-69`

### 6. Fixed Schedule Offset Order
**Rule**: `--fixed-schedule-start-offset` must be <= `--fixed-schedule-end-offset`

**Error**: `"The --fixed-schedule-start-offset must be less than or equal to the --fixed-schedule-end-offset"`

**Code**: `input_config.py:73-83`

### 7. Multi-Turn Request Count Exclusivity
**Rule**: `--request-count` and `--conversation-num` cannot both be set for multi-turn

**Error**: `"Both a request-count and number of conversations are set. This can result in confusing output. Use only --conversation-num for multi-turn scenarios."`

**Code**: `user_config.py:361-373`

### 8. Concurrency Limits
**Rule (Multi-Turn)**: `--concurrency` cannot exceed `--conversation-num`

**Error**: `"Concurrency ({concurrency}) cannot be greater than the number of conversations ({num}). Either reduce --concurrency or increase --conversation-num."`

**Code**: `user_config.py:375-400`

**Rule (Single-Turn)**: `--concurrency` cannot exceed `--request-count` (if explicitly set)

**Error**: `"Concurrency ({concurrency}) cannot be greater than the request count ({count}). Either reduce --concurrency or increase --request-count."`

---

## Implicit Implementation Constraints

These are enforced by code logic but may not have explicit validation:

### 1. Request Rate Generator Requirements

**CONSTANT/POISSON Mode**:
- **Requires**: `request_rate > 0`
- **Validation**: Runtime error if not met
- **Code**: `request_rate_strategy.py:397-400, 425-428`

**CONCURRENCY_BURST Mode**:
- **Requires**: `concurrency >= 1` AND `request_rate == None`
- **Validation**: Runtime error if not met
- **Code**: `request_rate_strategy.py:447-454`

### 2. Multi-Turn Delay Behavior

**WITH Concurrency**:
- `delay_ms` from dataset is **respected**
- Semaphore held for entire conversation
- Behavior is conversation-centric

**WITHOUT Concurrency**:
- `delay_ms` from dataset is **ignored**
- ⚠️ Warning logged at runtime when `delay_ms > 0`
- Behavior is rate-centric
- **Code**: `request_rate_strategy.py:372-376`

### 3. Fixed Schedule Turn Requirements

**First Turn (turn_index=0)**:
- **MUST** have `timestamp_ms`
- **Validation**: Raises `ValueError` if missing
- **Code**: `fixed_schedule_strategy.py:463-466`

**Subsequent Turns (turn_index > 0)**:
- **MUST** have either `timestamp_ms` OR `delay_ms` (at least one)
- **Validation**: Raises `ValueError` if both missing
- **Code**: `fixed_schedule_strategy.py:477-481`

### 4. Conversation Provider Selection (Implicit)

**PreSampledConversationProvider**:
- **Used when**: `num_sessions` (i.e., `--conversation-num`) is specified
- **Behavior**: Pre-samples exact number of conversation IDs
- **Use case**: Count-based benchmarks with fixed number of sessions

**LiveConversationProvider**:
- **Used when**: Duration-based OR simple count-based without `num_sessions`
- **Behavior**: Samples conversation IDs on-demand
- **Use case**: Duration-based or flexible count-based benchmarks

**Code**: `conversation_provider.py`

### 5. Timing Mode Selection Priority

**Priority Order**:
1. `--fixed-schedule` flag → FIXED_SCHEDULE
2. Mooncake trace with timestamps → FIXED_SCHEDULE (auto-detected)
3. `--request-rate` specified → REQUEST_RATE
4. Default → REQUEST_RATE with CONCURRENCY_BURST (auto-sets `concurrency=1`)

**Code**: `user_config.py:51-76`

---

## Usage Examples

### Example 1: Constant Rate Load Testing
```bash
aiperf \
  --endpoint-type openai \
  --model gpt-4 \
  --request-rate 50 \
  --request-rate-mode constant \
  --request-count 1000
```

Sends exactly 1000 requests at a fixed rate of 50 req/s (20ms between requests).

### Example 2: Poisson Distribution Load Testing (Realistic Traffic)
```bash
aiperf \
  --endpoint-type openai \
  --model gpt-4 \
  --request-rate 100 \
  --request-rate-mode poisson \
  --benchmark-duration 300 \
  --benchmark-grace-period 60
```

Sends requests at an average rate of 100 req/s with natural randomness, runs for 5 minutes, waits up to 60s for responses after duration ends.

### Example 3: Concurrency Burst Testing
```bash
aiperf \
  --endpoint-type openai \
  --model gpt-4 \
  --concurrency 100 \
  --request-count 5000
```

Bursts up to 100 concurrent requests as fast as possible, totaling 5000 requests.

### Example 4: Combined Rate and Concurrency Limiting
```bash
aiperf \
  --endpoint-type openai \
  --model gpt-4 \
  --request-rate 200 \
  --request-rate-mode poisson \
  --concurrency 50 \
  --benchmark-duration 600
```

Attempts to send ~200 req/s (Poisson distribution) but never exceeds 50 concurrent requests. Runs for 10 minutes.

### Example 5: Fixed Schedule Trace Replay
```bash
aiperf \
  --input-file production_trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --fixed-schedule-auto-offset \
  --endpoint-type openai \
  --model gpt-4
```

Replays production trace with precise timing, auto-offsetting timestamps to start at 0.

### Example 6: Partial Trace Replay
```bash
aiperf \
  --input-file long_trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --fixed-schedule-start-offset 300000 \
  --fixed-schedule-end-offset 600000 \
  --endpoint-type openai \
  --model gpt-4
```

Replays only 5 minutes of trace (from 5min to 10min mark).

### Example 7: Multi-Turn with Conversation Timing (Conversation-Centric)
```bash
aiperf \
  --input-file conversations.jsonl \
  --custom-dataset-type multi_turn \
  --request-rate 10 \
  --concurrency 20 \
  --conversation-num 100 \
  --endpoint-type openai \
  --model gpt-4
```

Sends 100 conversations at ~10 req/s, respects `delay_ms` from dataset, max 20 concurrent conversations.

### Example 8: Multi-Turn with Rate Priority (Rate-Centric)
```bash
aiperf \
  --input-file conversations.jsonl \
  --custom-dataset-type multi_turn \
  --request-rate 100 \
  --request-rate-mode constant \
  --conversation-num 1000 \
  --endpoint-type openai \
  --model gpt-4
```

Sends 1000 conversations at exactly 100 req/s, ignores `delay_ms` from dataset to maintain rate precision.

### Example 9: Duration-Based Benchmarking
```bash
aiperf \
  --endpoint-type openai \
  --model gpt-4 \
  --request-rate 50 \
  --benchmark-duration 1800 \
  --benchmark-grace-period 30
```

Runs for 30 minutes at 50 req/s, waits up to 30s for responses after benchmark ends.

### Example 10: Count-Based Benchmarking with Warmup
```bash
aiperf \
  --endpoint-type openai \
  --model gpt-4 \
  --request-rate 25 \
  --warmup-request-count 50 \
  --request-count 500
```

Sends 50 warmup requests (not measured), then 500 measured requests at 25 req/s.

---

## Decision Tree: Choosing the Right Timing Mode

```
START: What's your benchmarking goal?
│
├─ Replay real-world trace with precise timing?
│  │
│  └─ YES → Use FIXED_SCHEDULE mode
│     │
│     ├─ Have mooncake_trace with timestamps?
│     │  └─ YES → Automatically enabled
│     │
│     └─ OR set --fixed-schedule flag
│        │
│        ├─ Want to replay entire trace?
│        │  └─ Use --fixed-schedule-auto-offset
│        │
│        └─ Want to replay portion of trace?
│           └─ Use --fixed-schedule-start-offset and/or --fixed-schedule-end-offset
│
└─ NO → Use REQUEST_RATE mode (synthetic load)
   │
   ├─ Need precise request rate control?
   │  │
   │  └─ YES → Set --request-rate
   │     │
   │     ├─ Want consistent timing?
   │     │  └─ Use --request-rate-mode constant
   │     │
   │     ├─ Want realistic traffic patterns?
   │     │  └─ Use --request-rate-mode poisson (DEFAULT)
   │     │
   │     └─ Also need concurrency limit?
   │        └─ Set both --request-rate and --concurrency
   │
   └─ NO (want max load) → Use CONCURRENCY_BURST mode
      │
      └─ Set only --concurrency (don't set --request-rate)
         │
         ├─ Multi-turn conversations?
         │  │
         │  ├─ Want to preserve conversation timing (delay_ms)?
         │  │  └─ Set --concurrency (conversation-centric mode)
         │  │
         │  └─ Want to prioritize rate precision?
         │     └─ Don't set --concurrency (rate-centric mode)
         │     └─ ⚠️ delay_ms will be ignored
         │
         └─ Single-turn?
            └─ Set --concurrency for max load
```

---

## Known Issues and Gaps

This section documents gaps, issues, and edge cases discovered through code analysis. These represent areas where behavior may be unclear, undocumented, or potentially problematic.

### 🔍 Issue #1: Concurrency + Request Rate (Undocumented Feature)

**Status**: ✅ Valid configuration, but undocumented

**Finding**: You CAN specify both `--concurrency` and `--request-rate` together with CONSTANT or POISSON modes.

**Current State**:
- Validation only rejects CONCURRENCY_BURST mode with request_rate
- No validation for CONSTANT/POISSON modes with concurrency
- Actual behavior: semaphore limits concurrency, rate generator controls timing

**Impact**: Users don't know they can rate-limit AND concurrency-limit simultaneously. This is actually a very useful configuration!

**Code Reference**: `request_rate_strategy.py:168-171`

**Recommendation**: Document this as a valid and powerful configuration option (now documented above in "Advanced: Combining Concurrency and Request Rate")

---

### ⚠️ Issue #2: Multi-Turn Delay Behavior (Runtime-Only Warning)

**Status**: ⚠️ No config-time validation

**Finding**: Delays are ignored in rate-centric mode (no concurrency), but you only find out at runtime via warning logs.

**Current Behavior**:
- Warning logged: `Conv {conversation_id} turn {turn_index}: delay_ms={delay_ms} ignored (rate-centric mode)`

**Problem**: No config-time validation to warn users their dataset delays will be ignored

**Impact**: Users load multi-turn dataset with delays, don't set concurrency, and wonder why timing isn't respected

**Code Reference**: `request_rate_strategy.py:372-376`

**Workaround**: Always set `--concurrency` when using multi-turn datasets with delays

**Recommendation**: Add config-time warning or validation when multi-turn dataset has delays but concurrency is not set

---

### ⚠️ Issue #3: Mooncake Trace Auto-Detection Can Fail Silently

**Status**: ⚠️ Silent fallback behavior

**Finding**: File read failures during timestamp detection silently fall back to REQUEST_RATE mode.

**Current Behavior**:
- Only logs warning: `"Could not read dataset file {file} to check for timestamps"`
- Returns `False`, falls back to REQUEST_RATE

**Problem**: User expects FIXED_SCHEDULE but gets REQUEST_RATE instead

**Impact**: Incorrect timing mode selected with minimal visibility

**Code Reference**: `user_config.py:136-150`

**Workaround**: Explicitly set `--fixed-schedule` flag to force FIXED_SCHEDULE mode

**Recommendation**: Make failure more visible or error out instead of silent fallback

---

### ❓ Issue #4: Fixed Schedule + Warmup (Unclear Behavior)

**Status**: ❓ Unclear whether supported

**Finding**: No explicit warmup phase setup in fixed_schedule_strategy.py

**Question**: Is `--warmup-request-count` supported with FIXED_SCHEDULE mode?

**Impact**: Unclear whether warmup applies to trace replay, and if so, how it works

**Code Reference**: `fixed_schedule_strategy.py` (no warmup phase setup found)

**Recommendation**: Document or validate this combination

---

### ❓ Issue #5: Fixed Schedule with Concurrency/Request Rate (Unvalidated)

**Status**: ❓ No validation, unclear behavior

**Finding**: No validation prevents `--concurrency` or `--request-rate` with `--fixed-schedule`

**Questions**:
- Does concurrency limit apply to fixed schedule?
- Does request rate override trace timing?
- Should these be rejected as incompatible?

**Impact**: Undefined behavior when combined

**Recommendation**: Either document the behavior or add validation to reject incompatible combinations

---

### ❓ Issue #6: Fixed Schedule with Benchmark Duration (Unvalidated)

**Status**: ❓ No validation, unclear precedence

**Finding**: No validation prevents `--benchmark-duration` with `--fixed-schedule`

**Questions**:
- Does duration override the trace duration?
- Does it cap the trace at the specified duration?
- Which takes precedence?

**Impact**: Unclear which parameter takes precedence

**Recommendation**: Validate or document precedence rules

---

### ⚠️ Issue #7: Synthetic Turn Delay Options with Fixed Schedule (Not Validated)

**Status**: ⚠️ Silently ignored

**Finding**: `--conversation-turn-delay-mean/stddev/ratio` not validated with fixed schedule

**Problem**: These synthetic options don't apply to trace replay (which uses file data)

**Impact**: Users might set these expecting them to work with trace replay

**Code Reference**: `conversation_config.py:26-70`

**Recommendation**: Add validation to reject or warn when these options are used with fixed schedule

---

### ⚠️ Issue #8: Dataset Sampling Strategy with Fixed Schedule (Not Validated)

**Status**: ⚠️ Likely ignored

**Finding**: `--dataset-sampling-strategy` (sequential/random/shuffle) not validated with fixed schedule

**Problem**: Sampling strategy doesn't make sense for trace replay (order determined by timestamps)

**Impact**: Options accepted but likely ignored

**Code Reference**: `input_config.py:257-270`

**Recommendation**: Validate incompatible combinations

---

### 🔒 Issue #9: Request Count vs Conversation Num (Overly Restrictive)

**Status**: 🔒 Possibly too restrictive

**Finding**: Cannot set both `--request-count` and `--conversation-num`

**Current State**: Raises `ValueError`

**Problem**: Use case "Run 100 conversations BUT cap at 1000 total requests" cannot be expressed

**Impact**: Can't express "whichever limit comes first" semantics

**Code Reference**: `user_config.py:363-371`

**Recommendation**: Consider allowing both with "first limit wins" semantics

---

### 🔒 Issue #10: Grace Period Only for Duration-Based (Limitation)

**Status**: 🔒 Intentional limitation, but limiting

**Finding**: `--benchmark-grace-period` requires `--benchmark-duration`

**Problem**: Can't have grace period with count-based benchmarks

**Use Case**: "Send 1000 requests, wait 30s for responses after last sent"

**Impact**: Late responses in count-based mode might not be counted

**Code Reference**: `user_config.py:90-97`

**Recommendation**: Consider allowing grace period for count-based benchmarks

---

### 📝 Issue #11: Conversation Provider Selection (Implicit)

**Status**: 📝 Implicit selection, not user-controllable

**Finding**: Provider automatically selected based on `num_sessions`:
- PreSampledConversationProvider when `num_sessions` specified
- LiveConversationProvider otherwise

**Problem**: No way to explicitly choose or override

**Impact**: Can't use live sampling with num_sessions or vice versa

**Code Reference**: `conversation_provider.py`

**Recommendation**: Document this implicit selection clearly (now documented above)

---

### ❓ Issue #12: CONCURRENCY_BURST with Multi-Turn (Unclear Semantics)

**Status**: ❓ Behavior not documented

**Finding**: Behavior of burst mode with multi-turn conversations not documented

**Question**: Does it burst first turns then handle subsequent turns normally?

**Impact**: Unclear if this is a sensible combination

**Recommendation**: Document the behavior or add validation if it doesn't make sense

---

### ⚠️ Issue #13: Empty Dataset Handling (Inconsistent)

**Status**: ⚠️ Inconsistent validation

**Finding**: Fixed schedule validates empty dataset, request_rate_strategy may not

**Impact**: Inconsistent validation between strategies

**Code Reference**: `fixed_schedule_strategy.py:455-458`

**Recommendation**: Add consistent empty dataset validation across all strategies

---

### ❓ Issue #14: Request Cancellation with Fixed Schedule (Unclear Intent)

**Status**: ❓ Unclear whether intended

**Finding**: Both strategies support `--request-cancellation-rate/delay`

**Question**: Does request cancellation make sense for trace replay? Might violate trace fidelity.

**Impact**: Cancelling requests in trace replay may not be desired behavior

**Recommendation**: Document whether this is intended or add validation to warn/reject

---

### ⚠️ Issue #15: Synthetic Dataset Options with Fixed Schedule (Silently Ignored)

**Status**: ⚠️ Silently ignored

**Finding**: Many synthetic options not validated with fixed schedule:
- `--num-dataset-entries`
- `--num-prompts`
- Prompt generation options
- Turn generation options

**Problem**: These options don't apply to file-based traces

**Impact**: Users set options that are silently ignored

**Recommendation**: Add validation to warn/reject incompatible synthetic options with fixed schedule

---

## Common Pitfalls

### Pitfall #1: Delays Ignored Without Concurrency
**Problem**: Load multi-turn dataset with delays, forget to set `--concurrency`, delays silently ignored

**Example**:
```bash
# ❌ WRONG: Delays will be ignored!
aiperf \
  --input-file multi_turn_with_delays.jsonl \
  --custom-dataset-type multi_turn \
  --request-rate 10
```

**Solution**: Set `--concurrency` to preserve conversation timing
```bash
# ✅ CORRECT: Delays respected
aiperf \
  --input-file multi_turn_with_delays.jsonl \
  --custom-dataset-type multi_turn \
  --request-rate 10 \
  --concurrency 20
```

---

### Pitfall #2: Expecting Both Request Count and Conversation Num
**Problem**: Try to set both `--request-count` and `--conversation-num`, get error

**Example**:
```bash
# ❌ ERROR: Cannot use both
aiperf \
  --request-count 1000 \
  --conversation-num 100 \
  --endpoint-type openai \
  --model gpt-4
```

**Solution**: Use only `--conversation-num` for multi-turn
```bash
# ✅ CORRECT: Use conversation-num
aiperf \
  --conversation-num 100 \
  --endpoint-type openai \
  --model gpt-4
```

---

### Pitfall #3: Not Realizing Concurrency + Rate Can Combine
**Problem**: Think you can only use one or the other

**Tip**: You CAN combine `--concurrency` and `--request-rate` for powerful control!

**Example**:
```bash
# ✅ This works! Rate-limited AND concurrency-limited
aiperf \
  --request-rate 100 \
  --concurrency 50 \
  --endpoint-type openai \
  --model gpt-4
```

---

### Pitfall #4: Using Synthetic Options with Fixed Schedule
**Problem**: Set synthetic dataset options with fixed schedule, expecting them to work

**Example**:
```bash
# ⚠️ IGNORED: Synthetic options don't apply to trace replay
aiperf \
  --input-file trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --conversation-turn-delay-mean 1000 \  # Ignored!
  --num-dataset-entries 500  # Ignored!
```

**Solution**: Don't use synthetic options with fixed schedule - they're for synthetic generation only

---

### Pitfall #5: Mooncake Trace Mode Switching Silently
**Problem**: Expect FIXED_SCHEDULE but get REQUEST_RATE due to file read failure

**Solution**: Explicitly set `--fixed-schedule` to force mode
```bash
# ✅ Explicit mode
aiperf \
  --input-file trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --fixed-schedule  # Force fixed schedule mode
```

---

### Pitfall #6: No Grace Period for Count-Based
**Problem**: Try to use `--benchmark-grace-period` with `--request-count`

**Example**:
```bash
# ❌ ERROR: Grace period requires duration-based
aiperf \
  --request-count 1000 \
  --benchmark-grace-period 30 \
  --endpoint-type openai \
  --model gpt-4
```

**Solution**: Use `--benchmark-duration` instead
```bash
# ✅ CORRECT: Duration-based with grace period
aiperf \
  --benchmark-duration 300 \
  --benchmark-grace-period 30 \
  --endpoint-type openai \
  --model gpt-4
```

---

## Troubleshooting

### Problem: Delays not being respected in multi-turn

**Symptom**: Multi-turn conversations not respecting `delay_ms` from dataset

**Solution**: Ensure `--concurrency` is set for conversation-centric mode

**Check**: Look for warnings in logs: `delay_ms={X} ignored (rate-centric mode)`

---

### Problem: Mooncake trace not using FIXED_SCHEDULE mode

**Symptom**: Trace replay using REQUEST_RATE instead of FIXED_SCHEDULE

**Solutions**:
1. Check that file has valid timestamps
2. Explicitly set `--fixed-schedule` flag
3. Check logs for: `"Could not read dataset file..."`

---

### Problem: Concurrency exceeds conversation count

**Symptom**: Error about concurrency being greater than conversation count

**Solution**: Reduce `--concurrency` or increase `--conversation-num`

```bash
# Make sure concurrency <= conversation_num
aiperf \
  --concurrency 50 \
  --conversation-num 100  # Must be >= concurrency
```

---

### Problem: Can't use both request-count and conversation-num

**Symptom**: Error about both being set

**Solution**: Use only `--conversation-num` for multi-turn scenarios

---

### Problem: Grace period not working

**Symptom**: Responses arriving late not counted in metrics

**Solution**: Ensure using `--benchmark-duration` (not `--request-count`)

---

## Code Improvements Recommendations

Based on the identified issues, we recommend the following code improvements:

1. **Add config-time validation** for runtime-only checks (Issue #2)
   - Warn at config time when multi-turn dataset has delays but concurrency is not set

2. **Make implicit behaviors explicit** through warnings or documentation (Issue #11)
   - Log which conversation provider is being used
   - Document automatic timing mode selection more prominently

3. **Relax overly restrictive validations** where semantics are clear (Issues #9, #10)
   - Consider allowing `request_count` + `conversation_num` with "first limit wins" semantics
   - Consider allowing grace period for count-based benchmarks

4. **Add cross-mode validation** (Issues #5, #6, #7, #8, #15)
   - Validate fixed schedule incompatible with synthetic options
   - Validate fixed schedule with concurrency/request_rate
   - Validate fixed schedule with benchmark_duration

5. **Improve error messages** for auto-detection failures (Issue #3)
   - Make mooncake trace detection failures more visible
   - Error out instead of silent fallback when expected mode doesn't work

---

## References

**Source Files**:
- `src/aiperf/timing/request_rate_strategy.py` - REQUEST_RATE mode implementation
- `src/aiperf/timing/fixed_schedule_strategy.py` - FIXED_SCHEDULE mode implementation
- `src/aiperf/common/config/user_config.py` - Timing mode selection and validation
- `src/aiperf/common/config/loadgen_config.py` - Load generator options
- `src/aiperf/common/config/input_config.py` - Input and fixed schedule options
- `src/aiperf/timing/conversation_provider.py` - Conversation provider selection

**Related Documentation**:
- [Multi-Turn Tutorial](tutorials/multi-turn.md) - Multi-turn conversation examples
- [CLI Options](cli_options.md) - Complete CLI reference
- [Architecture](architecture.md) - System architecture overview
