<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Architecture Proposal: Dataset Inspector for Early Config Validation

## Problem Statement

**Current Issue**: Config validation needs dataset properties to make decisions, but dataset processing happens asynchronously in DatasetManager service after config is validated.

**Specific Issues**:
1. Can't warn about "multi-turn delays without concurrency" at config time
2. Can't validate incompatible option combinations that depend on dataset structure
3. Mooncake trace auto-detection does lightweight file reading, but can't extract full properties
4. Timing mode selection needs to know: has_timestamps, has_delays, is_multi_turn

## Proposed Solution: Lightweight Dataset Inspector

### Design Principles

1. **Separation of Concerns**: Dataset structure inspection vs. full dataset loading
2. **Minimal I/O**: Read only first N entries to infer properties
3. **Stateless**: Pure function that returns properties, no state
4. **Reusable**: Can be used during config validation OR by other components
5. **Fast**: Synchronous, minimal parsing, suitable for CLI validation

---

## Architecture

### High-Level Integration

```
┌─────────────────────────────────────────────────────────────────────────┐
│ EXISTING: Dataset Loaders (Already Registered)                          │
│ ─────────────────────────────────────────────────────────────────────   │
│                                                                          │
│  ┌────────────────────────┐  ┌──────────────────────┐  ┌─────────────┐ │
│  │ MooncakeTraceLoader    │  │ MultiTurnLoader      │  │ Others...   │ │
│  │ ─────────────────────  │  │ ───────────────────  │  │ ──────────  │ │
│  │ • can_load(data)       │  │ • can_load(data)     │  │ • can_load  │ │
│  │ • get_preferred_...()  │  │ • get_preferred_...()│  │ • get_...   │ │
│  └────────────────────────┘  └──────────────────────┘  └─────────────┘ │
│                       ▲                 ▲                     ▲          │
│                       └─────────────────┴─────────────────────┘          │
│                                         │                                │
│                          CustomDatasetFactory.get_registered_classes()  │
└─────────────────────────────────────────┬───────────────────────────────┘
                                          │
                                          │ Iterates & calls can_load()
                                          │
┌─────────────────────────────────────────▼───────────────────────────────┐
│ NEW: DatasetInspector (Lightweight Inspection)                          │
│ ─────────────────────────────────────────────────────────────────────   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ inspect_file(path)                                               │   │
│  │ ────────────────────────────────────────────────────────────     │   │
│  │ 1. Read first N entries from file                                │   │
│  │ 2. For each entry:                                               │   │
│  │    • Try each loader's can_load(data)                            │   │
│  │    • When match found → detected_type                            │   │
│  │    • Call loader.get_preferred_sampling_strategy()               │   │
│  │    • Extract timing properties (timestamps, delays)              │   │
│  │ 3. Aggregate properties                                          │   │
│  │ 4. Return DatasetProperties                                      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  DatasetProperties:                                                     │
│  • detected_type: from can_load() ← REUSED LOGIC                        │
│  • preferred_sampling_strategy: from get_preferred...() ← REUSED LOGIC  │
│  • has_timestamps: extracted from data                                  │
│  • has_delays: extracted from data                                      │
│  • is_multi_turn: extracted from data                                   │
└──────────────────────────────────────────┬───────────────────────────────┘
                                           │
                                           │ Returns properties
                                           │
┌──────────────────────────────────────────▼───────────────────────────────┐
│ UserConfig Validation (Enhanced)                                         │
│ ─────────────────────────────────────────────────────────────────────    │
│                                                                           │
│  Uses DatasetProperties for:                                             │
│  ✅ Timing mode selection (has_timestamps → FIXED_SCHEDULE)              │
│  ✅ Multi-turn delay warnings (has_delays + no concurrency)              │
│  ✅ Sampling strategy warnings (user override vs preferred)              │
│  ✅ Fixed schedule validation (detected_type compatibility)              │
│  ✅ Early error detection (before services start)                        │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Handling Public Datasets

**Key Difference**: Public datasets (e.g., ShareGPT) are downloaded from URLs, not loaded from local files.

### Public Dataset Characteristics

```python
# ShareGPT is the primary public dataset
class ShareGPTLoader(BasePublicDatasetLoader):
    tag = "ShareGPT"
    url = "https://huggingface.co/.../ShareGPT_V3_unfiltered_cleaned_split.json"

    # Properties:
    # - Always multi-turn (conversations from ShareGPT)
    # - No timing data (no timestamps or delays in source)
    # - Downloaded and cached locally
```

### DatasetInspector Approach for Public Datasets

**Recommended: Hardcoded Properties**

Since public datasets are well-known types, use hardcoded properties:

```python
class DatasetInspector:

    # Known public dataset properties (no inspection needed)
    PUBLIC_DATASET_PROPERTIES = {
        PublicDatasetType.SHAREGPT: DatasetProperties(
            has_timestamps=False,  # ShareGPT has no timing data
            has_delays=False,      # ShareGPT has no timing data
            is_multi_turn=True,    # ShareGPT is multi-turn conversations
            max_turns_seen=0,      # Unknown until loaded
            estimated_entry_count=0,  # Unknown until downloaded
            detected_type=None,    # Public, not custom
            has_session_ids=True,  # ShareGPT has conversation IDs
            preferred_sampling_strategy=DatasetSamplingStrategy.SHUFFLE,
        ),
        # Add more public datasets as they're added
    }

    @classmethod
    def get_public_dataset_properties(
        cls,
        dataset_type: PublicDatasetType
    ) -> DatasetProperties:
        """Get hardcoded properties for public datasets."""
        return cls.PUBLIC_DATASET_PROPERTIES.get(
            dataset_type,
            DatasetProperties()  # Default if unknown
        )
```

### Integration with UserConfig

```python
def _get_dataset_properties(self) -> DatasetProperties | None:
    """Get dataset properties from any source."""

    # 1. Public dataset (hardcoded properties)
    if self.input.public_dataset:
        return DatasetInspector.get_public_dataset_properties(
            self.input.public_dataset
        )

    # 2. File-based custom dataset (inspect file)
    elif self.input.file:
        return DatasetInspector.inspect_file(self.input.file)

    # 3. Synthetic dataset (extract from config)
    else:
        return self._extract_synthetic_properties()
```

### Public Dataset Example

```bash
aiperf \
  --public-dataset sharegpt \
  --request-rate 10 \
  --endpoint-type openai \
  --model gpt-4

# Config validation uses hardcoded properties:
# - has_timestamps=False → Cannot use FIXED_SCHEDULE
# - is_multi_turn=True → Multi-turn validations apply
# - has_delays=False → No delay warnings needed
# - preferred_sampling=SHUFFLE → Default for ShareGPT
```

---

## Handling Batch Size (Cross-Entry Batching)

**Key Insight**: For file-based datasets, `batch_size` selects **multiple file entries** and combines them into a single batched request.

### What Batch Size Actually Means

```python
# Cross-entry batching (file-based datasets)
--prompt-batch-size 5

# Behavior:
# 1. Select 5 entries from dataset file
# 2. Extract prompt from each entry
# 3. Combine into single API request
# 4. Send as one batched request

# Example:
# File has: ["Hello", "Hi", "Hey", "Greetings", "Howdy"]
# batch_size=5 → One request with all 5 prompts
# batch_size=1 → Five requests with one prompt each
```

### Impact on Request Count

**Critical**: Batch size affects total number of requests!

```
File entries: 100
--request-rate 10
--prompt-batch-size 5

Calculation:
- Total requests = 100 / 5 = 20 requests
- Each request contains 5 prompts from 5 different file entries
- Duration at 10 req/s = 2 seconds
- NOT 100 requests!
```

### Batch Size in File Formats

File entries are individual, unbatched:

```jsonl
{"text": "Hello"}
{"text": "Hi there"}
{"text": "What is AI?"}
{"text": "Explain ML"}
{"text": "Define NLP"}
```

With `--prompt-batch-size 5`, these 5 entries become **1 request**:
```json
{
  "model": "text-embedding-3-small",
  "input": ["Hello", "Hi there", "What is AI?", "Explain ML", "Define NLP"]
}
```

### Within-Entry Batching (Different Concept)

Some formats support batching **within an entry** (less common):

```jsonl
# This entry already contains multiple prompts
{"texts": ["prompt1", "prompt2", "prompt3"]}
```

This is **separate** from cross-entry batching via `--prompt-batch-size`.

### Request Count Calculation with Batching

**Critical for Config Validation**: Need to calculate effective request count!

```python
class DatasetProperties(AIPerfBaseModel):
    """Extended with effective request count calculation."""

    # ... existing fields ...

    def get_effective_request_count(self, batch_size: int = 1) -> int:
        """Calculate effective request count with batching.

        Args:
            batch_size: Number of file entries per request

        Returns:
            Effective number of requests that will be sent

        Example:
            estimated_entry_count=100, batch_size=5 → 20 requests
            estimated_entry_count=100, batch_size=3 → 34 requests (33 full + 1 partial)
        """
        if batch_size <= 1:
            return self.estimated_entry_count

        # Calculate batched requests (round up for remainder)
        import math
        return math.ceil(self.estimated_entry_count / batch_size)
```

### Batch Size Validation

```python
# In UserConfig validation:
@model_validator(mode="after")
def validate_batch_size_with_request_count(self) -> Self:
    """Validate batch size against request count and concurrency."""
    if not self._dataset_properties:
        return self

    batch_size = self.input.prompt.batch_size
    if batch_size <= 1 or not self.input.file:
        return self

    # Calculate effective request count
    effective_requests = self._dataset_properties.get_effective_request_count(batch_size)

    # Warn about significant reduction
    if effective_requests < self._dataset_properties.estimated_entry_count / 2:
        _logger.info(
            f"Batch size {batch_size} will reduce {self._dataset_properties.estimated_entry_count} "
            f"file entries to ~{effective_requests} batched requests."
        )

    # Validate concurrency doesn't exceed effective requests
    if (
        self.loadgen.concurrency
        and self.loadgen.concurrency > effective_requests
    ):
        _logger.warning(
            f"Concurrency ({self.loadgen.concurrency}) exceeds effective request count "
            f"({effective_requests}) with batch_size={batch_size}. "
            f"Consider reducing concurrency or batch size."
        )

    # Validate with request_count if set
    if (
        "request_count" in self.loadgen.model_fields_set
        and self.loadgen.request_count != effective_requests
    ):
        _logger.warning(
            f"--request-count={self.loadgen.request_count} specified, but with "
            f"--prompt-batch-size={batch_size}, the dataset will produce "
            f"{effective_requests} batched requests. These numbers should match."
        )

    return self
```

### Batch Size Examples

```bash
# Example 1: Cross-entry batching with file
aiperf \
  --input-file prompts.jsonl \              # 1000 entries
  --custom-dataset-type single_turn \
  --endpoint-type openai_embeddings \
  --model text-embedding-3-small \
  --prompt-batch-size 10 \                  # Batch 10 entries per request
  --request-rate 5

# File: 1000 single-line entries
# {"text": "prompt1"}
# {"text": "prompt2"}
# ...
# {"text": "prompt1000"}

# Calculation:
# - File entries: 1000
# - Batch size: 10
# - Effective requests: 1000 / 10 = 100 requests
# - Timing: 5 requests/second
# - Duration: 100 / 5 = 20 seconds
# - Each request contains 10 prompts from 10 different file entries

# Inspector helps:
ℹ️  Batch size 10 will reduce 1000 file entries to ~100 batched requests.
```

```bash
# Example 2: Batch size mismatch detection
aiperf \
  --input-file prompts.jsonl \              # 100 entries
  --prompt-batch-size 10 \
  --concurrency 50                          # Too high!

# Validation:
⚠️  WARNING: Concurrency (50) exceeds effective request count (10) with batch_size=10.
    Consider reducing concurrency or batch size.

# Explanation:
# - 100 entries / batch_size 10 = 10 requests
# - Concurrency 50 means max 50 concurrent requests
# - But only 10 requests total → only 10 will ever be in flight
```

```bash
# Example 3: Batch size with multi-turn (probably invalid!)
aiperf \
  --input-file conversations.jsonl \        # Multi-turn conversations
  --custom-dataset-type multi_turn \
  --prompt-batch-size 5 \                   # Doesn't make sense for multi-turn!
  --request-rate 10

# Potential validation:
❌ ERROR: --prompt-batch-size cannot be used with multi-turn datasets.
   Batching combines multiple file entries, but multi-turn entries represent
   complete conversations that cannot be combined.
   Use batch_size=1 (default) for multi-turn datasets.
```

### Batch Size & Timing Mode Interaction

**Important**: Batch size affects **request count**, which impacts timing!

| Timing Mode | Batch Size Impact |
|-------------|-------------------|
| REQUEST_RATE (CONSTANT) | Reduces total requests by batch_size factor<br>Fixed interval per batch, not per entry |
| REQUEST_RATE (POISSON) | Reduces total requests by batch_size factor<br>Poisson interval per batch, not per entry |
| REQUEST_RATE (CONCURRENCY_BURST) | Reduces total requests by batch_size factor<br>Each batch counts as 1 concurrent request |
| FIXED_SCHEDULE | ❓ **Unclear/Invalid**: Timestamps are per-entry, batching would break timing |

**Key Point**:
- Timing controls **when** requests are sent (interval)
- Batch size controls **how many entries** per request
- **Together**: Fewer requests sent at the same rate

### New Constraints Discovered

**ISSUE #16: Batch Size with Multi-Turn Datasets**
- **Problem**: Batching combines multiple file entries into one request
- **Conflict**: Multi-turn entries are complete conversations (cannot be combined)
- **Current State**: No validation prevents `batch_size > 1` with multi-turn
- **Recommendation**: Validate and reject batch_size > 1 with multi-turn custom datasets

**ISSUE #17: Batch Size with Fixed Schedule**
- **Problem**: Fixed schedule uses timestamps per-entry
- **Conflict**: Batching combines entries, breaks timing semantics
- **Current State**: No validation prevents batch_size > 1 with fixed schedule
- **Recommendation**: Validate and reject batch_size > 1 with FIXED_SCHEDULE mode

**ISSUE #18: Request Count vs Effective Request Count**
- **Problem**: User sets `--request-count 1000` but batch_size=10 → only 100 requests
- **Current State**: No warning about mismatch
- **Recommendation**: Warn when request_count doesn't match effective_request_count

**ISSUE #19: Concurrency vs Effective Request Count**
- **Problem**: User sets `--concurrency 50` but batch_size creates only 10 requests
- **Current State**: No warning, most concurrency slots unused
- **Recommendation**: Warn when concurrency > effective_request_count

---

## Complete Dataset Type Matrix

| Dataset Type | Inspector Method | Properties Source | Timing Data | Cross-Entry Batch Support | Sampling Strategy |
|--------------|-----------------|-------------------|-------------|---------------------------|-------------------|
| **Synthetic** | `_extract_synthetic_properties()` | Config options | ❌ No timestamps<br>✅ Has delays (if delay.mean > 0) | ✅ Via `--prompt-batch-size` config<br>Generates N items per turn | User-specified or default |
| **Custom: SINGLE_TURN** | `inspect_file()` + loader `can_load()` | File inspection | Maybe (rare) | ✅ **YES** - Primary use case<br>Combines N entries per request | From loader |
| **Custom: MULTI_TURN** | `inspect_file()` + loader `can_load()` | File inspection | ✅ timestamps or delays | ❌ **NO** - Conversations can't be combined<br>Should validate batch_size=1 | From loader |
| **Custom: RANDOM_POOL** | `inspect_file()` + loader `can_load()` | File inspection | ❌ Usually none | ✅ **YES** - Pool of prompts<br>Combines N entries per request | From loader |
| **Custom: MOONCAKE_TRACE** | `inspect_file()` + loader `can_load()` | File inspection | ✅ timestamps or delays | ❌ **NO** - Trace replay<br>Should validate batch_size=1 | SEQUENTIAL (from loader) |
| **Public: SHAREGPT** | `get_public_dataset_properties()` | Hardcoded | ❌ No timing data | ❌ **NO** - Multi-turn conversations<br>Should validate batch_size=1 | SHUFFLE (hardcoded) |

### Batch Size Validation Matrix

| Dataset Type | Batch Size > 1? | Rationale | Validation Needed |
|--------------|-----------------|-----------|-------------------|
| **Synthetic** | ✅ **VALID** | Generates N items per turn | None - works as designed |
| **Custom: SINGLE_TURN** | ✅ **VALID** | Primary use case for embeddings<br>Combines N file entries | ✅ Warn about effective request count |
| **Custom: MULTI_TURN** | ❌ **INVALID** | Conversations are atomic units<br>Cannot combine conversations | ✅ **Reject with error** |
| **Custom: RANDOM_POOL** | ✅ **VALID** | Pool entries can be batched | ✅ Warn about effective request count |
| **Custom: MOONCAKE_TRACE** | ❌ **INVALID** | Trace replay needs precise timing<br>Per-entry timestamps | ✅ **Reject with error** |
| **Public: SHAREGPT** | ❌ **INVALID** | Multi-turn conversations<br>Cannot combine | ✅ **Reject with error** |

### Updated DatasetProperties Model

```python
class DatasetProperties(AIPerfBaseModel):
    """Extended with batch compatibility."""

    # ... existing fields ...

    supports_cross_entry_batching: bool = Field(
        default=True,
        description="True if dataset supports combining multiple entries into batched requests"
    )

    @property
    def can_use_batch_size(self) -> bool:
        """Check if batch_size > 1 is valid for this dataset."""
        # Multi-turn datasets cannot be batched (conversations are atomic)
        if self.is_multi_turn:
            return False

        # Fixed schedule cannot be batched (timestamps are per-entry)
        if self.has_timestamps:
            return False

        # Otherwise, batching is valid (single-turn, random_pool, synthetic)
        return True
```

### Complete Validation Logic

```python
@model_validator(mode="after")
def validate_batch_size_compatibility(self) -> Self:
    """Comprehensive batch size validation."""
    if not self._dataset_properties:
        return self

    batch_size = self.input.prompt.batch_size
    if batch_size <= 1:
        return self  # No batching, nothing to validate

    # REJECT: Batch size with multi-turn datasets
    if self._dataset_properties.is_multi_turn:
        raise ValueError(
            f"--prompt-batch-size={batch_size} cannot be used with multi-turn datasets. "
            f"Multi-turn entries represent complete conversations that cannot be combined. "
            f"Detected dataset type: {self._dataset_properties.detected_type}"
        )

    # REJECT: Batch size with fixed schedule (timestamps per-entry)
    if self._timing_mode == TimingMode.FIXED_SCHEDULE:
        raise ValueError(
            f"--prompt-batch-size={batch_size} cannot be used with fixed schedule mode. "
            f"Fixed schedule requires one request per entry to preserve timing. "
            f"Batching would combine entries and break timestamp semantics."
        )

    # Calculate effective request count for file-based datasets
    if self.input.file and self._dataset_properties.estimated_entry_count > 0:
        effective_requests = self._dataset_properties.get_effective_request_count(batch_size)

        # INFO: Notify about request count reduction
        _logger.info(
            f"Cross-entry batching: {self._dataset_properties.estimated_entry_count} "
            f"file entries → {effective_requests} batched requests (batch_size={batch_size})"
        )

        # WARN: Concurrency exceeds effective requests
        if self.loadgen.concurrency and self.loadgen.concurrency > effective_requests:
            _logger.warning(
                f"Concurrency ({self.loadgen.concurrency}) exceeds effective request count "
                f"({effective_requests}). With batch_size={batch_size}, only {effective_requests} "
                f"requests will be sent. Consider reducing --concurrency to {effective_requests}."
            )

        # WARN: Request count mismatch
        if (
            "request_count" in self.loadgen.model_fields_set
            and self.loadgen.request_count != effective_requests
        ):
            _logger.warning(
                f"--request-count={self.loadgen.request_count} specified, but dataset has "
                f"{self._dataset_properties.estimated_entry_count} entries with batch_size={batch_size}, "
                f"which produces {effective_requests} batched requests. "
                f"Consider setting --request-count={effective_requests} or adjusting batch size."
            )

    return self
```

### Three-Way Decision: Dataset Source

```
┌───────────────────────────────────────────────────────────┐
│ UserConfig._get_dataset_properties()                      │
│                                                            │
│  Check dataset source type:                               │
│  │                                                         │
│  ├─ Public dataset? (--public-dataset sharegpt)           │
│  │  │                                                      │
│  │  └─► get_public_dataset_properties(SHAREGPT)           │
│  │      • Hardcoded properties                            │
│  │      • No inspection needed                            │
│  │      • Returns known ShareGPT properties               │
│  │                                                         │
│  ├─ File-based? (--input-file data.jsonl)                 │
│  │  │                                                      │
│  │  └─► inspect_file(data.jsonl)                          │
│  │      • Reads first 10 entries                          │
│  │      • Calls loader.can_load() for type detection      │
│  │      • Extracts timing/batch properties                │
│  │      • Returns discovered properties                   │
│  │                                                         │
│  └─ Synthetic? (no file, no public dataset)               │
│     │                                                      │
│     └─► _extract_synthetic_properties()                   │
│         • Reads config options                            │
│         • turn.mean → is_multi_turn                       │
│         • turn.delay.mean → has_delays                    │
│         • Returns inferred properties                     │
│                                                            │
│  All paths return: DatasetProperties                      │
│                                                            │
│  Use for validation ──────────────────────────────────────►│
└───────────────────────────────────────────────────────────┘
```

---

## Summary: Clean Integration with Existing Infrastructure

### What Gets Reused

| Component | Existing Method | DatasetInspector Uses |
|-----------|----------------|----------------------|
| **MooncakeTraceLoader** | `can_load(data)` | ✅ Type detection |
| **MooncakeTraceLoader** | `get_preferred_sampling_strategy()` | ✅ Returns SEQUENTIAL |
| **MultiTurnLoader** | `can_load(data)` | ✅ Type detection |
| **MultiTurnLoader** | `get_preferred_sampling_strategy()` | ✅ Returns strategy |
| **SingleTurnLoader** | `can_load(data)` | ✅ Type detection |
| **RandomPoolLoader** | `can_load(data)` | ✅ Type detection |
| **CustomDatasetFactory** | `get_registered_classes()` | ✅ Iterate all loaders |

### What's New (DatasetInspector Only)

DatasetInspector adds **only** the timing property extraction:
- `has_timestamps`: Check for timestamp/timestamp_ms fields
- `has_delays`: Check for delay/delay_ms fields
- `is_multi_turn`: Check for turns array structure
- `max_turns_seen`: Count turns in conversations

Everything else is **reused from existing loaders**!

### Division of Responsibilities

```
┌──────────────────────────────────────────────────────────────┐
│ Dataset Loaders (EXISTING - No Changes Needed)               │
│ ──────────────────────────────────────────────────────────   │
│ Responsibility: "Can I load this format?"                    │
│                                                               │
│ Provides:                                                     │
│ • can_load(data) → bool                                      │
│ • get_preferred_sampling_strategy() → DatasetSamplingStrategy│
│ • load_dataset() → full data loading                         │
│ • convert_to_conversations() → metadata generation           │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ DatasetInspector (NEW - Lightweight Layer)                   │
│ ──────────────────────────────────────────────────────────   │
│ Responsibility: "What properties does this dataset have?"    │
│                                                               │
│ Provides:                                                     │
│ • inspect_file() → DatasetProperties                         │
│ • Uses loader.can_load() for type detection ← REUSE          │
│ • Uses loader.get_preferred_sampling_strategy() ← REUSE      │
│ • Extracts timing properties (new logic, minimal)            │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ UserConfig Validation (ENHANCED - Uses Inspector)            │
│ ──────────────────────────────────────────────────────────   │
│ Responsibility: "Is this configuration valid?"               │
│                                                               │
│ Uses DatasetProperties:                                       │
│ • Timing mode selection                                      │
│ • Multi-turn delay warnings                                  │
│ • Sampling strategy validation                               │
│ • Early error detection                                      │
└──────────────────────────────────────────────────────────────┘
```

### Zero Duplication!

**Type Detection**:
- ❌ **Don't**: Duplicate format detection in inspector
- ✅ **Do**: Call `loader.can_load(data)` to leverage existing logic

**Sampling Strategy**:
- ❌ **Don't**: Hardcode preferred strategies in inspector
- ✅ **Do**: Call `loader.get_preferred_sampling_strategy()` to leverage existing logic

**Timing Properties**:
- ✅ **New Logic**: Inspector adds minimal timing extraction (timestamps, delays)
- ✅ **Justification**: Loaders don't expose this info, it's inspector's responsibility

---

## Implementation

### 1. DatasetProperties Model

```python
# src/aiperf/common/dataset_inspector.py

from pathlib import Path
from typing import Any

from pydantic import Field

from aiperf.common.enums import CustomDatasetType
from aiperf.common.models import AIPerfBaseModel
from aiperf.common.utils import load_json_str


class DatasetProperties(AIPerfBaseModel):
    """Lightweight properties of a dataset file, suitable for config validation.

    This model contains only the information needed for config validation and
    timing mode selection, without fully parsing or loading the dataset.
    """

    has_timestamps: bool = Field(
        default=False,
        description="True if any entry has timestamp or timestamp_ms fields"
    )
    has_delays: bool = Field(
        default=False,
        description="True if any entry has delay or delay_ms fields"
    )
    is_multi_turn: bool = Field(
        default=False,
        description="True if dataset contains multi-turn conversations"
    )
    max_turns_seen: int = Field(
        default=1,
        ge=1,
        description="Maximum number of turns seen in any conversation"
    )
    estimated_entry_count: int = Field(
        default=0,
        ge=0,
        description="Approximate number of entries in the file (counts non-empty lines)"
    )
    detected_type: CustomDatasetType | None = Field(
        default=None,
        description="Auto-detected custom dataset type using loader's can_load() method"
    )
    has_session_ids: bool = Field(
        default=False,
        description="True if entries have session_id or conversation_id fields"
    )
    preferred_sampling_strategy: DatasetSamplingStrategy | None = Field(
        default=None,
        description="Preferred sampling strategy from detected loader"
    )


class DatasetInspector:
    """Lightweight dataset file inspector for config validation.

    Inspects dataset files to extract properties needed for config validation
    and timing mode selection WITHOUT fully loading or processing the dataset.

    Design:
    - Reads only first N entries (default: 10) for fast inspection
    - Leverages existing loader's can_load() for type detection
    - Leverages loader's get_preferred_sampling_strategy() for sampling info
    - Stateless utility functions
    - Minimal parsing (JSON decode only)
    - Returns DatasetProperties model

    Key Integration:
    - Uses CustomDatasetFactory to get all registered loaders
    - Calls each loader's can_load(data) to detect type
    - Calls detected loader's get_preferred_sampling_strategy()
    - No duplication of type detection logic!

    Usage:
        properties = DatasetInspector.inspect_file("dataset.jsonl")
        if properties.has_delays and not user_config.concurrency:
            warn("Delays will be ignored without concurrency")
    """

    DEFAULT_SAMPLE_SIZE = 10
    MAX_SAMPLE_SIZE = 100

    @classmethod
    def inspect_file(
        cls,
        file_path: str | Path,
        sample_size: int | None = None
    ) -> DatasetProperties:
        """Inspect a dataset file to extract properties for validation.

        Args:
            file_path: Path to dataset file (JSONL format)
            sample_size: Number of entries to sample (default: 10, max: 100)
                        None means use default

        Returns:
            DatasetProperties with inferred properties

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not valid JSONL or sample_size invalid
        """
        if sample_size is None:
            sample_size = cls.DEFAULT_SAMPLE_SIZE
        elif sample_size < 1:
            raise ValueError(f"sample_size must be >= 1, got {sample_size}")
        elif sample_size > cls.MAX_SAMPLE_SIZE:
            raise ValueError(
                f"sample_size must be <= {cls.MAX_SAMPLE_SIZE}, got {sample_size}"
            )

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Initialize aggregated properties
        has_timestamps = False
        has_delays = False
        is_multi_turn = False
        max_turns_seen = 1
        has_session_ids = False

        # Track detected types from sampled entries
        detected_types: set[CustomDatasetType] = set()
        detected_loader_class: type | None = None

        entry_count = 0
        sampled_count = 0

        try:
            with open(file_path) as f:
                for line in f:
                    # Count all non-empty lines for estimated entry count
                    if line := line.strip():
                        entry_count += 1
                    else:
                        continue

                    # Only parse first N entries for sampling
                    if sampled_count >= sample_size:
                        continue  # Still counting entries, but not parsing

                    try:
                        data = load_json_str(line)
                        sampled_count += 1

                        # Inspect this entry
                        entry_props = cls._inspect_entry(data)

                        # Aggregate properties
                        has_timestamps = has_timestamps or entry_props["has_timestamps"]
                        has_delays = has_delays or entry_props["has_delays"]
                        is_multi_turn = is_multi_turn or entry_props["is_multi_turn"]
                        max_turns_seen = max(max_turns_seen, entry_props["num_turns"])
                        has_session_ids = has_session_ids or entry_props["has_session_id"]

                        if entry_props["detected_type"]:
                            detected_types.add(entry_props["detected_type"])
                        if entry_props["loader_class"]:
                            detected_loader_class = entry_props["loader_class"]

                    except Exception:
                        # Skip malformed entries during inspection
                        # They'll be caught during actual loading
                        continue

        except Exception as e:
            raise ValueError(f"Failed to inspect dataset file: {e}") from e

        # Determine single detected type (if all sampled entries agree)
        detected_type = None
        if len(detected_types) == 1:
            detected_type = detected_types.pop()
        # If multiple types detected, leave as None (ambiguous)

        # Get preferred sampling strategy from detected loader
        preferred_sampling_strategy = None
        if detected_loader_class:
            try:
                preferred_sampling_strategy = detected_loader_class.get_preferred_sampling_strategy()
            except Exception:
                # Loader doesn't support preferred strategy
                pass

        return DatasetProperties(
            has_timestamps=has_timestamps,
            has_delays=has_delays,
            is_multi_turn=is_multi_turn,
            max_turns_seen=max_turns_seen,
            estimated_entry_count=entry_count,
            detected_type=detected_type,
            has_session_ids=has_session_ids,
            preferred_sampling_strategy=preferred_sampling_strategy,
        )

    @classmethod
    def _inspect_entry(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Inspect a single dataset entry to extract properties.

        Leverages existing loader's can_load() method for type detection!

        Returns dict with:
            - has_timestamps: bool
            - has_delays: bool
            - is_multi_turn: bool
            - num_turns: int
            - has_session_id: bool
            - detected_type: CustomDatasetType | None
            - loader_class: type | None (the loader that can handle this data)
        """
        from aiperf.common.factories import CustomDatasetFactory

        has_timestamps = False
        has_delays = False
        is_multi_turn = False
        num_turns = 1
        has_session_id = False
        detected_type = None
        loader_class = None

        # USE EXISTING LOADER LOGIC: Try each registered loader's can_load()
        for dataset_type, loader_cls in CustomDatasetFactory.get_registered_classes().items():
            try:
                if loader_cls.can_load(data=data):
                    detected_type = dataset_type
                    loader_class = loader_cls
                    break  # Found matching loader
            except Exception:
                # Loader's can_load() failed, try next one
                continue

        # Check for session/conversation IDs
        has_session_id = (
            "session_id" in data or
            "conversation_id" in data
        )

        # Extract timing properties (timestamps, delays)
        # This is still needed since can_load() doesn't tell us about timing

        # Check for multi-turn structure
        if "turns" in data and isinstance(data["turns"], list):
            is_multi_turn = True
            num_turns = len(data["turns"])

            # Check turns for timestamps/delays
            for turn in data["turns"]:
                if isinstance(turn, dict):
                    has_timestamps = has_timestamps or (
                        "timestamp" in turn or "timestamp_ms" in turn
                    )
                    has_delays = has_delays or (
                        "delay" in turn or "delay_ms" in turn
                    )

        # Check for top-level timestamp/delay (mooncake trace format)
        if "timestamp" in data or "timestamp_ms" in data:
            has_timestamps = True
        if "delay" in data or "delay_ms" in data:
            has_delays = True

        return {
            "has_timestamps": has_timestamps,
            "has_delays": has_delays,
            "is_multi_turn": is_multi_turn,
            "num_turns": num_turns,
            "has_session_id": has_session_id,
            "detected_type": detected_type,
            "loader_class": loader_class,
        }
```

---

### 2. Leveraging Existing Loader Infrastructure

**Key Benefit**: DatasetInspector reuses existing loader logic instead of duplicating type detection!

#### How It Works

```python
# Each loader already has can_load() method
@CustomDatasetFactory.register(CustomDatasetType.MOONCAKE_TRACE)
class MooncakeTraceDatasetLoader(BaseFileLoader):
    @classmethod
    def can_load(cls, data: dict[str, Any] | None = None, ...) -> bool:
        """Check if this loader can handle the given data format."""
        if data is None:
            return False
        try:
            MooncakeTrace.model_validate(data)
            return True
        except ValidationError:
            return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred dataset sampling strategy for MooncakeTrace."""
        return DatasetSamplingStrategy.SEQUENTIAL
```

#### DatasetInspector Uses These

```python
# In _inspect_entry():
for dataset_type, loader_cls in CustomDatasetFactory.get_registered_classes().items():
    if loader_cls.can_load(data=data):
        detected_type = dataset_type
        loader_class = loader_cls
        break

# Then get preferred strategy:
if detected_loader_class:
    preferred_sampling_strategy = detected_loader_class.get_preferred_sampling_strategy()
```

#### Benefits

| Without Integration | With Integration |
|---------------------|------------------|
| ❌ Duplicate detection logic in inspector | ✅ Reuse loader's can_load() |
| ❌ Inspector needs to know all formats | ✅ Loader knows its own format |
| ❌ Two places to update when adding types | ✅ One place: just the loader |
| ❌ Can't get preferred sampling strategy | ✅ Get strategy from loader |
| ❌ Logic divergence over time | ✅ Single source of truth |

#### Example: Adding New Dataset Type

**Without Integration** (duplicated logic):
```python
# 1. Add loader
class NewDatasetLoader:
    @classmethod
    def can_load(cls, data): ...  # Detection logic here

# 2. ALSO update inspector (duplicate!)
class DatasetInspector:
    def _inspect_entry(self, data):
        if "new_format_field" in data:  # Duplicate detection!
            detected_type = CustomDatasetType.NEW_TYPE
```

**With Integration** (single source):
```python
# 1. Add loader (that's it!)
@CustomDatasetFactory.register(CustomDatasetType.NEW_TYPE)
class NewDatasetLoader:
    @classmethod
    def can_load(cls, data): ...  # Detection logic here

    @classmethod
    def get_preferred_sampling_strategy(cls):
        return DatasetSamplingStrategy.SHUFFLE

# 2. Inspector automatically uses it!
# No changes needed - it iterates registered loaders
```

#### Validation Strategy Usage

```python
# In UserConfig validation:
@model_validator(mode="after")
def validate_sampling_strategy(self) -> Self:
    """Warn if user overrides preferred sampling strategy."""
    if not self._dataset_properties:
        return self

    # Get preferred strategy from loader
    preferred = self._dataset_properties.preferred_sampling_strategy

    # Check if user explicitly set different strategy
    if (
        preferred
        and self.input.dataset_sampling_strategy
        and self.input.dataset_sampling_strategy != preferred
    ):
        _logger.warning(
            f"Dataset type {self._dataset_properties.detected_type} prefers "
            f"{preferred} sampling, but {self.input.dataset_sampling_strategy} was specified. "
            f"Consider using {preferred} for optimal behavior."
        )

    return self
```

#### Complete Flow Example

```bash
# User runs command with mooncake trace
aiperf \
  --input-file production_trace.jsonl \
  --dataset-sampling-strategy random \
  --endpoint-type openai \
  --model gpt-4

# During config validation:

# 1. DatasetInspector reads first 10 entries
# 2. For each entry, tries loaders' can_load():
#    - MultiTurnDatasetLoader.can_load(data) → False
#    - MooncakeTraceDatasetLoader.can_load(data) → True ✓
# 3. Detected: CustomDatasetType.MOONCAKE_TRACE
# 4. Gets preferred strategy:
#    MooncakeTraceDatasetLoader.get_preferred_sampling_strategy()
#    → DatasetSamplingStrategy.SEQUENTIAL
# 5. Also extracts timing properties:
#    - has_timestamps=True (from data inspection)
#    - has_delays=False
# 6. Returns DatasetProperties

# Validation runs:
ℹ️  Automatically enabling fixed schedule mode for mooncake_trace dataset with timestamps
⚠️  WARNING: Dataset type mooncake_trace prefers sequential sampling, but random was specified.
    Consider using sequential for optimal behavior.

# Result:
DatasetProperties(
    detected_type=MOONCAKE_TRACE,        # From can_load()
    preferred_sampling_strategy=SEQUENTIAL, # From get_preferred_sampling_strategy()
    has_timestamps=True,                 # From data inspection
    has_delays=False,                    # From data inspection
    is_multi_turn=False,                 # From data inspection
    ...
)
```

#### Maintainability Win

When you add a new dataset type, you only update the loader:

```python
@CustomDatasetFactory.register(CustomDatasetType.NEW_CUSTOM_TYPE)
class NewCustomDatasetLoader(BaseFileLoader):
    @classmethod
    def can_load(cls, data: dict[str, Any] | None = None) -> bool:
        """Check if this is the new format."""
        if data is None:
            return False
        # Your detection logic here
        return "new_format_marker" in data and "special_field" in data

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """New format works best with shuffle sampling."""
        return DatasetSamplingStrategy.SHUFFLE

    def load_dataset(self): ...
    def convert_to_conversations(self): ...
```

**That's it!** DatasetInspector automatically:
- ✅ Detects the new type via `can_load()`
- ✅ Gets preferred strategy via `get_preferred_sampling_strategy()`
- ✅ No changes to inspector code needed
- ✅ Single source of truth maintained

---

### 3. Integration with UserConfig

```python
# src/aiperf/common/config/user_config.py (MODIFIED)

from aiperf.common.dataset_inspector import DatasetInspector, DatasetProperties

class UserConfig(BaseConfig):
    """User configuration with dataset-aware validation."""

    _timing_mode: TimingMode = TimingMode.REQUEST_RATE
    _dataset_properties: DatasetProperties | None = None

    @model_validator(mode="after")
    def validate_timing_mode(self) -> Self:
        """Set the timing mode based on the user config."""

        # Get dataset properties (file-based OR synthetic)
        self._dataset_properties = self._get_dataset_properties()

        if self._dataset_properties:
            self.debug(
                f"Dataset properties: has_timestamps={self._dataset_properties.has_timestamps}, "
                f"has_delays={self._dataset_properties.has_delays}, "
                f"is_multi_turn={self._dataset_properties.is_multi_turn}"
            )

        # Timing mode selection (now with dataset properties)
        if self.input.fixed_schedule:
            self._timing_mode = TimingMode.FIXED_SCHEDULE
        elif self._should_use_fixed_schedule_based_on_dataset():
            self._timing_mode = TimingMode.FIXED_SCHEDULE
            _logger.info(
                "Automatically enabling fixed schedule mode based on dataset properties"
            )
        elif self.loadgen.request_rate is not None:
            self._timing_mode = TimingMode.REQUEST_RATE
            if self.loadgen.request_rate_mode == RequestRateMode.CONCURRENCY_BURST:
                raise ValueError(
                    f"Request rate mode cannot be {RequestRateMode.CONCURRENCY_BURST!r} "
                    "when a request rate is specified."
                )
        else:
            # Default to concurrency burst mode
            if self.loadgen.concurrency is None:
                self.loadgen.concurrency = 1
            self._timing_mode = TimingMode.REQUEST_RATE
            self.loadgen.request_rate_mode = RequestRateMode.CONCURRENCY_BURST

        return self

    def _get_dataset_properties(self) -> DatasetProperties | None:
        """Get dataset properties from file inspection OR synthetic config.

        Two sources:
        1. File-based: Inspect file to discover properties
        2. Synthetic: Extract properties from config options
        """
        # File-based datasets: Inspect file
        if self.input.file:
            try:
                return DatasetInspector.inspect_file(
                    self.input.file,
                    sample_size=10  # Fast inspection
                )
            except Exception as e:
                _logger.warning(f"Could not inspect dataset file: {e}")
                return None

        # Synthetic datasets: Extract properties from config
        else:
            return self._extract_synthetic_properties()

    def _extract_synthetic_properties(self) -> DatasetProperties:
        """Extract dataset properties from synthetic generation config.

        For synthetic datasets, all properties are explicit in config options.
        No file inspection needed.
        """
        # Determine if multi-turn based on conversation.turn.mean
        is_multi_turn = self.input.conversation.turn.mean > 1
        max_turns = self.input.conversation.turn.mean  # Mean as estimate

        # Synthetic always uses delays (delay_ms), never absolute timestamps
        has_delays = (
            is_multi_turn and
            self.input.conversation.turn.delay.mean > 0
        )
        has_timestamps = False  # Synthetic doesn't use absolute timestamps

        # Estimated entry count
        estimated_count = self.input.conversation.num or 0

        return DatasetProperties(
            has_timestamps=False,  # Synthetic uses delays, not timestamps
            has_delays=has_delays,
            is_multi_turn=is_multi_turn,
            max_turns_seen=max_turns,
            estimated_entry_count=estimated_count,
            detected_type=None,  # Synthetic, not file-based
            has_session_ids=True,  # Synthetic always has session IDs
        )

    def _should_use_fixed_schedule_based_on_dataset(self) -> bool:
        """Determine if fixed schedule should be used based on dataset properties."""
        if not self._dataset_properties:
            return False

        # Use fixed schedule if mooncake trace with timestamps
        if (
            self._dataset_properties.detected_type == CustomDatasetType.MOONCAKE_TRACE
            and self._dataset_properties.has_timestamps
        ):
            return True

        # Use fixed schedule if multi-turn with timestamps (new capability!)
        if (
            self._dataset_properties.is_multi_turn
            and self._dataset_properties.has_timestamps
        ):
            return True

        return False

    @model_validator(mode="after")
    def validate_multi_turn_delay_warning(self) -> Self:
        """Warn if dataset has delays but concurrency is not set (NEW)."""
        if not self._dataset_properties:
            return self

        if (
            self._dataset_properties.has_delays
            and self._dataset_properties.is_multi_turn
            and self.loadgen.concurrency is None
            and self._timing_mode == TimingMode.REQUEST_RATE
        ):
            _logger.warning(
                "Dataset contains multi-turn conversations with delays, but --concurrency is not set. "
                "Delays will be IGNORED to maintain precise request rate. "
                "Set --concurrency to enable conversation-centric mode and respect delays."
            )

        return self

    @model_validator(mode="after")
    def validate_fixed_schedule_with_synthetic_options(self) -> Self:
        """Validate fixed schedule incompatible with synthetic options (NEW)."""
        if self._timing_mode != TimingMode.FIXED_SCHEDULE:
            return self

        # Check for synthetic turn delay options
        if (
            "mean" in self.input.conversation.turn.delay.model_fields_set
            or "stddev" in self.input.conversation.turn.delay.model_fields_set
            or "ratio" in self.input.conversation.turn.delay.model_fields_set
        ):
            raise ValueError(
                "Synthetic turn delay options (--conversation-turn-delay-mean/stddev/ratio) "
                "cannot be used with fixed schedule mode (trace replay). "
                "Fixed schedule uses timing data from the input file."
            )

        # Check for synthetic dataset options
        if "num_dataset_entries" in self.input.conversation.model_fields_set:
            _logger.warning(
                "--num-dataset-entries is ignored with fixed schedule mode. "
                "Entry count is determined by the input file."
            )

        return self

    @property
    def dataset_properties(self) -> DatasetProperties | None:
        """Get the inspected dataset properties (if available)."""
        return self._dataset_properties
```

---

### 3. Handling Synthetic Datasets

**Key Insight**: Synthetic datasets don't need file inspection because all properties are explicit in config options.

#### Synthetic vs File-Based Decision Tree

```
UserConfig.validate_timing_mode()
│
├─ Has input file?
│  │
│  ├─ YES → File-Based Dataset
│  │         │
│  │         ├─ Use DatasetInspector.inspect_file()
│  │         │  • Reads first N entries
│  │         │  • Discovers properties (timestamps, delays, etc.)
│  │         │  • Returns DatasetProperties
│  │         │
│  │         └─ Properties: DISCOVERED from file
│  │
│  └─ NO → Synthetic Dataset
│            │
│            ├─ Extract properties from config
│            │  • conversation.turn.mean → is_multi_turn
│            │  • conversation.turn.delay.mean → has_delays
│            │  • No timestamps (synthetic uses delays)
│            │
│            └─ Properties: EXTRACTED from config
│
└─ Use properties for validation & mode selection
```

#### Synthetic Properties Extraction

```python
def _extract_synthetic_properties(self) -> DatasetProperties:
    """For synthetic, properties come from config options."""

    # Multi-turn if conversation.turn.mean > 1
    is_multi_turn = self.input.conversation.turn.mean > 1

    # Has delays if multi-turn AND delay.mean > 0
    has_delays = (
        is_multi_turn and
        self.input.conversation.turn.delay.mean > 0
    )

    # Synthetic NEVER uses absolute timestamps
    has_timestamps = False

    return DatasetProperties(
        has_timestamps=False,      # Synthetic uses delays, not timestamps
        has_delays=has_delays,     # From turn.delay.mean
        is_multi_turn=is_multi_turn,  # From turn.mean
        max_turns_seen=self.input.conversation.turn.mean,
        estimated_entry_count=self.input.conversation.num or 0,
        detected_type=None,        # Not file-based
        has_session_ids=True,      # Synthetic always has sessions
    )
```

#### Example: Synthetic Multi-Turn with Delays

```bash
# Synthetic multi-turn with delays
aiperf \
  --conversation-turn-mean 3 \
  --conversation-turn-delay-mean 1000 \
  --request-rate 10 \
  --endpoint-type openai \
  --model gpt-4

# Config validation extracts properties:
# - has_delays=True (from delay.mean=1000)
# - is_multi_turn=True (from turn.mean=3)
# - No file inspection needed!

# Output:
⚠️  WARNING: Dataset contains multi-turn conversations with delays, but --concurrency is not set.
    Delays will be IGNORED to maintain precise request rate.
    Set --concurrency to enable conversation-centric mode and respect delays.
```

#### Synthetic + Concurrency (Correct Usage)

```bash
# Synthetic multi-turn with delays AND concurrency
aiperf \
  --conversation-turn-mean 3 \
  --conversation-turn-delay-mean 1000 \
  --request-rate 10 \
  --concurrency 20 \
  --endpoint-type openai \
  --model gpt-4

# Config validation extracts properties:
# - has_delays=True
# - concurrency=20
# - No warning! Delays will be respected.
```

#### Why This Works

| Aspect | File-Based | Synthetic |
|--------|------------|-----------|
| **Properties Source** | Discovered via inspection | Extracted from config |
| **Inspection Needed?** | ✅ Yes | ❌ No |
| **Has Timestamps?** | Maybe (check file) | ❌ Never (uses delays) |
| **Has Delays?** | Maybe (check file) | ✅ If delay.mean > 0 |
| **Is Multi-Turn?** | Maybe (check file) | ✅ If turn.mean > 1 |
| **Fast?** | ✅ Samples only 10 entries | ✅ No I/O at all |

---

### 4. Complete Validation Flow

```python
# UserConfig validation flow with both file-based and synthetic

@model_validator(mode="after")
def validate_multi_turn_delay_warning(self) -> Self:
    """Warn if dataset has delays but concurrency not set.

    Works for BOTH file-based AND synthetic datasets!
    """
    if not self._dataset_properties:
        return self

    # This check works regardless of dataset source
    if (
        self._dataset_properties.has_delays
        and self._dataset_properties.is_multi_turn
        and self.loadgen.concurrency is None
        and self._timing_mode == TimingMode.REQUEST_RATE
    ):
        _logger.warning(
            "Dataset contains multi-turn conversations with delays, but --concurrency is not set. "
            "Delays will be IGNORED to maintain precise request rate. "
            "Set --concurrency to enable conversation-centric mode and respect delays."
        )

    return self
```

**Key Point**: Validation logic is **identical** for file-based and synthetic because both produce `DatasetProperties`. The validator doesn't care where properties came from!

---

### 5. Usage Examples

#### Example 1: File-Based with Early Warning

```bash
# File-based dataset with delays, no concurrency
aiperf \
  --input-file conversations_with_delays.jsonl \
  --custom-dataset-type multi_turn \
  --request-rate 100 \
  --endpoint-type openai \
  --model gpt-4

# DatasetInspector reads file and discovers:
# - has_delays=True
# - is_multi_turn=True

# Output (during config validation, BEFORE services start):
⚠️  WARNING: Dataset contains multi-turn conversations with delays, but --concurrency is not set.
    Delays will be IGNORED to maintain precise request rate.
    Set --concurrency to enable conversation-centric mode and respect delays.

✓ Configuration validated successfully
Starting services...
```

#### Example 2: Incompatible Options Detected

```bash
aiperf \
  --input-file trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --conversation-turn-delay-mean 1000

# Output:
❌ ERROR: Synthetic turn delay options (--conversation-turn-delay-mean/stddev/ratio)
   cannot be used with fixed schedule mode (trace replay).
   Fixed schedule uses timing data from the input file.
```

#### Example 3: Auto-Detection with Properties

```bash
aiperf \
  --input-file multi_turn_with_timestamps.jsonl \
  --custom-dataset-type multi_turn \
  --endpoint-type openai \
  --model gpt-4

# Output:
ℹ️  Dataset inspection: has_timestamps=True, has_delays=False, is_multi_turn=True
ℹ️  Automatically enabling fixed schedule mode based on dataset properties
✓ Configuration validated successfully
Using Fixed Schedule strategy
```

---

## Benefits

### 1. Early Validation
- ✅ Catch configuration errors **before** services start
- ✅ Provide clear, actionable error messages at CLI level
- ✅ Fail fast with helpful guidance

### 2. Better User Experience
- ✅ Warnings about ignored options (delays without concurrency)
- ✅ Clear explanation of auto-detected modes
- ✅ No surprises during benchmark execution

### 3. Clean Separation of Concerns
- ✅ **DatasetInspector**: Lightweight, fast, stateless inspection
- ✅ **DatasetManager**: Full dataset loading, processing, metadata generation
- ✅ **UserConfig**: Validation with dataset-aware rules
- ✅ No circular dependencies

### 4. Maintainability
- ✅ Reusable utility for other components
- ✅ Easy to extend with new properties
- ✅ Testable in isolation
- ✅ Clear API and responsibilities

### 5. Performance
- ✅ Samples only first N entries (default: 10)
- ✅ Synchronous, suitable for CLI validation
- ✅ Minimal file I/O
- ✅ No full dataset parsing overhead

---

## Migration Path

### Phase 1: Add DatasetInspector (Non-Breaking)
1. Create `src/aiperf/common/dataset_inspector.py`
2. Add tests
3. No changes to existing code yet

### Phase 2: Integrate with UserConfig (Breaking for edge cases)
1. Modify `user_config.py` to use DatasetInspector
2. Add new validation rules
3. Keep existing mooncake detection as fallback

### Phase 3: Enhance Validation (Gradual)
1. Add more validation rules as needed
2. Improve detection heuristics
3. Add more properties to DatasetProperties

### Phase 4: Cleanup (Optional)
1. Remove duplicate detection logic
2. Consolidate validation rules
3. Document new validation flow

---

## Testing Strategy

```python
# tests/unit/common/test_dataset_inspector.py

def test_inspect_mooncake_trace_with_timestamps():
    """Test detection of mooncake trace with timestamps."""
    # Create temp file with mooncake trace data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write('{"timestamp": 1000, "input_length": 100, "output_length": 50}\n')
        f.write('{"timestamp": 2000, "input_length": 200, "output_length": 75}\n')
        temp_file = f.name

    try:
        props = DatasetInspector.inspect_file(temp_file)

        assert props.has_timestamps is True
        assert props.has_delays is False
        assert props.is_multi_turn is False
        assert props.detected_type == CustomDatasetType.MOONCAKE_TRACE
        assert props.estimated_entry_count == 2
    finally:
        Path(temp_file).unlink()


def test_inspect_multi_turn_with_delays():
    """Test detection of multi-turn conversations with delays."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write('{"session_id": "s1", "turns": [{"text": "Hi", "delay": 0}, {"text": "Hello", "delay": 1000}]}\n')
        temp_file = f.name

    try:
        props = DatasetInspector.inspect_file(temp_file)

        assert props.has_delays is True
        assert props.has_timestamps is False
        assert props.is_multi_turn is True
        assert props.max_turns_seen == 2
        assert props.has_session_ids is True
        assert props.detected_type == CustomDatasetType.MULTI_TURN
    finally:
        Path(temp_file).unlink()


def test_inspect_with_custom_sample_size():
    """Test inspection with custom sample size."""
    # Create file with 100 entries
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(100):
            f.write(f'{{"text": "prompt {i}"}}\n')
        temp_file = f.name

    try:
        # Sample only first 5
        props = DatasetInspector.inspect_file(temp_file, sample_size=5)

        assert props.estimated_entry_count == 100  # Counts all
        # But only parsed first 5 for detection
    finally:
        Path(temp_file).unlink()


def test_user_config_warns_about_delays_without_concurrency():
    """Test that UserConfig warns when delays exist but concurrency not set."""
    # This would be an integration test
    # Create temp file with delays
    # Create UserConfig
    # Assert warning was logged
    pass
```

---

## Alternative Approaches Considered

### Alternative 1: Two-Phase Validation
**Approach**: Validate basic config at CLI, then re-validate after dataset loads

**Pros**:
- No need for dataset inspector
- Full dataset metadata available for validation

**Cons**:
- ❌ Services already started before full validation
- ❌ Late error discovery (poor UX)
- ❌ Validation logic split across two phases (confusing)
- ❌ Can't provide helpful warnings before benchmark starts

**Verdict**: ❌ Rejected - poor user experience

---

### Alternative 2: Require Explicit Dataset Properties
**Approach**: Force users to declare properties via CLI flags

**Example**:
```bash
aiperf \
  --input-file dataset.jsonl \
  --dataset-has-delays \
  --dataset-is-multi-turn \
  --custom-dataset-type multi_turn
```

**Pros**:
- No file reading during validation
- Explicit, no magic

**Cons**:
- ❌ Terrible UX (too many flags)
- ❌ Users will get it wrong
- ❌ Doesn't solve auto-detection problem

**Verdict**: ❌ Rejected - poor usability

---

### Alternative 3: Deferred Mode Selection
**Approach**: Don't select timing mode during config validation, defer to TimingManager

**Pros**:
- Full dataset metadata available when selecting mode

**Cons**:
- ❌ Config becomes less predictable
- ❌ Can't provide early warnings
- ❌ Harder to understand what will happen
- ❌ Services already started before mode selection

**Verdict**: ❌ Rejected - poor predictability

---

## Recommendation

**Implement the Lightweight Dataset Inspector** as proposed.

### Why This Approach Wins:

1. ✅ **Clean Separation**: Inspector is independent, reusable utility
2. ✅ **Early Validation**: Catches issues before services start
3. ✅ **Fast & Efficient**: Samples only what's needed
4. ✅ **Good UX**: Clear warnings and errors at CLI level
5. ✅ **Extensible**: Easy to add new properties and validations
6. ✅ **Testable**: Pure functions, easy to unit test
7. ✅ **Non-Breaking**: Can be added incrementally

### Next Steps:

1. **Create DatasetInspector module** with DatasetProperties model
2. **Add unit tests** for inspector with various dataset formats
3. **Integrate with UserConfig** for timing mode selection
4. **Add validation rules** for common pitfalls (delays without concurrency, etc.)
5. **Update documentation** to explain new validation behavior
6. **Add integration tests** for end-to-end validation flow

---

## Open Questions

1. **Sample size**: Is 10 entries enough? Should it be configurable?
   - **Recommendation**: Start with 10, make it configurable if needed

2. **Caching**: Should we cache inspection results?
   - **Recommendation**: No - config validation runs once, caching not needed

3. **Error handling**: What if file is malformed?
   - **Recommendation**: Log warning, continue with partial properties

4. **Performance**: What if file is huge (GB)?
   - **Recommendation**: Sample-based approach handles this well (only reads first N lines)

5. **Dataset type detection**: What if ambiguous?
   - **Recommendation**: Return `None` for detected_type, require explicit `--custom-dataset-type`
