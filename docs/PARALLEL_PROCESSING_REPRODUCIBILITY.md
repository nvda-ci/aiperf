<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Parallel Processing with Reproducibility for Mooncake Traces

## Overview

This document explains the design and implementation of parallel processing support for Mooncake trace loading with perfect reproducibility guarantees. The solution enables multiple workers to process requests with hash IDs independently while ensuring identical results across runs and workers.

## Problem Statement

When processing Mooncake traces with KV cache hash IDs in parallel:

1. **Cache Requirement**: Each hash ID represents a KV block that should produce identical token sequences across all workers
2. **Independence Requirement**: Different workers may process different subsets of hash IDs
3. **Order Independence**: The order in which hash IDs are processed should not affect results
4. **Reproducibility**: Given the same global seed, all workers must produce identical content for the same hash ID

Traditional approaches fail because:
- Shared RNG state creates order dependencies
- Worker-specific seeds break reproducibility across workers
- Cache misses on different workers would generate different content

## Solution: Hash-ID-Based Re-Seeding

### Architecture

```
Global Seed (42)
    ↓
Base RNG Derivation ("dataset.prompt.corpus")
    ↓
HashIdRandomGenerator (wraps base RNG)
    ↓
For each hash_id: reseed_for_hash_id(hash_id)
    ↓
Deterministic seed = SHA256(base_seed + ":" + hash_id)
    ↓
Independent, reproducible random sequence
```

### Key Components

#### 1. `HashIdRandomGenerator` (Subclass of `RandomGenerator`)

Located in: `aiperf/common/hash_id_random_generator.py`

**Features:**
- Subclass of `RandomGenerator` with added `reseed_for_hash_id()` method
- Efficiently re-seeds the same RNG instance (no new instance creation)
- Deterministic seed derivation using SHA-256 hash
- Maintains base seed for reproducibility

**Usage:**
```python
from aiperf.common import random_generator as rng
from aiperf.common.hash_id_random_generator import HashIdRandomGenerator

# Initialize once per worker
base_rng = rng.derive("dataset.prompt.corpus")
hash_rng = HashIdRandomGenerator.from_base_rng(base_rng)

# For each hash_id that needs processing
hash_rng.reseed_for_hash_id(123)
tokens = sample_tokens(hash_rng)  # Deterministic for hash_id=123
```

#### 2. Enhanced `PromptGenerator`

Located in: `aiperf/dataset/generator/prompt.py`

**Changes:**
- Added `_hash_id_corpus_rng`: `HashIdRandomGenerator` instance for hash-ID-based generation
- Modified `_generate_cached_prompt()`: Re-seeds RNG before generating tokens for each hash ID
- Modified `_sample_tokens()`: Accepts optional `use_rng` parameter

**Flow:**
```python
def _generate_cached_prompt(self, num_tokens, hash_ids, block_size):
    for hash_id in hash_ids:
        if hash_id not in self._cache:
            # Re-seed for deterministic generation
            self._hash_id_corpus_rng.reseed_for_hash_id(hash_id)

            # Generate tokens using the re-seeded RNG
            tokens = self._sample_tokens(
                current_block_size,
                use_rng=self._hash_id_corpus_rng
            )

            # Cache for this hash_id
            self._cache[hash_id] = tokens
```

## Guarantees

### 1. **Deterministic Per Hash ID**
Same hash ID always produces the same token sequence:
```python
# Run 1
hash_rng.reseed_for_hash_id(123)
tokens_1 = generate_tokens()

# Run 2 (same hash_id)
hash_rng.reseed_for_hash_id(123)
tokens_2 = generate_tokens()

assert tokens_1 == tokens_2  # ✓ Always true
```

### 2. **Worker Independence**
Different workers produce identical results for the same hash ID:
```python
# Worker 1
worker1_rng = HashIdRandomGenerator.from_base_rng(rng.derive("worker.corpus"))
worker1_rng.reseed_for_hash_id(456)
worker1_tokens = generate_tokens()

# Worker 2 (different process/machine, same global seed)
worker2_rng = HashIdRandomGenerator.from_base_rng(rng.derive("worker.corpus"))
worker2_rng.reseed_for_hash_id(456)
worker2_tokens = generate_tokens()

assert worker1_tokens == worker2_tokens  # ✓ Always true
```

### 3. **Order Independence**
Processing order doesn't affect results:
```python
# Process order: 100, 200, 300
for hash_id in [100, 200, 300]:
    hash_rng.reseed_for_hash_id(hash_id)
    cache[hash_id] = generate_tokens()

# Process order: 300, 100, 200 (different order)
for hash_id in [300, 100, 200]:
    hash_rng.reseed_for_hash_id(hash_id)
    cache2[hash_id] = generate_tokens()

assert cache[100] == cache2[100]  # ✓ All match
assert cache[200] == cache2[200]
assert cache[300] == cache2[300]
```

### 4. **Independent Caches**
Each worker maintains its own cache without coordination:
```python
# Worker 1 processes: hash_ids [1, 2, 3]
# Worker 2 processes: hash_ids [3, 4, 5]
# Both produce identical tokens for hash_id=3
# No cache synchronization needed
```

## Implementation Details

### Seed Derivation

The seed for each hash ID is derived using SHA-256:

```python
def reseed_for_hash_id(self, hash_id: int) -> None:
    seed_string = f"{self._base_seed}:hash_id:{hash_id}"
    hash_bytes = hashlib.sha256(seed_string.encode("utf-8")).digest()
    derived_seed = int.from_bytes(hash_bytes[:8], byteorder="big")

    # Re-seed both Python and NumPy RNGs
    self._seed = derived_seed
    self._python_rng.seed(derived_seed)
    self._numpy_rng = np.random.default_rng(derived_seed)
```

**Why SHA-256?**
- Cryptographically strong hash function
- Produces uniform distribution of seeds
- Same hash → same seed (deterministic)
- Different hashes → independent seeds (no collisions)
- Cross-platform stable

### Performance Considerations

**Re-seeding Cost:**
- Re-seeding is O(1) operation
- Much cheaper than creating new RNG instances
- Negligible overhead compared to token generation
- Enables efficient cache reuse

**Memory Efficiency:**
- Single `HashIdRandomGenerator` instance per worker
- Individual caches per worker (no shared state)
- No inter-process communication needed

## Usage Examples

### Basic Usage

```python
from aiperf.common import random_generator as rng
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.common.config import PromptConfig
from aiperf.common.tokenizer import Tokenizer

# Initialize global RNG (once at startup)
rng.init(42)

# Create prompt generator
config = PromptConfig(...)
tokenizer = Tokenizer(...)
generator = PromptGenerator(config, tokenizer)

# Generate prompt with hash IDs
prompt = generator.generate(
    mean=1000,
    stddev=100,
    hash_ids=[123, 456, 789]  # KV block hashes from Mooncake trace
)
```

### Parallel Worker Setup

```python
# Worker initialization (each process)
def worker_init(global_seed):
    rng.init(global_seed)  # Same seed for all workers

    # Each worker gets its own generator and cache
    generator = PromptGenerator(config, tokenizer)
    return generator

# Worker processing
def worker_process(generator, trace_entry):
    if trace_entry.hash_ids:
        prompt = generator.generate(
            mean=trace_entry.input_length,
            hash_ids=trace_entry.hash_ids
        )
    else:
        prompt = generator.generate(mean=trace_entry.input_length)

    return prompt
```

### Testing Reproducibility

```python
import multiprocessing as mp

def test_parallel_reproducibility():
    # Run 1: Process traces in parallel
    with mp.Pool(processes=4, initializer=worker_init, initargs=(42,)) as pool:
        results_1 = pool.map(worker_process, traces)

    # Run 2: Process same traces with same seed
    with mp.Pool(processes=4, initializer=worker_init, initargs=(42,)) as pool:
        results_2 = pool.map(worker_process, traces)

    # Results are identical
    assert results_1 == results_2
```

## Best Practices

### 1. **Initialize RNG Once**
```python
# ✓ Good: Initialize once at startup
rng.init(global_seed)

# ✗ Bad: Re-initializing during processing
for trace in traces:
    rng.init(global_seed)  # Don't do this
```

### 2. **One Generator Per Worker**
```python
# ✓ Good: Each worker has its own generator
def worker_init():
    return PromptGenerator(config, tokenizer)

# ✗ Bad: Sharing generator across workers
generator = PromptGenerator(config, tokenizer)  # Don't share
```

### 3. **Consistent Derivation Names**
```python
# ✓ Good: Use consistent derivation names
corpus_rng = rng.derive("dataset.prompt.corpus")

# ✗ Bad: Different names for same purpose
corpus_rng1 = rng.derive("corpus")
corpus_rng2 = rng.derive("dataset.corpus")
```

### 4. **Cache Locally**
```python
# ✓ Good: Each worker maintains its own cache
class PromptGenerator:
    def __init__(self):
        self._cache = {}  # Local to this instance

# ✗ Bad: Shared cache across workers
global_cache = {}  # Don't do this
```

## Testing

Comprehensive test suite in `tests/test_hash_id_random_generator.py`:

- ✓ Deterministic seeding per hash ID
- ✓ Independence across workers
- ✓ Order independence
- ✓ Parallel cache simulation
- ✓ Non-deterministic mode
- ✓ Multiple random operations
- ✓ Inheritance of all RandomGenerator methods
- ✓ Hash collision independence
- ✓ Edge cases (zero, negative, large hash IDs)

Run tests:
```bash
pytest tests/test_hash_id_random_generator.py -v
```

## Conclusion

The hash-ID-based re-seeding approach provides:
- ✓ Perfect reproducibility across runs and workers
- ✓ Order-independent processing
- ✓ Independent worker caches
- ✓ Minimal performance overhead
- ✓ Clean, maintainable implementation

This design enables efficient parallel processing of Mooncake traces while maintaining the strict reproducibility guarantees required for benchmarking and testing.

