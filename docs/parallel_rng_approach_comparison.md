<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Parallel Processing RNG Approach Comparison

This document compares three approaches for achieving reproducible random number generation during parallel processing of Mooncake traces.

## Problem Statement

When generating synthetic prompts for Mooncake traces, each `hash_id` needs to produce a deterministic slice of the corpus. In parallel processing:

- Multiple workers process different sessions/traces
- The same `hash_id` may appear in multiple sessions (KV cache reuse simulation)
- Processing order varies between runs
- **Requirement:** Same seed + same hash_id = identical output, regardless of worker or order

---

## Approach 1: HashIdRandomGenerator (Current Implementation)

### Description

Each `hash_id` derives its own deterministic seed via SHA256 hashing:

```python
seed_string = f"{base_seed}:hash_id:{hash_id}"
derived_seed = SHA256(seed_string)[:8]
rng.seed(derived_seed)
start_idx = rng.randrange(corpus_size)
```

Workers only need the `base_seed` - they independently compute the same `start_idx` for any given `hash_id`.

### Pros

| Benefit | Description |
|---------|-------------|
| **Already implemented** | Class exists, 16 unit tests pass, integrated into `PromptGenerator` |
| **Minimal coordination** | Workers only need `base_seed` (single integer) |
| **Streaming compatible** | Handles unknown hash_ids arriving at runtime |
| **Memory efficient** | No mapping to store - O(1) space for RNG state |
| **Order independent** | Any worker, any order, same result |
| **Stateless workers** | Workers are fully independent, no shared mutable state |

### Cons

| Drawback | Description |
|----------|-------------|
| **SHA256 overhead** | ~50-100ns per hash_id (negligible in practice) |
| **Conceptual complexity** | Deterministic derivation via hashing is less intuitive |
| **Over-engineered?** | Streaming capability unused - all hash_ids known upfront |
| **NumPy RNG disabled** | Intentionally blocked, limits some operations |

### Current Implementation Status

```
src/aiperf/common/hash_id_random_generator.py  - Complete
tests/test_hash_id_random_generator.py         - 16 tests passing
src/aiperf/dataset/generator/prompt.py         - Integrated (line 49, 223)
Actual parallelization                         - NOT IMPLEMENTED
```

---

## Approach 2: Pre-compute Indices

### Description

During the existing serial pass (session grouping), collect all unique hash_ids and pre-compute their corpus indices:

```python
# In load_dataset() - already iterates all traces
all_hash_ids = set()
for trace in all_traces:
    all_hash_ids.update(trace.hash_ids or [])

# Pre-compute mapping
index_mapping = {}
for hash_id in sorted(all_hash_ids):  # sorted for determinism
    index_mapping[hash_id] = rng.randrange(corpus_size)

# Workers receive index_mapping, do simple lookups
start_idx = index_mapping[hash_id]
```

### Pros

| Benefit | Description |
|---------|-------------|
| **Simpler mental model** | All randomness happens in one place, serially |
| **Faster lookups** | Dict lookup (~50ns) vs SHA256 + reseed (~100ns) |
| **Easier debugging** | Can inspect/log the entire mapping upfront |
| **No special RNG class** | Uses standard RandomGenerator |
| **Explicit over implicit** | The mapping is a concrete, inspectable artifact |

### Cons

| Drawback | Description |
|----------|-------------|
| **Requires all hash_ids upfront** | Cannot handle streaming/incremental traces |
| **Additional serial work** | Must iterate hash_ids and compute indices before parallelization |
| **Memory overhead** | O(n) storage for mapping where n = unique hash_ids |
| **Coordination required** | Must distribute mapping to all workers |
| **Throws away existing work** | HashIdRandomGenerator implementation discarded |
| **Coupling** | load_dataset() now responsible for RNG concerns |

### Implementation Status

```
Not implemented - would require new code
```

---

## Approach 3: Pre-compute Full Token Arrays

### Description

Pre-compute the entire token array for each unique hash_id upfront. Workers receive a mapping of `hash_id -> list[int]` and only need to concatenate tokens and detokenize - no corpus access, no RNG, no index lookups:

```python
# In load_dataset() - pre-compute full token arrays
all_hash_ids = set()
for trace in all_traces:
    all_hash_ids.update(trace.hash_ids or [])

# Pre-compute FULL token arrays for each hash_id
token_mapping: dict[int, list[int]] = {}
for hash_id in sorted(all_hash_ids):
    start_idx = rng.randrange(corpus_size)
    token_mapping[hash_id] = corpus[start_idx:start_idx + block_size]

# Workers receive token_mapping - just concatenate and decode
tokens = []
for hash_id in trace.hash_ids:
    tokens.extend(token_mapping[hash_id])
prompt = tokenizer.decode(tokens)
```

### Pros

| Benefit | Description |
|---------|-------------|
| **Simplest worker logic** | Workers just concatenate pre-computed tokens and decode |
| **No corpus distribution** | Workers don't need access to the corpus at all |
| **No RNG in workers** | All randomness happens upfront, serially |
| **No index lookups** | Direct access to tokens, no indirection |
| **Easiest to debug** | Can inspect exact tokens for any hash_id before parallelization |
| **Fastest worker execution** | Just list concatenation + detokenize |

### Cons

| Drawback | Description |
|----------|-------------|
| **Massive memory overhead** | O(n × block_size) integers where n = unique hash_ids |
| **Huge serialization cost** | Must pickle/serialize full token arrays to each worker |
| **Long serial pre-computation** | Must generate all token arrays before any parallelization |
| **Inflexible block sizes** | Pre-computed for fixed block_size; variable sizes need recomputation |
| **Requires all hash_ids upfront** | Cannot handle streaming/incremental traces |
| **Throws away existing work** | HashIdRandomGenerator implementation discarded |

### Memory Analysis

```
Assumptions:
- block_size = 512 tokens
- unique_hash_ids = 100,000 (typical for large trace file)
- Each token ID = 4 bytes (int32)

Memory for token_mapping:
= 100,000 hash_ids × 512 tokens × 4 bytes
= 204,800,000 bytes
≈ 195 MB just for the mapping

Compare to:
- HashIdRandomGenerator: ~100 bytes (RNG state)
- Pre-compute indices: 100,000 × 8 bytes ≈ 0.8 MB
```

For very large trace files with millions of unique hash_ids, this becomes prohibitive.

### Implementation Status

```
Not implemented - would require new code
```

---

## Side-by-Side Comparison

| Aspect | HashIdRandomGenerator | Pre-compute Indices | Pre-compute Tokens |
|--------|----------------------|---------------------|-------------------|
| Implementation effort | Low (finish existing) | Medium (start fresh) | Medium (start fresh) |
| Code already written | ~200 lines + tests | 0 | 0 |
| Data passed to workers | `base_seed` (8 bytes) | `Dict[int, int]` (~16 bytes × n) | `Dict[int, list[int]]` (~2KB × n) |
| Per-hash_id cost | SHA256 + reseed (~100ns) | Dict lookup (~50ns) | Dict lookup (~50ns) |
| Memory footprint | O(1) | O(n) | O(n × block_size) |
| Workers need corpus? | Yes | Yes | **No** |
| Workers need RNG? | Yes (reseed) | No | No |
| Streaming support | Yes | No | No |
| Debugging | Compute on demand | Inspect index mapping | Inspect full tokens |
| Conceptual simplicity | Medium | High | **Highest** |
| Serialization overhead | Minimal | Low | **High** |
| Scalability (large n) | Excellent | Good | Poor |

---

## Proposed Implementations

### Implementation A: Finish HashIdRandomGenerator Parallelization

```python
# src/aiperf/dataset/loader/mooncake_trace.py

from concurrent.futures import ProcessPoolExecutor
from functools import partial

class MooncakeTraceDatasetLoader:

    def convert_to_conversations_parallel(
        self,
        data: dict[str, list[MooncakeTrace]],
        num_workers: int = None
    ) -> list[Conversation]:
        """Convert traces to conversations using parallel workers.

        Each worker independently processes assigned sessions using
        HashIdRandomGenerator for reproducible prompt generation.
        """
        num_workers = num_workers or min(os.cpu_count() or 4, 8)
        sessions = list(data.items())

        # Partition sessions across workers
        # Each worker gets a subset of sessions to process
        chunk_size = max(1, len(sessions) // num_workers)
        session_chunks = [
            sessions[i:i + chunk_size]
            for i in range(0, len(sessions), chunk_size)
        ]

        # Get base seed for workers (they'll derive HashIdRandomGenerator from this)
        base_seed = rng.get_seed()

        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            worker_fn = partial(
                _process_session_chunk,
                prompt_config=self.prompt_generator.config,
                tokenizer_name=self.prompt_generator.tokenizer.model_name,
                base_seed=base_seed,
            )
            results = list(executor.map(worker_fn, session_chunks))

        # Flatten results
        return [conv for chunk_result in results for conv in chunk_result]


def _process_session_chunk(
    session_chunk: list[tuple[str, list[MooncakeTrace]]],
    prompt_config: PromptConfig,
    tokenizer_name: str,
    base_seed: int,
) -> list[Conversation]:
    """Worker function: process a chunk of sessions.

    Each worker creates its own PromptGenerator with HashIdRandomGenerator
    seeded from the shared base_seed, ensuring reproducibility.
    """
    # Each worker creates its own tokenizer and generator
    # HashIdRandomGenerator ensures same hash_id -> same output across workers
    tokenizer = Tokenizer.from_pretrained(tokenizer_name)

    # Initialize RNG with shared seed so HashIdRandomGenerator derives same base
    rng.reset()
    rng.init(base_seed)

    generator = PromptGenerator(prompt_config, tokenizer)

    conversations = []
    for session_id, traces in session_chunk:
        conversation = Conversation(session_id=session_id)
        for trace in traces:
            if trace.text_input is not None:
                prompt = trace.text_input
            else:
                prompt = generator.generate(
                    mean=trace.input_length,
                    stddev=0,
                    hash_ids=trace.hash_ids or [],
                )

            turn = Turn(
                timestamp=trace.timestamp,
                delay=trace.delay,
                texts=[Text(name="text", contents=[prompt])],
                max_tokens=trace.output_length,
            )
            conversation.turns.append(turn)
        conversations.append(conversation)

    return conversations
```

**Additional changes needed:**
1. Fix test bug at `tests/test_mooncake_trace_reproducibility.py:155`
2. Add `get_seed()` method to random_generator module if not present
3. Ensure `Tokenizer` and `PromptConfig` are picklable for multiprocessing

---

### Implementation B: Pre-compute Indices Approach

```python
# src/aiperf/dataset/loader/mooncake_trace.py

from concurrent.futures import ProcessPoolExecutor
from functools import partial

class MooncakeTraceDatasetLoader:

    def load_dataset(self) -> tuple[dict[str, list[MooncakeTrace]], dict[int, int]]:
        """Load Mooncake trace data and pre-compute hash_id index mapping.

        Returns:
            Tuple of (session_data, hash_id_to_corpus_index mapping)
        """
        data: dict[str, list[MooncakeTrace]] = defaultdict(list)
        all_hash_ids: set[int] = set()

        with open(self.filename) as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue

                trace_data = MooncakeTrace.model_validate_json(line)

                if (
                    trace_data.timestamp is not None
                    and not self._timestamp_within_offsets(trace_data.timestamp)
                ):
                    self._skipped_traces += 1
                    continue

                session_id = trace_data.session_id or str(uuid.uuid4())
                data[session_id].append(trace_data)

                # Collect hash_ids for pre-computation
                if trace_data.hash_ids:
                    all_hash_ids.update(trace_data.hash_ids)

        # Pre-compute corpus indices for all hash_ids
        index_mapping = self._precompute_hash_indices(all_hash_ids)

        return data, index_mapping

    def _precompute_hash_indices(self, hash_ids: set[int]) -> dict[int, int]:
        """Pre-compute corpus start indices for all hash_ids.

        Uses sorted iteration for deterministic ordering.
        """
        corpus_size = self.prompt_generator._corpus_size
        corpus_rng = rng.derive("dataset.prompt.corpus.precompute")

        mapping = {}
        for hash_id in sorted(hash_ids):  # sorted for determinism
            mapping[hash_id] = corpus_rng.randrange(corpus_size)

        self.debug(lambda: f"Pre-computed indices for {len(mapping):,} unique hash_ids")
        return mapping

    def convert_to_conversations_parallel(
        self,
        data: dict[str, list[MooncakeTrace]],
        index_mapping: dict[int, int],
        num_workers: int = None
    ) -> list[Conversation]:
        """Convert traces to conversations using parallel workers."""
        num_workers = num_workers or min(os.cpu_count() or 4, 8)
        sessions = list(data.items())

        chunk_size = max(1, len(sessions) // num_workers)
        session_chunks = [
            sessions[i:i + chunk_size]
            for i in range(0, len(sessions), chunk_size)
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            worker_fn = partial(
                _process_session_chunk_precompute,
                index_mapping=index_mapping,
                tokenizer_name=self.prompt_generator.tokenizer.model_name,
                corpus_path=self.prompt_generator._corpus_path,
                block_size=self.prompt_generator.config.input_tokens.block_size,
            )
            results = list(executor.map(worker_fn, session_chunks))

        return [conv for chunk_result in results for conv in chunk_result]


def _process_session_chunk_precompute(
    session_chunk: list[tuple[str, list[MooncakeTrace]]],
    index_mapping: dict[int, int],
    tokenizer_name: str,
    corpus_path: str,
    block_size: int,
) -> list[Conversation]:
    """Worker function using pre-computed index mapping."""
    tokenizer = Tokenizer.from_pretrained(tokenizer_name)

    # Load corpus in worker (or use shared memory for optimization)
    corpus = _load_tokenized_corpus(corpus_path, tokenizer)

    conversations = []
    for session_id, traces in session_chunk:
        conversation = Conversation(session_id=session_id)
        for trace in traces:
            if trace.text_input is not None:
                prompt = trace.text_input
            else:
                prompt = _generate_prompt_from_mapping(
                    hash_ids=trace.hash_ids or [],
                    num_tokens=trace.input_length,
                    index_mapping=index_mapping,
                    corpus=corpus,
                    block_size=block_size,
                    tokenizer=tokenizer,
                )

            turn = Turn(
                timestamp=trace.timestamp,
                delay=trace.delay,
                texts=[Text(name="text", contents=[prompt])],
                max_tokens=trace.output_length,
            )
            conversation.turns.append(turn)
        conversations.append(conversation)

    return conversations


def _generate_prompt_from_mapping(
    hash_ids: list[int],
    num_tokens: int,
    index_mapping: dict[int, int],
    corpus: list[int],
    block_size: int,
    tokenizer: Tokenizer,
) -> str:
    """Generate prompt using pre-computed index mapping."""
    final_tokens = []
    corpus_size = len(corpus)

    for i, hash_id in enumerate(hash_ids):
        # Determine block size (last block may be smaller)
        if i == len(hash_ids) - 1:
            current_block_size = num_tokens - (len(hash_ids) - 1) * block_size
        else:
            current_block_size = block_size

        # Look up pre-computed index
        start_idx = index_mapping[hash_id]

        # Extract tokens (with wraparound)
        end_idx = start_idx + current_block_size
        if end_idx <= corpus_size:
            tokens = corpus[start_idx:end_idx]
        else:
            tokens = corpus[start_idx:] + corpus[:end_idx - corpus_size]

        # Add block separator if available
        if tokenizer.block_separation_token_id is not None:
            tokens = [tokenizer.block_separation_token_id] + tokens[:-1]

        final_tokens.extend(tokens)

    return tokenizer.decode(final_tokens, skip_special_tokens=False)
```

**Additional changes needed:**
1. Modify `load_dataset()` signature (breaking change) or add new method
2. Extract corpus loading to standalone function
3. Handle corpus path storage in PromptGenerator
4. Potentially use shared memory for corpus to avoid copying to each worker

---

### Implementation C: Pre-compute Full Token Arrays

```python
# src/aiperf/dataset/loader/mooncake_trace.py

from concurrent.futures import ProcessPoolExecutor
from functools import partial

class MooncakeTraceDatasetLoader:

    def load_dataset_with_tokens(
        self,
    ) -> tuple[dict[str, list[MooncakeTrace]], dict[int, list[int]]]:
        """Load Mooncake trace data and pre-compute full token arrays.

        Returns:
            Tuple of (session_data, hash_id_to_token_array mapping)
        """
        data: dict[str, list[MooncakeTrace]] = defaultdict(list)
        all_hash_ids: set[int] = set()

        with open(self.filename) as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue

                trace_data = MooncakeTrace.model_validate_json(line)

                if (
                    trace_data.timestamp is not None
                    and not self._timestamp_within_offsets(trace_data.timestamp)
                ):
                    self._skipped_traces += 1
                    continue

                session_id = trace_data.session_id or str(uuid.uuid4())
                data[session_id].append(trace_data)

                if trace_data.hash_ids:
                    all_hash_ids.update(trace_data.hash_ids)

        # Pre-compute FULL token arrays for all hash_ids
        token_mapping = self._precompute_token_arrays(all_hash_ids)

        return data, token_mapping

    def _precompute_token_arrays(self, hash_ids: set[int]) -> dict[int, list[int]]:
        """Pre-compute full token arrays for all hash_ids.

        Warning: Memory usage is O(n × block_size) where n = len(hash_ids).
        For 100k hash_ids with block_size=512, this uses ~200MB.
        """
        corpus = self.prompt_generator._tokenized_corpus
        corpus_size = len(corpus)
        block_size = self.prompt_generator.config.input_tokens.block_size
        corpus_rng = rng.derive("dataset.prompt.corpus.precompute")

        # Optional: add block separator token
        sep_token = self.prompt_generator.tokenizer.block_separation_token_id

        mapping = {}
        for hash_id in sorted(hash_ids):  # sorted for determinism
            start_idx = corpus_rng.randrange(corpus_size)

            # Extract tokens with wraparound
            end_idx = start_idx + block_size
            if end_idx <= corpus_size:
                tokens = corpus[start_idx:end_idx]
            else:
                tokens = corpus[start_idx:] + corpus[:end_idx - corpus_size]

            # Add separator token if available
            if sep_token is not None:
                tokens = [sep_token] + tokens[:-1]

            mapping[hash_id] = tokens

        self.debug(
            lambda: f"Pre-computed token arrays for {len(mapping):,} unique hash_ids "
            f"(~{len(mapping) * block_size * 4 / 1024 / 1024:.1f} MB)"
        )
        return mapping

    def convert_to_conversations_parallel_tokens(
        self,
        data: dict[str, list[MooncakeTrace]],
        token_mapping: dict[int, list[int]],
        num_workers: int = None
    ) -> list[Conversation]:
        """Convert traces to conversations using pre-computed token arrays.

        Workers only need tokenizer for decoding - no corpus, no RNG.
        """
        num_workers = num_workers or min(os.cpu_count() or 4, 8)
        sessions = list(data.items())

        chunk_size = max(1, len(sessions) // num_workers)
        session_chunks = [
            sessions[i:i + chunk_size]
            for i in range(0, len(sessions), chunk_size)
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            worker_fn = partial(
                _process_session_chunk_tokens,
                token_mapping=token_mapping,
                tokenizer_name=self.prompt_generator.tokenizer.model_name,
                block_size=self.prompt_generator.config.input_tokens.block_size,
            )
            results = list(executor.map(worker_fn, session_chunks))

        return [conv for chunk_result in results for conv in chunk_result]


def _process_session_chunk_tokens(
    session_chunk: list[tuple[str, list[MooncakeTrace]]],
    token_mapping: dict[int, list[int]],
    tokenizer_name: str,
    block_size: int,
) -> list[Conversation]:
    """Worker function using pre-computed token arrays.

    This is the simplest worker implementation:
    - No corpus needed
    - No RNG needed
    - Just concatenate tokens and decode
    """
    tokenizer = Tokenizer.from_pretrained(tokenizer_name)

    conversations = []
    for session_id, traces in session_chunk:
        conversation = Conversation(session_id=session_id)
        for trace in traces:
            if trace.text_input is not None:
                prompt = trace.text_input
            else:
                prompt = _generate_prompt_from_tokens(
                    hash_ids=trace.hash_ids or [],
                    num_tokens=trace.input_length,
                    token_mapping=token_mapping,
                    block_size=block_size,
                    tokenizer=tokenizer,
                )

            turn = Turn(
                timestamp=trace.timestamp,
                delay=trace.delay,
                texts=[Text(name="text", contents=[prompt])],
                max_tokens=trace.output_length,
            )
            conversation.turns.append(turn)
        conversations.append(conversation)

    return conversations


def _generate_prompt_from_tokens(
    hash_ids: list[int],
    num_tokens: int,
    token_mapping: dict[int, list[int]],
    block_size: int,
    tokenizer: Tokenizer,
) -> str:
    """Generate prompt by concatenating pre-computed token arrays.

    Simplest possible implementation - just lookup, slice, concatenate, decode.
    """
    final_tokens = []

    for i, hash_id in enumerate(hash_ids):
        tokens = token_mapping[hash_id]

        # For the last hash_id, may need fewer tokens
        if i == len(hash_ids) - 1:
            needed = num_tokens - (len(hash_ids) - 1) * block_size
            tokens = tokens[:needed]

        final_tokens.extend(tokens)

    return tokenizer.decode(final_tokens, skip_special_tokens=False)
```

**Additional changes needed:**
1. Add new `load_dataset_with_tokens()` method (non-breaking)
2. Consider memory limits and add warnings for large hash_id counts
3. Potentially use shared memory (e.g., `multiprocessing.shared_memory`) to avoid copying token arrays to each worker
4. May need to chunk token_mapping for very large datasets

**Shared Memory Optimization (optional):**

```python
import multiprocessing.shared_memory as shm
import numpy as np

def _create_shared_token_mapping(token_mapping: dict[int, list[int]]) -> tuple[shm.SharedMemory, dict]:
    """Convert token mapping to shared memory for zero-copy worker access."""
    # Flatten all tokens into a single numpy array
    all_tokens = []
    index_info = {}  # hash_id -> (start_offset, length)

    offset = 0
    for hash_id in sorted(token_mapping.keys()):
        tokens = token_mapping[hash_id]
        all_tokens.extend(tokens)
        index_info[hash_id] = (offset, len(tokens))
        offset += len(tokens)

    # Create shared memory
    arr = np.array(all_tokens, dtype=np.int32)
    shared = shm.SharedMemory(create=True, size=arr.nbytes)
    shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shared.buf)
    shared_arr[:] = arr[:]

    return shared, index_info
```

---

## Final Decision and Recommendation

### Recommendation: Finish HashIdRandomGenerator (Approach A)

**Rationale:**

1. **Sunk cost matters here** - ~200 lines of working, tested code exists. The HashIdRandomGenerator class is complete and the integration is done. Throwing this away to implement pre-compute gains little.

2. **The hard work is identical** - All three approaches require solving:
   - Process spawning and management
   - Tokenizer instantiation per worker
   - Result collection

   The RNG strategy is the *easy* part. Neither pre-compute approach simplifies the actual parallelization significantly.

3. **Simpler worker interface** - Workers need only `base_seed` (8 bytes) vs `Dict[int, int]` (~0.8 MB for 100k hash_ids) vs `Dict[int, list[int]]` (~200 MB for 100k hash_ids).

4. **Future flexibility** - If streaming traces are ever needed, HashIdRandomGenerator handles it without changes. Pre-compute approaches would require redesign.

5. **Cleaner separation of concerns** - `load_dataset()` stays focused on loading. RNG concerns stay in the generator. Pre-compute would mix these responsibilities.

6. **Scalability** - HashIdRandomGenerator scales to any number of hash_ids with O(1) memory. Pre-compute tokens would struggle with large trace files.

### Approach Comparison Summary

| Criteria | Winner | Notes |
|----------|--------|-------|
| Implementation effort | **HashIdRandomGenerator** | Already done |
| Memory efficiency | **HashIdRandomGenerator** | O(1) vs O(n) vs O(n × block_size) |
| Worker simplicity | **Pre-compute Tokens** | No corpus, no RNG needed |
| Scalability | **HashIdRandomGenerator** | Handles millions of hash_ids |
| Debugging | **Pre-compute Tokens** | Full inspection upfront |
| Serialization | **HashIdRandomGenerator** | 8 bytes vs potentially 200MB |

### When to reconsider Pre-compute Indices (Approach B)

- If SHA256 overhead becomes measurable in profiling (unlikely)
- If you need to inspect/log all indices upfront for debugging
- If the codebase moves to a model where all hash_ids are always known at a central coordinator

### When to reconsider Pre-compute Tokens (Approach C)

- If the number of unique hash_ids is small and bounded (e.g., < 10,000)
- If worker startup time is critical and you want to avoid corpus loading per worker
- If you're willing to use shared memory to avoid serialization overhead
- If simplicity of worker logic is the primary concern

**Note:** Approach C becomes more attractive if combined with shared memory (see optional implementation above), which would eliminate the serialization overhead. However, this adds complexity that may negate the "simplicity" benefit.

### Immediate Next Steps

1. **Fix test bug** - `tests/test_mooncake_trace_reproducibility.py:155` references undefined `test_traces`

2. **Add parallel conversion method** - Implement `convert_to_conversations_parallel()` as shown in Implementation A

3. **Test reproducibility** - Verify sequential vs parallel produces identical results

4. **Benchmark** - Compare single-threaded vs parallel performance on real trace files

---

## Appendix: Decision Matrix

Score each approach 1-5 (5 = best) for your priorities, multiply by weight, sum for final score:

| Criteria | Weight | HashIdRNG | Pre-compute Idx | Pre-compute Tokens |
|----------|--------|-----------|-----------------|-------------------|
| Implementation effort | __ | 5 | 2 | 2 |
| Memory efficiency | __ | 5 | 4 | 1 |
| Worker simplicity | __ | 3 | 4 | 5 |
| Scalability | __ | 5 | 4 | 2 |
| Debugging ease | __ | 2 | 4 | 5 |
| Serialization overhead | __ | 5 | 4 | 1 |
| **TOTAL** | | | | |

Fill in weights based on your priorities. Higher total = better fit for your use case.
