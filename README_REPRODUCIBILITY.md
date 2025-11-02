<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Reproducibility Documentation

This directory contains comprehensive documentation on how reproducibility works in AIPerf and how user configurations map to random number generation (RNG) usage.

## Document Guide

### 1. **REPRODUCIBILITY_QUICK_REFERENCE.md** (START HERE)
   - **Length**: ~210 lines
   - **Purpose**: Quick answers to common questions
   - **Contains**:
     - TL;DR summary
     - Config option impact matrix (what RNG controls what)
     - Common Q&A
     - Reproducibility levels/examples
     - Testing reproducibility

   **Read this if**: You want quick answers about reproducibility

---

### 2. **REPRODUCIBILITY_MAPPING.md**
   - **Length**: ~358 lines
   - **Purpose**: Detailed mapping of config options to RNG components
   - **Contains**:
     - Visual RNG flow diagram
     - Component-by-component mapping (Prompts, Images, Audio, Dataset, Timing)
     - Component isolation matrix
     - Seed derivation formula
     - Example request tracing

   **Read this if**: You want to understand which config option affects which RNG

---

### 3. **REPRODUCIBILITY_ANALYSIS.md** (COMPREHENSIVE)
   - **Length**: ~562 lines
   - **Purpose**: Complete analysis for deep understanding
   - **Contains**:
     - User workflow and mental model
     - Detailed reproducibility guarantees
     - User expectations vs implementation
     - RNG component usage details
     - Initialization flow
     - Cross-unit interactions

   **Read this if**: You want comprehensive understanding of the entire reproducibility system

---

## Quick Start

### For Users

**How do I make my benchmarks reproducible?**

Add `--random-seed <value>` to your command:
```bash
aiperf profile \
  --model gpt-4 \
  --random-seed 42 \  # Add this line
  --prompt-input-tokens-mean 100 \
  --prompt-input-tokens-stddev 50 \
  ...
```

Same seed + same config = identical outputs every time, on any machine.

**What should I know?**
1. You only need ONE option: `--random-seed`
2. It affects ALL randomness (data generation, timing, dataset sampling)
3. Changing one config (e.g., image dimensions) doesn't affect other configs (e.g., prompts)

### For Developers

**How is reproducibility implemented?**

1. **Single entry point**: `bootstrap.py` line 106 calls `rng.init(user_seed)`
2. **Component isolation**: Each component derives its own RNG via `rng.derive(identifier)`
3. **Deterministic derivation**: Child seeds are SHA256-based, not counter-based

```python
# In your component:
class MyComponent:
    def __init__(self):
        self._rng = rng.derive("my.component.unique_id")  # Gets deterministic seed

    def generate(self):
        value = self._rng.choice([1, 2, 3])  # Reproducible if seed was set
        return value
```

**Key guarantees**:
- Same identifier always produces same child seed
- Order of initialization doesn't matter
- Cross-machine reproducibility guaranteed
- Each component's RNG is isolated from others

---

## Key Concepts

### What is a "Logical Unit"?

A logical unit is a conceptual grouping of configuration options that users think affects together:

1. **Input Data Content**: Prompt length/content, image dimensions, audio duration
2. **Input Selection Strategy**: How conversations are sampled (random, sequential, shuffle)
3. **Load Pattern**: Request rate, timing mode (constant vs poisson), cancellation
4. **Reproducibility Control**: `--random-seed` knob

### Component Isolation (Most Important)

When you change a config option, ONLY that component's output changes:

| Change | Affects | Stays Same |
|--------|---------|-----------|
| Image width stddev | Image dimensions | Prompts, timing, dataset order |
| Dataset sampling strategy | Conversation order | Prompt content, images, timing |
| Request rate mode (poisson) | Request timing | Prompts, images, dataset order |
| Random seed value | Everything changes deterministically | Config stays same |

### Reproducibility Matrix

Some configurations are deterministic WITHOUT needing a seed:

| Config | RNG Used? | Deterministic Without Seed? |
|--------|---|---|
| `--prompt-input-tokens-stddev 0` | No | Yes |
| `--prompt-input-tokens-stddev 50` | Yes | No |
| `--request-rate-mode constant` | No | Yes |
| `--request-rate-mode poisson` | Yes | No |
| `--dataset-sampling-strategy sequential` | No | Yes |
| `--dataset-sampling-strategy random` | Yes | No |

---

## Architecture Overview

### RNG Hierarchy

```
User provides: --random-seed VALUE
         |
         v
   Bootstrap initializes: rng.init(value)
         |
         +------------------+------------------+
         |                  |                  |
         v                  v                  v
   Image RNG        Prompt RNG         Dataset RNG
   (dimensions,     (lengths,          (selection
    format,         content,           order)
    source)         prefixes)
```

### Key Properties

1. **Global + Hierarchical**: One global RNG manager, each component gets its own child RNG
2. **Hash-Based**: Seed derivation uses SHA256(root_seed + identifier)
3. **Order-Independent**: Component initialization order doesn't affect results
4. **All-or-Nothing**: Either everything is deterministic (seed set) or nothing is (seed = None)

---

## Examples

### Example 1: Reproducible Variation

```bash
# Run 1
aiperf profile --random-seed 42 --image-width-stddev 50 ...
# Output: Image 1 (538px), Image 2 (496px), ...

# Run 2 (same seed, same config)
aiperf profile --random-seed 42 --image-width-stddev 50 ...
# Output: Image 1 (538px), Image 2 (496px), ... [IDENTICAL]

# Run 3 (different seed, same config)
aiperf profile --random-seed 99 --image-width-stddev 50 ...
# Output: Image 1 (512px), Image 2 (524px), ... [DIFFERENT]
```

### Example 2: Component Isolation

```bash
# Config A: With images
aiperf profile --random-seed 42 \
  --image-width-stddev 50 \
  --prompt-input-tokens-mean 100 ...

# Config B: Without images
aiperf profile --random-seed 42 \
  --prompt-input-tokens-mean 100 ...

# Result: Prompts are IDENTICAL
# (Image RNG doesn't affect prompt RNG)
```

### Example 3: Partial Determinism

```bash
# Fully deterministic (no seed needed)
aiperf profile \
  --prompt-input-tokens-stddev 0 \        # No variation
  --request-rate-mode constant \          # Fixed timing
  --dataset-sampling-strategy sequential  # Fixed order
# Result: Deterministic without seed

# Partially deterministic (seed needed for full reproducibility)
aiperf profile --random-seed 42 \
  --prompt-input-tokens-stddev 50 \       # Variable, now reproducible
  --request-rate-mode poisson \           # Random timing, now reproducible
  --dataset-sampling-strategy random      # Random order, now reproducible
# Result: Fully reproducible with seed
```

---

## Common Pitfalls

1. **Thinking one change affects everything**: Nope, each component is isolated
2. **Forgetting to set seed for randomness**: Set `--random-seed` if you want reproducibility with variation
3. **Assuming cross-version compatibility**: Same seed might produce different results across AIPerf versions
4. **Using multiple seeds**: AIPerf uses only ONE `--random-seed` for everything

---

## For More Information

- **Quick answers**: See `REPRODUCIBILITY_QUICK_REFERENCE.md`
- **Mapping details**: See `REPRODUCIBILITY_MAPPING.md`
- **Deep dive**: See `REPRODUCIBILITY_ANALYSIS.md`
- **Code**: `src/aiperf/common/random_generator.py` (main implementation)
- **Bootstrap**: `src/aiperf/common/bootstrap.py` (initialization)

