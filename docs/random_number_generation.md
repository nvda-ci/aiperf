<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Random Number Generation System

**Quick Links:** [User Guide](#user-guide) • [Developer Guide](#developer-guide) • [Pitfalls](#common-pitfalls) • [Reference](#reference)

---

## Overview

AIPerf's RNG system provides **perfect reproducibility** with **complete component isolation**. One seed controls all randomness, while components remain independent.

**Core guarantees:**
- Same seed + same config = identical results (across runs, machines, OS)
- Changing one config affects only its corresponding randomness
- Adding features doesn't break existing reproducibility

---

## Architecture

### Three-Layer Design

```
User: --random-seed 42
        ↓
Manager: SHA256-based seed derivation
        ↓ rng.derive(identifier)
Components: 23 isolated RNGs
```

**Components:**
- **RandomGenerator**: Pure RNG class (methods: `choice()`, `shuffle()`, `random()`, etc.)
- **_RNGManager**: Internal seed derivation using SHA-256
- **Module API**: `rng.init()`, `rng.derive()`, `rng.reset()`

**Key insight:** Each component gets a deterministic seed derived from `SHA256(root_seed:identifier)`. This makes creation order irrelevant.

---

## User Guide

### Basic Usage

```bash
# Reproducible (deterministic)
aiperf --random-seed 42 [options...]

# Non-reproducible (uses OS randomness)
aiperf [options...]
```

### What the Seed Controls

| Feature | Randomized? | Example |
|---------|-------------|---------|
| Synthetic prompts | ✅ Token count, content position, prefixes | `--input-tokens-mean 500 --input-tokens-stddev 50` |
| Synthetic images | ✅ Dimensions, format, source selection | `--image-width-mean 512 --image-format random` |
| Synthetic audio/video | ✅ Duration, format params, sample data | `--audio-length-mean 5.0` |
| Dataset sampling | ✅ Random/shuffle strategies (not sequential) | `--dataset-sampling-strategy random` |
| Model selection | ✅ Random selection (not round-robin) | `--model-selection-strategy random` |
| Request timing | ✅ Poisson intervals (not constant), cancellation | `--request-rate-mode poisson` |

### Configuration Isolation

**Key principle:** Changing one config only affects its randomness.

```bash
# Baseline
aiperf --random-seed 42 --input-tokens-mean 500 --image-width-mean 512

# Change ONLY prompt length
aiperf --random-seed 42 --input-tokens-mean 1000 --image-width-mean 512
# → Prompts: different lengths, SAME corpus positions
# → Images: IDENTICAL (same dimensions, same sources)
```

**Why?** Each config maps to a specific RNG identifier. Prompts use `dataset.prompt.length`, images use `dataset.image.dimensions`. They're completely isolated.

---

## Developer Guide

### Getting an RNG

```python
from aiperf.common import random_generator as rng

class MyComponent:
    def __init__(self, config):
        # Derive in __init__ with unique identifier
        self._rng = rng.derive("dataset.mycomponent.feature")

    def process(self):
        # Use the stored RNG
        return self._rng.choice([1, 2, 3, 4, 5])
```

**Rules:**
1. Import as `rng` (not individual functions)
2. Call `rng.derive()` in `__init__` (not in methods)
3. Store the RNG as instance variable
4. Use unique identifier following naming convention

### Naming Convention

**Pattern:** `<module>.<component>.<aspect>`

```python
# Dataset generators
rng.derive("dataset.prompt.length")      # Prompt token count
rng.derive("dataset.image.dimensions")   # Image width + height
rng.derive("dataset.audio.duration")     # Audio length

# Timing
rng.derive("timing.request.cancellation")

# Composers
rng.derive("composer.turn.model_selection")
```

**All 23 identifiers** are listed in the [Reference](#reference) section.

### Split vs Combine Decision Tree

```
Is this feature independently configurable by users?
├─ YES → Separate RNG
└─ NO → Check if parameters are mathematically coupled
    ├─ YES (e.g., width ↔ height) → Same RNG
    └─ NO → Separate RNG
```

**Examples:**

```python
# ✅ Separate: Independent configs
self._length_rng = rng.derive("dataset.prompt.length")  # --input-tokens-mean
self._corpus_rng = rng.derive("dataset.prompt.corpus")  # (always random)

# ✅ Combined: Coupled parameters
self._dimensions_rng = rng.derive("dataset.image.dimensions")
width = self._dimensions_rng.sample(...)   # Coupled to height
height = self._dimensions_rng.sample(...)  # (aspect ratio)
```

### Adding New Features (Future-Proof Pattern)

**Scenario:** Add brightness control to images

```python
class ImageGenerator:
    def __init__(self):
        self._dimensions_rng = rng.derive("dataset.image.dimensions")
        self._format_rng = rng.derive("dataset.image.format")
        self._source_rng = rng.derive("dataset.image.source")
        # NEW: Add separate RNG for new feature
        self._effects_rng = rng.derive("dataset.image.effects")

    def generate(self):
        width = self._dimensions_rng.sample(...)    # ✅ Unaffected
        format = self._format_rng.choice(...)       # ✅ Unaffected
        brightness = self._effects_rng.uniform(...) # NEW - isolated!
        source = self._source_rng.choice(...)       # ✅ Unaffected
```

**Result:** Existing users with `--random-seed 42` still get same dimensions, format, and source. Only the new brightness varies.

**Why this matters:** Without separate RNGs, adding `brightness` would shift all subsequent random calls, changing which source image is selected!

---

## Common Pitfalls

### 1. Using Python's `random` Module

```python
# ❌ Wrong
import random
value = random.choice([1, 2, 3])  # Not reproducible!

# ✅ Correct
self._rng = rng.derive("dataset.mycomponent")
value = self._rng.choice([1, 2, 3])
```

### 2. Sharing RNG Between Components

```python
# ❌ Wrong
class ComponentB:
    def __init__(self, component_a):
        self._rng = component_a._rng  # Sharing!

# ✅ Correct
class ComponentB:
    def __init__(self):
        self._rng = rng.derive("dataset.component_b")
```

### 3. Creating RNG in Method Instead of `__init__`

```python
# ❌ Wrong - creates new RNG each call, always returns same value
def process(self):
    temp_rng = rng.derive("dataset.component")
    return temp_rng.random()  # Always 0.639...

# ✅ Correct
def __init__(self):
    self._rng = rng.derive("dataset.component")

def process(self):
    return self._rng.random()  # Advances state each call
```

### 4. Adding Operations Without New RNG

```python
# ❌ Wrong - breaks downstream randomness
def generate(self):
    width = self._rng.sample(...)
    brightness = self._rng.uniform(...)  # NEW - shifts everything!
    source = self._rng.choice(...)       # Different image now!

# ✅ Correct - isolated
def generate(self):
    width = self._dimensions_rng.sample(...)
    brightness = self._effects_rng.uniform(...)  # NEW - isolated!
    source = self._source_rng.choice(...)        # Unaffected
```

### 5. Conditional RNG Usage

```python
# ❌ Wrong - order dependent
if config.feature_enabled:
    self._rng.random()  # Sometimes called, sometimes not
b = self._rng.random()  # Gets different value!

# ✅ Correct
if config.feature_enabled:
    self._feature_rng.random()
b = self._other_rng.random()  # Always same value
```

---

## Reference

### All RNG Identifiers (23)

**Dataset (14)**
```python
# Prompts
"dataset.prompt.length"        # Token count
"dataset.prompt.corpus"        # Content position
"dataset.prompt.prefix"        # Prefix selection

# Images
"dataset.image.dimensions"     # Width + height (coupled)
"dataset.image.format"         # PNG/JPEG selection
"dataset.image.source"         # Source image

# Audio
"dataset.audio.duration"       # Length
"dataset.audio.format"         # Sample rate + bit depth
"dataset.audio.data"           # Audio samples

# Video
"dataset.video.dimensions"     # Dimensions
"dataset.video.format"         # Format params
"dataset.video.data"           # Frame data

# Samplers & Loaders
"dataset.sampler.random_choice"  # Random sampling
"dataset.sampler.shuffle"        # Shuffle sampling
"dataset.loader.random_pool"     # Random pool loader
"dataset.loader.sharegpt"        # ShareGPT loader
```

**Timing (2)**
```python
"timing.request.cancellation"      # Cancellation decisions
"timing.request.poisson_interval"  # Poisson intervals
```

**Composer (4)**
```python
"composer.turn.model_selection"    # Model selection
"composer.turn.max_tokens"         # max_tokens sampling
"composer.conversation.turn_count" # Number of turns
"composer.conversation.turn_delay" # Turn delays
```

**Models (1)**
```python
"models.sequence.distribution"     # ISL/OSL sampling
```

### Module API

```python
from aiperf.common import random_generator as rng

# Initialize once at startup (bootstrap.py)
rng.init(seed: int | None)

# Derive component RNGs (in __init__)
my_rng = rng.derive(identifier: str) -> RandomGenerator

# Reset for testing
rng.reset()
```

### RandomGenerator Methods

**Selection:** `choice()`, `choices()`, `sample()`, `numpy_choice()`
**Integers:** `randint()`, `randrange()`, `integers()`
**Floats:** `random()`, `uniform()`, `random_batch()`
**Distributions:** `normal()`, `sample_normal()`, `sample_positive_normal()`, `sample_positive_normal_integer()`, `expovariate()`
**Array Ops:** `shuffle()`

---

## FAQ

**Q: What if I change the seed?**
A: All randomness changes. It's a global reset.

**Q: What if I change image width config but keep same seed?**
A: Image widths change. Everything else (heights, sources, prompts, audio) stays identical.

**Q: Can components be created in any order?**
A: Yes. SHA-256 derivation makes order irrelevant.

**Q: What if I add a new random feature?**
A: Create a new RNG with new identifier. Existing RNGs are unaffected. See "Adding New Features" section.

**Q: Why can't I construct `RandomGenerator` directly?**
A: Enforces proper seed derivation. Use `rng.derive()`.

**Q: Are RNGs thread-safe?**
A: No. Each thread/worker needs its own RNG instance.

**Q: What's the difference between `seed=42` and `seed=None`?**
A: `42` = deterministic/reproducible. `None` = OS randomness.

**Q: Will results change between AIPerf versions?**
A: Only if RNG identifiers or behavior changes (documented in release notes).

---

## Quick Reference Card

```python
# Import
from aiperf.common import random_generator as rng

# In __init__
self._rng = rng.derive("module.component.aspect")

# Use it
value = self._rng.choice([1, 2, 3])
count = self._rng.sample_positive_normal_integer(100, 10)

# DON'T
❌ RandomGenerator(seed=42)           # Blocked
❌ import random; random.choice(...)  # Not managed
❌ self._rng = other._rng             # Sharing
```

---

**Last Updated:** 2025-11-01 | **Version:** 1.0
