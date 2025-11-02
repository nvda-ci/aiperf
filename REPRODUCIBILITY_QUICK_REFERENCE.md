<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Reproducibility Quick Reference

## TL;DR

- **Single reproducibility option**: `--random-seed <value>`
- **When seed is set**: Everything with randomness becomes reproducible
- **When seed is not set**: Everything is non-deterministic

## Config Option Impact Matrix

### What RNG Controls

| Aspect | Config Options | RNG Used? | Deterministic Without Seed? |
|--------|---|---|---|
| **Prompt Length** | `--prompt-input-tokens-stddev` | Yes | No (only if stddev=0) |
| **Prompt Content** | (corpus selection) | Yes | No |
| **Image Dimensions** | `--image-width-stddev`, `--image-height-stddev` | Yes | No (only if stddev=0) |
| **Image Format** | `--image-format random` | Yes | No |
| **Audio Duration** | `--audio-length-stddev` | Yes | No (only if stddev=0) |
| **Audio Properties** | `--audio-depths`, `--audio-sample-rates` | Yes (selection) | No |
| **Dataset Order** | `--dataset-sampling-strategy random` | Yes | No |
| **Dataset Shuffle** | `--dataset-sampling-strategy shuffle` | Yes | No (reshuffles each pass) |
| **Dataset Sequential** | `--dataset-sampling-strategy sequential` | No | Yes (always same order) |
| **Turn Delays** | `--conversation-turn-delay-stddev` | Yes | No (only if stddev=0) |
| **Request Timing** | `--request-rate-mode poisson` | Yes | No |
| **Request Timing** | `--request-rate-mode constant` | No | Yes (fixed intervals) |
| **Request Cancellation** | `--request-cancellation-rate` | Yes | No |

## RNG Isolation: What Stays the Same When You Change Something

### Scenario 1: Change Image Dimensions
```bash
# Config A: no image generation
aiperf profile --prompt-input-tokens-mean 100 --random-seed 42

# Config B: add image generation with variation
aiperf profile --prompt-input-tokens-mean 100 \
               --image-width-stddev 50 \
               --random-seed 42
```
**Result**: Prompts are IDENTICAL, images are different

---

### Scenario 2: Change Request Timing Pattern
```bash
# Config A: constant rate
aiperf profile --request-rate 50 --request-rate-mode constant --random-seed 42

# Config B: poisson rate
aiperf profile --request-rate 50 --request-rate-mode poisson --random-seed 42
```
**Result**: Request timing changes, prompts/images stay IDENTICAL

---

### Scenario 3: Change Dataset Sampling
```bash
# Config A: sequential
aiperf profile --dataset-sampling-strategy sequential --random-seed 42

# Config B: random
aiperf profile --dataset-sampling-strategy random --random-seed 42
```
**Result**: Dataset order changes, prompt/image content stays IDENTICAL

---

## Component RNG Isolation

Each component has its own RNG stream derived from the root seed:

```
root_seed (--random-seed)
  ├─→ dataset.image_generator          (image dimensions, format, source)
  ├─→ dataset.audio_generator         (audio duration, bit depth, sample rate)
  ├─→ generator.prompt.length         (prompt token counts)
  ├─→ generator.prompt.corpus         (corpus token selection)
  ├─→ generator.prompt.prefix         (prefix prompt generation)
  ├─→ dataset.random_sampler          (random conversation selection)
  ├─→ dataset.shuffle_sampler         (dataset shuffling)
  ├─→ timing.request_rate_poisson     (poisson request intervals)
  └─→ timing.request_cancellation     (which requests to cancel)
```

**Key Insight**: Changing ONE component's RNG behavior does NOT affect OTHER components because each has isolated RNG stream.

---

## Common User Questions

### Q: If I set a seed, will results be identical on different machines?
**A**: Yes. Same `--random-seed` + same config = identical outputs across any machine, Python version, or OS (assuming same AIPerf version).

### Q: If I only change `--image-width-stddev`, do prompts change?
**A**: No. Prompts stay identical (unless you also changed prompt settings). Each component's randomness is isolated.

### Q: Is `--dataset-sampling-strategy sequential` deterministic without a seed?
**A**: Yes. Sequential iteration is deterministic regardless of seed. But the CONTENT of each dataset entry might still be random (if using synthetic data with variance).

### Q: What happens if I use `--request-rate-mode constant` without a seed?
**A**: Timing is deterministic (fixed intervals). Data generation and dataset selection still random. Use `--random-seed` for full reproducibility.

### Q: Does changing `--prompt-input-tokens-mean` affect images?
**A**: No. Prompt and image generation use independent RNG streams.

---

## Reproducibility Levels

### Level 1: Deterministic Config Only
```bash
aiperf profile \
  --prompt-input-tokens-mean 100 \
  --prompt-input-tokens-stddev 0 \      # No variation = deterministic
  --image-width-stddev 0 \               # No variation = deterministic
  --request-rate-mode constant \         # Fixed intervals = deterministic
  --dataset-sampling-strategy sequential # Sequential iteration = deterministic
  # Result: Fully deterministic without seed
```

### Level 2: Deterministic With Seed
```bash
aiperf profile \
  --prompt-input-tokens-mean 100 \
  --prompt-input-tokens-stddev 50 \     # With seed, variation is reproducible
  --image-width-stddev 50 \              # With seed, variation is reproducible
  --request-rate-mode poisson \          # With seed, random timing is reproducible
  --dataset-sampling-strategy random \   # With seed, random order is reproducible
  --random-seed 42                       # Makes everything reproducible
  # Result: Fully reproducible
```

### Level 3: Non-Deterministic (No Seed)
```bash
aiperf profile \
  --prompt-input-tokens-stddev 50 \
  --image-width-stddev 50 \
  --request-rate-mode poisson \
  --dataset-sampling-strategy random
  # Result: Non-deterministic, different every run
```

---

## Implementation Details for Developers

### How Seed Derivation Works
```python
root_seed = 42                              # User provides via --random-seed
identifier = "dataset.image_generator"

# Derive child seed
seed_string = f"{root_seed}:{identifier}"  # "42:dataset.image_generator"
child_seed = SHA256(seed_string)[0:8]      # First 8 bytes of hash as uint64

# Same identifier always produces same seed
rng.derive("dataset.image_generator")  # With seed=42 → always child_seed=X
rng.derive("dataset.image_generator")  # With seed=42 → always child_seed=X
```

### Where RNG is Initialized
File: `bootstrap.py` line 106
```python
rng.reset()
rng.init(user_config.input.random_seed)  # Initialize with --random-seed value
```

### Component Registration Pattern
```python
class MyComponent:
    def __init__(self):
        # Each component creates its own RNG in __init__
        self._rng = rng.derive("my.component.identifier")

    def generate(self):
        # All randomness goes through this component's RNG
        value = self._rng.choice([1, 2, 3])
        return value
```

---

## Testing Reproducibility

```bash
# Run 1
aiperf profile --model gpt-4 --random-seed 42 ... > run1.log

# Run 2 (same seed)
aiperf profile --model gpt-4 --random-seed 42 ... > run2.log

# Run 3 (different seed)
aiperf profile --model gpt-4 --random-seed 99 ... > run3.log

# Verify:
# - run1.log and run2.log should have identical synthetic data
# - run3.log should have different synthetic data (unless stddev=0)
```

---

## Architecture Benefits

1. **Single Reproducibility Knob**: Not requiring separate seeds per component
2. **No Call-Order Dependencies**: Seed derivation uses hash-based, not counter-based
3. **Component Isolation**: Changing one component doesn't ripple effects
4. **Cross-Machine Stability**: Same seed produces identical results anywhere
5. **Transparent to Users**: Appears as single `--random-seed` option

