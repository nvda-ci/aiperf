<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf User Workflow and Reproducibility Analysis

## Executive Summary

AIPerf provides a single configuration option for reproducibility: `--random-seed`. When this seed is provided, ALL random operations across synthetic data generation, dataset sampling, timing, and request generation are deterministic and reproducible. The RNG system is global and hierarchical, with seed-derived child RNGs for each component.

---

## 1. USER-FACING CONFIGURATION OPTIONS

### Primary Reproducibility Control

**`--random-seed` (CLI Option)**
- **Description**: "The seed used to generate random values. Set to some value to make the synthetic data generation deterministic. It will use system default if not provided."
- **Config Location**: `input_config.py:InputConfig.random_seed`
- **Type**: `int | None`
- **Semantics**:
  - `None` (default): Non-deterministic RNG from OS entropy
  - Integer value (0 to 2^64-1): Deterministic reproducible RNG across all runs
  - Same seed guarantees identical output sequences across different machines/runs

---

## 2. CONFIGURATION OPTIONS THAT ARE AFFECTED BY RNG

### Synthetic Data Generation Options

#### Prompt Generation (Text Input)
```
--prompt-input-tokens-mean         # Mean input sequence length
--prompt-input-tokens-stddev       # Std dev for input length randomization
--prompt-output-tokens-mean        # Mean output sequence length
--prompt-output-tokens-stddev      # Std dev for output length randomization
--prompt-prefix-pool-size          # Size of K-V cache prefix pool
--prompt-prefix-length             # Tokens per prefix prompt
--seq-dist                         # Sequence distribution specification
```

**RNG Impact**:
- Token count sampling: Uses normal distribution sampling
- Corpus selection: Randomly selects tokens from Shakespeare corpus
- Prefix prompt generation: Each prefix is randomly generated from corpus

#### Image Generation (Synthetic Images)
```
--image-width-mean                 # Mean image width
--image-width-stddev               # Std dev of width
--image-height-mean                # Mean image height
--image-height-stddev              # Std dev of height
--image-format                     # Format (png, jpeg, random)
--image-batch-size                 # Batch size
```

**RNG Impact**:
- Width/height sampling: Normal distribution sampling for each image
- Format selection: If format is "random", randomly chooses between png/jpeg
- Source image selection: Randomly picks which source image to resize

#### Audio Generation (Synthetic Audio)
```
--audio-length-mean                # Mean audio duration (seconds)
--audio-length-stddev              # Std dev of duration
--audio-format                     # Format (wav or mp3)
--audio-depths                     # List of bit depths to choose from
--audio-sample-rates               # List of sample rates to choose from
--audio-num-channels               # Number of channels (1 or 2)
--audio-batch-size                 # Batch size
```

**RNG Impact**:
- Duration sampling: Normal distribution sampling
- Bit depth selection: Random choice from list
- Sample rate selection: Random choice from list
- Audio waveform generation: Random noise generation

#### Video Generation (Synthetic Video)
```
--video-duration                   # Duration per clip (seconds)
--video-fps                        # Frames per second
--video-width                      # Video width in pixels
--video-height                     # Video height in pixels
--video-synth-type                 # Synthetic generator type
--video-format                     # Video format
--video-codec                      # Video codec
--video-batch-size                 # Batch size
```

**RNG Impact**: Frame generation randomization depends on synth_type

### Dataset Sampling Options

```
--custom-dataset-type              # [single_turn, multi_turn, random_pool, mooncake_trace]
--dataset-sampling-strategy        # [sequential, random, shuffle]
--conversation-num                 # Number of unique conversations
--num-dataset-entries              # Number of unique dataset entries
--conversation-turn-mean           # Mean turns per conversation
--conversation-turn-stddev         # Std dev of turns
--conversation-turn-delay-mean     # Mean delay between turns (ms)
--conversation-turn-delay-stddev   # Std dev of turn delays
```

**RNG Impact**:
- **RandomSampler**: Randomly selects conversations with replacement
- **ShuffleSampler**: Shuffles dataset order, reshuffles when exhausted
- **SequentialSampler**: NO RNG impact - deterministic iteration
- Turn count sampling: Normal distribution for multi-turn
- Turn delay sampling: Normal distribution for inter-turn delays

### Timing/Load Generation Options

```
--request-rate                     # Requests per second
--request-rate-mode                # [constant, poisson]
--benchmark-duration               # Duration of benchmark (seconds)
--concurrency                      # Max concurrent requests
--request-count                    # Number of requests to send
--warmup-request-count             # Warmup requests before measurement
--request-cancellation-rate        # % of requests to cancel
--request-cancellation-delay       # Delay before cancellation
```

**RNG Impact**:
- **Poisson mode**: Exponentially-distributed inter-arrival times (uses RNG)
- **Constant mode**: Fixed interval timing (NO RNG impact)
- **Request cancellation**: Random selection of which requests to cancel

---

## 3. RNG MAPPING TO OUTPUT CHANGES

### Deterministic Hierarchy (with same seed)

```
GlobalRNG
  ├── dataset.image_generator
  ├── dataset.audio_generator
  ├── dataset.video_generator
  ├── generator.prompt.length
  ├── generator.prompt.corpus
  ├── generator.prompt.prefix
  ├── dataset.random_sampler
  ├── dataset.shuffle_sampler
  ├── timing.request_rate_poisson
  ├── timing.request_cancellation
  └── [other components]
```

### Example Scenario: User Changes Image Width Stddev

**Configuration A**: `--image-width-stddev 0`
```
Image 1: 512x256
Image 2: 512x256 (deterministic, no variation)
...
```

**Configuration B**: `--image-width-stddev 50` (same seed, same rest of config)
```
Image 1: 538x256 (randomly sampled from N(512, 50))
Image 2: 496x256 (next random value)
...
```

**Effect**:
- Image dimensions CHANGE (different RNG calls produce different dimensions)
- Prompt data REMAINS IDENTICAL (uses different RNG, "generator.prompt.*")
- Request timing REMAINS IDENTICAL (uses different RNG, "timing.*")
- Dataset sampling REMAINS IDENTICAL (uses different RNG, "dataset.*_sampler")

**Key Insight**: Changing a config that controls RNG stddev/mean affects ONLY that component's RNG stream, not others, because each component has its own derived RNG.

---

## 4. USER MENTAL MODEL: "Logical Units" from User Perspective

### What Is a Logical Unit?

A **logical unit** is a conceptual grouping of configuration that users think affects together. In AIPerf:

#### Logical Unit 1: "Input Data Content"
```
prompt-input-tokens-mean/stddev      → What text is sent
image-width/height-mean/stddev       → Image dimensions
image-format                         → Image encoding
audio-length-mean/stddev             → Audio duration
audio-sample-rates, audio-depths     → Audio characteristics
video-duration, fps, width/height    → Video characteristics
```

**User Expectation**: "When I change how my inputs are sized/formatted, the actual inputs sent change, but..."

#### Logical Unit 2: "Input Selection Strategy"
```
dataset-sampling-strategy            → How conversations picked
conversation-num                     → Conversation variety
--public-dataset vs --custom-dataset-type vs synthetic
```

**User Expectation**: "When I change how conversations are selected, the order/selection changes, but..."

#### Logical Unit 3: "Load Pattern"
```
request-rate                         → Traffic rate
request-rate-mode                    → Traffic distribution (constant vs poisson)
concurrency                          → Parallel requests
request-cancellation-rate/delay      → Request lifecycle
```

**User Expectation**: "When I change how I'm sending requests, the timing changes, but..."

#### Logical Unit 4: "Reproducibility Control"
```
--random-seed
```

**User Expectation**: "When I set a seed, everything with randomness becomes reproducible."

### Cross-Unit Interactions

Users expect that **changing one logical unit doesn't affect other units**:

| Change | Affects | Stays Same |
|--------|---------|-----------|
| Increase `image-width-stddev` | Image dimensions, output quality | Prompts, timing, dataset order |
| Change `dataset-sampling-strategy` to RANDOM | Conversation selection order | Prompt content, image dimensions, timing |
| Change `request-rate` to poisson | Request inter-arrival times | Prompt content, images, conversation order |
| Add `--random-seed 42` | EVERYTHING deterministic | Nothing changes, just now reproducible |
| Remove `--random-seed` | Everything now random | Becomes non-deterministic |

---

## 5. REPRODUCIBILITY GUARANTEES

### Explicit Guarantees

From `random_generator.py` docstring:
> "Same seed guarantees identical random sequences across program runs."

From `input_config.py`:
> "Set to some value to make the synthetic data generation deterministic."

### Implicit Guarantees (from architecture)

1. **Component Isolation**: Each component derives its own child RNG via `rng.derive(identifier)`. Same identifier always produces same seed derivation (SHA-256 hash based).

2. **Order Independence**: Child RNG creation is deterministic based on identifier, not call order. This means:
   ```
   rng.derive("dataset.image_generator")  # Always same seed
   rng.derive("generator.prompt.length")  # Always same seed
   ```
   regardless of initialization order.

3. **Cross-Run Stability**: Hash-based seed derivation ensures:
   ```
   Run 1: --random-seed 42 → Image RNG gets seed X
   Run 2: --random-seed 42 → Image RNG gets seed X  (identical)
   Machine A vs Machine B: Same seed X
   ```

4. **Thread Safety Note**: RandomGenerator instances are NOT thread-safe. Each thread/component should call `rng.derive()` to get its own instance.

### Documented Expectations

From documentation:
- `--random-seed 42` → "reproducible conversation patterns"
- `--request-rate-mode poisson` with `--random-seed 42` → "reproducible random patterns"
- `--request-rate-mode constant` → "precisely evenly-spaced intervals... highly reproducible" (deterministic, seed not needed)

### What's NOT Guaranteed

1. **Output length constraints**: "Output sequence length constraints (`--output-tokens-mean`) cannot be guaranteed unless you pass `ignore_eos` and/or `min_tokens`" (depends on inference server)

2. **Cross-version reproducibility**: Same seed might NOT produce identical results across AIPerf versions if RNG component identifiers change

3. **Dataset content**: For custom datasets (not synthetic), reproducibility is about selection order, not generation:
   - `--dataset-sampling-strategy sequential` → Same order every time (deterministic)
   - `--dataset-sampling-strategy random` + seed → Same random order every time (reproducible)
   - `--dataset-sampling-strategy shuffle` + seed → Same shuffle every time (reproducible)

---

## 6. CLI OPTIONS AND USER MENTAL MODEL

### Grouped by "Logical Unit"

#### Endpoint Configuration (No RNG)
```
--model-names, --endpoint-type, --url, --streaming, --api-key
```
These are deterministic routing - seed doesn't affect them.

#### Input Data Generation (Uses RNG)
```
Input Sequence Length (ISL)
  --prompt-input-tokens-mean (default: 550)
  --prompt-input-tokens-stddev (default: 0.0)
  --prompt-input-tokens-block-size (default: 512)

Output Sequence Length (OSL)
  --prompt-output-tokens-mean (default: None)
  --prompt-output-tokens-stddev (default: 0)

Image Input
  --image-width-mean (default: 0.0)
  --image-width-stddev (default: 0.0)
  --image-height-mean (default: 0.0)
  --image-height-stddev (default: 0.0)
  --image-batch-size (default: 1)
  --image-format (default: png)

Audio Input
  --audio-length-mean (default: 0.0)
  --audio-length-stddev (default: 0.0)
  --audio-format (default: wav)
  --audio-depths (default: [16])
  --audio-sample-rates (default: [16.0])
  --audio-num-channels (default: 1)
  --audio-batch-size (default: 1)

Video Input
  --video-duration, --video-fps, --video-width, --video-height
  --video-synth-type, --video-format, --video-codec, --video-batch-size

Prefix Prompts
  --prompt-prefix-pool-size (default: 0)
  --prompt-prefix-length (default: 0)
```

#### Dataset Selection (Uses RNG)
```
Conversation Configuration
  --conversation-num (default: not set)
  --num-dataset-entries (default: 100)
  --conversation-turn-mean (default: 1)
  --conversation-turn-stddev (default: 0)
  --conversation-turn-delay-mean (default: 0.0)
  --conversation-turn-delay-stddev (default: 0.0)
  --conversation-turn-delay-ratio (default: 1.0)

Dataset Input
  --public-dataset (choices: sharegpt)
  --custom-dataset-type (choices: single_turn, multi_turn, random_pool, mooncake_trace)
  --input-file
  --dataset-sampling-strategy (choices: sequential, random, shuffle)
```

#### Load Generation (Uses RNG for Poisson)
```
Load Generation
  --benchmark-duration
  --benchmark-grace-period
  --concurrency
  --request-rate
  --request-rate-mode (choices: constant, poisson) [default: poisson]
  --request-count (default: 10)
  --warmup-request-count (default: 0)
  --request-cancellation-rate (default: 0.0)
  --request-cancellation-delay (default: 0.0)
```

#### Reproducibility Control
```
--random-seed (default: None)
```

---

## 7. WHAT CHANGES AFFECT WHAT OUTPUTS

### Reproducibility Matrix

| Config Change | Synthetic Data | Dataset Order | Request Timing | Notes |
|---|---|---|---|---|
| `--image-width-stddev` | ✓ Changes | - | - | Only image dimensions change |
| `--prompt-input-tokens-stddev` | ✓ Changes | - | - | Only prompt lengths change |
| `--audio-length-mean` | ✓ Changes | - | - | Only audio durations change |
| `--dataset-sampling-strategy` | - | ✓ Changes | - | Only conversation order changes |
| `--conversation-turn-stddev` | - | - | ✓ Changes* | Multi-turn delays affected |
| `--request-rate-mode poisson` | - | - | ✓ Changes | Request intervals randomized |
| `--request-rate-mode constant` | - | - | - | No RNG, deterministic timing |
| `--random-seed X` | ✓ Fixed | ✓ Fixed | ✓ Fixed | Everything deterministic |
| Remove `--random-seed` | ✓ Random | ✓ Random | ✓ Random | Everything non-deterministic |

*Conversation turn delays use RNG even when creating dataset structure

---

## 8. COMPONENT-LEVEL RNG USAGE

### Image Generator
```python
self._rng = rng.derive("dataset.image_generator")

# Uses RNG for:
- Image format selection (if format is RANDOM)
- Image width sampling: sample_positive_normal_integer(mean, stddev)
- Image height sampling: sample_positive_normal_integer(mean, stddev)
- Source image selection: choice() from pre-loaded images
```

### Audio Generator
```python
self._rng = rng.derive("dataset.audio_generator")

# Uses RNG for:
- Audio length sampling: sample_normal(mean, stddev)
- Bit depth selection: choice() from configured list
- Sample rate selection: choice() from configured list
- Audio waveform generation: random noise generation
```

### Prompt Generator
```python
self._length_rng = rng.derive("generator.prompt.length")
self._corpus_rng = rng.derive("generator.prompt.corpus")
self._prefix_rng = rng.derive("generator.prompt.prefix")

# Uses RNG for:
- Token count sampling: sample_positive_normal_integer(mean, stddev)
- Corpus token selection: randint() to pick from corpus
- Prefix prompt generation: full prompts generated from corpus
```

### Dataset Samplers
```python
# RandomSampler
self._rng = rng.derive("dataset.random_sampler")
# Uses RNG for: choice() to pick conversation ID with replacement

# ShuffleSampler
self._rng = rng.derive("dataset.shuffle_sampler")
# Uses RNG for: shuffle() to reorder conversations

# SequentialSampler
# NO RNG - purely deterministic iteration
```

### Request Rate Generator (Poisson)
```python
self._rng = rng.derive("timing.request_rate_poisson")

# Uses RNG for:
- expovariate(lambd) to generate exponentially distributed intervals
```

### Request Cancellation
```python
self._rng = rng.derive("timing.request_cancellation")

# Uses RNG for:
- random() to determine if request should be cancelled
```

---

## 9. INITIALIZATION FLOW

### Bootstrap Process
```python
# bootstrap.py line 106
rng.reset()  # Clear any previous state
rng.init(user_config.input.random_seed)  # Initialize with user's seed
```

### RNG Initialization (random_generator.py)
```python
def init(seed: int | None) -> None:
    """Initialize global RNG manager.

    Args:
        seed: Root seed (0 to 2^64-1) for deterministic behavior,
              or None for non-deterministic behavior.
    """
    global _manager
    _manager = _RNGManager(seed)
```

### Seed Derivation
```python
def derive(identifier: str) -> RandomGenerator:
    """Derive a child RNG with deterministic seed from identifier.

    If root_seed is not None:
        seed_string = f"{root_seed}:{identifier}"
        child_seed = SHA256(seed_string.encode())[0:8] as uint64
        return RandomGenerator(child_seed)

    If root_seed is None:
        return RandomGenerator(None)  # Non-deterministic
    """
```

---

## 10. SUMMARY: USER REPRODUCIBILITY EXPECTATIONS

### What Users Should Expect

1. **Single Reproducibility Knob**: `--random-seed <value>` makes all randomness deterministic
   - Same seed + same config = identical output every time, on any machine
   - No seed = non-deterministic output

2. **Config Isolation**: Changing a parameter in one logical unit (e.g., image dimensions)
   - AFFECTS outputs from that unit (images look different)
   - DOES NOT AFFECT outputs from other units (prompts, timing, conversation order stay same)

3. **Component Independence**: Different components have independent RNG streams
   - Adjusting one component's randomization (e.g., stddev) doesn't affect other components
   - Each component's RNG is "sealed off" from others

4. **No Reproducibility Across Versions**: Same seed might produce different results across AIPerf versions if component identifiers change

5. **Partial Determinism**: Some options are deterministic without a seed:
   - `--request-rate-mode constant` = deterministic timing (no seed needed)
   - `--dataset-sampling-strategy sequential` = deterministic order (no seed needed)
   - These don't randomize regardless of seed setting

### Documented Reproducibility Patterns

From tutorials and docs:
```bash
# For reproducible synthetic data patterns
aiperf profile --random-seed 42 \
    --image-width-stddev 50 \
    --prompt-input-tokens-stddev 100 \
    ...

# For reproducible traffic patterns with randomness
aiperf profile --random-seed 42 \
    --request-rate-mode poisson \
    --request-rate 50 \
    ...

# For reproducible request cancellation
aiperf profile --random-seed 42 \
    --request-cancellation-rate 0.1 \
    --request-cancellation-delay 0.5 \
    ...
```

---

## KEY ARCHITECTURAL INSIGHTS

1. **No Configuration Inheritance**: Changing one config parameter doesn't have "ripple effects" on unrelated outputs because RNGs are component-scoped

2. **Hash-Based Determinism**: All determinism is based on SHA-256(root_seed + identifier), meaning:
   - No need to track call ordering
   - Parallel initialization possible (though not done currently)
   - Reproducible across Python versions, machines, architectures

3. **Graceful Degradation**: If seed is None:
   - All components still work
   - Results are just non-deterministic
   - No "partially seeded" state (all-or-nothing design)

4. **User Mental Model Alignment**: The single `--random-seed` knob matches how users think about reproducibility
   - Not requiring separate seeds for each component
   - Each change is transparent and isolated
   - Simple and predictable behavior

